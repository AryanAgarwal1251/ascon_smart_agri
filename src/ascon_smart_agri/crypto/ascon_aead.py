r"""Ascon-AEAD128 wrapper, NIST SP 800-232 (Phase 6, Section III-G).

    Enc:  (c, t) = Enc_ke(nu, a, p)                        (Eq. 25)
    Dec:  Dec_ke(nu, a, c, t) in {p, bottom}               (Eq. 26)

128-bit key, 128-bit nonce, 128-bit tag. Decryption returns bottom (a verification failure),
never plaintext, for any modified ciphertext OR associated data --- the property that matters
here, since a farm must not act on a tampered reading. ``tests/test_ascon_tamper.py`` mutates
one bit of ciphertext and one bit of associated data and asserts rejection in both cases.

Associated data (authenticated, not encrypted), Eq. (27):
    a = <edge_id, device_id, counter, schema_version>
The cloud receiver reads edge_id in the clear to select a decryption key; the monotonic
counter gives replay detection.

    SPEC EXTENSION (design paper, CLAUDE.md golden rule 1 waived by the user for this decision):
    Eq. (27) defines the associated-data *tuple* but does NOT specify a byte serialisation for it.
    Naive concatenation of the variable-length ``edge_id``/``device_id``/``schema_version`` strings
    is ambiguous --- e.g. ``edge_id="ab", device_id="c"`` and ``edge_id="a", device_id="bc"``
    concatenate to the same bytes, so a receiver could authenticate a reading against the wrong
    device's metadata. The resolved encoding is a **length-prefixed (TLV-style)** framing,
    injective and canonically parseable by construction (see
    ``docs/plans/phase6-ad-serialization.md``); fixed-width padding was rejected because
    truncation/padding silently re-introduce the cross-device collision and it needs id-length
    figures Phase 1 has not produced. :meth:`AssociatedData.to_bytes` /
    :meth:`AssociatedData.from_bytes` implement it, versioned by a leading ``fmt_version`` byte
    (``ad_encoding = "length-prefixed-v1"`` in :func:`backend_provenance`). The AEAD facade stays
    decoupled: :class:`AsconAEAD128` still operates on already-serialised ``bytes``, so callers
    pass ``associated_data.to_bytes()``. This is an intentional extension of the paper; reconcile
    with the design paper if it is later revised to define an AD serialisation.

Nonce discipline (III-G3): a fresh 16-byte nonce is drawn per message from a CSPRNG
(:func:`secrets.token_bytes`). :class:`NonceRegistry` accumulates every nonce issued per key so
a test can assert zero collisions (R6) --- nonce reuse under a fixed key is catastrophic and
unrecoverable.

VARIANT + CONFORMANCE (III-G1, risk R5): SP 800-232 Ascon-AEAD128 is NOT the pre-standard
Ascon-128/128a of Ascon v1.2 (revised initial values, little-endian formatting). The PyPI
``ascon`` package implements only the v1.2 variants (``Ascon-128``/``Ascon-128a``) and does not
expose ``Ascon-AEAD128`` at all, so it cannot pass the standard's KAT. Per the paper-consistent
fallback (III-G1: "no primitive implemented from scratch"), the backend is the OFFICIAL
reference implementation, vendored under ``crypto/_vendor/pyascon`` (see its ``PROVENANCE.md``)
and gated behind :func:`verify_kat_conformance` / ``tests/test_ascon_kat.py`` against the
official known-answer vectors before any integration. :func:`backend_provenance` returns the
package/commit/variant record for the run manifest (III-G1).
"""

from __future__ import annotations

import secrets
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ._vendor.pyascon import ascon_decrypt, ascon_encrypt

_VARIANT: Final = "Ascon-AEAD128"
_KEY_BYTES = 16
_NONCE_BYTES = 16
_TAG_BYTES = 16

# Associated-data (Eq. 27) length-prefixed encoding (docs/plans/phase6-ad-serialization.md).
_AD_ENCODING: Final = "length-prefixed-v1"
_AD_FMT_VERSION: Final = 0x01  # leading wire-format version byte; receivers reject unknown values
_U16_MAX: Final = 0xFFFF  # max UTF-8 byte length per string field (u16 length prefix)
_U64_MAX: Final = 2**64 - 1  # counter is a fixed-width u64

# Official SP 800-232 KAT vectors, committed under tests/ (see tests/kat/PROVENANCE.md). Resolved
# relative to the repo root: this file is src/ascon_smart_agri/crypto/ascon_aead.py.
_KAT_PATH = Path(__file__).resolve().parents[3] / "tests" / "kat" / "LWC_AEAD_KAT_128_128.txt"


class NonceReuseError(Exception):
    """Raised when a (key, nonce) pair is issued twice --- fatal for AEAD confidentiality (R6)."""


@dataclass(frozen=True)
class AssociatedData:
    """Authenticated-but-not-encrypted metadata, Eq. (27).

    :meth:`to_bytes` / :meth:`from_bytes` implement the resolved length-prefixed (TLV-style)
    serialisation (see the module docstring and ``docs/plans/phase6-ad-serialization.md``): an
    injective, canonical encoding so a reading can never authenticate against the wrong device's
    metadata. Callers pass ``associated_data.to_bytes()`` to :class:`AsconAEAD128`, which stays
    decoupled by operating on ``bytes``.

    REPLAY ENFORCEMENT IS OUT OF SCOPE FOR THIS MODULE. The paper (Section III-G2, near Eq. 27)
    notes that authenticating the monotonic ``counter`` "also gives replay detection", but the
    actual check --- tracking the last-seen counter and rejecting stale/duplicate messages --- is
    receiver/routing logic that belongs on the cloud-ingest path (``routing/cloud_sink.py``), not
    in the crypto core. This class only *carries* the counter in the authenticated AD and exposes
    it via :meth:`from_bytes`; the AEAD guarantees the counter is authentic (a tampered counter
    yields bottom), which is the precondition the receiver's replay window relies on. Do NOT add
    replay/window state here. The policy that is still open (per-key vs. per-device scope, window
    size, persistence across restarts) is deferred to ``docs/plans/phase6-replay-window.md``.
    """

    edge_id: str
    device_id: str
    counter: int  # monotonic; authenticated here, replay-checked on the receiver (see class doc)
    schema_version: str

    def to_bytes(self) -> bytes:
        """Serialize the Eq. (27) tuple to authenticated-data bytes (phase6-ad-serialization.md).

        Length-prefixed, big-endian, UTF-8:
        ``fmt_version:u8 || L(edge_id):u16 || edge_id || L(device_id):u16 || device_id ||
        counter:u64 || L(schema_version):u16 || schema_version``. Raises :class:`ValueError` if
        any string field exceeds 65535 UTF-8 bytes or if ``counter`` is out of u64 range. Never
        truncates.
        """
        if not 0 <= self.counter <= _U64_MAX:
            raise ValueError(f"counter must be in [0, 2**64), got {self.counter}")

        def prefixed(name: str, value: str) -> bytes:
            encoded = value.encode("utf-8")
            if len(encoded) > _U16_MAX:
                raise ValueError(
                    f"{name} is {len(encoded)} UTF-8 bytes, exceeds u16 max {_U16_MAX}"
                )
            return struct.pack(">H", len(encoded)) + encoded

        return b"".join(
            (
                struct.pack(">B", _AD_FMT_VERSION),
                prefixed("edge_id", self.edge_id),
                prefixed("device_id", self.device_id),
                struct.pack(">Q", self.counter),
                prefixed("schema_version", self.schema_version),
            )
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> AssociatedData:
        """Parse AD bytes back to the tuple; the receiver reads ``edge_id`` from it in the clear.

        Strict, single-valid-encoding parse: raises :class:`ValueError` on an unknown
        ``fmt_version``, input shorter than the declared lengths, or any trailing bytes after the
        last field. Guarantees ``from_bytes(to_bytes(x)) == x``.
        """
        view = memoryview(data)
        offset = 0

        def take(n: int, what: str) -> bytes:
            nonlocal offset
            if offset + n > len(view):
                raise ValueError(f"truncated AD: need {n} bytes for {what} at offset {offset}")
            chunk = bytes(view[offset : offset + n])
            offset += n
            return chunk

        (fmt_version,) = struct.unpack(">B", take(1, "fmt_version"))
        if fmt_version != _AD_FMT_VERSION:
            raise ValueError(f"unknown AD fmt_version {fmt_version:#04x}")

        def take_str(what: str) -> str:
            (length,) = struct.unpack(">H", take(2, f"{what} length"))
            return take(length, what).decode("utf-8")

        edge_id = take_str("edge_id")
        device_id = take_str("device_id")
        (counter,) = struct.unpack(">Q", take(8, "counter"))
        schema_version = take_str("schema_version")

        if offset != len(view):
            raise ValueError(f"trailing bytes after AD: {len(view) - offset} extra")
        return cls(
            edge_id=edge_id, device_id=device_id, counter=counter, schema_version=schema_version
        )


class NonceRegistry:
    """Accumulates every nonce issued under each key; used to assert zero collisions (R6)."""

    def __init__(self) -> None:
        self._issued: dict[bytes, set[bytes]] = {}

    def register(self, key: bytes, nonce: bytes) -> None:
        """Record a (key, nonce) pair, raising :class:`NonceReuseError` if already used.

        Nonce reuse under a fixed key is unrecoverable (III-G3), so this is a hard error, not a
        warning. Validates the nonce width to catch malformed callers early.
        """
        if len(nonce) != _NONCE_BYTES:
            raise ValueError(f"nonce must be {_NONCE_BYTES} bytes, got {len(nonce)}")
        used = self._issued.setdefault(key, set())
        if nonce in used:
            raise NonceReuseError(f"nonce {nonce.hex()} already issued under this key")
        used.add(nonce)

    def issue(self, key: bytes) -> bytes:
        """Draw a fresh CSPRNG nonce, register it under ``key``, and return it.

        This is the intended way to obtain a nonce per message (III-G3): drawn from
        :func:`secrets.token_bytes`, never ``random``. Registration guards against the
        astronomically unlikely CSPRNG collision.
        """
        nonce = secrets.token_bytes(_NONCE_BYTES)
        self.register(key, nonce)
        return nonce


class AsconAEAD128:
    """SP 800-232 Ascon-AEAD128 facade over the KAT-verified vendored reference backend."""

    def __init__(self, key: bytes) -> None:
        if len(key) != _KEY_BYTES:
            raise ValueError(f"key must be {_KEY_BYTES} bytes, got {len(key)}")
        self.key = key

    def encrypt(self, nonce: bytes, associated_data: bytes, plaintext: bytes) -> bytes:
        """Return ciphertext||tag for the given nonce, associated data, and plaintext (Eq. 25)."""
        if len(nonce) != _NONCE_BYTES:
            raise ValueError(f"nonce must be {_NONCE_BYTES} bytes, got {len(nonce)}")
        return ascon_encrypt(self.key, nonce, associated_data, plaintext, _VARIANT)

    def decrypt(self, nonce: bytes, associated_data: bytes, ciphertext: bytes) -> bytes | None:
        """Return the plaintext, or ``None`` (bottom) if verification fails (Eq. 26).

        Returns ``None`` --- never plaintext, never an exception --- for any tampered ciphertext
        or associated data, and for a ciphertext too short to carry a tag.
        """
        if len(nonce) != _NONCE_BYTES:
            raise ValueError(f"nonce must be {_NONCE_BYTES} bytes, got {len(nonce)}")
        if len(ciphertext) < _TAG_BYTES:
            return None
        return ascon_decrypt(self.key, nonce, associated_data, ciphertext, _VARIANT)


def _parse_kat(path: Path) -> list[dict[str, bytes]]:
    """Parse CAVP-style KAT records (Count/Key/Nonce/PT/AD/CT) into a list of field dicts."""
    vectors: list[dict[str, bytes]] = []
    current: dict[str, bytes] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            if current:
                vectors.append(current)
                current = {}
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name == "Count":
            continue
        current[name] = bytes.fromhex(value.strip())
    if current:
        vectors.append(current)
    return vectors


def verify_kat_conformance(kat_path: Path | None = None) -> bool:
    """Verify the selected backend against the official SP 800-232 known-answer vectors.

    This is the GATING check (R5): integration must not proceed unless this returns ``True``.
    Asserts, for every committed vector, that ``encrypt`` reproduces the expected ciphertext||tag
    and that ``decrypt`` recovers the plaintext.
    """
    path = kat_path if kat_path is not None else _KAT_PATH
    for vec in _parse_kat(path):
        key, nonce = vec["Key"], vec["Nonce"]
        ad, pt, ct = vec["AD"], vec["PT"], vec["CT"]
        cipher = AsconAEAD128(key)
        if cipher.encrypt(nonce, ad, pt) != ct:
            return False
        if cipher.decrypt(nonce, ad, ct) != pt:
            return False
    return True


def backend_provenance() -> dict[str, str]:
    """Return the Ascon backend identity for the run manifest (III-G1, R5).

    Records package/source, pinned commit, variant string, and KAT vector source so no headline
    number is reported without an auditable crypto-conformance record.
    """
    return {
        "backend": "vendored:meichlseder/pyascon (official reference implementation)",
        "source_url": "https://github.com/meichlseder/pyascon",
        "commit": "ed24e54abf9507d26fa49b46a56091570c7e743e",
        "variant": _VARIANT,
        "ad_encoding": _AD_ENCODING,
        "standard": "NIST SP 800-232",
        "kat_source": "https://github.com/ascon/ascon-c (crypto_aead/asconaead128)",
        "kat_source_commit": "446347f21b209f3921c65ece70027c366cbe1693",
        "kat_file": _KAT_PATH.name,
    }
