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

Nonce discipline (III-G3): a fresh 16-byte nonce is drawn per message from a CSPRNG.
:class:`NonceRegistry` accumulates every nonce issued per key so a test can assert zero
collisions (R6) --- nonce reuse under a fixed key is catastrophic and unrecoverable.

VARIANT + CONFORMANCE (III-G1, risk R5): SP 800-232 Ascon-AEAD128 is NOT the pre-standard
Ascon-128/128a of Ascon v1.2 (revised initial values, little-endian formatting). The widely
installed ``ascon`` PyPI package implements the v1.2 variants, so its conformance is treated
as UNVERIFIED until ``tests/test_ascon_kat.py`` passes against the official known-answer test
vectors, as a GATING check before any integration. The package name, version, and variant
string are recorded in the run manifest.

    DESIGN TENSION TO RESOLVE AT PHASE 6: III-G1 states "No primitive is implemented from
    scratch and no placeholder encryption is used at any stage." If the candidate library
    fails the KATs, the paper-consistent fallback is to bind the OFFICIAL REFERENCE
    implementation (treated as the reference), NOT to hand-roll the primitive here. Do not
    write a from-scratch Ascon in this module.

TODO(Phase 6): select the conformant backend behind the KAT gate; implement encrypt/decrypt,
the nonce registry, and manifest recording of the variant string.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssociatedData:
    """Authenticated-but-not-encrypted metadata, Eq. (27)."""

    edge_id: str
    device_id: str
    counter: int  # monotonic, for replay detection
    schema_version: str


class NonceRegistry:
    """Accumulates every nonce issued under each key; used to assert zero collisions (R6)."""

    def __init__(self) -> None:
        self._issued: dict[bytes, set[bytes]] = {}

    def register(self, key: bytes, nonce: bytes) -> None:
        """Record a (key, nonce) pair, raising if the nonce was already used under that key."""
        del key, nonce
        raise NotImplementedError("Phase 6: nonce registry not implemented yet.")


class AsconAEAD128:
    """SP 800-232 Ascon-AEAD128 facade over a KAT-verified backend."""

    def __init__(self, key: bytes) -> None:
        self.key = key

    def encrypt(self, nonce: bytes, associated_data: bytes, plaintext: bytes) -> bytes:
        """Return ciphertext||tag for the given nonce, associated data, and plaintext."""
        del nonce, associated_data, plaintext
        raise NotImplementedError("Phase 6: Ascon encryption not implemented yet.")

    def decrypt(self, nonce: bytes, associated_data: bytes, ciphertext: bytes) -> bytes | None:
        """Return the plaintext, or ``None`` (bottom) if verification fails (Eq. 26)."""
        del nonce, associated_data, ciphertext
        raise NotImplementedError("Phase 6: Ascon decryption not implemented yet.")


def verify_kat_conformance() -> bool:
    """Verify the selected backend against the official SP 800-232 known-answer vectors.

    This is the GATING check (R5): integration must not proceed unless this returns True.
    """
    raise NotImplementedError("Phase 6: KAT conformance check not implemented yet.")
