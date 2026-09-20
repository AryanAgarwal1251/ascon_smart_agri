"""Ascon-AEAD128 protection of the client<->server weight transport (Channel 3).

    IMPLEMENTATION DEVIATION FROM THE DESIGN PAPER (flagged per CLAUDE.md golden rule 1, decided
    with the user; see ``docs/design_paper.md``'s "Implementation deviation" section and
    ``crypto/ascon_aead.py``'s module docstring). The paper couples Ascon-AEAD128 to Channel 2
    (gateway-to-cloud telemetry, Eq. 5/25-29) and leaves Channel 3 (client<->aggregator model
    updates, Section I-B) explicitly unsolved: "federated learning addresses [reconstruction]
    only partially and [poisoning] not at all." This module moves the AEAD protection to Channel
    3 instead: every weight blob, in BOTH directions of every round (client upload and server
    broadcast), is encrypted on send and decrypted on receive. This still does not provide secure
    aggregation, does not defend against a curious-but-honest aggregator inspecting the plaintext
    it decrypts (assumption A4 stands), and does not defend against gradient inversion or a
    poisoned update -- those stay out of scope exactly as the paper places them. What this adds is
    confidentiality and tamper-evidence for the wire itself, matching Eq. (25)/(26)'s guarantee
    (``Dec`` returns bottom, never a corrupted state, on any modification).

Associated data, mirroring Eq. (27)'s tuple shape but for this channel:
    a = <client_id, round_index, direction, schema_version>
``direction`` (``"upload"`` or ``"broadcast"``) stops a ciphertext legitimately produced for one
leg of a round from being replayed as if it were the other leg; ``round_index`` gives ordering
protection analogous to Eq. (27)'s monotonic counter. The wire encoding reuses the same
length-prefixed (TLV) framing as ``crypto/ascon_aead.py``'s ``AssociatedData``, via the shared
helpers in ``crypto/_ad_wire.py``, for the same injectivity reasons (see that module's docstring).

Key model: one pre-shared 128-bit symmetric key per client<->server pair (an extension of
assumption A5, "Ascon keys are pre-shared out of band"), reused for both directions across all
rounds with fresh nonces every message -- chosen because both legs of a round are between the
same two parties, and it keeps the key material a compromised client cannot use to read another
client's traffic. Key provisioning is the caller's responsibility (``federated/server.py``
receives a ``client_keys: dict[int, bytes]`` mapping), mirroring the existing
``routing/cloud_sink.py`` precedent of "a plain mapping the caller constructs" rather than a new
key-management abstraction -- production key management is out of scope (III-J3).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final, Literal

from ..crypto import _ad_wire
from ..crypto.ascon_aead import AsconAEAD128, NonceRegistry
from .serialization import StateDict, deserialize_state, serialize_state

_AD_FMT_VERSION: Final = 0x01
Direction = Literal["upload", "broadcast"]
_DIRECTIONS: Final = ("upload", "broadcast")


class WeightIntegrityError(Exception):
    """Raised when a weight blob fails AEAD verification -- a corrupted round, not a routing
    decision. Unlike the telemetry path (which diverts a bad message to the alert sink),
    there is no alternate path for a federated round: the round cannot proceed on unverified
    parameters, so this is a hard failure by design."""


@dataclass(frozen=True)
class WeightAssociatedData:
    """The associated-data tuple for one weight blob: ``<client_id, round_index, direction,
    schema_version>``, mirroring ``crypto/ascon_aead.py``'s ``AssociatedData`` (Eq. 27) shape."""

    client_id: int
    round_index: int
    direction: Direction
    schema_version: str

    def to_bytes(self) -> bytes:
        """Length-prefixed, big-endian encoding (see the module docstring)."""
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {self.direction!r}")
        return b"".join(
            (
                struct.pack(">B", _AD_FMT_VERSION),
                _ad_wire.pack_u64("client_id", self.client_id),
                _ad_wire.pack_u64("round_index", self.round_index),
                _ad_wire.pack_str("direction", self.direction),
                _ad_wire.pack_str("schema_version", self.schema_version),
            )
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> WeightAssociatedData:
        """Inverse of :meth:`to_bytes`; ``from_bytes(to_bytes(x)) == x`` for any valid ``x``."""
        cursor = _ad_wire.Cursor(data)
        fmt_version = cursor.take_u8("fmt_version")
        if fmt_version != _AD_FMT_VERSION:
            raise ValueError(f"unknown AD fmt_version {fmt_version:#04x}")

        client_id = cursor.take_u64("client_id")
        round_index = cursor.take_u64("round_index")
        direction = cursor.take_str("direction")
        schema_version = cursor.take_str("schema_version")
        cursor.assert_exhausted()

        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
        return cls(
            client_id=client_id,
            round_index=round_index,
            direction=direction,  # type: ignore[arg-type]  # validated above
            schema_version=schema_version,
        )


def protect_state(
    cipher: AsconAEAD128,
    registry: NonceRegistry,
    state: StateDict,
    sequence_count: int,
    *,
    ad: WeightAssociatedData,
) -> tuple[bytes, bytes, bytes]:
    """Serialize ``state``/``sequence_count`` and encrypt it (Eq. 25 applied to Channel 3).

    Returns ``(nonce, ad_bytes, ciphertext)`` -- everything a receiver needs to call
    :func:`unprotect_state`. A fresh CSPRNG nonce is drawn per call (III-G3); reuse across calls
    under the same key would be caught by ``registry``.
    """
    plaintext = serialize_state(state, sequence_count)
    ad_bytes = ad.to_bytes()
    nonce = registry.issue(cipher.key)
    ciphertext = cipher.encrypt(nonce, ad_bytes, plaintext)
    return nonce, ad_bytes, ciphertext


def unprotect_state(
    cipher: AsconAEAD128, nonce: bytes, ad_bytes: bytes, ciphertext: bytes
) -> tuple[StateDict, int]:
    """Decrypt and deserialize a weight blob produced by :func:`protect_state` (Eq. 26).

    Raises :class:`WeightIntegrityError` if the tag fails to verify -- Eq. (26)'s bottom outcome,
    surfaced as a hard failure rather than returned as ``None``, since there is no alternate path
    for a federated round to take on unverified parameters (see the class docstring).
    """
    plaintext = cipher.decrypt(nonce, ad_bytes, ciphertext)
    if plaintext is None:
        raise WeightIntegrityError(
            "Ascon tag verification failed for a federated weight blob; refusing to aggregate "
            "unverified parameters"
        )
    return deserialize_state(plaintext)
