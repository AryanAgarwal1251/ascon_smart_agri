"""Mock cloud receiver --- the benign-path endpoint (Phase 6, Sections III-A, III-G, A6).

Receives Ascon-protected telemetry, reads the edge identifier from the associated data,
selects the decryption key, and returns the plaintext only if the tag verifies (Eq. 26). A
mock only (A6): no real cloud analytics.

Implementation notes:

* **Verification order is fixed and load-bearing**: decrypt/verify (Eq. 26) FIRST, replay check
  SECOND, never the reverse. A message whose tag fails must never touch replay state, or a
  forged counter could poison the window for a legitimate later message. See
  ``routing/replay_guard.py``.
* **``send_encrypted`` returns ``None``, matching the ``CloudTransport`` Protocol exactly** --
  a real transport is fire-and-forget from the sender's side, and the router must not depend on
  a return value that only a mock can give. Outcomes are observable only through this mock's
  OWN instance state (``received_count``, ``rejected_count``, ``accepted_payloads``), which is
  not part of the Protocol and exists purely for test/demo inspection.
* **``keys`` is a DEMO key store, not production key management** (explicitly out of scope --
  the paper's own exclusions list names "production key management and device attestation").
  It is a plain ``edge_id -> 16-byte key`` mapping the caller constructs and is responsible for
  keeping out of version control (III-J3); this class does not generate, rotate, or persist
  keys.
* **A decrypted AD's `edge_id` must match the cleartext `edge_id` parameter.** The AEAD only
  guarantees the AD bytes are authentic; it says nothing about which channel/key selector was
  used to reach this call. Without this check, a message routed (or replayed) under the wrong
  edge's key selector but somehow verifying would be accepted under the wrong identity. In
  practice this can only happen if the caller mismatches `edge_id` against the key it selected,
  but the check costs nothing and closes the gap structurally rather than by caller discipline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..crypto.ascon_aead import AsconAEAD128, AssociatedData
from .replay_guard import ReplayGuard


@runtime_checkable
class CloudTransport(Protocol):
    """The benign-path transport interface. The malicious path must never hold one of these."""

    def send_encrypted(
        self, edge_id: str, nonce: bytes, associated_data: bytes, blob: bytes
    ) -> None:
        """Transmit an Ascon-protected payload toward the cloud receiver."""
        del edge_id, nonce, associated_data, blob  # interface declaration; see MockCloudReceiver


class MockCloudReceiver:
    """In-process mock implementing :class:`CloudTransport` for the benign path."""

    def __init__(self, keys: dict[str, bytes]) -> None:
        # DEMO keys only -- see the module docstring. Never committed (III-J3).
        self._keys = keys
        self._replay_guard = ReplayGuard()
        # Counts encrypted payloads accepted; the disjointness test asserts this stays 0
        # during a malicious-only run.
        self.received_count = 0
        self.rejected_count = 0
        self.accepted_payloads: list[bytes] = []

    def send_encrypted(
        self, edge_id: str, nonce: bytes, associated_data: bytes, blob: bytes
    ) -> None:
        """Accept and verify an encrypted payload (mock)."""
        key = self._keys.get(edge_id)
        if key is None:
            self.rejected_count += 1  # unknown edge_id: no key to attempt verification with
            return

        plaintext = AsconAEAD128(key).decrypt(nonce, associated_data, blob)
        if plaintext is None:
            self.rejected_count += 1  # tag failed (Eq. 26): tampered ciphertext or AD
            return

        ad = AssociatedData.from_bytes(associated_data)
        if ad.edge_id != edge_id:
            self.rejected_count += 1  # authenticated edge_id disagrees with the transport param
            return

        # Replay check AFTER verification succeeds -- see the module docstring.
        if not self._replay_guard.check(ad.edge_id, ad.device_id, ad.counter):
            self.rejected_count += 1
            return

        self.received_count += 1
        self.accepted_payloads.append(plaintext)
