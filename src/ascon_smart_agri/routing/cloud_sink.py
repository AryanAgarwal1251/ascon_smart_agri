"""Mock cloud receiver --- the benign-path endpoint (Phase 6, Sections III-A, III-G, A6).

Receives benign-verdict telemetry, reads the edge identifier and monotonic counter from the
routing metadata, and accepts it after an edge-identity check and replay-window check. A mock
only (A6): no real cloud analytics.

    IMPLEMENTATION DEVIATION FROM THE DESIGN PAPER (flagged per CLAUDE.md golden rule 1, decided
    with the user; see ``docs/design_paper.md``'s "Implementation deviation" section). This
    receiver no longer decrypts anything -- Eq. (26)'s tag-verification step is gone from this
    path, since cloud-payload confidentiality/integrity is now assumed to be handled by
    mechanisms outside this codebase. What remains is exactly what does not depend on
    encryption: matching the cleartext ``edge_id`` against the authenticated-at-the-time
    ``edge_id`` the metadata carries, and the replay window. See
    ``crypto/ascon_aead.py``/``federated/crypto.py`` for where Ascon-AEAD128 now lives instead
    (Channel 3, the federated weight transport).

Implementation notes:

* **``send_plaintext`` returns ``None``, matching the ``CloudTransport`` Protocol exactly** -- a
  real transport is fire-and-forget from the sender's side, and the router must not depend on a
  return value that only a mock can give. Outcomes are observable only through this mock's OWN
  instance state (``received_count``, ``rejected_count``, ``accepted_payloads``), which is not
  part of the Protocol and exists purely for test/demo inspection.
* **Malformed metadata is rejected, not allowed to crash the receiver.** ``AssociatedData.
  from_bytes`` raises :class:`ValueError` on truncated/malformed input; that is caught here and
  counted as a rejection, since a real network can deliver corrupted framing without any
  adversary being involved.
* **A parsed metadata's ``edge_id`` must match the cleartext ``edge_id`` parameter.** Without
  this check, metadata built for one edge but delivered under another edge's identity would be
  accepted under the wrong identity. In practice this can only happen if the caller mismatches
  the two, but the check costs nothing and closes the gap structurally.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..crypto.ascon_aead import AssociatedData
from .replay_guard import ReplayGuard


@runtime_checkable
class CloudTransport(Protocol):
    """The benign-path transport interface. The malicious path must never hold one of these."""

    def send_plaintext(self, edge_id: str, metadata: bytes, payload: bytes) -> None:
        """Transmit a benign-verdict payload toward the cloud receiver."""
        del edge_id, metadata, payload  # interface declaration; see MockCloudReceiver


class MockCloudReceiver:
    """In-process mock implementing :class:`CloudTransport` for the benign path."""

    def __init__(self) -> None:
        self._replay_guard = ReplayGuard()
        # Counts payloads accepted; the disjointness test asserts this stays 0 during a
        # malicious-only run.
        self.received_count = 0
        self.rejected_count = 0
        self.accepted_payloads: list[bytes] = []

    def send_plaintext(self, edge_id: str, metadata: bytes, payload: bytes) -> None:
        """Accept a benign-verdict payload (mock)."""
        try:
            ad = AssociatedData.from_bytes(metadata)
        except ValueError:
            self.rejected_count += 1  # malformed routing metadata
            return

        if ad.edge_id != edge_id:
            self.rejected_count += 1  # metadata's edge_id disagrees with the transport param
            return

        if not self._replay_guard.check(ad.edge_id, ad.device_id, ad.counter):
            self.rejected_count += 1
            return

        self.received_count += 1
        self.accepted_payloads.append(payload)
