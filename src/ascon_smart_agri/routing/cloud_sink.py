"""Mock cloud receiver --- the benign-path endpoint (Phase 6, Sections III-A, III-G, A6).

Receives Ascon-protected telemetry, reads the edge identifier from the associated data,
selects the decryption key, and returns the plaintext only if the tag verifies (Eq. 26). A
mock only (A6): no real cloud analytics.

TODO(Phase 6): implement receive() with key selection + Ascon verification.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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

    def __init__(self) -> None:
        # Counts encrypted payloads accepted; the disjointness test asserts this stays 0
        # during a malicious-only run.
        self.received_count = 0

    def send_encrypted(
        self, edge_id: str, nonce: bytes, associated_data: bytes, blob: bytes
    ) -> None:
        """Accept and verify an encrypted payload (mock)."""
        del edge_id, nonce, associated_data, blob
        raise NotImplementedError("Phase 6: mock cloud receiver not implemented yet.")
