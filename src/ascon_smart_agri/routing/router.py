"""Verdict router --- implements the routing rule of Eq. (5) (Phase 6, Section III-A, gap G1).

The classifier verdict selects the data path:

    benign     -> encrypt with Ascon and hand to the CloudTransport (Pi_cloud)
    malicious  -> hand to the AlertSink (Pi_alert), which cannot reach the cloud

The two paths are disjoint by construction (see alert_sink.py). The router is the ONLY object
that holds the cloud transport; it never passes it to the alert sink.

TODO(Phase 6): implement route(); benign -> encrypt+send, malicious -> alert only.
"""

from __future__ import annotations

from ..crypto.ascon_aead import AsconAEAD128
from .alert_sink import AlertSink
from .cloud_sink import CloudTransport


class VerdictRouter:
    """Routes each telemetry item to exactly one disjoint path based on the verdict."""

    def __init__(self, cloud: CloudTransport, alert: AlertSink, cipher: AsconAEAD128) -> None:
        # The cloud transport lives ONLY here; it is never handed to `alert`.
        self._cloud = cloud
        self._alert = alert
        self._cipher = cipher

    def route(self, *, verdict_benign: bool, edge_id: str, payload: bytes) -> None:
        """Send benign payloads (encrypted) to the cloud, malicious ones to the alert sink."""
        del verdict_benign, edge_id, payload
        raise NotImplementedError("Phase 6: verdict routing not implemented yet.")
