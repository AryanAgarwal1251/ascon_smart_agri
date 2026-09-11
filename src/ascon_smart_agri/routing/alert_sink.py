"""Alerting sink --- the malicious-path endpoint (Phase 6, Section III-A, gap G1).

PATH DISJOINTNESS BY CONSTRUCTION: this handler holds NO reference whatsoever to the cloud
transport. It is not constructed with, and is never given access to, a CloudTransport. The
guarantee is therefore structural, not merely logical: there is no code path from a malicious
verdict to the cloud channel because the object that handles malicious verdicts cannot reach
that channel. ``tests/test_path_disjointness.py`` asserts that no AlertSink references a
cloud transport and that zero encrypted payloads are emitted during a malicious-only run.

Do NOT add a CloudTransport (or anything that can encrypt/transmit toward the cloud) to this
class --- doing so would silently break the core guarantee of the architecture.

TODO(Phase 6): implement alert emission (log/queue), still with no cloud reference.
"""

from __future__ import annotations


class AlertSink:
    """Receives malicious-verdict payloads. Deliberately constructed with no cloud transport."""

    def __init__(self) -> None:
        self.alert_count = 0

    def raise_alert(self, edge_id: str, reason: str) -> None:
        """Record/emit an alert for a malicious verdict. No cloud transport is reachable here."""
        del edge_id, reason
        raise NotImplementedError("Phase 6: alert emission not implemented yet.")
