"""Alerting sink --- the malicious-path endpoint (Phase 6, Section III-A, gap G1).

PATH DISJOINTNESS BY CONSTRUCTION: this handler holds NO reference whatsoever to the cloud
transport. It is not constructed with, and is never given access to, a CloudTransport. The
guarantee is therefore structural, not merely logical: there is no code path from a malicious
verdict to the cloud channel because the object that handles malicious verdicts cannot reach
that channel. ``tests/test_path_disjointness.py`` asserts that no AlertSink references a
cloud transport and that zero encrypted payloads are emitted during a malicious-only run.

Do NOT add a CloudTransport (or anything that can encrypt/transmit toward the cloud) to this
class --- doing so would silently break the core guarantee of the architecture.

Alerts are kept in an in-process log (a list of dicts), which is enough for the demo/mock scope
(A6) this project targets -- a real deployment's alert queue (paging, a SIEM, etc.) is exactly
the kind of production integration this project does not build.
"""

from __future__ import annotations


class AlertSink:
    """Receives malicious-verdict payloads. Deliberately constructed with no cloud transport."""

    def __init__(self) -> None:
        self.alert_count = 0
        self._alerts: list[dict[str, str]] = []

    def raise_alert(self, edge_id: str, reason: str) -> None:
        """Record/emit an alert for a malicious verdict. No cloud transport is reachable here."""
        if not edge_id:
            raise ValueError("edge_id must be non-empty")
        self.alert_count += 1
        self._alerts.append(
            {"sequence": str(self.alert_count), "edge_id": edge_id, "reason": reason}
        )

    def alerts(self) -> list[dict[str, str]]:
        """The alert log, oldest first -- for demo narration and test inspection."""
        return list(self._alerts)
