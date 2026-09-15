"""Unit tests for the alert sink (Phase 6, Section III-A, gap G1).

The structural no-cloud-reference guarantee is gating and lives in test_path_disjointness.py.
This covers the behavioural correctness of the alert log itself.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.routing.alert_sink import AlertSink


def test_raise_alert_increments_the_count() -> None:
    sink = AlertSink()

    sink.raise_alert("edge01", "malicious verdict")
    sink.raise_alert("edge01", "malicious verdict")

    assert sink.alert_count == 2


def test_alerts_are_logged_oldest_first_with_edge_id_and_reason() -> None:
    sink = AlertSink()

    sink.raise_alert("edge01", "reason A")
    sink.raise_alert("edge02", "reason B")

    log = sink.alerts()
    assert [a["edge_id"] for a in log] == ["edge01", "edge02"]
    assert [a["reason"] for a in log] == ["reason A", "reason B"]


def test_alerts_returns_a_copy_not_the_live_log() -> None:
    sink = AlertSink()
    sink.raise_alert("edge01", "x")

    log = sink.alerts()
    log.append({"sequence": "999", "edge_id": "forged", "reason": "y"})

    assert sink.alert_count == 1  # the mutation above did not reach internal state


def test_rejects_an_empty_edge_id() -> None:
    sink = AlertSink()

    with pytest.raises(ValueError, match="edge_id"):
        sink.raise_alert("", "reason")
