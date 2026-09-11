"""GATING TEST (G1) --- disjoint benign/malicious data paths (Section III-A).

Two guarantees:
  (a) STRUCTURAL (active now): the malicious-path handler (AlertSink) holds no reference to the
      cloud transport --- it is not even constructed with one. Checked by introspection.
  (b) BEHAVIOURAL (Phase 6): zero encrypted payloads are emitted during a malicious-only run.
"""

from __future__ import annotations

import inspect

import pytest

from ascon_smart_agri.routing.alert_sink import AlertSink
from ascon_smart_agri.routing.cloud_sink import CloudTransport, MockCloudReceiver


@pytest.mark.gating
def test_alert_sink_constructor_takes_no_transport() -> None:
    # AlertSink.__init__ must not accept a cloud/transport dependency: disjointness is
    # structural, not conventional.
    params = [p for p in inspect.signature(AlertSink.__init__).parameters if p != "self"]
    assert params == [], f"AlertSink.__init__ must take no dependencies, got {params}"


@pytest.mark.gating
def test_alert_sink_holds_no_cloud_reference() -> None:
    alert = AlertSink()
    for name, value in vars(alert).items():
        assert not isinstance(value, MockCloudReceiver), f"{name} references the cloud receiver"
        assert not isinstance(value, CloudTransport), f"{name} is a CloudTransport"


@pytest.mark.gating
@pytest.mark.skip(reason="pending Phase 6: routing/crypto not implemented yet")
def test_malicious_only_run_emits_zero_encrypted_payloads() -> None:
    # Phase 6 will run a malicious-only stream through VerdictRouter and assert the
    # MockCloudReceiver.received_count stays 0.
    cloud = MockCloudReceiver()
    assert cloud.received_count == 0
    raise AssertionError("implement in Phase 6")
