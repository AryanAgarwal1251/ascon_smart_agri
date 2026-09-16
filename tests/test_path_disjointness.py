"""GATING TEST (G1) --- disjoint benign/malicious data paths (Section III-A).

Two guarantees:
  (a) STRUCTURAL (active now): the malicious-path handler (AlertSink) holds no reference to the
      cloud transport --- it is not even constructed with one. Checked by introspection.
  (b) BEHAVIOURAL (Phase 6): zero encrypted payloads are emitted during a malicious-only run.
"""

from __future__ import annotations

import inspect

import pytest

from ascon_smart_agri.crypto.ascon_aead import AsconAEAD128, AssociatedData
from ascon_smart_agri.routing.alert_sink import AlertSink
from ascon_smart_agri.routing.cloud_sink import CloudTransport, MockCloudReceiver
from ascon_smart_agri.routing.router import VerdictRouter


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
def test_malicious_only_run_emits_zero_encrypted_payloads() -> None:
    """The behavioural half: route N malicious-verdict messages through the real VerdictRouter
    and assert the cloud receiver never accepts a payload -- not because nothing was sent past
    the router, but because AlertSink structurally cannot reach the cloud transport at all."""
    edge_id, device_id = "edge01", "sensor01"
    key = b"\x00" * 16  # fixed test key; never a real/demo secret
    cloud = MockCloudReceiver({edge_id: key})
    alert = AlertSink()
    router = VerdictRouter(cloud, alert, AsconAEAD128(key))

    for counter in range(5):
        ad = AssociatedData(edge_id, device_id, counter, "v1")
        router.route(verdict_benign=False, associated_data=ad, payload=b"malicious payload")

    assert cloud.received_count == 0
    assert cloud.rejected_count == 0  # nothing was even SENT to the cloud to reject
    assert alert.alert_count == 5
