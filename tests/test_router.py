"""Unit tests for the verdict router (Eq. 5, Section III-A, gap G1).

The structural disjointness guarantee has its own gating file, test_path_disjointness.py. This
covers the behavioural correctness of routing itself: benign encrypts and reaches the cloud
intact; malicious never does; nonces are never reused across calls.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.crypto.ascon_aead import AsconAEAD128, AssociatedData, NonceRegistry
from ascon_smart_agri.routing.alert_sink import AlertSink
from ascon_smart_agri.routing.cloud_sink import MockCloudReceiver
from ascon_smart_agri.routing.router import VerdictRouter

KEY = b"\x33" * 16


def _router(
    cloud: MockCloudReceiver | None = None,
) -> tuple[VerdictRouter, MockCloudReceiver, AlertSink]:
    cloud = cloud if cloud is not None else MockCloudReceiver({"edge01": KEY})
    alert = AlertSink()
    return VerdictRouter(cloud, alert, AsconAEAD128(KEY)), cloud, alert


def test_benign_verdict_reaches_the_cloud_and_decrypts_to_the_original_payload() -> None:
    router, cloud, alert = _router()
    ad = AssociatedData("edge01", "dev01", 0, "v1")

    router.route(verdict_benign=True, associated_data=ad, payload=b"soil reading")

    assert cloud.received_count == 1
    assert cloud.accepted_payloads == [b"soil reading"]
    assert alert.alert_count == 0  # the benign path never touches the alert sink


def test_malicious_verdict_reaches_the_alert_sink_and_never_the_cloud() -> None:
    router, cloud, alert = _router()
    ad = AssociatedData("edge01", "dev01", 0, "v1")

    router.route(verdict_benign=False, associated_data=ad, payload=b"attack traffic")

    assert cloud.received_count == 0
    assert cloud.rejected_count == 0  # never even attempted delivery
    assert alert.alert_count == 1
    assert alert.alerts()[0]["edge_id"] == "edge01"


def test_alert_reason_carries_device_and_counter_context() -> None:
    router, _, alert = _router()
    ad = AssociatedData("edge01", "dev07", 42, "v1")

    router.route(verdict_benign=False, associated_data=ad, payload=b"x")

    reason = alert.alerts()[0]["reason"]
    assert "dev07" in reason
    assert "42" in reason


def test_successive_benign_messages_never_reuse_a_nonce() -> None:
    router, cloud, _ = _router()
    registry = router._nonces  # inspecting internal state deliberately, for this one assertion

    for counter in range(10):
        ad = AssociatedData("edge01", "dev01", counter, "v1")
        router.route(verdict_benign=True, associated_data=ad, payload=b"x")

    assert cloud.received_count == 10
    # NonceRegistry itself raises on reuse (tested in test_nonce_collision.py); the absence
    # of an exception across 10 real routed messages is the behavioural proof here.
    assert isinstance(registry, NonceRegistry)


def test_router_accepts_an_injected_nonce_registry() -> None:
    registry = NonceRegistry()
    cloud = MockCloudReceiver({"edge01": KEY})
    alert = AlertSink()
    router = VerdictRouter(cloud, alert, AsconAEAD128(KEY), nonce_registry=registry)
    ad = AssociatedData("edge01", "dev01", 0, "v1")

    router.route(verdict_benign=True, associated_data=ad, payload=b"x")

    assert cloud.received_count == 1


def test_different_verdicts_for_the_same_device_route_independently() -> None:
    router, cloud, alert = _router()

    router.route(
        verdict_benign=True,
        associated_data=AssociatedData("edge01", "dev01", 0, "v1"),
        payload=b"benign",
    )
    router.route(
        verdict_benign=False,
        associated_data=AssociatedData("edge01", "dev01", 1, "v1"),
        payload=b"attack",
    )

    assert cloud.received_count == 1
    assert alert.alert_count == 1


@pytest.mark.gating
def test_router_never_hands_its_cloud_transport_to_the_alert_sink() -> None:
    """A second structural check, at the router's own construction site rather than AlertSink's:
    the router must not leak its cloud reference into the alert sink it holds."""
    _, cloud, alert = _router()

    for name, value in vars(alert).items():
        assert value is not cloud, f"AlertSink.{name} references the router's cloud transport"
