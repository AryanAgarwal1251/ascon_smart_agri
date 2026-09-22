"""GATING TESTS --- the TLS benign path keeps G1 path disjointness (Phase 8, plan §1, §4).

The router is the only holder of the cloud transport; the alert sink never sees one; a
malicious-only run emits nothing to the cloud. Plus the receiver's replay check, and a real
HTTP round-trip through ``HttpsCloudClient`` -> ``CloudReceiverServer`` on loopback.
"""

from __future__ import annotations

import gc

import pytest

from ascon_smart_agri.routing.alert_sink import AlertSink
from ascon_smart_agri.routing.tls_path import (
    CloudReceiverServer,
    CloudReceiverState,
    HttpsCloudClient,
    InMemoryCloudTransport,
    PlainCloudTransport,
    ReadingEnvelope,
    TlsVerdictRouter,
)


def _env(device: str, counter: int, edge: str = "pi-1") -> ReadingEnvelope:
    return ReadingEnvelope(edge, device, counter, "v1", {"temperature": 24.8, "seq": counter})


@pytest.mark.gating
def test_malicious_only_run_emits_nothing_to_the_cloud() -> None:
    cloud, alert = InMemoryCloudTransport(), AlertSink()
    router = TlsVerdictRouter(cloud, alert)
    for i in range(25):
        router.route(verdict_benign=False, envelope=_env("soil01", i))
    assert cloud.sent == []
    assert alert.alert_count == 25


@pytest.mark.gating
def test_alert_sink_holds_no_reference_to_any_cloud_transport() -> None:
    cloud, alert = InMemoryCloudTransport(), AlertSink()
    TlsVerdictRouter(cloud, alert).route(verdict_benign=False, envelope=_env("soil01", 1))
    # Structural check: nothing reachable from the alert sink is a cloud transport.
    referrers = gc.get_referents(alert)
    assert not any(isinstance(obj, PlainCloudTransport) for obj in referrers)
    assert all(not isinstance(v, PlainCloudTransport) for v in vars(alert).values())


def test_benign_readings_reach_the_transport_with_their_metadata() -> None:
    cloud, alert = InMemoryCloudTransport(), AlertSink()
    router = TlsVerdictRouter(cloud, alert)
    router.route(verdict_benign=True, envelope=_env("soil01", 7))
    router.route(verdict_benign=False, envelope=_env("soil02", 1))
    router.route(verdict_benign=True, envelope=_env("soil01", 8))
    assert [(e.device_id, e.counter) for e in cloud.sent] == [("soil01", 7), ("soil01", 8)]
    assert alert.alert_count == 1


def test_envelope_json_roundtrip() -> None:
    env = _env("soil01", 3)
    assert ReadingEnvelope.from_json(env.to_json()) == env


def test_receiver_rejects_replays_unknown_edges_and_garbage() -> None:
    state = CloudReceiverState(known_edges=frozenset({"pi-1"}))
    assert state.ingest(_env("soil01", 1).to_json()) == (202, "accepted")
    assert state.ingest(_env("soil01", 2).to_json()) == (202, "accepted")
    assert state.ingest(_env("soil01", 2).to_json())[0] == 409  # replay
    assert state.ingest(_env("soil01", 1).to_json())[0] == 409  # stale
    assert state.ingest(_env("soil01", 3, edge="pi-9").to_json())[0] == 403
    assert state.ingest(b"{not json")[0] == 400
    assert (state.accepted, state.rejected) == (2, 4)


@pytest.mark.gating
def test_http_roundtrip_to_the_receiver_on_loopback() -> None:
    server = CloudReceiverServer("127.0.0.1", 0, known_edges=frozenset({"pi-1"}))
    server.start()
    try:
        client = HttpsCloudClient(server.url)
        router = TlsVerdictRouter(client, AlertSink())
        for i in range(5):
            router.route(verdict_benign=True, envelope=_env("soil01", i))
        router.route(verdict_benign=True, envelope=_env("soil01", 2))  # replay -> 409
        for i in range(3):
            router.route(verdict_benign=False, envelope=_env("soil02", i))
    finally:
        server.stop()
    assert client.sent == 5
    assert client.failed == 1 and "409" in (client.last_error or "")
    assert server.state.accepted == 5
    assert server.state.rejected == 1
    assert [r["counter"] for r in server.state.readings] == [0, 1, 2, 3, 4]
    # G1 on the wire: only benign readings ever reached the receiver.
    assert all(r["deviceId"] == "soil01" for r in server.state.readings)
