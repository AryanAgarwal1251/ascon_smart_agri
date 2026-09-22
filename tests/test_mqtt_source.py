"""The sensor-node contract as the Pi runtime sees it (Phase 8 software twin)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ascon_smart_agri.telemetry.mqtt_source import MqttSensorSource
from ascon_smart_agri.telemetry.provenance import FeatureProvenanceAdapter


def _reading(device: str, seq: int, **extra: float) -> bytes:
    body = {"deviceId": device, "temperature": 24.8, "soilMoisture": 42.5, "seq": seq, **extra}
    return json.dumps(body).encode()


def test_readings_are_parsed_ordered_and_replay_filtered() -> None:
    src = MqttSensorSource("farm1")
    src.handle("farm/farm1/sensor/soil01", _reading("soil01", 1))
    src.handle("farm/farm1/sensor/soil02", _reading("soil02", 1))
    src.handle("farm/farm1/sensor/soil01", _reading("soil01", 2))
    src.handle("farm/farm1/sensor/soil01", _reading("soil01", 2))  # replayed
    src.handle("farm/farm1/sensor/soil01", _reading("soil01", 1))  # stale

    got = list(src.readings(timeout_s=0.01))
    assert [(r.device_id, r.counter, r.stream_index) for r in got] == [
        ("soil01", 1, 0),
        ("soil02", 1, 1),
        ("soil01", 2, 2),
    ]
    assert all(r.scenario == "benign" for r in got)
    assert [reason for _, reason in src.dropped] == [
        "seq 2 does not advance past 2 (replay)",
        "seq 1 does not advance past 2 (replay)",
    ]


def test_malformed_foreign_and_mismatched_messages_are_dropped() -> None:
    src = MqttSensorSource("farm1")
    src.handle("farm/farm2/sensor/soil01", _reading("soil01", 1))  # other farm
    src.handle("farm/farm1/sensor/soil01", b"not json")
    src.handle("farm/farm1/sensor/soil01", json.dumps({"deviceId": "soil01"}).encode())  # no seq
    src.handle("farm/farm1/sensor/soil01", _reading("soil09", 1))  # payload id != topic id
    src.handle("weather/today", b"{}")
    assert list(src.readings(timeout_s=0.01)) == []
    assert len(src.dropped) == 5


def test_scenario_topic_switches_farm_or_one_device() -> None:
    src = MqttSensorSource("farm1")
    src.handle("farm/farm1/scenario", b"attack")
    src.handle("farm/farm1/sensor/soil01", _reading("soil01", 1))
    src.handle("farm/farm1/scenario", json.dumps({"deviceId": "soil02", "mode": "benign"}).encode())
    src.handle("farm/farm1/sensor/soil02", _reading("soil02", 1))
    src.handle("farm/farm1/sensor/soil01", _reading("soil01", 2))
    src.handle("farm/farm1/scenario", b"benign")  # farm-wide reset clears per-device overrides
    src.handle("farm/farm1/sensor/soil02", _reading("soil02", 2))
    src.handle("farm/farm1/scenario", b"chaos")  # unknown -> dropped, state unchanged

    got = list(src.readings(timeout_s=0.01))
    assert [(r.device_id, r.scenario) for r in got] == [
        ("soil01", "attack"),
        ("soil02", "benign"),
        ("soil01", "attack"),
        ("soil02", "benign"),
    ]
    assert src.scenario_for("soil01") == "benign"
    assert src.dropped[-1][1] == "unknown scenario 'chaos'"


def test_paho_client_is_only_needed_for_a_real_connection() -> None:
    class Fake:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def connect(self, host: str, port: int, keepalive: int) -> None:
            self.calls.append(("connect", (host, port)))

        def subscribe(self, topic: str) -> None:
            self.calls.append(("subscribe", topic))

        def loop_start(self) -> None:
            self.calls.append(("loop_start", None))

        def loop_stop(self) -> None:
            self.calls.append(("loop_stop", None))

        def disconnect(self) -> None:
            self.calls.append(("disconnect", None))

    fake = Fake()
    src = MqttSensorSource("farm1", client=fake)
    src.connect("broker", 1883)
    src.close()
    assert fake.calls == [
        ("connect", ("broker", 1883)),
        ("subscribe", "farm/farm1/sensor/#"),
        ("subscribe", "farm/farm1/scenario"),
        ("loop_start", None),
        ("loop_stop", None),
        ("disconnect", None),
    ]


# -------------------------------------------------- scenario-restricted provenance (G6)


def _adapter() -> FeatureProvenanceAdapter:
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels = ["BenignTraffic", "DDoS-SYN_Flood", "BenignTraffic", "Mirai-udpplain", "Benign", "XSS"]
    return FeatureProvenanceAdapter(features, [f"f:{i}" for i in range(6)], labels=labels, seed=0)


def test_scenario_pools_draw_only_held_out_records_of_that_kind() -> None:
    adapter = _adapter()
    benign = {adapter.network_features_for(i, scenario="benign").true_label for i in range(20)}
    attack = {adapter.network_features_for(i, scenario="attack").true_label for i in range(20)}
    assert benign == {"BenignTraffic", "Benign"}
    assert attack == {"DDoS-SYN_Flood", "Mirai-udpplain", "XSS"}
    # Every record still comes from the held-out pool, with its provenance reference intact.
    assert adapter.network_features_for(3, scenario="attack").origin == "held_out_ciciot2023_record"
    assert adapter.network_features_for(3, scenario="attack").source_record_ref.startswith("f:")


def test_scenario_pool_is_deterministic_and_default_is_unchanged() -> None:
    a, b = _adapter(), _adapter()
    assert [a.network_features_for(i, scenario="attack").source_record_ref for i in range(6)] == [
        b.network_features_for(i, scenario="attack").source_record_ref for i in range(6)
    ]
    assert (
        a.network_features_for(4).source_record_ref == b.network_features_for(4).source_record_ref
    )


def test_scenario_without_records_refuses_rather_than_synthesising() -> None:
    unlabelled = FeatureProvenanceAdapter(np.zeros((3, 2), np.float32), ["a", "b", "c"], seed=0)
    with pytest.raises(ValueError, match="forbidden"):
        unlabelled.network_features_for(0, scenario="attack")
    with pytest.raises(ValueError, match="scenario must be"):
        _adapter().network_features_for(0, scenario="mixed")
