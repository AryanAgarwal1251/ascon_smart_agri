"""Farm-local MQTT sensor source for the Pi runtime (Phase 8, plan §2, §4; software twin).

The sensor node is a *contract*, not a component. Anything that honours it --- an ESP32 on
the farm Wi-Fi, a Wokwi-simulated ESP32, or ``scripts/run_virtual_sensor.py`` in a container
--- is a valid sensor node, and the Pi runtime cannot tell them apart:

    topic    farm/<farm_id>/sensor/<device_id>
    payload  {"deviceId": "soil01", "temperature": 24.8, "soilMoisture": 42.5, "seq": 17}

    topic    farm/<farm_id>/scenario
    payload  "benign" | "attack"                      -> every device on the farm
             {"deviceId": "soil02", "mode": "attack"}  -> one device

The scenario topic is how a demonstration switches a node into an attack scenario. It does
NOT change what the node sends; it tells the Pi's provenance adapter which held-out record
pool to pair that device's messages with (plan §8 option A: scenario-triggered replay of real
CICIoT2023 attack records, never synthesised traffic --- the G6 boundary holds).

The class is written against a tiny client interface so the tests drive it with an in-memory
fake and no broker; ``paho-mqtt`` is only imported when a real connection is requested.
``seq`` from the payload becomes the per-device replay counter; a reading whose ``seq`` does
not advance is dropped here, before it can reach the model.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

SCENARIOS = ("benign", "attack")
_TOPIC = re.compile(r"^farm/(?P<farm>[^/]+)/(?P<kind>sensor|scenario)(?:/(?P<device>[^/]+))?$")


@dataclass(frozen=True)
class SensorReading:
    """One application-plane reading as received from the farm broker."""

    farm_id: str
    device_id: str
    payload: dict[str, Any]
    counter: int  # the node's ``seq``: monotonic per device
    stream_index: int  # monotonic across everything this source has accepted
    scenario: str  # the device's scenario at the moment of receipt
    received_at: float = field(default_factory=time.time)


class MqttClientLike(Protocol):
    """The slice of ``paho.mqtt.client.Client`` this module uses."""

    def connect(self, host: str, port: int, keepalive: int) -> Any: ...
    def subscribe(self, topic: str) -> Any: ...
    def loop_start(self) -> Any: ...
    def loop_stop(self) -> Any: ...
    def disconnect(self) -> Any: ...


class MqttSensorSource:
    """Subscribes to one farm's sensor and scenario topics; yields validated readings."""

    def __init__(self, farm_id: str, *, client: MqttClientLike | None = None) -> None:
        self.farm_id = farm_id
        self._client = client
        self._queue: queue.Queue[SensorReading] = queue.Queue()
        self._lock = threading.Lock()
        self._scenario_by_device: dict[str, str] = {}
        self._farm_scenario = "benign"
        self._last_counter: dict[str, int] = {}
        self._stream_index = 0
        self.dropped: list[tuple[str, str]] = []  # (topic, reason), for the node's log

    # -- wiring -----------------------------------------------------------------------

    def connect(self, host: str, port: int = 1883, *, keepalive: int = 30) -> None:
        """Connect a real paho client (created here if none was injected) and subscribe."""
        if self._client is None:
            import paho.mqtt.client as mqtt
            from paho.mqtt.enums import CallbackAPIVersion

            client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=f"pi-{self.farm_id}")
            client.on_message = lambda _c, _u, msg: self.handle(msg.topic, bytes(msg.payload))
            self._client = client
        self._client.connect(host, port, keepalive)
        self._client.subscribe(f"farm/{self.farm_id}/sensor/#")
        self._client.subscribe(f"farm/{self.farm_id}/scenario")
        self._client.loop_start()

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()

    # -- the contract -----------------------------------------------------------------

    def scenario_for(self, device_id: str) -> str:
        with self._lock:
            return self._scenario_by_device.get(device_id, self._farm_scenario)

    def handle(self, topic: str, payload: bytes) -> None:
        """Process one raw MQTT message (also the entry point the tests use)."""
        match = _TOPIC.match(topic)
        if match is None or match["farm"] != self.farm_id:
            self.dropped.append((topic, "topic not on this farm's contract"))
            return
        if match["kind"] == "scenario":
            self._handle_scenario(topic, payload)
            return
        self._handle_reading(topic, match["device"], payload)

    def _handle_scenario(self, topic: str, payload: bytes) -> None:
        text = payload.decode("utf-8", errors="replace").strip()
        device: str | None = None
        mode = text.strip('"')
        if text.startswith("{"):
            try:
                body = json.loads(text)
                device = str(body["deviceId"]) if "deviceId" in body else None
                mode = str(body["mode"])
            except (ValueError, KeyError, TypeError):
                self.dropped.append((topic, "malformed scenario message"))
                return
        if mode not in SCENARIOS:
            self.dropped.append((topic, f"unknown scenario {mode!r}"))
            return
        with self._lock:
            if device is None:
                self._farm_scenario = mode
                self._scenario_by_device.clear()
            else:
                self._scenario_by_device[device] = mode

    def _handle_reading(self, topic: str, device_id: str | None, payload: bytes) -> None:
        if not device_id:
            self.dropped.append((topic, "sensor topic without a device id"))
            return
        try:
            body = json.loads(payload.decode("utf-8"))
            seq = int(body["seq"])
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            self.dropped.append((topic, "malformed reading (needs JSON with integer seq)"))
            return
        if str(body.get("deviceId", device_id)) != device_id:
            self.dropped.append((topic, "deviceId in payload disagrees with topic"))
            return
        with self._lock:
            last = self._last_counter.get(device_id)
            if last is not None and seq <= last:
                self.dropped.append((topic, f"seq {seq} does not advance past {last} (replay)"))
                return
            self._last_counter[device_id] = seq
            reading = SensorReading(
                farm_id=self.farm_id,
                device_id=device_id,
                payload=body,
                counter=seq,
                stream_index=self._stream_index,
                scenario=self._scenario_by_device.get(device_id, self._farm_scenario),
            )
            self._stream_index += 1
        self._queue.put(reading)

    # -- consumption ------------------------------------------------------------------

    def readings(self, *, timeout_s: float | None = None) -> Iterator[SensorReading]:
        """Yield readings as they arrive; stops after ``timeout_s`` of silence (None = never)."""
        while True:
            try:
                yield self._queue.get(timeout=timeout_s)
            except queue.Empty:
                return
