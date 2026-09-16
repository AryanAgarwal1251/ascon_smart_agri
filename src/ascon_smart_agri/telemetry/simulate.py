"""MQTT/JSON telemetry simulation (Phase 5, Section III-H).

Generates application-layer sensor payloads such as::

    {"deviceId": "soil01", "temperature": 24.8, "soilMoisture": 42.5}

These are application data, protected by Ascon on the benign path. They contain NO
flow-derived features and are never parsed into the model's input --- see provenance.py.

Implementation notes:

* **The counter is monotonic PER DEVICE, not global across the stream.** Eq. (27)'s associated
  data is ``<edge_id, device_id, counter, schema_version>``: the tuple that scopes replay
  protection includes ``device_id`` alongside ``counter``, which only makes sense if each
  device runs its own counter sequence starting at 0 -- a global counter shared across devices
  would make ``device_id`` redundant in the AD tuple. Messages are still emitted in a single
  round-robin stream (device 0's message 0, device 1's message 0, ..., device 0's message 1,
  ...), so the stream itself has one well-defined order even though each device's counter is
  independent.
* **Payload generation is seeded and deterministic**, matching the rest of this project's
  reproducibility discipline -- the same ``seed`` always produces the same stream, which is
  what lets a demo run be replayed exactly.
* This module has NO knowledge of the network-feature plane (``provenance.py``) and never will:
  the two planes are paired later, by message counter, only in the Phase 7 runtime pipeline.
  Coupling them here would blur exactly the boundary Section III-H exists to keep sharp.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TelemetryMessage:
    """One simulated message: the application-plane JSON payload and its metadata.

    Carries TWO distinct counters, deliberately kept apart:

    * ``counter`` -- monotonic PER DEVICE, restarting at 0 for each device. This is the crypto
      layer's replay-protection counter, authenticated as associated data (Eq. 27); its scope
      matches the AD tuple ``<edge_id, device_id, counter, schema_version>``, which only makes
      sense per (device, key) pair.
    * ``stream_index`` -- monotonic across the WHOLE emitted stream, never restarting. This is
      what ``telemetry.provenance.FeatureProvenanceAdapter.network_features_for`` should be
      called with, not ``counter``: since every device's ``counter`` starts at 0, pairing on
      ``counter`` alone would give every device's first message the identical held-out network
      record, second message the identical next one, and so on -- the provenance pairing needs
      a key that actually differentiates every message in the stream, which only
      ``stream_index`` does.
    """

    device_id: str
    payload: dict[str, float | str]
    counter: int  # monotonic PER DEVICE; authenticated as associated data for replay detection
    stream_index: int  # monotonic across the WHOLE stream; use this to pair with provenance.py
    schema_version: str


def simulate_stream(
    device_ids: list[str],
    *,
    n_messages: int,
    seed: int,
    schema_version: str = "v1",
    temperature_range_c: tuple[float, float] = (15.0, 35.0),
    soil_moisture_range_pct: tuple[float, float] = (0.0, 100.0),
) -> Iterator[TelemetryMessage]:
    """Yield a deterministic stream of simulated telemetry messages.

    ``n_messages`` is the TOTAL across all devices, round-robin: with 3 devices and
    ``n_messages=10``, devices get 4/3/3 messages respectively, in emission order.
    """
    if not device_ids:
        raise ValueError("device_ids must be non-empty")
    if n_messages <= 0:
        raise ValueError(f"n_messages must be positive, got {n_messages}")
    if temperature_range_c[0] >= temperature_range_c[1]:
        raise ValueError(f"temperature_range_c must be (low, high), got {temperature_range_c}")
    if soil_moisture_range_pct[0] >= soil_moisture_range_pct[1]:
        raise ValueError(
            f"soil_moisture_range_pct must be (low, high), got {soil_moisture_range_pct}"
        )

    rng = np.random.default_rng(seed)
    counters = dict.fromkeys(device_ids, 0)

    for i in range(n_messages):
        device_id = device_ids[i % len(device_ids)]
        payload: dict[str, float | str] = {
            "deviceId": device_id,
            "temperature": round(float(rng.uniform(*temperature_range_c)), 1),
            "soilMoisture": round(float(rng.uniform(*soil_moisture_range_pct)), 1),
        }
        yield TelemetryMessage(
            device_id=device_id,
            payload=payload,
            counter=counters[device_id],
            stream_index=i,
            schema_version=schema_version,
        )
        counters[device_id] += 1
