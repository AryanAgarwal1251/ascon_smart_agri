"""MQTT/JSON telemetry simulation (Phase 5, Section III-H).

Generates application-layer sensor payloads such as::

    {"deviceId": "soil01", "temperature": 24.8, "soilMoisture": 42.5}

These are application data, protected by Ascon on the benign path. They contain NO
flow-derived features and are never parsed into the model's input --- see provenance.py.

TODO(Phase 5): implement a payload stream (device ids, schema version, monotonic counter).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetryMessage:
    """One simulated message: the application-plane JSON payload and its metadata."""

    device_id: str
    payload: dict[str, float | str]
    counter: int  # monotonic; authenticated as associated data for replay detection
    schema_version: str


def simulate_stream(
    device_ids: list[str], *, n_messages: int, seed: int
) -> Iterator[TelemetryMessage]:
    """Yield a deterministic stream of simulated telemetry messages."""
    del device_ids, n_messages, seed
    raise NotImplementedError("Phase 5: telemetry simulation not implemented yet.")
