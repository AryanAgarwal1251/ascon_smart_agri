"""Unit tests for MQTT/JSON telemetry simulation (Section III-H).

Covers the property that matters for the crypto layer downstream (Section III-G): the counter
is a real, gap-free, per-device monotonic sequence, since it is authenticated as associated
data (Eq. 27) and used for replay detection.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.telemetry.simulate import TelemetryMessage, simulate_stream


def test_stream_yields_the_requested_total_message_count() -> None:
    messages = list(simulate_stream(["a", "b", "c"], n_messages=10, seed=0))

    assert len(messages) == 10


def test_devices_are_visited_round_robin() -> None:
    messages = list(simulate_stream(["a", "b"], n_messages=5, seed=0))

    assert [m.device_id for m in messages] == ["a", "b", "a", "b", "a"]


def test_counter_is_monotonic_per_device_starting_at_zero() -> None:
    """Not a single global counter: Eq. (27)'s AD tuple pairs device_id with counter, which
    only makes sense if each device runs its own sequence."""
    messages = list(simulate_stream(["a", "b"], n_messages=6, seed=0))

    per_device = {"a": [], "b": []}
    for m in messages:
        per_device[m.device_id].append(m.counter)

    assert per_device["a"] == [0, 1, 2]
    assert per_device["b"] == [0, 1, 2]


def test_stream_index_is_monotonic_across_the_whole_stream_not_per_device() -> None:
    """The property that fixes the provenance-pairing bug: unlike counter, stream_index must
    never repeat across devices, or every device's Nth message would pair with the same
    held-out network record (see telemetry/provenance.py's module docstring)."""
    messages = list(simulate_stream(["a", "b", "c"], n_messages=9, seed=0))

    stream_indices = [m.stream_index for m in messages]

    assert stream_indices == list(range(9))  # strictly increasing, no repeats, no gaps
    # In particular: every device's message-0 (counter=0) has a DIFFERENT stream_index.
    firsts = [m.stream_index for m in messages if m.counter == 0]
    assert len(firsts) == len(set(firsts)) == 3


def test_payload_matches_the_papers_own_example_shape() -> None:
    # Section III-H: {"deviceId": "soil01", "temperature": 24.8, "soilMoisture": 42.5}
    message = next(simulate_stream(["soil01"], n_messages=1, seed=0))

    assert set(message.payload) == {"deviceId", "temperature", "soilMoisture"}
    assert message.payload["deviceId"] == "soil01"
    assert isinstance(message.payload["temperature"], float)
    assert isinstance(message.payload["soilMoisture"], float)


def test_payload_values_stay_within_the_configured_ranges() -> None:
    messages = list(
        simulate_stream(
            ["a"],
            n_messages=50,
            seed=0,
            temperature_range_c=(10.0, 20.0),
            soil_moisture_range_pct=(30.0, 40.0),
        )
    )

    assert all(10.0 <= m.payload["temperature"] <= 20.0 for m in messages)  # type: ignore[operator]
    assert all(30.0 <= m.payload["soilMoisture"] <= 40.0 for m in messages)  # type: ignore[operator]


def test_stream_is_deterministic_given_a_seed() -> None:
    first = list(simulate_stream(["a", "b"], n_messages=8, seed=42))
    second = list(simulate_stream(["a", "b"], n_messages=8, seed=42))

    assert first == second


def test_different_seeds_produce_different_payloads() -> None:
    first = list(simulate_stream(["a"], n_messages=5, seed=1))
    second = list(simulate_stream(["a"], n_messages=5, seed=2))

    assert [m.payload["temperature"] for m in first] != [m.payload["temperature"] for m in second]


def test_schema_version_is_carried_on_every_message() -> None:
    messages = list(simulate_stream(["a"], n_messages=3, seed=0, schema_version="v2"))

    assert all(m.schema_version == "v2" for m in messages)


def test_stream_is_a_lazy_iterator_not_a_list() -> None:
    stream = simulate_stream(["a"], n_messages=1_000_000, seed=0)

    first = next(stream)  # must not require materialising all 1M messages

    assert isinstance(first, TelemetryMessage)


def test_rejects_degenerate_arguments() -> None:
    with pytest.raises(ValueError, match="device_ids"):
        list(simulate_stream([], n_messages=5, seed=0))
    with pytest.raises(ValueError, match="n_messages"):
        list(simulate_stream(["a"], n_messages=0, seed=0))
    with pytest.raises(ValueError, match="temperature_range_c"):
        list(simulate_stream(["a"], n_messages=1, seed=0, temperature_range_c=(20.0, 10.0)))
    with pytest.raises(ValueError, match="soil_moisture_range_pct"):
        list(simulate_stream(["a"], n_messages=1, seed=0, soil_moisture_range_pct=(50.0, 10.0)))
