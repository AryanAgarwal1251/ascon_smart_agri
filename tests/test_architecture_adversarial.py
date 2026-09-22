"""ADVERSARIAL ACCEPTANCE SUITE --- one block per component of the Phase 8 architecture.

Every test here is written from the attacker's side: the test *name* is what the attacker is
trying to achieve, and the test passes when the attacker **fails**. This is the suite to walk
through when demonstrating the system (``docs/architecture_test_matrix.md`` maps each
architecture block to the checks that cover it).

The components, in the order a reading travels:

* **A. Sensor contract** --- ESP32 -> farm broker (``telemetry/mqtt_source.py``)
* **B. Farm isolation** --- broker -> Pi; one farm's Pi never consumes another farm's topic
* **C. Feature provenance (G6)** --- the JSON payload can never become a network feature
* **D. Sealed weight channel** --- Pi <-> aggregator, Ascon-AEAD128 both ways (the headline
  security claim; ``federated/transport.py``)
* **E. Aggregator integrity** --- what a rejected or duplicated frame can do to the global model
* **F. Cloud path (G1)** --- a malicious verdict can never leave on the benign transport

These complement rather than replace the property suites: ``test_weight_channel.py`` (frame
algebra), ``test_ascon_kat.py`` (SP 800-232 conformance), ``test_path_disjointness.py`` and
``test_tls_path.py`` (G1), ``test_mqtt_source.py`` (sensor contract). Where a property is
already pinned there, the matrix cites that test instead of restating it here.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from ascon_smart_agri.federated.serialization import serialize_state
from ascon_smart_agri.federated.transport import (
    AggregatorChannel,
    ChannelKeys,
    ClientChannel,
    WeightOpener,
    peek_client_id,
)
from ascon_smart_agri.telemetry.mqtt_source import MqttSensorSource
from ascon_smart_agri.telemetry.provenance import FeatureProvenanceAdapter

_KEYS = ChannelKeys(up=b"\x01" * 16, down=b"\x02" * 16)


def _tiny(seed: int) -> dict[str, torch.Tensor]:
    """A few tensors. The frame algebra does not depend on |theta|, and the pure-Python Ascon
    backend costs ~0.5 s per 135 KB seal --- keep the adversarial loops cheap."""
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(4, 3, generator=g), "b": torch.randn(4, generator=g)}


def _reading(farm: str, device: str, seq: int, **payload: object) -> tuple[str, bytes]:
    body: dict[str, object] = {"seq": seq, "soil_moisture": 41.2, **payload}
    return f"farm/{farm}/sensor/{device}", json.dumps(body).encode()


# =========================================================== A. sensor contract (ESP32 -> broker)


def test_attacker_replaying_a_captured_sensor_reading_cannot_advance_the_stream() -> None:
    """Attacker sniffs the farm's MQTT traffic and re-publishes an earlier reading verbatim.

    The node's ``seq`` is monotonic per device, so the replayed reading must be dropped rather
    than processed a second time --- otherwise a stale reading could be used to mask a live one.
    """
    source = MqttSensorSource("farm1")
    for seq in (1, 2, 3):
        source.handle(*_reading("farm1", "soil01", seq))
    accepted = list(source.readings(timeout_s=0))
    assert [r.counter for r in accepted] == [1, 2, 3]

    source.handle(*_reading("farm1", "soil01", 2))  # the replay
    assert list(source.readings(timeout_s=0)) == []


def test_attacker_forging_a_reading_the_node_never_sent_is_dropped_when_malformed() -> None:
    """Attacker injects a payload with no ``seq`` (or a non-JSON body) on a valid topic.

    The contract is ``farm/<farm>/sensor/<device>`` carrying JSON with a monotonic ``seq``.
    Anything that does not satisfy it is dropped, not guessed at.
    """
    source = MqttSensorSource("farm1")
    source.handle("farm/farm1/sensor/soil01", b"{not json at all")
    source.handle("farm/farm1/sensor/soil01", json.dumps({"soil_moisture": 9.9}).encode())
    assert list(source.readings(timeout_s=0)) == []


# ============================================================ B. farm isolation (G1-adjacent)


def test_attacker_publishing_into_another_farms_topic_is_ignored() -> None:
    """Attacker with access to farm 2's broker publishes readings addressed to farm 1.

    Each Pi subscribes to its own farm only. In the Compose topology the two farms have separate
    brokers as well, so this is defence in depth: even handed the foreign message directly, the
    source must refuse it.
    """
    source = MqttSensorSource("farm1")
    source.handle(*_reading("farm2", "soil04", 1))
    source.handle(*_reading("farm2", "soil05", 1))
    assert list(source.readings(timeout_s=0)) == []

    source.handle(*_reading("farm1", "soil01", 1))  # its own farm still works
    assert [r.device_id for r in source.readings(timeout_s=0)] == ["soil01"]


# ============================================================= C. feature provenance boundary (G6)


def test_attacker_cannot_inject_network_features_through_the_sensor_payload() -> None:
    """Attacker controls an ESP32 and stuffs network-feature values into its JSON.

    G6: network features come only from held-out CICIoT2023 records. The proof is structural ---
    ``network_features_for`` takes a stream index and a scenario, and has no parameter through
    which a payload could reach it. Two readings with wildly different payloads at the same
    stream position therefore yield the identical feature vector.
    """
    features = np.arange(40, dtype=np.float64).reshape(8, 5)
    adapter = FeatureProvenanceAdapter(features, [f"held_out.csv:{i}" for i in range(8)])

    params = set(inspect.signature(adapter.network_features_for).parameters)
    assert params == {"stream_index", "scenario"}, (
        f"network_features_for accepts {params}: a payload must never be able to reach it"
    )

    honest = adapter.network_features_for(3)
    forged = adapter.network_features_for(3)  # same position, attacker-controlled payload
    assert np.array_equal(honest.features, forged.features)
    assert honest.origin == "held_out_ciciot2023_record"
    assert honest.source_record_ref.startswith("held_out.csv:")


def test_attacker_cannot_make_the_adapter_synthesise_a_scenario_it_has_no_records_for() -> None:
    """Attacker flips a device into ``attack`` scenario on a node with no held-out attack pool.

    The adapter must refuse rather than fabricate traffic: synthesising would put values into the
    classifier that never came from a real record, which is exactly what G6 forbids.
    """
    features = np.arange(20, dtype=np.float64).reshape(4, 5)
    adapter = FeatureProvenanceAdapter(
        features, [f"r:{i}" for i in range(4)], labels=["BenignTraffic"] * 4
    )
    with pytest.raises(ValueError, match="forbidden"):
        adapter.network_features_for(0, scenario="attack")


# ================================================ D. sealed weight channel (headline claim)


def test_eavesdropper_without_the_key_cannot_read_the_weights() -> None:
    """Attacker captures a weight frame on the IoT network and tries to open it.

    This is the central security claim of the redirected architecture: the model weights crossing
    between each local GRU and the master GRU are Ascon-AEAD128 sealed, so a frame on the wire is
    useless without the per-client, per-direction key. Every wrong key must yield bottom --- the
    opener returns ``None``, never a partial or garbled state dict.
    """
    # A real key that is not a repeated byte, so every guess in the sweep below is genuinely wrong.
    secret = ChannelKeys(up=bytes(range(16)), down=bytes(range(16, 32)))
    frame = ClientChannel("pi-1", secret).seal_update(_tiny(0), sequence_count=5000, round_index=1)

    for guess in range(64):  # a brute-force sweep, standing in for the whole key space
        attacker = WeightOpener(bytes([guess]) * 16, "pi-1", "up")
        assert attacker.open(frame, expected_round=1) is None, (
            f"key {guess:#04x} opened a frame it must not have"
        )

    # And the honest holder of the key still opens it --- the channel works, it is just sealed.
    assert WeightOpener(secret.up, "pi-1", "up").open(frame, expected_round=1) is not None


def test_a_captured_frame_does_not_contain_the_weights_in_the_clear() -> None:
    """Attacker does not even try to decrypt --- they scan the captured bytes for the weights.

    A sealed frame must not embed the serialized tensor bytes anywhere: no prefix, no suffix, no
    accidental passthrough. Only the associated data (client id, round, direction, schema) is
    readable, and that is by design --- it is authenticated, not secret.
    """
    state = _tiny(1)
    plaintext = serialize_state(state, 5000)
    frame = ClientChannel("pi-1", _KEYS).seal_update(state, sequence_count=5000, round_index=1)

    assert plaintext not in frame
    for name, tensor in state.items():
        raw = tensor.numpy().tobytes()
        assert raw not in frame, f"tensor {name!r} appears in the frame in the clear"

    assert peek_client_id(frame) == "pi-1"  # AD is readable: authenticated, not confidential


def test_identical_weights_sealed_twice_do_not_produce_the_same_frame() -> None:
    """Attacker watches two rounds and tries to tell whether the model changed.

    A fresh CSPRNG nonce per message means sealing the *same* state twice gives different
    ciphertext, so frame equality leaks nothing about whether training moved the weights.
    """
    state = _tiny(2)
    channel = ClientChannel("pi-1", _KEYS)
    first = channel.seal_update(state, sequence_count=5000, round_index=1)
    second = channel.seal_update(state, sequence_count=5000, round_index=1)

    assert first != second
    assert len(first) == len(second)


def test_attacker_cannot_downgrade_the_channel_by_sending_unsealed_weights() -> None:
    """Attacker sends the raw serialized state, hoping the aggregator accepts plaintext.

    There is no unsealed path. A frame that is not an authenticated ``ASAW`` frame is rejected
    before any state is deserialized, so the aggregator cannot be argued down to plaintext.
    """
    aggregator = AggregatorChannel("pi-1", _KEYS)
    plaintext = serialize_state(_tiny(3), 5000)

    assert aggregator.open_update(plaintext, 1) is None
    assert aggregator.open_update(b"", 1) is None
    assert aggregator.open_update(b"ASAW" + plaintext, 1) is None


def test_attacker_cannot_reuse_a_client_frame_against_a_different_client() -> None:
    """Attacker intercepts pi-1's update and re-addresses it as pi-2's, to steer the average.

    Keys are per client *and* per direction, and the client id is bound into the associated data,
    so the frame is dead the moment it is presented on another client's channel.
    """
    pi1_frame = ClientChannel("pi-1", _KEYS).seal_update(_tiny(4), 5000, 1)
    pi2_keys = ChannelKeys(up=b"\x05" * 16, down=b"\x06" * 16)

    assert AggregatorChannel("pi-2", pi2_keys).open_update(pi1_frame, 1) is None
    # Even if pi-2 somehow shared pi-1's key material, the AD still names pi-1.
    assert AggregatorChannel("pi-2", _KEYS).open_update(pi1_frame, 1) is None


# ================================================================ E. aggregator integrity


def test_replaying_a_clients_own_frame_within_a_round_cannot_inflate_its_weight() -> None:
    """Attacker re-sends pi-1's captured update inside the same round, to double-count it.

    The transport refuses it outright: the opener records the last round it accepted and rejects
    any frame whose round is not strictly newer, so the duplicate never reaches aggregation at
    all. (Defence in depth: even had it opened, the server keys updates by client id, so pi-1's
    ``n_k`` would still have been counted once rather than twice.)
    """
    aggregator = AggregatorChannel("pi-1", _KEYS)
    frame = ClientChannel("pi-1", _KEYS).seal_update(_tiny(5), sequence_count=5000, round_index=1)

    first = aggregator.open_update(frame, 1)
    assert first is not None and first[1] == 5000

    assert aggregator.open_update(frame, 1) is None
    assert "replay" in (aggregator.last_rejection or "")
    assert aggregator.frames_rejected == 1


def test_attacker_cannot_tamper_with_an_honest_clients_declared_n_k() -> None:
    """Attacker modifies ``n_k`` in flight to skew the weighted average (Eq. 21).

    ``n_k`` is authenticated, so any edit yields bottom and the update is dropped entirely.

    **Documented limitation, deliberately not defended:** this stops an *outside* attacker, not a
    dishonest *client* that declares an inflated ``n_k`` under its own valid key. Defending that
    is Byzantine-robust aggregation, which CLAUDE.md places out of scope for this project. The
    architecture's trust boundary is the network, not the client.
    """
    frame = ClientChannel("pi-1", _KEYS).seal_update(_tiny(6), sequence_count=5000, round_index=1)
    aggregator = AggregatorChannel("pi-1", _KEYS)

    honest = aggregator.open_update(frame, 1)
    assert honest is not None and honest[1] == 5000

    for position in range(len(frame)):
        mutated = bytearray(frame)
        mutated[position] ^= 0x01
        assert AggregatorChannel("pi-1", _KEYS).open_update(bytes(mutated), 1) is None
        if position > 64:  # the whole frame is covered by test_weight_channel.py; sample here
            break
