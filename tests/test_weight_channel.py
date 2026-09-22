"""GATING TESTS (Phase 8, plan Section 6) --- the Ascon-sealed weight channel.

Three properties, each a hard requirement of the redirected architecture:

* **Roundtrip**: seal -> open reproduces the state dict bit-for-bit AND the authenticated
  ``n_k``; FedAvg over opened frames equals in-process FedAvg exactly.
* **Tamper**: flipping any byte of nonce, associated data, ciphertext or tag yields bottom
  (``None``), and the opener applies nothing.
* **Replay / reflection**: a round-r frame is dead after round r; an uplink frame presented as
  a downlink, a frame for another client, or one under the wrong key, is rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ascon_smart_agri.crypto.ascon_aead import NonceRegistry, WeightAssociatedData
from ascon_smart_agri.federated.aggregation import weighted_fedavg
from ascon_smart_agri.federated.serialization import serialize_state
from ascon_smart_agri.federated.transport import (
    AggregatorChannel,
    ChannelKeys,
    ClientChannel,
    WeightOpener,
    WeightSealer,
    frame_overhead_bytes,
    generate_demo_keys,
    load_keys,
    write_keys,
)
from ascon_smart_agri.model.gru import build_detector

_KEYS = ChannelKeys(up=b"\x01" * 16, down=b"\x02" * 16)
_OTHER = ChannelKeys(up=b"\x03" * 16, down=b"\x04" * 16)


def _state(seed: int) -> dict[str, torch.Tensor]:
    """The real 33,800-parameter detector state (Eq. 19) -- for the roundtrip tests."""
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in build_detector(16, 96, 8).state_dict().items()}


def _tiny(seed: int) -> dict[str, torch.Tensor]:
    """A few tensors: the pure-Python backend costs ~0.5 s per 135 KB seal, and the tamper,
    replay and nonce properties do not depend on |theta|."""
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(4, 3, generator=g), "b": torch.randn(4, generator=g)}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- roundtrip


@pytest.mark.gating
def test_uplink_roundtrip_is_bit_exact_and_authenticates_n_k() -> None:
    client = ClientChannel("pi-1", _KEYS)
    server = AggregatorChannel("pi-1", _KEYS)
    state = _state(0)

    frame = client.seal_update(state, 284_500, round_index=1)
    opened = server.open_update(frame, round_index=1)

    assert opened is not None
    got_state, n_k = opened
    assert _same(got_state, state)
    assert n_k == 284_500


@pytest.mark.gating
def test_downlink_roundtrip_is_bit_exact() -> None:
    client = ClientChannel("pi-1", _KEYS)
    server = AggregatorChannel("pi-1", _KEYS)
    global_state = _state(7)

    frame = server.seal_global(global_state, round_index=3)
    assert _same(client.open_global(frame, round_index=3) or {}, global_state)


@pytest.mark.gating
def test_fedavg_over_channel_equals_in_process_fedavg() -> None:
    keys = generate_demo_keys(["pi-1", "pi-2", "sim-3"])
    clients = {cid: ClientChannel(cid, k) for cid, k in keys.items()}
    server = {cid: AggregatorChannel(cid, k) for cid, k in keys.items()}
    states = [_state(i) for i in range(3)]
    counts = [284_500, 458_447, 452_390]

    opened_states, opened_counts = [], []
    for (cid, channel), state, n_k in zip(clients.items(), states, counts, strict=True):
        opened = server[cid].open_update(channel.seal_update(state, n_k, 1), 1)
        assert opened is not None
        opened_states.append(opened[0])
        opened_counts.append(opened[1])

    via_channel = weighted_fedavg(opened_states, opened_counts)
    in_process = weighted_fedavg(states, counts)
    assert _same(via_channel, in_process)


def test_overhead_is_constant_and_small_relative_to_weights() -> None:
    sealer = WeightSealer(_KEYS.up, "pi-1", "up")
    state = _state(0)
    plaintext_len = len(serialize_state(state, 1))
    frame = sealer.seal(state, 1, 1)
    overhead = frame_overhead_bytes(frame, plaintext_len)
    # nonce(16) + tag(16) + magic(4) + ad length(2) + AD (~20 bytes): independent of |theta|.
    assert 40 <= overhead <= 96
    assert overhead / plaintext_len < 0.001  # ~0.05 %, against 33 % for 96-byte telemetry


# ------------------------------------------------------------------------------ tamper


def _flip(data: bytes, index: int) -> bytes:
    return data[:index] + bytes([data[index] ^ 0x01]) + data[index + 1 :]


@pytest.mark.gating
@pytest.mark.parametrize(
    "region",
    ["magic", "nonce", "ad", "ciphertext_first", "ciphertext_middle", "tag_last"],
)
def test_single_bit_tamper_anywhere_yields_bottom(region: str) -> None:
    client = ClientChannel("pi-1", _KEYS)
    server = AggregatorChannel("pi-1", _KEYS)
    frame = client.seal_update(_tiny(0), 10, round_index=1)

    ad_len = int.from_bytes(frame[20:22], "big")
    index = {
        "magic": 0,
        "nonce": 4,
        "ad": 22,
        "ciphertext_first": 22 + ad_len,
        "ciphertext_middle": 22 + ad_len + (len(frame) - 22 - ad_len) // 2,
        "tag_last": len(frame) - 1,
    }[region]
    tampered = _flip(frame, index)
    assert tampered != frame

    assert server.open_update(tampered, round_index=1) is None
    assert server.last_rejection is not None
    # Nothing was applied: the opener still expects round 1 and accepts the genuine frame.
    assert server.open_update(frame, round_index=1) is not None


@pytest.mark.gating
def test_truncated_and_foreign_protocol_frames_are_bottom() -> None:
    server = AggregatorChannel("pi-1", _KEYS)
    frame = ClientChannel("pi-1", _KEYS).seal_update(_tiny(0), 10, 1)
    assert server.open_update(frame[:21], round_index=1) is None
    assert server.open_update(b"", round_index=1) is None
    assert server.open_update(b"HTTP/1.1 200 OK\r\n\r\n" + frame, round_index=1) is None


@pytest.mark.gating
def test_wrong_key_is_bottom() -> None:
    frame = ClientChannel("pi-1", _KEYS).seal_update(_tiny(0), 10, 1)
    assert AggregatorChannel("pi-1", _OTHER).open_update(frame, 1) is None


# ----------------------------------------------------------------------- replay/reflect


@pytest.mark.gating
def test_round_r_frame_is_rejected_at_round_r_plus_1() -> None:
    client = ClientChannel("pi-1", _KEYS)
    server = AggregatorChannel("pi-1", _KEYS)
    captured = client.seal_update(_tiny(0), 10, round_index=1)
    assert server.open_update(captured, 1) is not None

    assert server.open_update(captured, 2) is None  # replayed into the next round
    assert server.open_update(captured, 1) is None  # replayed into the same round
    assert "round" in (server.last_rejection or "")


@pytest.mark.gating
def test_uplink_frame_reflected_as_downlink_is_rejected() -> None:
    # Even under a shared key, direction is authenticated: reflection fails.
    shared = ChannelKeys(up=b"\x09" * 16, down=b"\x0a" * 16)
    sealer = WeightSealer(shared.up, "pi-1", "up")
    frame = sealer.seal(_tiny(0), 10, 1)
    reflected_opener = WeightOpener(shared.up, "pi-1", "down")
    assert reflected_opener.open(frame, expected_round=1) is None
    assert "direction" in (reflected_opener.last_rejection or "")


@pytest.mark.gating
def test_frame_for_another_client_is_rejected() -> None:
    frame = ClientChannel("pi-1", _KEYS).seal_update(_tiny(0), 10, 1)
    assert AggregatorChannel("pi-2", _KEYS).open_update(frame, 1) is None


def test_schema_mismatch_is_rejected() -> None:
    sealer = WeightSealer(_KEYS.up, "pi-1", "up", schema_version="w0")
    opener = WeightOpener(_KEYS.up, "pi-1", "up")
    assert opener.open(sealer.seal(_tiny(0), 1, 1), expected_round=1) is None


# --------------------------------------------------------------------- nonces and keys


@pytest.mark.gating
def test_sealers_share_one_registry_and_never_reuse_a_nonce() -> None:
    registry = NonceRegistry()
    keys = generate_demo_keys(["pi-1", "pi-2"])
    channels = [AggregatorChannel(cid, k, registry=registry) for cid, k in keys.items()]
    state = _tiny(0)
    nonces = set()
    for round_index in range(1, 51):
        for channel in channels:
            frame = channel.seal_global(state, round_index)
            nonces.add(frame[4:20])
    assert len(nonces) == 100


def test_generate_and_load_demo_keys_roundtrip(tmp_path: Path) -> None:
    keys = generate_demo_keys(["pi-1", "pi-2", "sim-3"])
    path = tmp_path / "phase8.demo.key"
    write_keys(path, keys)
    assert load_keys(path) == keys


def test_unlabelled_key_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "prod.key"
    path.write_text(json.dumps({"clients": {"pi-1": {"up": "00" * 16, "down": "11" * 16}}}))
    with pytest.raises(ValueError, match="labelled demo key file"):
        load_keys(path)


def test_channel_keys_must_differ_per_direction() -> None:
    with pytest.raises(ValueError, match="must differ"):
        ChannelKeys(up=b"\x05" * 16, down=b"\x05" * 16)


# --------------------------------------------------------------- weight AD encoding


def test_weight_ad_roundtrip_and_strictness() -> None:
    ad = WeightAssociatedData(client_id="pi-1", round=17, direction="down", schema_version="w1")
    assert WeightAssociatedData.from_bytes(ad.to_bytes()) == ad
    with pytest.raises(ValueError, match="trailing"):
        WeightAssociatedData.from_bytes(ad.to_bytes() + b"\x00")
    with pytest.raises(ValueError, match="direction"):
        WeightAssociatedData(client_id="pi-1", round=1, direction="sideways", schema_version="w1")


def test_weight_ad_and_telemetry_ad_are_mutually_unparseable() -> None:
    from ascon_smart_agri.crypto.ascon_aead import AssociatedData

    telemetry = AssociatedData("edge-1", "soil01", 42, "v1").to_bytes()
    weight = WeightAssociatedData("pi-1", 1, "up", "w1").to_bytes()
    with pytest.raises(ValueError, match="fmt_version"):
        WeightAssociatedData.from_bytes(telemetry)
    with pytest.raises(ValueError, match="fmt_version"):
        AssociatedData.from_bytes(weight)
