"""The in-process sealed federation (Phase 9) is the Phase 4 federation plus the Phase 8 channel.

Two invariants: sealing changes nothing about the arithmetic (the global model after R rounds
is bit-for-bit the plain server's), and a frame that fails to open is dropped -- the client
contributes weight 0 that round rather than a parameter vector nobody authenticated.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ascon_smart_agri.federated.client import FederatedClient
from ascon_smart_agri.federated.server import FederatedServer, SealedFederatedServer
from ascon_smart_agri.federated.transport import ClientChannel, generate_demo_keys
from ascon_smart_agri.model.gru import build_detector

_F, _H, _C, _W = 4, 8, 8, 3
_IDS = ["ciciot2023", "ciciomt2024", "edge_iiotset"]


def _client(cid: int, seed: int, rows: int = 40) -> FederatedClient:
    rng = np.random.default_rng(seed + cid)
    x = rng.normal(size=(rows, _F)).astype(np.float32)
    y = rng.integers(0, _C, size=rows)
    starts = np.arange(0, rows - _W + 1)
    seqs = np.stack([x[s : s + _W] for s in starts])
    labels = y[starts + _W - 1].astype(np.int64)
    return FederatedClient(
        cid, seqs, labels, seed=seed, hidden_size=_H, n_classes=_C, batch_size=16
    )


def _initial(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in build_detector(_F, _H, _C).state_dict().items()}


def _run(server: FederatedServer, rounds: int, seed: int) -> dict[str, torch.Tensor]:
    state = _initial(seed)
    for _ in range(rounds):
        state = server.run_round(state, local_epochs=1)
    return state


@pytest.mark.gating
def test_sealed_server_matches_plain_server_bit_for_bit() -> None:
    seed = 3
    plain = FederatedServer([_client(i, seed) for i in range(3)])
    sealed = SealedFederatedServer(
        [_client(i, seed) for i in range(3)], _IDS, generate_demo_keys(_IDS)
    )
    a = _run(plain, rounds=3, seed=seed)
    b = _run(sealed, rounds=3, seed=seed)
    for name in a:
        assert torch.equal(a[name], b[name]), name
    assert sealed.frames_dropped == 0
    assert all(v == 0 for v in sealed.frames_rejected_by_client.values())
    # Sealing is not free: the wire carries nonce + AD + tag on every frame, both directions.
    assert sealed.last_round_bytes > plain.last_round_bytes


def test_tampered_update_frame_is_dropped_not_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    seed = 5
    keys = generate_demo_keys(_IDS)
    sealed = SealedFederatedServer([_client(i, seed) for i in range(3)], _IDS, keys)

    # Client 1's update is corrupted on the wire (one ciphertext byte flipped) every round.
    real_seal = ClientChannel.seal_update

    def corrupting_seal(self: ClientChannel, state, n_k, round_index):  # type: ignore[no-untyped-def]
        frame = bytearray(real_seal(self, state, n_k, round_index))
        if self.client_id == _IDS[1]:
            frame[-1] ^= 0x01
        return bytes(frame)

    monkeypatch.setattr(ClientChannel, "seal_update", corrupting_seal)
    tampered = _run(sealed, rounds=2, seed=seed)
    assert sealed.frames_dropped == 2  # one per round, client 1 only

    # The aggregate equals a federation of the other two clients alone: the tampered client
    # was not merely down-weighted, it was absent.
    monkeypatch.undo()
    two = SealedFederatedServer([_client(0, seed), _client(2, seed)], [_IDS[0], _IDS[2]], keys)
    reference = _run(two, rounds=2, seed=seed)
    for name in reference:
        assert torch.equal(tampered[name], reference[name]), name
