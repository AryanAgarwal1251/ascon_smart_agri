"""GATING TEST (Phase 8, plan Section 6): networked federation == in-process federation.

Three :class:`ClientNode` threads talk to one :class:`AggregatorNode` over loopback TCP with
every frame Ascon-sealed. The global state they converge to must equal, bit for bit, what
:class:`FederatedServer` produces in-process from the same clients, seed and rounds --- the
network and the cipher must change nothing about what is computed. A second test puts a
tampering proxy on the wire and checks the aggregator applies nothing and the client recovers.
"""

from __future__ import annotations

import socket
import threading

import numpy as np
import pytest
import torch

from ascon_smart_agri.federated.client import FederatedClient
from ascon_smart_agri.federated.node import (
    SCALER_ROUND,
    AggregatorNode,
    ClientNode,
    initial_global_state,
    recv_message,
    send_message,
)
from ascon_smart_agri.federated.scaler_stats import combine_stats, local_sufficient_stats
from ascon_smart_agri.federated.server import FederatedServer
from ascon_smart_agri.federated.transport import generate_demo_keys

_F, _H, _C, _W = 4, 8, 8, 3
_IDS = ["pi-1", "pi-2", "sim-3"]


def _raw_partition(seed: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, _F)) * (seed + 1) + seed
    y = rng.integers(0, _C, size=rows)
    return x, y


def _windows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(x) - _W + 1)
    seqs = np.stack([x[s : s + _W] for s in starts]).astype(np.float32)
    return seqs, y[starts + _W - 1].astype(np.int64)


def _client(cid: int, x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray, seed: int):
    seqs, labels = _windows((x - mean) / std, y)
    return FederatedClient(
        cid, seqs, labels, seed=seed, hidden_size=_H, n_classes=_C, batch_size=16
    )


@pytest.mark.gating
def test_loopback_federation_matches_in_process_bit_for_bit() -> None:
    torch.set_num_threads(1)  # deterministic reductions across the three client threads
    seed, rounds, epochs = 0, 2, 1
    raw = [_raw_partition(i, 40 + 10 * i) for i in range(3)]
    stats = [local_sufficient_stats(x) for x, _ in raw]

    # ---- reference: in-process, exactly as Phase 4 does it ----------------------------
    mean, std = combine_stats(stats)
    ref_clients = [_client(i, x, y, mean, std, seed) for i, (x, y) in enumerate(raw)]
    server = FederatedServer(ref_clients)
    reference = initial_global_state(seed, _F, _H, _C)
    for _ in range(rounds):
        reference = server.run_round(reference, local_epochs=epochs)

    # ---- networked: three client threads, one aggregator, all frames sealed -------------
    keys = generate_demo_keys(_IDS)
    aggregator = AggregatorNode(_IDS, keys, rounds=rounds, port=0)
    aggregator.start()
    nodes = [
        ClientNode(
            cid,
            keys[cid],
            server=(aggregator.host, aggregator.port),
            local_stats=stats[i],
            make_client=lambda m, s, i=i: _client(i, raw[i][0], raw[i][1], m, s, seed),
            initial_state=initial_global_state(seed, _F, _H, _C),
            rounds=rounds,
            local_epochs=epochs,
        )
        for i, cid in enumerate(_IDS)
    ]
    threads = [threading.Thread(target=node.run) for node in nodes]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    aggregator.stop()

    assert aggregator.finished
    assert aggregator.global_state is not None
    assert aggregator.rejections == []
    for node in nodes:
        assert node.global_state is not None
        assert set(node.global_state) == set(reference)
        assert all(torch.equal(node.global_state[k], reference[k]) for k in reference)
    assert all(torch.equal(aggregator.global_state[k], reference[k]) for k in reference)

    # The scaler round travelled sealed too, and combined to the same (mean, std).
    assert aggregator.scaler is not None
    np.testing.assert_allclose(aggregator.scaler[0], mean)
    np.testing.assert_allclose(aggregator.scaler[1], std)
    assert [r.round_index for r in aggregator.records] == [SCALER_ROUND, 1, 2]
    assert aggregator.records[1].sequence_counts == {
        cid: c.n_sequences for cid, c in zip(_IDS, ref_clients, strict=True)
    }


class _TamperingProxy:
    """Forwards client -> aggregator, flipping one ciphertext bit of the first frame only."""

    def __init__(self, upstream: tuple[str, int]) -> None:
        self.upstream = upstream
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        self.tampered = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._relay, args=(conn,), daemon=True).start()

    def _relay(self, conn: socket.socket) -> None:
        with conn:
            frame = recv_message(conn)
            if self.tampered == 0:
                self.tampered += 1
                frame = frame[:-1] + bytes([frame[-1] ^ 0x01])  # last tag byte
            try:
                with socket.create_connection(self.upstream) as up:
                    send_message(up, frame)
                    send_message(conn, recv_message(up))
            except OSError:
                return  # aggregator dropped it: the client sees a closed socket and retries

    def close(self) -> None:
        self.listener.close()


@pytest.mark.gating
def test_tampered_frame_on_the_wire_is_dropped_and_client_recovers() -> None:
    torch.set_num_threads(1)
    keys = generate_demo_keys(["pi-1"])
    aggregator = AggregatorNode(["pi-1"], keys, rounds=1, port=0)
    aggregator.start()
    proxy = _TamperingProxy((aggregator.host, aggregator.port))

    x, y = _raw_partition(0, 40)
    node = ClientNode(
        "pi-1",
        keys["pi-1"],
        server=("127.0.0.1", proxy.port),
        local_stats=local_sufficient_stats(x),
        make_client=lambda m, s: _client(0, x, y, m, s, 0),
        initial_state=initial_global_state(0, _F, _H, _C),
        rounds=1,
        local_epochs=1,
        retry_delay_s=0.05,
    )
    final = node.run()
    proxy.close()
    aggregator.stop()

    assert proxy.tampered == 1
    assert len(aggregator.rejections) == 1
    assert aggregator.rejections[0][0] == "pi-1"
    assert "authentication failed" in aggregator.rejections[0][1]
    assert aggregator.finished and aggregator.global_state is not None
    assert all(torch.equal(final[k], aggregator.global_state[k]) for k in final)
