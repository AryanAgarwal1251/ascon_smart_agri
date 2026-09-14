"""Unit tests for Dirichlet partitioning, serialization and the federated round (Section III-F).

The two blocking invariants have their own files (``test_fedavg_weighting.py``,
``test_scaler_equivalence.py``). This covers the rest of Phase 4: that the partition is a real
partition and actually responds to alpha, that nothing crosses the client/server boundary except
parameters and n_k, and that Algorithm 1's round does what it says.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ascon_smart_agri.federated.client import FederatedClient
from ascon_smart_agri.federated.partition import (
    dirichlet_block_partition,
    per_client_class_histograms,
)
from ascon_smart_agri.federated.serialization import deserialize_state, serialize_state
from ascon_smart_agri.federated.server import FederatedServer, theoretical_bytes_per_round
from ascon_smart_agri.model.gru import build_detector, count_parameters

# ---------------------------------------------------------------- partitioning (III-F1)


def _labels(n_per_class: int = 20, n_classes: int = 4) -> np.ndarray:
    return np.repeat([f"c{i}" for i in range(n_classes)], n_per_class)


def test_partition_assigns_every_block_exactly_once() -> None:
    labels = _labels()

    clients = dirichlet_block_partition(labels, n_clients=3, alpha=0.5, seed=0)

    assigned = sorted(b for blocks in clients for b in blocks)
    assert assigned == list(range(len(labels)))  # no block dropped, none duplicated


def test_partition_is_deterministic_given_a_seed() -> None:
    labels = _labels()

    first = dirichlet_block_partition(labels, n_clients=3, alpha=0.5, seed=42)
    second = dirichlet_block_partition(labels, n_clients=3, alpha=0.5, seed=42)

    assert first == second


def test_large_alpha_is_near_uniform_and_small_alpha_is_skewed() -> None:
    """alpha is the heterogeneity knob (Eq. 20); if it does nothing the sweep is meaningless."""
    labels = _labels(n_per_class=60, n_classes=4)

    near_iid = dirichlet_block_partition(labels, n_clients=3, alpha=100.0, seed=0)
    skewed = dirichlet_block_partition(labels, n_clients=3, alpha=0.05, seed=0)

    def spread(clients: list[list[int]]) -> float:
        sizes = np.array([len(c) for c in clients], dtype=float)
        return float(sizes.std() / sizes.mean())

    assert spread(near_iid) < spread(skewed)
    assert spread(near_iid) < 0.25  # large alpha -> clients look alike


def test_each_class_gets_its_own_dirichlet_draw() -> None:
    """One draw reused across classes would be a uniform partition in disguise (L2)."""
    labels = _labels(n_per_class=40, n_classes=5)

    clients = dirichlet_block_partition(labels, n_clients=3, alpha=0.3, seed=1)
    histograms = per_client_class_histograms(clients, labels)

    # Client 0's share of each class, as a fraction. With per-class draws these differ; with a
    # single shared draw they would all be equal.
    shares = [histograms[0][f"c{i}"] / 40 for i in range(5)]
    assert len(set(np.round(shares, 6))) > 1


def test_histograms_list_every_class_including_absent_ones() -> None:
    """An absent class is the most informative thing a skewed partition can report (G3)."""
    labels = _labels(n_per_class=10, n_classes=3)
    clients = dirichlet_block_partition(labels, n_clients=3, alpha=0.05, seed=3)

    histograms = per_client_class_histograms(clients, labels)

    assert len(histograms) == 3
    for histogram in histograms:
        assert set(histogram) == {"c0", "c1", "c2"}  # never omits a class
    # Block counts across clients reconstruct the corpus exactly.
    for klass in ("c0", "c1", "c2"):
        assert sum(h[klass] for h in histograms) == 10


def test_partition_rejects_bad_arguments() -> None:
    labels = _labels()
    with pytest.raises(ValueError, match="n_clients"):
        dirichlet_block_partition(labels, n_clients=0, alpha=0.5, seed=0)
    with pytest.raises(ValueError, match="alpha"):
        dirichlet_block_partition(labels, n_clients=3, alpha=0.0, seed=0)
    with pytest.raises(ValueError, match="1-D"):
        dirichlet_block_partition(np.ones((3, 3)), n_clients=3, alpha=0.5, seed=0)


def test_histograms_reject_an_out_of_range_block() -> None:
    with pytest.raises(ValueError, match="outside the corpus"):
        per_client_class_histograms([[99]], _labels(n_per_class=2, n_classes=1))


# ---------------------------------------------------------------- serialization (III-F2)


def test_state_round_trips_through_safetensors_with_n_k() -> None:
    state = {"w": torch.randn(3, 4), "b": torch.zeros(4)}

    restored, n_k = deserialize_state(serialize_state(state, 1234))

    assert n_k == 1234
    assert set(restored) == {"w", "b"}
    torch.testing.assert_close(restored["w"], state["w"])


def test_serialized_blob_is_safetensors_not_pickle() -> None:
    """Arbitrary-object deserialisation across a transport boundary is a code-execution hazard."""
    blob = serialize_state({"w": torch.ones(2)}, 5)

    assert not blob.startswith(b"\x80")  # pickle protocol marker
    assert b"__metadata__" in blob[:256]  # safetensors JSON header


def test_n_k_travels_with_the_parameters_not_as_a_tensor() -> None:
    """Keeping n_k out of the tensor map means the state loads into a module unmodified."""
    restored, _ = deserialize_state(serialize_state({"w": torch.ones(2)}, 7))

    assert "sequence_count" not in restored


def test_deserialize_rejects_a_missing_count_rather_than_defaulting() -> None:
    from safetensors.torch import save

    with pytest.raises(ValueError, match="no 'sequence_count' metadata"):
        deserialize_state(save({"w": torch.ones(2)}))


def test_deserialize_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="too short"):
        deserialize_state(b"abc")
    with pytest.raises(ValueError, match="exceeds the blob"):
        deserialize_state((10**6).to_bytes(8, "little") + b"{}")


def test_serialize_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        serialize_state({"w": torch.ones(1)}, -1)


# ---------------------------------------------------------------- the round (Algorithm 1)


def _client(client_id: int, n: int, *, n_features: int = 4, window: int = 3) -> FederatedClient:
    rng = np.random.default_rng(client_id)
    y = (np.arange(n) % 2).astype(np.int64)
    x = (rng.normal(0, 0.3, size=(n, window, n_features)) + y[:, None, None] * 3.0).astype(
        np.float32
    )
    return FederatedClient(client_id, x, y, hidden_size=8, n_classes=2)


def test_round_returns_an_aggregated_state_of_the_right_shape() -> None:
    clients = [_client(0, 40), _client(1, 60)]
    server = FederatedServer(clients)
    model = build_detector(4, 8, 2)
    global_state = {k: v.clone() for k, v in model.state_dict().items()}

    new_state = server.run_round(global_state, local_epochs=1)

    assert set(new_state) == set(global_state)
    for name, tensor in new_state.items():
        assert tensor.shape == global_state[name].shape


def test_round_actually_changes_the_global_parameters() -> None:
    server = FederatedServer([_client(0, 60), _client(1, 60)])
    model = build_detector(4, 8, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    after = server.run_round(before, local_epochs=2)

    assert not torch.allclose(after["head.weight"], before["head.weight"])


def test_server_reports_per_client_sequence_counts() -> None:
    server = FederatedServer([_client(0, 40), _client(1, 75)])

    assert server.sequence_counts() == [40, 75]


def test_client_with_no_data_returns_the_global_state_and_zero() -> None:
    """Legitimate under a strongly heterogeneous partition, not an error."""
    empty = FederatedClient(3, None, None, hidden_size=8, n_classes=2)
    model = build_detector(4, 8, 2)
    state = model.state_dict()

    returned, n_k = empty.local_train(state, local_epochs=1)

    assert n_k == 0
    torch.testing.assert_close(returned["head.bias"], state["head.bias"])


def test_client_does_not_expose_its_data_publicly() -> None:
    """A1: independence is enforced in code, not merely documented."""
    client = _client(0, 20)

    public = [name for name in vars(client) if not name.startswith("_")]

    assert "sequences" not in public
    assert "labels" not in public
    assert client.n_sequences == 20  # only the count is public


def test_server_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="at least one client"):
        FederatedServer([])
    with pytest.raises(ValueError, match="weighted"):
        FederatedServer([_client(0, 10)], aggregation="median")


def test_client_rejects_non_positive_local_epochs() -> None:
    with pytest.raises(ValueError, match="local_epochs"):
        _client(0, 10).local_train(build_detector(4, 8, 2).state_dict(), local_epochs=0)


# ---------------------------------------------------------------- communication cost (Eq. 22)


def test_theoretical_round_cost_matches_the_papers_figure() -> None:
    """Section III-F3: K=3, |theta|=33,800, b=4 -> 811,200 bytes, about 0.77 MiB."""
    assert count_parameters(build_detector(16, 96, 8)) == 33_800

    cost = theoretical_bytes_per_round(3, 33_800)

    assert cost == 811_200
    assert cost / 1024**2 == pytest.approx(0.7736, abs=1e-4)
    # A 20-round schedule: the paper says roughly 15.5 MiB.
    assert (cost * 20) / 1024**2 == pytest.approx(15.47, abs=0.05)


def test_measured_round_cost_is_close_to_the_theoretical_figure() -> None:
    """The claim is about what is actually sent, so measure it rather than trust Eq. (22)."""
    model = build_detector(16, 96, 8)  # the reference sizing, 33,800 parameters
    clients = [_client(i, 30, n_features=16) for i in range(3)]
    for client in clients:
        client.hidden_size, client.n_classes = 96, 8
    server = FederatedServer(clients)

    server.run_round({k: v.clone() for k, v in model.state_dict().items()}, local_epochs=1)

    theoretical = theoretical_bytes_per_round(3, 33_800)
    # Serialised blobs carry a small JSON header on top of the raw tensor bytes.
    assert server.last_round_bytes == pytest.approx(theoretical, rel=0.02)


def test_theoretical_cost_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        theoretical_bytes_per_round(0, 33_800)


# --- Local training seed (Section III-I4) ---------------------------------------------------


def test_training_seed_separates_run_client_and_round() -> None:
    """Seeding with the bare client id makes rounds repeat and seeds stop mattering.

    Both consequences are invisible at runtime and only distort the reported numbers: a client
    would replay one batch permutation every round, and the >= 3-seed std of Section III-I4
    would sample only the initial parameters and the Dirichlet draw.
    """
    client = FederatedClient(0, seed=0)

    # Rounds must differ from one another...
    per_round = {client.training_seed(r) for r in range(8)}
    assert len(per_round) == 8

    # ...the run seed must change the schedule...
    assert FederatedClient(0, seed=1).training_seed(1) != client.training_seed(1)

    # ...and two clients must not train identically within a round.
    assert FederatedClient(1, seed=0).training_seed(1) != client.training_seed(1)


def test_training_seed_is_reproducible() -> None:
    """Varying is not the same as random: the manifest must be able to reproduce the run."""
    assert FederatedClient(2, seed=7).training_seed(3) == FederatedClient(2, seed=7).training_seed(
        3
    )


def test_local_train_advances_the_round_counter() -> None:
    """The round index must actually advance, or every round re-seeds identically again."""
    client = _client(0, 40)
    state = build_detector(4, 8, 2).state_dict()

    seeds = []
    for _ in range(3):
        seeds.append(client.training_seed(client._round + 1))
        client.local_train({k: v.clone() for k, v in state.items()}, local_epochs=1)

    assert len(set(seeds)) == 3
