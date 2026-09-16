"""FedAvg weighting correctness (Section III-F2, Eq. 21).

The aggregation weight n_k must be the number of TRAINING SEQUENCES per client, not raw rows.
Using row counts over-weights clients whose data fragments into short runs (fewer sequences
per row, Eq. 12). This is a common, silent bug. Activates in Phase 4.
"""

from __future__ import annotations

import pytest
import torch

from ascon_smart_agri.federated.aggregation import (
    fedprox_proximal_term,
    unweighted_average,
    weighted_fedavg,
)


def _state(value: float) -> dict[str, torch.Tensor]:
    return {"w": torch.full((2, 2), value), "b": torch.tensor([value])}


@pytest.mark.gating
def test_weight_uses_sequence_count_not_row_count() -> None:
    """The invariant: weights are SEQUENCE counts, and row counts give a different answer.

    Two clients hold the same number of rows but different run structure, so client 0 yields
    far more sequences (Eq. 12). Weighting by rows would average them equally; weighting by
    sequences must pull the result towards client 0.
    """
    states = [_state(0.0), _state(10.0)]

    # Same rows each (say 1000), but client 0's data forms 900 windows and client 1's only 100.
    sequence_counts = [900, 100]
    row_counts = [1000, 1000]

    by_sequences = weighted_fedavg(states, sequence_counts)
    by_rows = weighted_fedavg(states, row_counts)

    # Eq. (21) with n_k = sequences: (900*0 + 100*10) / 1000 = 1.0
    assert by_sequences["w"].flatten()[0].item() == pytest.approx(1.0)
    # The row-count mistake would give the plain mean, 5.0 -- a materially different model.
    assert by_rows["w"].flatten()[0].item() == pytest.approx(5.0)
    assert by_sequences["w"].flatten()[0].item() != by_rows["w"].flatten()[0].item()


@pytest.mark.gating
def test_weighted_average_matches_equation_21_exactly() -> None:
    states = [_state(1.0), _state(2.0), _state(4.0)]
    counts = [10, 30, 60]

    result = weighted_fedavg(states, counts)

    # (10*1 + 30*2 + 60*4) / 100 = 3.1
    assert result["w"].flatten()[0].item() == pytest.approx(3.1)
    assert result["b"].item() == pytest.approx(3.1)


def test_equal_counts_reduce_to_the_plain_mean() -> None:
    states = [_state(2.0), _state(4.0)]

    weighted = weighted_fedavg(states, [50, 50])
    unweighted = unweighted_average(states)

    assert weighted["w"].flatten()[0].item() == pytest.approx(3.0)
    assert unweighted["w"].flatten()[0].item() == pytest.approx(3.0)


def test_client_with_zero_sequences_contributes_nothing() -> None:
    """Legitimate at alpha=0.1: a client can receive too few blocks to form any window."""
    states = [_state(1.0), _state(99.0)]

    result = weighted_fedavg(states, [40, 0])

    assert result["w"].flatten()[0].item() == pytest.approx(1.0)


def test_aggregation_preserves_shapes_and_dtypes() -> None:
    states = [_state(1.0), _state(3.0)]

    result = weighted_fedavg(states, [1, 1])

    assert result["w"].shape == (2, 2)
    assert result["w"].dtype == torch.float32  # accumulated in float64, returned as float32
    assert result["b"].shape == (1,)


def test_non_float_buffers_are_not_averaged() -> None:
    """A weighted mean of a counter or mask is not meaningful; take it from the first client."""
    states = [
        {"w": torch.tensor([2.0]), "steps": torch.tensor([10])},
        {"w": torch.tensor([4.0]), "steps": torch.tensor([20])},
    ]

    result = weighted_fedavg(states, [1, 1])

    assert result["w"].item() == pytest.approx(3.0)
    assert result["steps"].item() == 10  # not 15
    assert result["steps"].dtype == torch.int64


def test_aggregation_rejects_misaligned_states() -> None:
    with pytest.raises(ValueError, match="does not match client 0"):
        weighted_fedavg([{"a": torch.ones(1)}, {"b": torch.ones(1)}], [1, 1])


def test_aggregation_rejects_bad_counts() -> None:
    states = [_state(1.0), _state(2.0)]
    with pytest.raises(ValueError, match="sequence counts"):
        weighted_fedavg(states, [1])
    with pytest.raises(ValueError, match="non-negative"):
        weighted_fedavg(states, [1, -1])
    with pytest.raises(ValueError, match="no training sequences"):
        weighted_fedavg(states, [0, 0])


def test_aggregation_rejects_an_empty_client_list() -> None:
    with pytest.raises(ValueError, match="empty list"):
        weighted_fedavg([], [])


def test_fedprox_proximal_term_is_zero_at_the_global_point() -> None:
    state = _state(3.0)

    assert fedprox_proximal_term(state, state, mu=0.1).item() == pytest.approx(0.0)


def test_fedprox_proximal_term_grows_with_distance() -> None:
    global_state = _state(0.0)
    near, far = _state(1.0), _state(2.0)

    # mu/2 * ||theta_k - theta||^2 over 5 elements: 0.1/2 * 5 = 0.25, and 0.1/2 * 20 = 1.0.
    assert fedprox_proximal_term(near, global_state, mu=0.1).item() == pytest.approx(0.25)
    assert fedprox_proximal_term(far, global_state, mu=0.1).item() == pytest.approx(1.0)


def test_fedprox_rejects_negative_mu_and_misaligned_states() -> None:
    state = _state(1.0)
    with pytest.raises(ValueError, match="mu must be non-negative"):
        fedprox_proximal_term(state, state, mu=-1.0)
    with pytest.raises(ValueError, match="same parameter names"):
        fedprox_proximal_term({"a": torch.ones(1)}, {"b": torch.ones(1)}, mu=0.1)
