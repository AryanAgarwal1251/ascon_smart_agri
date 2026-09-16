"""FedAvg aggregation (Phase 4, Section III-F2, Eq. 21).

Sample-weighted FedAvg:  theta = sum_k (n_k / n) * theta_k, where n_k is the number of
TRAINING SEQUENCES held by client k --- NOT the number of raw rows. Using row counts would
over-weight clients whose data fragments into short runs (fewer sequences per row, Eq. 12).
This is a common bug; ``tests/test_fedavg_weighting.py`` guards it.

    * Unweighted averaging is retained as a selectable ablation.
    * FedProx (a proximal term mu/2 * ||theta_k - theta_global||^2) is the documented R4
      fallback if plain FedAvg diverges under strong heterogeneity.
    * Optimiser state is local and NOT aggregated (a fresh optimiser is built each round);
      averaging momentum buffers would make the method not FedAvg.

Implementation notes:

* **Averaging is done in float64 and cast back**, so that summing K weighted float32 tensors
  does not accumulate rounding drift across 20 rounds. The weights themselves are exact
  rationals n_k/n.
* **Integer and boolean buffers are not averaged**; they are taken from the first client. A
  module can carry non-floating state (counters, masks) for which a weighted mean is not a
  meaningful operation and would silently corrupt the value. The GRU detector has none, so this
  is defence in depth rather than an expected path -- but it is the kind of thing that fails
  quietly when a model gains a buffer later.
* **A client reporting n_k = 0 contributes nothing but is not an error**: under a strongly
  heterogeneous Dirichlet partition (alpha = 0.1) a client can legitimately receive too few
  blocks to form a single sequence at W = 16. Dropping it from the average is correct; treating
  it as an error would make a legitimate experimental configuration unrunnable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

# A model's parameters as a name -> tensor mapping (structured, never pickled). Inputs are
# typed as the wider Mapping and results returned as a concrete dict, so these functions
# compose with serialization.py's dict-typed StateDict without a cast at every call site.
StateDict = Mapping[str, "torch.Tensor"]


def _check_alignment(states: Sequence[StateDict]) -> None:
    if not states:
        raise ValueError("cannot aggregate an empty list of client states")
    reference = set(states[0])
    for index, state in enumerate(states[1:], start=1):
        if set(state) != reference:
            missing = reference - set(state)
            extra = set(state) - reference
            raise ValueError(
                f"client {index} state does not match client 0 "
                f"(missing: {sorted(missing)}, unexpected: {sorted(extra)})"
            )


def _combine(states: Sequence[StateDict], weights: Sequence[float]) -> dict[str, torch.Tensor]:
    """Weighted sum of aligned state dicts, accumulated in float64."""
    combined: dict[str, torch.Tensor] = {}
    for name in states[0]:
        reference = states[0][name]
        if not reference.is_floating_point():
            # Non-float buffers are taken as-is; a weighted mean of them is not meaningful.
            combined[name] = reference.clone()
            continue
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for state, weight in zip(states, weights, strict=True):
            accumulator += state[name].to(torch.float64) * weight
        combined[name] = accumulator.to(reference.dtype)
    return combined


def weighted_fedavg(
    states: Sequence[StateDict], sequence_counts: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Weighted FedAvg (Eq. 21) with weights n_k = per-client TRAINING SEQUENCE counts."""
    _check_alignment(states)
    if len(states) != len(sequence_counts):
        raise ValueError(f"got {len(states)} states but {len(sequence_counts)} sequence counts")
    if any(count < 0 for count in sequence_counts):
        raise ValueError(f"sequence counts must be non-negative, got {list(sequence_counts)}")

    total = sum(sequence_counts)
    if total == 0:
        raise ValueError(
            "every client reported n_k = 0; there are no training sequences to weight by"
        )

    # Clients holding no sequences contribute weight 0 (see the module docstring).
    weights = [count / total for count in sequence_counts]
    return _combine(states, weights)


def unweighted_average(states: Sequence[StateDict]) -> dict[str, torch.Tensor]:
    """Plain mean of client states (selectable ablation)."""
    _check_alignment(states)
    return _combine(states, [1.0 / len(states)] * len(states))


def fedprox_proximal_term(
    local_state: StateDict, global_state: StateDict, mu: float
) -> torch.Tensor:
    """FedProx proximal penalty mu/2 * ||theta_k - theta_global||^2 (R4 fallback)."""
    if mu < 0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    if set(local_state) != set(global_state):
        raise ValueError("local and global states do not have the same parameter names")

    squared = torch.zeros((), dtype=torch.float64)
    for name, local in local_state.items():
        if not local.is_floating_point():
            continue
        difference = local.to(torch.float64) - global_state[name].to(torch.float64)
        squared = squared + (difference**2).sum()
    return (mu / 2.0) * squared
