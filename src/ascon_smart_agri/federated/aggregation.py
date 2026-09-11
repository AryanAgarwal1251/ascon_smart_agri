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

TODO(Phase 4): implement weighted/unweighted aggregation over safetensors state dicts and
the FedProx proximal term.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# A model's parameters as a name -> tensor mapping (structured, never pickled).
StateDict = Mapping[str, "torch.Tensor"]


def weighted_fedavg(states: Sequence[StateDict], sequence_counts: Sequence[int]) -> StateDict:
    """Weighted FedAvg (Eq. 21) with weights n_k = per-client TRAINING SEQUENCE counts."""
    del states, sequence_counts
    raise NotImplementedError("Phase 4: weighted FedAvg not implemented yet.")


def unweighted_average(states: Sequence[StateDict]) -> StateDict:
    """Plain mean of client states (selectable ablation)."""
    del states
    raise NotImplementedError("Phase 4: unweighted averaging not implemented yet.")


def fedprox_proximal_term(
    local_state: StateDict, global_state: StateDict, mu: float
) -> torch.Tensor:
    """FedProx proximal penalty mu/2 * ||theta_k - theta_global||^2 (R4 fallback)."""
    del local_state, global_state, mu
    raise NotImplementedError("Phase 4: FedProx proximal term not implemented yet.")
