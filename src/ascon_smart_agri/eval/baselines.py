"""The five required baselines (Phase 3/4, Section III-I1).

Reported for EVERY configuration:
    1. random forest on single records      (motivated by S13's dissenting result)
    2. MLP on single records                 (isolates the contribution of recurrence)
    3. centralised GRU                        (upper bound attainable by pooling)
    4. three local-only GRUs                  (lower bound attainable without federating)
    5. federated global GRU                   (the method under test)

Baselines 3 and 4 bracket the federated result and together answer gap G4 (local-only is what
an operator gets by declining to federate). Every headline number is mean +/- std over >= 3
seeds (III-I4).

TODO(Phase 3/4): implement each baseline's train/eval entry point.
"""

from __future__ import annotations

from .._types import Array
from .metrics import MulticlassMetrics


def random_forest_single_record(x: Array, y: Array, *, seed: int) -> MulticlassMetrics:
    """Baseline 1: random forest on single records."""
    del x, y, seed
    raise NotImplementedError("Phase 3: RF baseline not implemented yet.")


def mlp_single_record(x: Array, y: Array, *, seed: int) -> MulticlassMetrics:
    """Baseline 2: MLP on single records (no recurrence)."""
    del x, y, seed
    raise NotImplementedError("Phase 3: MLP baseline not implemented yet.")


def centralized_gru(seqs: Array, y: Array, *, seed: int) -> MulticlassMetrics:
    """Baseline 3: centralised GRU (upper bound)."""
    del seqs, y, seed
    raise NotImplementedError("Phase 3: centralised-GRU baseline not implemented yet.")


def local_only_grus(
    client_seqs: list[Array], client_y: list[Array], *, seed: int
) -> list[MulticlassMetrics]:
    """Baseline 4: one GRU per client, no federation (lower bound)."""
    del client_seqs, client_y, seed
    raise NotImplementedError("Phase 4: local-only baseline not implemented yet.")


def federated_global_gru(
    client_seqs: list[Array], client_y: list[Array], *, seed: int
) -> MulticlassMetrics:
    """Baseline 5: the federated global GRU (method under test)."""
    del client_seqs, client_y, seed
    raise NotImplementedError("Phase 4: federated-global baseline not implemented yet.")
