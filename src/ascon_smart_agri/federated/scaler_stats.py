"""Federated preprocessing statistics (Phase 4, Section III-F4, Eqs. 23-24).

If each client fits its own scaler, the global model sees three different input distributions
and a shared test set is ill-defined; if one scaler is fitted on pooled raw data, the
no-raw-data constraint is violated. We resolve this by having clients transmit ONLY sufficient
statistics --- count, sum, and sum of squares per feature --- which the server combines with
the parallel (Chan's) formulation:

    n   = sum_k n_k
    mu  = (1/n) sum_k n_k mu_k
    M2  = sum_k ( M2_k + n_k (mu_k - mu)^2 )
    var = M2 / n

The result is IDENTICAL to a scaler fitted on the pooled training data, but only aggregates
cross the client boundary. The global scaler is distributed once, before round one.
``tests/test_scaler_equivalence.py`` asserts equality with a pooled fit.

TODO(Phase 4): implement local sufficient statistics + Chan combination.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._types import Array


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature sufficient statistics; only these cross the client boundary."""

    count: int
    mean: Array  # mu_k
    m2: Array  # sum of squared deviations from the mean (M2_k)


def local_sufficient_stats(x: Array) -> FeatureStats:
    """Compute per-feature count/mean/M2 on a client's local training features."""
    del x
    raise NotImplementedError("Phase 4: local sufficient statistics not implemented yet.")


def combine_stats(parts: list[FeatureStats]) -> tuple[Array, Array]:
    """Combine client stats via Eqs. 23-24; return global ``(mean, std)``."""
    del parts
    raise NotImplementedError("Phase 4: Chan combination not implemented yet.")
