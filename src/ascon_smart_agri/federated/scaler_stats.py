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

Implementation notes:

* **The statistics are held as (count, mean, M2), not (count, sum, sum-of-squares).** The paper
  names the latter, and the two carry the same information, but the naive sum-of-squares route
  computes the variance as ``E[x^2] - E[x]^2``, which catastrophically cancels when the mean is
  large relative to the spread -- precisely the regime several CICIoT2023 features sit in
  (``Tot sum`` has mean ~25,800). Chan's update is the numerically stable form and is what
  Eqs. (23-24) actually specify, so the transmitted triple matches the scaffold's
  :class:`FeatureStats` rather than the prose.
* The combination is **exact, not approximate**: for any split of the same rows it reproduces
  the pooled fit to floating-point tolerance, which is what makes the equivalence test a real
  check rather than a loose one.
* Zero-variance features report a standard deviation of 1.0, matching ``data/scaling.py`` so the
  federated and pooled scalers agree on that edge case too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._types import Array

_Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature sufficient statistics; only these cross the client boundary."""

    count: int
    mean: Array  # mu_k
    m2: Array  # sum of squared deviations from the mean (M2_k)


def local_sufficient_stats(x: Array) -> FeatureStats:
    """Compute per-feature count/mean/M2 on a client's local training features.

    This is the ONLY thing a client sends for scaling. No record of ``x`` itself crosses the
    boundary (Section III-F4).
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D (rows, F) array, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("cannot compute sufficient statistics on zero rows")
    if not np.isfinite(values).all():
        raise ValueError("features contain non-finite values; resolve them before fitting")

    mean = values.mean(axis=0)
    m2 = ((values - mean) ** 2).sum(axis=0)
    return FeatureStats(count=len(values), mean=mean, m2=m2)


def combine_stats(parts: list[FeatureStats]) -> tuple[Array, Array]:
    """Combine client stats via Eqs. 23-24; return global ``(mean, std)``."""
    if not parts:
        raise ValueError("cannot combine an empty list of client statistics")

    widths = {len(np.asarray(p.mean)) for p in parts}
    if len(widths) != 1:
        raise ValueError(f"clients disagree on feature count: {sorted(widths)}")
    if any(p.count < 0 for p in parts):
        raise ValueError("client counts must be non-negative")

    contributing = [p for p in parts if p.count > 0]
    if not contributing:
        raise ValueError("every client reported zero rows; nothing to combine")

    n = sum(p.count for p in contributing)
    counts = np.array([p.count for p in contributing], dtype=np.float64)
    means = np.stack([np.asarray(p.mean, dtype=np.float64) for p in contributing])
    m2s = np.stack([np.asarray(p.m2, dtype=np.float64) for p in contributing])

    # Eq. (23): the count-weighted mean of the client means.
    mean: _Floats = (counts[:, None] * means).sum(axis=0) / n
    # Eq. (24): each client's M2 plus the shift of its mean away from the global mean.
    m2: _Floats = (m2s + counts[:, None] * (means - mean) ** 2).sum(axis=0)

    std = np.sqrt(m2 / n)
    # Match data/scaling.py: a constant feature is divided by 1.0, never 0.0.
    return mean, np.where(std > 0.0, std, 1.0)
