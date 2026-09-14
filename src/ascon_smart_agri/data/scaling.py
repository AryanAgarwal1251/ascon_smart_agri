"""Pooled feature standardisation, fitted on TRAINING DATA ONLY (Phase 3, Section III-F4).

The GRU sees raw flow features whose magnitudes differ by orders of magnitude (``Tot sum`` runs
to tens of thousands while ``IAT`` sits near zero), so the inputs are standardised to zero mean
and unit variance before windowing.

This module is the **pooled** (centralised) scaler that Phase 3 trains against. Phase 4's
``federated/scaler_stats.py`` computes the same statistics from per-client sufficient statistics
combined by Chan's parallel formula (Eqs. 23-24), and ``tests/test_scaler_equivalence.py``
asserts the two agree -- so the statistics are deliberately stored in the ``(count, mean, M2)``
form that Chan's formula combines, not as a bare ``(mean, std)`` pair. Keeping the pooled fit
here rather than in ``federated/`` keeps Phase 4's module closed until its gate opens, while
giving the equivalence test something real to compare against.

LEAKAGE CONTRACT: fit on training rows only. A scaler fitted on train+test leaks test
distribution into training, which is the same class of error as R3's duplicate leakage --
subtler, because it moves no records across the split, only their summary statistics.

Zero-variance features are scaled by 1.0 rather than 0.0: a constant column carries no
information and dividing by its zero standard deviation would produce NaN or inf. Stage 1 of
feature selection already drops them, so this is defence in depth rather than an expected path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ScalerStats:
    """Per-feature sufficient statistics in the form Chan's parallel formula combines."""

    count: int
    mean: _Floats
    m2: _Floats  # sum of squared deviations from the mean

    @property
    def variance(self) -> _Floats:
        """Population variance, M2 / n."""
        if self.count == 0:
            raise ValueError("cannot take the variance of an empty scaler fit")
        return self.m2 / self.count

    @property
    def std(self) -> _Floats:
        """Population standard deviation, with zero-variance features left at 1.0."""
        std = np.sqrt(self.variance)
        return np.where(std > 0.0, std, 1.0)


def fit_scaler(x: npt.NDArray[np.floating]) -> ScalerStats:
    """Fit the pooled scaler on ``(rows, F)`` TRAINING features only."""
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D (rows, F) array, got shape {x.shape}")
    if len(x) == 0:
        raise ValueError("cannot fit a scaler on zero rows")
    if not np.isfinite(x).all():
        raise ValueError(
            "features contain non-finite values; drop or resolve them before fitting "
            "(this module never imputes)"
        )
    values = np.asarray(x, dtype=np.float64)
    mean = values.mean(axis=0)
    m2 = ((values - mean) ** 2).sum(axis=0)
    return ScalerStats(count=len(values), mean=mean, m2=m2)


def apply_scaler(x: npt.NDArray[np.floating], stats: ScalerStats) -> _Floats:
    """Standardise ``(rows, F)`` features with an already-fitted scaler."""
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D (rows, F) array, got shape {x.shape}")
    if x.shape[1] != len(stats.mean):
        raise ValueError(f"scaler was fitted on {len(stats.mean)} features, got {x.shape[1]}")
    scaled: _Floats = (np.asarray(x, dtype=np.float64) - stats.mean) / stats.std
    return scaled
