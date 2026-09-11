"""Federated scaler equivalence (Section III-F4, Eqs. 23-24).

Combining per-client sufficient statistics with the parallel (Chan's) formula must yield a
global mean/std IDENTICAL (up to floating-point tolerance) to a scaler fitted on the pooled
data --- without pooling the raw data. Activates in Phase 4.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.federated.scaler_stats import combine_stats, local_sufficient_stats


@pytest.mark.skip(reason="pending Phase 4: scaler statistics not implemented yet")
def test_combined_stats_equal_pooled_fit() -> None:
    assert callable(local_sufficient_stats)
    assert callable(combine_stats)
    raise AssertionError("implement in Phase 4: split data across clients, combine, compare")
