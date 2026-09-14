"""Federated scaler equivalence (Section III-F4, Eqs. 23-24).

Combining per-client sufficient statistics with the parallel (Chan's) formula must yield a
global mean/std IDENTICAL (up to floating-point tolerance) to a scaler fitted on the pooled
data --- without pooling the raw data. Activates in Phase 4.
"""

from __future__ import annotations

import numpy as np
import pytest

from ascon_smart_agri.data.scaling import fit_scaler
from ascon_smart_agri.federated.scaler_stats import combine_stats, local_sufficient_stats


@pytest.mark.gating
def test_combined_stats_equal_pooled_fit() -> None:
    """The invariant: federated combination == pooled fit, without the raw data ever pooling."""
    rng = np.random.default_rng(0)
    pooled = rng.normal(loc=[50.0, -3.0, 1000.0], scale=[7.0, 0.5, 250.0], size=(3000, 3))

    # Three clients with DELIBERATELY UNEQUAL shares -- an equal split would pass even with a
    # naive unweighted mean of client means, hiding the n_k weighting of Eq. (23).
    shares = [pooled[:1700], pooled[1700:2100], pooled[2100:]]
    assert [len(s) for s in shares] == [1700, 400, 900]

    combined_mean, combined_std = combine_stats([local_sufficient_stats(s) for s in shares])

    reference = fit_scaler(pooled)
    np.testing.assert_allclose(combined_mean, reference.mean, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(combined_std, reference.std, rtol=1e-12, atol=1e-10)


@pytest.mark.gating
def test_equivalence_holds_for_many_random_splits() -> None:
    """Not a lucky split: the identity must hold for any partition of the same rows."""
    rng = np.random.default_rng(7)
    pooled = rng.normal(loc=20.0, scale=3.0, size=(900, 4))
    reference = fit_scaler(pooled)

    for trial in range(10):
        split_rng = np.random.default_rng(trial)
        cuts = sorted(split_rng.choice(np.arange(1, 900), size=2, replace=False))
        shares = np.split(pooled, cuts)

        mean, std = combine_stats([local_sufficient_stats(s) for s in shares])

        np.testing.assert_allclose(mean, reference.mean, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(std, reference.std, rtol=1e-12, atol=1e-10)


def test_unweighted_mean_of_client_means_would_be_wrong() -> None:
    """Negative control: proves the n_k weighting in Eq. (23) is doing real work."""
    pooled = np.concatenate([np.zeros((900, 1)), np.full((100, 1), 10.0)])
    shares = [pooled[:900], pooled[900:]]

    mean, _ = combine_stats([local_sufficient_stats(s) for s in shares])

    assert mean[0] == pytest.approx(1.0)  # count-weighted: (900*0 + 100*10)/1000
    assert mean[0] != pytest.approx(5.0)  # the unweighted mean-of-means mistake


def test_large_magnitude_features_do_not_lose_precision() -> None:
    """Chan's form, not E[x^2]-E[x]^2: several CICIoT2023 features have huge means."""
    rng = np.random.default_rng(3)
    # Mean ~1e8 with spread ~1: the naive sum-of-squares route cancels catastrophically here.
    pooled = 1e8 + rng.normal(0.0, 1.0, size=(2000, 1))
    shares = [pooled[:1200], pooled[1200:]]

    _, std = combine_stats([local_sufficient_stats(s) for s in shares])

    np.testing.assert_allclose(std, fit_scaler(pooled).std, rtol=1e-9)
    assert 0.5 < float(std[0]) < 2.0  # the true spread, not a cancellation artefact


def test_single_client_is_the_pooled_fit() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(200, 2))

    mean, std = combine_stats([local_sufficient_stats(data)])

    np.testing.assert_allclose(mean, fit_scaler(data).mean, rtol=1e-12)
    np.testing.assert_allclose(std, fit_scaler(data).std, rtol=1e-12)


def test_zero_variance_feature_matches_the_pooled_convention() -> None:
    pooled = np.column_stack([np.full(100, 5.0), np.arange(100.0)])
    shares = [pooled[:60], pooled[60:]]

    _, std = combine_stats([local_sufficient_stats(s) for s in shares])

    assert std[0] == pytest.approx(1.0)  # constant feature -> 1.0, never 0.0
    np.testing.assert_allclose(std, fit_scaler(pooled).std, rtol=1e-12)


def test_client_with_no_rows_is_skipped_not_fatal() -> None:
    rng = np.random.default_rng(2)
    data = rng.normal(size=(300, 2))
    stats = [local_sufficient_stats(data[:200]), local_sufficient_stats(data[200:])]
    empty = type(stats[0])(count=0, mean=np.zeros(2), m2=np.zeros(2))

    mean, std = combine_stats([stats[0], empty, stats[1]])

    np.testing.assert_allclose(mean, fit_scaler(data).mean, rtol=1e-12)
    np.testing.assert_allclose(std, fit_scaler(data).std, rtol=1e-12)


def test_combine_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="empty list"):
        combine_stats([])
    with pytest.raises(ValueError, match="disagree on feature count"):
        combine_stats(
            [local_sufficient_stats(np.ones((5, 2))), local_sufficient_stats(np.ones((5, 3)))]
        )


def test_local_stats_reject_non_finite_and_empty_input() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        local_sufficient_stats(np.array([[1.0], [np.inf]]))
    with pytest.raises(ValueError, match="zero rows"):
        local_sufficient_stats(np.empty((0, 2)))
