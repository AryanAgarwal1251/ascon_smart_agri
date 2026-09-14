"""Unit tests for pooled feature standardisation (Section III-F4).

The statistics are deliberately stored as (count, mean, M2) so Phase 4's Chan combination can
reproduce them; that shape is asserted here so a later refactor cannot quietly drop it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ascon_smart_agri.data.scaling import ScalerStats, apply_scaler, fit_scaler


def test_scaling_produces_zero_mean_unit_variance() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=50.0, scale=7.0, size=(1000, 3))

    scaled = apply_scaler(x, fit_scaler(x))

    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-10)


def test_stats_are_stored_as_count_mean_m2_for_chans_formula() -> None:
    x = np.array([[1.0], [2.0], [3.0], [4.0]])

    stats = fit_scaler(x)

    assert stats.count == 4
    np.testing.assert_allclose(stats.mean, [2.5])
    np.testing.assert_allclose(stats.m2, [5.0])  # sum (x - mean)^2
    np.testing.assert_allclose(stats.variance, [1.25])  # M2 / n, population variance


def test_zero_variance_feature_is_divided_by_one_not_zero() -> None:
    x = np.column_stack([np.full(10, 3.0), np.arange(10.0)])

    scaled = apply_scaler(x, fit_scaler(x))

    # The constant column becomes exactly 0, never NaN or inf.
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled[:, 0], 0.0)


def test_test_data_is_scaled_with_the_training_fit_not_its_own() -> None:
    """Fitting on train+test would leak the test distribution (the R3 error in subtler form)."""
    train = np.array([[0.0], [10.0]])
    test = np.array([[20.0], [30.0]])
    stats = fit_scaler(train)

    scaled = apply_scaler(test, stats)

    # Train mean 5, population std 5 -> test maps to 3 and 5, far outside [-1, 1].
    np.testing.assert_allclose(scaled.ravel(), [3.0, 5.0])


def test_fit_rejects_non_finite_input_rather_than_imputing() -> None:
    x = np.array([[1.0], [np.nan]])

    with pytest.raises(ValueError, match="non-finite"):
        fit_scaler(x)


def test_fit_rejects_empty_and_non_2d_input() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        fit_scaler(np.empty((0, 3)))
    with pytest.raises(ValueError, match="2-D"):
        fit_scaler(np.array([1.0, 2.0, 3.0]))


def test_apply_rejects_a_feature_count_mismatch() -> None:
    stats = fit_scaler(np.ones((5, 2)) * np.arange(2))

    with pytest.raises(ValueError, match="fitted on 2 features"):
        apply_scaler(np.ones((5, 3)), stats)


def test_variance_of_an_empty_fit_raises() -> None:
    empty = ScalerStats(count=0, mean=np.zeros(1), m2=np.zeros(1))

    with pytest.raises(ValueError, match="empty scaler fit"):
        _ = empty.variance
