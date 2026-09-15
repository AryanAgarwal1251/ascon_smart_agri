"""Unit tests for centralised training (Section III-E) and the reporting guard (III-I5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ascon_smart_agri.eval.report import format_seed_summary, mean_std, near_ceiling_note
from ascon_smart_agri.model.gru import build_detector
from ascon_smart_agri.model.train import class_weights, predict, train_centralized, train_module


def test_class_weights_match_equation_18() -> None:
    # w_c = n / (C * n_c); n = 100, C = 2.
    weights = class_weights({"a": 90, "b": 10})

    assert weights["a"] == pytest.approx(100 / (2 * 90))
    assert weights["b"] == pytest.approx(100 / (2 * 10))
    assert weights["b"] > weights["a"]  # the rare class is up-weighted


def test_class_weights_are_uniform_on_balanced_data() -> None:
    weights = class_weights({"a": 50, "b": 50})

    assert weights == pytest.approx({"a": 1.0, "b": 1.0})


def test_absent_class_gets_zero_weight_not_infinity() -> None:
    """Eq. (18) divides by n_c, undefined at zero; an infinite weight would dominate the loss."""
    weights = class_weights({"a": 10, "b": 0})

    assert weights["b"] == 0.0
    assert np.isfinite(weights["a"])


def test_class_weights_reject_degenerate_input() -> None:
    with pytest.raises(ValueError, match="empty count map"):
        class_weights({})
    with pytest.raises(ValueError, match="negative class counts"):
        class_weights({"a": -1})
    with pytest.raises(ValueError, match="every class count is zero"):
        class_weights({"a": 0, "b": 0})


def _separable_task(n: int = 240, window: int = 4, n_features: int = 3, seed: int = 0):
    """Two classes separated by a constant offset: trainable in a couple of epochs."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(0, 0.3, size=(n, window, n_features)) + y[:, None, None] * 3.0
    return x.astype(np.float32), y


def test_training_learns_a_separable_task() -> None:
    x, y = _separable_task()

    model = train_centralized(x, y, hidden_size=16, n_classes=2, epochs=6, seed=0)
    predicted, _ = predict(model, x)

    assert (predicted == y).mean() > 0.9


def test_training_is_reproducible_given_a_seed() -> None:
    x, y = _separable_task()

    first, _ = predict(train_centralized(x, y, hidden_size=8, n_classes=2, epochs=2, seed=7), x)
    second, _ = predict(train_centralized(x, y, hidden_size=8, n_classes=2, epochs=2, seed=7), x)

    np.testing.assert_array_equal(first, second)


def test_predict_returns_normalised_probabilities() -> None:
    x, y = _separable_task(n=40)
    model = train_centralized(x, y, hidden_size=8, n_classes=2, epochs=1, seed=0)

    predicted, probabilities = predict(model, x)

    assert probabilities.shape == (40, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(predicted, probabilities.argmax(axis=1))


def test_training_rejects_degenerate_input() -> None:
    x, y = _separable_task(n=20)

    with pytest.raises(ValueError, match=r"\(N, W, F\)"):
        train_centralized(x[:, 0], y, hidden_size=8, n_classes=2, epochs=1, seed=0)
    with pytest.raises(ValueError, match="lengths disagree"):
        train_centralized(x, y[:5], hidden_size=8, n_classes=2, epochs=1, seed=0)
    with pytest.raises(ValueError, match="epochs must be positive"):
        train_centralized(x, y, hidden_size=8, n_classes=2, epochs=0, seed=0)


def test_training_rejects_labels_outside_the_class_range() -> None:
    x, y = _separable_task(n=20)
    y = y.copy()
    y[0] = 5

    with pytest.raises(ValueError, match=r"labels must lie in \[0, 2\)"):
        train_centralized(x, y, hidden_size=8, n_classes=2, epochs=1, seed=0)


def test_near_ceiling_note_fires_at_and_above_the_threshold() -> None:
    assert near_ceiling_note(0.985, threshold=0.98) is not None
    assert near_ceiling_note(0.98, threshold=0.98) is not None  # at the threshold too
    assert near_ceiling_note(0.97, threshold=0.98) is None


def test_near_ceiling_note_points_at_macro_f1_and_fpr() -> None:
    note = near_ceiling_note(0.99, threshold=0.98)

    assert note is not None
    assert "macro-F1" in note
    assert "false-positive rate" in note


def test_near_ceiling_note_rejects_a_non_fraction_accuracy() -> None:
    with pytest.raises(ValueError, match="fraction"):
        near_ceiling_note(99.0, threshold=0.98)


def test_mean_std_over_seeds() -> None:
    mean, std = mean_std([0.80, 0.84, 0.88])

    assert mean == pytest.approx(0.84)
    assert std == pytest.approx(np.std([0.80, 0.84, 0.88]))


def test_mean_std_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        mean_std([])


def test_format_seed_summary_reports_mean_std_and_n() -> None:
    summary = format_seed_summary("macro_f1", [0.5, 0.6, 0.7])

    assert "macro_f1" in summary
    assert "0.6000" in summary
    assert "n=3" in summary


def test_fedprox_pulls_training_back_toward_the_reference() -> None:
    """A real behavioural check: high mu must keep parameters closer to the round's start
    than plain FedAvg (mu=None) training on the same data does."""
    x, y = _separable_task(n=200, seed=1)
    reference = {
        name: tensor.clone()
        for name, tensor in build_detector(x.shape[2], 8, 2).state_dict().items()
    }

    # Load the SAME starting point into both, so only mu differs.
    plain = build_detector(x.shape[2], 8, 2)
    plain.load_state_dict(reference)
    plain = train_module(plain, x, y, n_classes=2, epochs=8, seed=0)

    constrained = build_detector(x.shape[2], 8, 2)
    constrained.load_state_dict(reference)
    constrained = train_module(
        constrained,
        x,
        y,
        n_classes=2,
        epochs=8,
        seed=0,
        fedprox_mu=10.0,
        fedprox_reference=reference,
    )

    def drift(trained: object) -> float:
        return float(
            sum(
                (trained.state_dict()[name] - reference[name]).pow(2).sum()  # type: ignore[attr-defined]
                for name in reference
            )
        )

    assert drift(constrained) < drift(plain)


def test_fedprox_with_mu_zero_barely_differs_from_plain_training() -> None:
    """mu=0 should behave like plain FedAvg -- the penalty term is a no-op at mu=0."""
    x, y = _separable_task(n=120, seed=2)
    reference = {
        name: tensor.clone()
        for name, tensor in build_detector(x.shape[2], 8, 2).state_dict().items()
    }

    plain = build_detector(x.shape[2], 8, 2)
    plain.load_state_dict(reference)
    plain = train_module(plain, x, y, n_classes=2, epochs=5, seed=3)

    zero_mu = build_detector(x.shape[2], 8, 2)
    zero_mu.load_state_dict(reference)
    zero_mu = train_module(
        zero_mu, x, y, n_classes=2, epochs=5, seed=3, fedprox_mu=0.0, fedprox_reference=reference
    )

    for name in reference:
        torch.testing.assert_close(
            plain.state_dict()[name], zero_mu.state_dict()[name], atol=1e-5, rtol=1e-4
        )


def test_fedprox_requires_mu_and_reference_together() -> None:
    x, y = _separable_task(n=40)
    reference = {
        name: tensor.clone()
        for name, tensor in build_detector(x.shape[2], 8, 2).state_dict().items()
    }

    with pytest.raises(ValueError, match="must be given together"):
        train_module(
            build_detector(x.shape[2], 8, 2), x, y, n_classes=2, epochs=1, seed=0, fedprox_mu=1.0
        )
    with pytest.raises(ValueError, match="must be given together"):
        train_module(
            build_detector(x.shape[2], 8, 2),
            x,
            y,
            n_classes=2,
            epochs=1,
            seed=0,
            fedprox_reference=reference,
        )


def test_fedprox_rejects_negative_mu() -> None:
    x, y = _separable_task(n=40)
    reference = {
        name: tensor.clone()
        for name, tensor in build_detector(x.shape[2], 8, 2).state_dict().items()
    }

    with pytest.raises(ValueError, match="fedprox_mu must be non-negative"):
        train_module(
            build_detector(x.shape[2], 8, 2),
            x,
            y,
            n_classes=2,
            epochs=1,
            seed=0,
            fedprox_mu=-1.0,
            fedprox_reference=reference,
        )
