"""Unit tests for the Section III-I1 baselines (Phase 3: baselines 1-3).

The property that makes these baselines meaningful is that they are *comparable* to the GRU:
same test targets, same training regime, matched capacity. Anything else leaves a difference
in the result confounded with a difference in the setup rather than the architecture.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ascon_smart_agri.eval.baselines import (
    MLPDetector,
    centralized_gru,
    federated_global_gru,
    local_only_grus,
    mlp_hidden_for_parameter_budget,
    mlp_single_record,
    random_forest_single_record,
)
from ascon_smart_agri.model.gru import build_detector, count_parameters, expected_param_count

CLASSES = ["Benign", "Attack"]


def _task(n: int = 300, window: int = 4, n_features: int = 3, seed: int = 0):
    """A separable task whose signal sits in the FINAL record, so single-record models can win."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(0, 0.3, size=(n, window, n_features))
    x[:, -1, :] += y[:, None] * 4.0
    return x.astype(np.float32), y


def test_mlp_is_parameter_matched_to_the_gru() -> None:
    """Matched capacity is what makes the MLP isolate RECURRENCE rather than model size."""
    budget = expected_param_count(16, 96, 8)
    hidden = mlp_hidden_for_parameter_budget(16, 8, budget)
    mlp_params = sum(p.numel() for p in MLPDetector(16, hidden, 8).parameters())

    assert count_parameters(build_detector(16, 96, 8)) == budget
    assert abs(mlp_params - budget) / budget < 0.01  # within 1% of the GRU's budget


def test_mlp_hidden_rejects_degenerate_budgets() -> None:
    with pytest.raises(ValueError, match="must all be positive"):
        mlp_hidden_for_parameter_budget(16, 8, 0)


def test_mlp_detector_reads_only_the_final_record() -> None:
    """The whole point of the baseline: it must not be able to use history."""
    torch.manual_seed(0)
    model = MLPDetector(3, 8, 2).eval()
    x = torch.randn(2, 5, 3)
    changed_history = x.clone()
    changed_history[:, :-1, :] += 10.0  # every step EXCEPT the last

    with torch.no_grad():
        assert torch.allclose(model(x), model(changed_history))

    changed_last = x.clone()
    changed_last[:, -1, :] += 10.0
    with torch.no_grad():
        assert not torch.allclose(model(x), model(changed_last))


def test_mlp_detector_rejects_bad_shapes() -> None:
    model = MLPDetector(3, 8, 2)
    with pytest.raises(ValueError, match=r"\(B, W, F\)"):
        model(torch.randn(5, 3))
    with pytest.raises(ValueError, match="expected 3 features"):
        model(torch.randn(2, 5, 7))
    with pytest.raises(ValueError, match="must all be positive"):
        MLPDetector(3, 0, 2)


def test_random_forest_baseline_learns_and_returns_the_metric_bundle() -> None:
    x, y = _task()

    metrics = random_forest_single_record(
        x, y, x, y, seed=0, class_names=CLASSES, n_estimators=20, max_train_rows=None
    )

    assert metrics.macro_f1 > 0.9
    assert set(metrics.per_class_f1) == set(CLASSES)
    assert metrics.confusion.shape == (2, 2)


def test_random_forest_respects_the_training_row_cap() -> None:
    x, y = _task(n=400)

    # A cap far below the data size must still produce a usable model, not an error.
    metrics = random_forest_single_record(
        x, y, x, y, seed=0, class_names=CLASSES, n_estimators=20, max_train_rows=50
    )

    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_mlp_baseline_learns_a_final_record_task() -> None:
    x, y = _task()

    metrics = mlp_single_record(
        x, y, x, y, seed=0, class_names=CLASSES, n_classes=2, epochs=15, gru_hidden_size=16
    )

    assert metrics.macro_f1 > 0.9


def test_centralized_gru_baseline_learns_and_matches_the_bundle() -> None:
    x, y = _task()

    metrics = centralized_gru(
        x, y, x, y, seed=0, class_names=CLASSES, n_classes=2, epochs=15, hidden_size=16
    )

    assert metrics.macro_f1 > 0.9
    assert set(metrics.per_class_f1) == set(CLASSES)


def test_all_baselines_are_evaluated_on_identical_targets() -> None:
    """Same test tensors in, so any difference is the model -- not the evaluation population."""
    x, y = _task(n=120)
    kwargs = {"seed": 0, "class_names": CLASSES}

    rf = random_forest_single_record(x, y, x, y, n_estimators=10, max_train_rows=None, **kwargs)
    mlp = mlp_single_record(x, y, x, y, n_classes=2, epochs=3, gru_hidden_size=8, **kwargs)
    gru = centralized_gru(x, y, x, y, n_classes=2, epochs=3, hidden_size=8, **kwargs)

    # Every confusion matrix accounts for exactly the same number of test items.
    assert rf.confusion.sum() == mlp.confusion.sum() == gru.confusion.sum() == len(y)


# ---------------------------------------------------------------- baselines 4-5 (Phase 4)


def _client_tasks(n_clients: int = 3, n: int = 120, window: int = 3, n_features: int = 4):
    """n_clients separable tasks, all evaluated against ONE shared test set (Section III-B3)."""
    seqs, labels = [], []
    for client_id in range(n_clients):
        x, y = _task(n=n, window=window, n_features=n_features, seed=client_id)
        seqs.append(x)
        labels.append(y)
    x_test, y_test = _task(n=40, window=window, n_features=n_features, seed=99)
    return seqs, labels, x_test, y_test


def test_local_only_grus_returns_one_result_per_client() -> None:
    seqs, labels, x_test, y_test = _client_tasks()

    results = local_only_grus(
        seqs,
        labels,
        x_test,
        y_test,
        seed=0,
        class_names=CLASSES,
        n_classes=2,
        epochs=40,
        hidden_size=16,
    )

    assert len(results) == 3
    assert all(r.macro_f1 > 0.9 for r in results)  # each client's task is individually learnable


def test_local_only_client_with_zero_sequences_is_reported_not_raised() -> None:
    """Legitimate under strong heterogeneity: a report, not a filtered list."""
    seqs, labels, x_test, y_test = _client_tasks(n_clients=2)
    seqs[1] = np.empty((0, 3, 4), dtype=np.float32)
    labels[1] = np.empty((0,), dtype=np.int64)

    results = local_only_grus(
        seqs,
        labels,
        x_test,
        y_test,
        seed=0,
        class_names=CLASSES,
        n_classes=2,
        epochs=15,
        hidden_size=16,
    )

    assert len(results) == 2
    assert results[1].confusion.sum() == len(y_test)  # still a full, well-formed metric bundle


def test_federated_global_gru_learns_and_returns_convergence() -> None:
    seqs, labels, x_test, y_test = _client_tasks()

    metrics, convergence, measured_bytes = federated_global_gru(
        seqs,
        labels,
        x_test,
        y_test,
        seed=0,
        class_names=CLASSES,
        n_classes=2,
        rounds=8,
        local_epochs=5,
        hidden_size=16,
    )

    assert metrics.macro_f1 > 0.9
    assert len(convergence) == 8  # one macro-F1 per round
    assert measured_bytes > 0


def test_federated_global_gru_weighted_vs_unweighted_are_both_runnable() -> None:
    seqs, labels, x_test, y_test = _client_tasks()

    for aggregation in ("weighted", "unweighted"):
        metrics, convergence, _ = federated_global_gru(
            seqs,
            labels,
            x_test,
            y_test,
            seed=0,
            class_names=CLASSES,
            n_classes=2,
            rounds=2,
            local_epochs=1,
            hidden_size=16,
            aggregation=aggregation,
        )
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert len(convergence) == 2


def test_federated_global_gru_raises_if_every_client_is_empty() -> None:
    seqs = [np.empty((0, 3, 4), dtype=np.float32)] * 3
    labels = [np.empty((0,), dtype=np.int64)] * 3
    _, _, x_test, y_test = _client_tasks()

    with pytest.raises(ValueError, match="every client holds zero sequences"):
        federated_global_gru(
            seqs,
            labels,
            x_test,
            y_test,
            seed=0,
            class_names=CLASSES,
            n_classes=2,
            rounds=1,
            local_epochs=1,
        )
