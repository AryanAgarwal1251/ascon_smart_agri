"""Unit tests for the evaluation metrics (Section III-I2).

The property that matters most here is the one CLAUDE.md states as an invariant: evaluation
never reports accuracy alone, and the metrics that carry the real signal under imbalance
(macro-F1, per-class F1, balanced accuracy, MCC, FPR) must actually punish a degenerate
majority-class detector that accuracy would flatter.
"""

from __future__ import annotations

import numpy as np
import pytest

from ascon_smart_agri.eval.metrics import binary_metrics, multiclass_metrics

LABELS = ["Benign", "DDoS", "Rare"]


def test_perfect_predictions_score_one_everywhere() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])

    metrics = multiclass_metrics(y, y, LABELS)

    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(1.0)
    assert metrics.mcc == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(1.0)


def test_majority_class_detector_is_punished_by_macro_f1_not_by_accuracy() -> None:
    """The exact failure mode Section III-I5 warns about: high accuracy, useless detector."""
    # 96 benign, 2 DDoS, 2 rare -- predict benign always.
    y_true = np.array([0] * 96 + [1, 1, 2, 2])
    y_pred = np.zeros_like(y_true)

    metrics = multiclass_metrics(y_true, y_pred, LABELS)

    assert metrics.accuracy == pytest.approx(0.96)  # looks excellent
    assert metrics.macro_f1 < 0.35  # and is not
    assert metrics.per_class_f1["DDoS"] == pytest.approx(0.0)
    assert metrics.per_class_f1["Rare"] == pytest.approx(0.0)
    assert metrics.balanced_accuracy == pytest.approx(1 / 3)
    assert metrics.mcc == pytest.approx(0.0)


def test_a_class_absent_from_the_split_still_counts_in_the_macro_average() -> None:
    """Dropping an absent class from the denominator would flatter the rare families."""
    y_true = np.array([0, 0, 1, 1])  # "Rare" never appears
    y_pred = np.array([0, 0, 1, 1])

    metrics = multiclass_metrics(y_true, y_pred, LABELS)

    assert set(metrics.per_class_f1) == set(LABELS)
    assert metrics.per_class_f1["Rare"] == pytest.approx(0.0)
    assert metrics.macro_f1 == pytest.approx(2 / 3)  # not 1.0
    assert metrics.confusion.shape == (3, 3)


def test_confusion_matrix_is_square_over_all_declared_classes() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 0])

    confusion = multiclass_metrics(y_true, y_pred, LABELS).confusion

    assert confusion.shape == (3, 3)
    assert confusion.sum() == 3
    assert confusion[0, 0] == 1


def test_multiclass_rejects_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError, match="lengths disagree"):
        multiclass_metrics(np.array([0, 1]), np.array([0]), LABELS)
    with pytest.raises(ValueError, match="zero predictions"):
        multiclass_metrics(np.array([]), np.array([]), LABELS)


def test_binary_false_positive_rate_is_computed_as_fp_over_fp_plus_tn() -> None:
    # 4 benign (2 wrongly flagged), 2 attacks (both caught).
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.7])

    metrics = binary_metrics(y_true, y_score)

    assert metrics.false_positive_rate == pytest.approx(2 / 4)  # Eq. (31)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(2 / 4)


def test_binary_pr_auc_is_one_for_a_perfectly_separable_score() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.01, 0.02, 0.98, 0.99])

    assert binary_metrics(y_true, y_score).pr_auc == pytest.approx(1.0)


def test_binary_pr_auc_is_nan_when_only_one_class_is_present() -> None:
    # Reporting 0.0 or 1.0 here would be a fabricated number, not an undefined one.
    metrics = binary_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))

    assert np.isnan(metrics.pr_auc)


def test_binary_threshold_is_honoured() -> None:
    y_true = np.array([0, 1])
    y_score = np.array([0.6, 0.6])

    assert binary_metrics(y_true, y_score, threshold=0.5).false_positive_rate == pytest.approx(1.0)
    assert binary_metrics(y_true, y_score, threshold=0.7).false_positive_rate == pytest.approx(0.0)
