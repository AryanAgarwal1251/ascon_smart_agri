"""Metrics (Phase 3+, Section III-I2).

NEVER report accuracy alone: at IR ~ 5751 a majority-class classifier looks excellent and
detects nothing. Primary metrics for the eight-class task: macro-averaged precision/recall/F1
(Eq. 30), per-class F1, balanced accuracy, Matthews correlation coefficient, and the confusion
matrix. For the binary projection (Eq. 5): precision, recall, PR-AUC, and --- treated as a
first-class metric --- the false-positive rate FP/(FP+TN) (Eq. 31), because in production a
high false-alarm rate is what gets a detector switched off. Accuracy may be reported ALONGSIDE
these, never on its own.

TODO(Phase 3): implement the metric functions over predictions/labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._types import Array


@dataclass(frozen=True)
class MulticlassMetrics:
    """Eight-class metric bundle (no bare accuracy as a headline)."""

    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_f1: dict[str, float]
    balanced_accuracy: float
    mcc: float
    confusion: Array
    accuracy: float  # reported only alongside the above


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary-projection metric bundle; FPR is first-class."""

    precision: float
    recall: float
    pr_auc: float
    false_positive_rate: float  # FP / (FP + TN), Eq. (31)


def multiclass_metrics(y_true: Array, y_pred: Array, labels: list[str]) -> MulticlassMetrics:
    """Compute the eight-class metric bundle."""
    del y_true, y_pred, labels
    raise NotImplementedError("Phase 3: multiclass metrics not implemented yet.")


def binary_metrics(y_true: Array, y_score: Array) -> BinaryMetrics:
    """Compute the binary-projection metric bundle (incl. FPR and PR-AUC)."""
    del y_true, y_score
    raise NotImplementedError("Phase 3: binary metrics not implemented yet.")
