"""Metrics (Phase 3+, Section III-I2).

NEVER report accuracy alone: at IR ~ 5751 a majority-class classifier looks excellent and
detects nothing. Primary metrics for the eight-class task: macro-averaged precision/recall/F1
(Eq. 30), per-class F1, balanced accuracy, Matthews correlation coefficient, and the confusion
matrix. For the binary projection (Eq. 5): precision, recall, PR-AUC, and --- treated as a
first-class metric --- the false-positive rate FP/(FP+TN) (Eq. 31), because in production a
high false-alarm rate is what gets a detector switched off. Accuracy may be reported ALONGSIDE
these, never on its own.

Implementation notes:

* **Macro averages are taken over every class in ``labels``, present in the predictions or
  not.** A family that the model never predicts and that never appears in this particular test
  split still contributes an F1 of 0 rather than being quietly dropped from the denominator,
  which would flatter a detector precisely on the rare families Section III-I cares about.
* **The binary projection is a deterministic function of the eight-class output** (Eq. 5), so
  ``binary_metrics`` takes the attack score directly rather than a second model's output; the
  caller forms it as ``1 - P(benign)``, which is exactly the "y > 0" projection of
  ``data/taxonomy.py``'s class order.
* **PR-AUC is average precision**, the interpolation-free summary of the precision-recall curve.
  It is preferred to ROC-AUC under heavy imbalance because the false-positive axis of an ROC is
  dominated by the abundant negative class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

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
    """Compute the eight-class metric bundle.

    ``labels`` names every class in index order (``data/taxonomy.CLASS_NAMES``); metrics are
    averaged over all of them, including any absent from this split.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true/y_pred lengths disagree: {len(y_true)}, {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on zero predictions")

    indices = list(range(len(labels)))
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average="macro", zero_division=0
    )
    return MulticlassMetrics(
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        per_class_f1={name: float(f1[i]) for i, name in enumerate(labels)},
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        confusion=confusion_matrix(y_true, y_pred, labels=indices),
        accuracy=float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    )


def binary_metrics(y_true: Array, y_score: Array, *, threshold: float = 0.5) -> BinaryMetrics:
    """Compute the binary-projection metric bundle (incl. FPR and PR-AUC).

    ``y_true`` is 1 for attack and 0 for benign (the ``y > 0`` projection of Eq. 5); ``y_score``
    is the predicted attack score. PR-AUC is threshold-free; precision, recall and FPR are taken
    at ``threshold``.
    """
    if len(y_true) != len(y_score):
        raise ValueError(f"y_true/y_score lengths disagree: {len(y_true)}, {len(y_score)}")
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on zero predictions")

    truth = np.asarray(y_true).astype(int)
    score = np.asarray(y_score, dtype=np.float64)
    predicted = (score >= threshold).astype(int)

    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))

    # A split with only one class present cannot support average precision; report NaN rather
    # than a misleading 0.0 or 1.0.
    pr_auc = (
        float(average_precision_score(truth, score)) if len(np.unique(truth)) > 1 else float("nan")
    )
    return BinaryMetrics(
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        pr_auc=pr_auc,
        false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0,
    )
