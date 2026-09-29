"""Baseline 4's reporting contract: the G4 lower bound carries the full headline metric set.

Section III-I2 forbids reporting accuracy alone and makes FPR (Eq. 31) first-class. The G4
bracket of Section III-I1 compares three baselines -- centralised (3), local-only (4) and
federated global (5) -- and is only readable if all three rows are measured on the same
columns. Baseline 4 previously reported macro-F1, balanced accuracy and MCC but *not* accuracy
or FPR, because the per-client confusion matrix was never retained; these tests pin the
repaired contract so the lower bound cannot silently lose a column again.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scripts.run_phase4 import HEADLINE, binary_fpr, local_only_summary

from ascon_smart_agri.eval.metrics import MulticlassMetrics


def make_metrics(
    *,
    macro_f1: float,
    confusion: np.ndarray,
    accuracy: float = 0.5,
    balanced_accuracy: float = 0.5,
    mcc: float = 0.5,
) -> MulticlassMetrics:
    """A MulticlassMetrics bundle with only the fields these tests exercise set meaningfully."""
    return MulticlassMetrics(
        macro_precision=macro_f1,
        macro_recall=macro_f1,
        macro_f1=macro_f1,
        per_class_f1={"Benign": macro_f1},
        balanced_accuracy=balanced_accuracy,
        mcc=mcc,
        confusion=confusion,
        accuracy=accuracy,
    )


def confusion_with(*, true_negative: float, false_positive: float) -> np.ndarray:
    """A 3x3 matrix whose benign row (index 0) has the requested TN/FP split."""
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0, 0] = true_negative
    matrix[0, 1] = false_positive
    return matrix


class TestBinaryFpr:
    """Eq. (31) is a pure function of the confusion matrix's benign row."""

    def test_counts_off_diagonal_benign_predictions_as_false_alarms(self) -> None:
        fpr = binary_fpr(confusion_with(true_negative=90, false_positive=10), 0)
        assert fpr == pytest.approx(0.1)

    def test_constant_benign_model_has_a_genuine_fpr_of_zero(self) -> None:
        """A model that never raises an alarm has FPR 0 -- the best value, the worst model.

        This is the subtlety ``local_only_summary``'s docstring warns about: it is not a bug
        and not a sentinel, so it must not be special-cased away. The pairing with a poor
        macro-F1 is what makes it readable.
        """
        assert binary_fpr(confusion_with(true_negative=50, false_positive=0), 0) == 0.0

    def test_no_benign_rows_is_undefined_not_zero(self) -> None:
        """An empty benign row means FPR was not measured; nan says so, 0.0 would lie."""
        assert math.isnan(binary_fpr(np.zeros((3, 3)), 0))


class TestLocalOnlySummary:
    """Baseline 4 must be reported on the same columns as baselines 3 and 5."""

    def test_reports_every_headline_metric(self) -> None:
        """The regression guard: a missing column here is what this change repaired."""
        summary = local_only_summary(
            [
                make_metrics(
                    macro_f1=0.7, confusion=confusion_with(true_negative=90, false_positive=10)
                )
            ],
            seed=0,
            empty_clients=[],
            benign_index=0,
            federated_macro_f1=0.8,
        )
        missing = [metric for metric in HEADLINE if metric not in summary]
        assert not missing, f"baseline 4 is missing headline metrics: {missing}"

    def test_every_metric_is_the_mean_over_clients(self) -> None:
        summary = local_only_summary(
            [
                make_metrics(
                    macro_f1=0.6,
                    accuracy=0.90,
                    balanced_accuracy=0.50,
                    mcc=0.40,
                    confusion=confusion_with(true_negative=80, false_positive=20),
                ),
                make_metrics(
                    macro_f1=0.8,
                    accuracy=0.70,
                    balanced_accuracy=0.70,
                    mcc=0.60,
                    confusion=confusion_with(true_negative=90, false_positive=10),
                ),
            ],
            seed=0,
            empty_clients=[],
            benign_index=0,
            federated_macro_f1=0.85,
        )
        assert summary["macro_f1"] == pytest.approx(0.7)
        assert summary["accuracy"] == pytest.approx(0.8)
        assert summary["balanced_accuracy"] == pytest.approx(0.6)
        assert summary["mcc"] == pytest.approx(0.5)
        # mean(0.20, 0.10), not the pooled matrix's FPR -- the lower bound is a mean over the
        # K independent models, since that is what declining to federate actually gives you.
        assert summary["false_positive_rate"] == pytest.approx(0.15)

    def test_a_client_with_no_data_still_counts_toward_the_lower_bound(self) -> None:
        """Excluding it would flatter the bound; III-I1 wants the honest lower bound."""
        starved = make_metrics(
            macro_f1=0.05,  # constant-benign: near-useless
            confusion=confusion_with(true_negative=100, false_positive=0),  # ...yet FPR 0.0
        )
        trained = make_metrics(
            macro_f1=0.85, confusion=confusion_with(true_negative=80, false_positive=20)
        )
        summary = local_only_summary(
            [starved, trained],
            seed=0,
            empty_clients=[0],
            benign_index=0,
            federated_macro_f1=0.9,
        )
        assert summary["clients_without_data"] == 1
        assert summary["macro_f1"] == pytest.approx(0.45)  # (0.05 + 0.85) / 2, nothing dropped
        assert summary["per_client_macro_f1"] == [0.05, 0.85]
        # The documented trap, pinned so nobody "fixes" it into a misleading number: the
        # starved client's perfect FPR drags the bound's FPR *down* while its macro-F1 drags
        # that column down too. Reading either alone misleads; that is why III-I2 pairs them.
        assert summary["false_positive_rate"] == pytest.approx(0.10)  # mean(0.0, 0.20)

    def test_client_to_global_gap_is_positive_when_federating_helped(self) -> None:
        summary = local_only_summary(
            [
                make_metrics(
                    macro_f1=0.60, confusion=confusion_with(true_negative=90, false_positive=10)
                ),
                make_metrics(
                    macro_f1=0.70, confusion=confusion_with(true_negative=90, false_positive=10)
                ),
            ],
            seed=0,
            empty_clients=[],
            benign_index=0,
            federated_macro_f1=0.80,
        )
        gap = summary["client_to_global_gap"]
        assert isinstance(gap, dict)
        assert gap["0"] == pytest.approx(0.20)
        assert gap["1"] == pytest.approx(0.10)
