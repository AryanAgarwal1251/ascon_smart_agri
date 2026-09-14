"""Unit tests for Phase 2 four-stage feature selection (Section III-C).

Synthetic frames with a known-correct answer built in: a constant column Stage 1 must drop, an
exact copy of a feature Stage 2 must prune, informative features that must outrank noise, and a
Stage-4 curve whose knee is placed deliberately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ascon_smart_agri.features.selection import FeatureSelector, SelectionResult


def _labelled_frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series[str]]:
    """Two informative features, one pure copy, one noise column, one constant column."""
    rng = np.random.default_rng(seed)
    y = pd.Series(["benign"] * (n // 2) + ["attack"] * (n // 2), name="label")
    signal = np.where(y == "attack", 1.0, 0.0) + rng.normal(0, 0.05, n)
    frame = pd.DataFrame(
        {
            "informative_a": signal,
            "informative_b": signal * 2.0 + rng.normal(0, 0.05, n),
            "copy_of_a": signal,  # perfectly rank-correlated with informative_a
            "noise": rng.normal(0, 1.0, n),
            "constant": np.ones(n),
        }
    )
    return frame, y


def test_stage1_drops_constant_label_and_provenance_columns() -> None:
    frame, y = _labelled_frame()
    frame["label"] = y  # the label itself must never survive into the feature set
    frame["source_file"] = "Benign_Final/BenignTraffic.pcap.csv"
    frame["block_id"] = 3

    result = FeatureSelector().fit(frame, y, tau=0.95, f=2)

    assert set(result.dropped_stage1) == {"constant", "label", "source_file", "block_id"}
    assert "label" not in result.selected_columns
    assert "source_file" not in result.selected_columns


def test_stage2_prunes_a_correlated_pair_keeping_the_more_relevant_member() -> None:
    frame, y = _labelled_frame()

    result = FeatureSelector().fit(frame, y, tau=0.95, f=3)

    # "copy_of_a" is rank-identical to "informative_a": exactly one of the pair may survive.
    survivors = set(result.mi_ranking)
    assert len({"informative_a", "copy_of_a"} & survivors) == 1
    assert len({"informative_a", "copy_of_a"} & set(result.dropped_stage2)) == 1


def test_informative_features_outrank_noise_in_both_rankings() -> None:
    frame, y = _labelled_frame()

    result = FeatureSelector().fit(frame, y, tau=0.99, f=2)

    assert result.mi_ranking[-1] == "noise"  # least relevant under mutual information
    assert result.rf_ranking[-1] == "noise"  # and under impurity importance
    assert "noise" not in result.selected_columns


def test_selected_columns_follow_the_fused_ranking_and_respect_f() -> None:
    frame, y = _labelled_frame()

    result = FeatureSelector().fit(frame, y, tau=0.99, f=2)

    assert len(result.selected_columns) == 2
    assert result.selected_f == 2
    assert result.selected_columns == result.fused_ranking[:2]


def test_stage4_without_an_evaluator_falls_back_to_configured_f() -> None:
    """Phase 2 has no detector yet, so no macro-F1 curve may be produced (Section III-C)."""
    frame, y = _labelled_frame()

    result = FeatureSelector().fit(frame, y, tau=0.99, f=3, f_sweep=[1, 2, 3])

    assert result.f_sweep_scores == {}  # no curve was fabricated
    assert result.selected_f == 3


def test_stage4_with_an_evaluator_picks_the_knee_of_the_curve() -> None:
    frame, y = _labelled_frame()
    # Stage 2 prunes "copy_of_a" and Stage 1 drops "constant", so F0 = 3 and a candidate F=4 is
    # correctly filtered out as unsatisfiable -- the sweep can only offer 1..3.
    # Macro-F1 rises steeply to F=2 then plateaus: the knee is F=2, not the best-scoring F=3.
    curve = {1: 0.50, 2: 0.90, 3: 0.91}

    result = FeatureSelector().fit(
        frame,
        y,
        tau=0.99,
        f=3,
        f_sweep=[1, 2, 3, 4],
        evaluator=lambda columns: curve[len(columns)],
    )

    assert result.f_sweep_scores == pytest.approx(curve)
    assert result.selected_f == 2
    assert len(result.selected_columns) == 2


def test_nonfinite_rows_are_excluded_from_statistics_and_counted() -> None:
    frame, y = _labelled_frame()
    frame.loc[0, "noise"] = np.inf  # genuine +inf, as CICIoT2023's "Rate" column carries
    frame.loc[1, "noise"] = -np.inf

    result = FeatureSelector().fit(frame, y, tau=0.99, f=2)

    assert result.n_nonfinite_rows_excluded == 2
    assert result.selected_columns  # fit still succeeds on the finite remainder


def test_rank_disagreements_are_reported() -> None:
    frame, y = _labelled_frame()

    result = FeatureSelector().fit(frame, y, tau=0.99, f=1)

    # Whatever the two methods decide, a disagreement is a top-F membership difference and must
    # never be silently averaged away by the fusion step (Section III-C, Stage 3).
    for column in result.rank_disagreements:
        in_mi = column in set(result.mi_ranking[:1])
        in_rf = column in set(result.rf_ranking[:1])
        assert in_mi != in_rf


def test_transform_projects_onto_selected_columns() -> None:
    frame, y = _labelled_frame()
    selector = FeatureSelector()
    result = selector.fit(frame, y, tau=0.99, f=2)

    projected = selector.transform(frame)

    assert list(projected.columns) == result.selected_columns
    assert len(projected) == len(frame)


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        FeatureSelector().transform(pd.DataFrame({"a": [1]}))


def test_transform_rejects_a_frame_missing_selected_columns() -> None:
    frame, y = _labelled_frame()
    selector = FeatureSelector()
    selector.fit(frame, y, tau=0.99, f=2)

    with pytest.raises(ValueError, match="missing selected columns"):
        selector.transform(pd.DataFrame({"unrelated": [1, 2, 3]}))


def test_fit_is_deterministic_given_a_seed() -> None:
    frame, y = _labelled_frame()

    first = FeatureSelector().fit(frame, y, tau=0.95, f=2, seed=11)
    second = FeatureSelector().fit(frame, y, tau=0.95, f=2, seed=11)

    assert first == second
    assert isinstance(first, SelectionResult)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tau": 0.0}, "tau"),
        ({"tau": 1.5}, "tau"),
        ({"f": 0}, "f must be positive"),
        ({"rrf_k": 0}, "rrf_k"),
    ],
)
def test_fit_rejects_invalid_arguments(kwargs: dict[str, float], match: str) -> None:
    frame, y = _labelled_frame(n=40)
    base: dict[str, float] = {"tau": 0.95, "f": 2}
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        FeatureSelector().fit(frame, y, **base)  # type: ignore[arg-type]


def test_fit_rejects_mismatched_x_and_y_lengths() -> None:
    frame, y = _labelled_frame(n=40)

    with pytest.raises(ValueError, match="rows"):
        FeatureSelector().fit(frame, y.iloc[:10], tau=0.95, f=2)


def test_fit_raises_when_stage1_eliminates_everything() -> None:
    frame = pd.DataFrame({"constant": [1, 1, 1, 1], "label": ["a", "b", "a", "b"]})
    y = pd.Series(["a", "b", "a", "b"])

    with pytest.raises(ValueError, match="Stage 1 eliminated every column"):
        FeatureSelector().fit(frame, y, tau=0.95, f=1)
