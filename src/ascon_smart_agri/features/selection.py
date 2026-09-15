"""Four-stage feature selection, fitted on TRAINING BLOCKS ONLY (Phase 2, Section III-C).

    Stage 1  rule-based elimination  -- drop labels/identifiers/zero-variance/columns not
                                        available at inference time.
    Stage 2  correlation pruning     -- Spearman |rho| >= tau (default 0.95); keep the member
                                        of a correlated pair with higher univariate relevance.
    Stage 3  relevance ranking       -- mutual information AND random-forest impurity
                                        importance, combined by reciprocal rank fusion;
                                        DISAGREEMENTS are reported, not averaged away.
    Stage 4  cardinality selection   -- sweep F in {8,12,16,24,F0}; pick the knee of the
                                        validation macro-F1 curve.

Explicitly NO PCA: components are linear mixtures that destroy the column-level
interpretability the runtime provenance adapter (III-H) and an operator acting on an alert
both need. The fitted selector is persisted so inference uses exactly the same columns.

LEAKAGE CONTRACT: every statistic below is computed from the ``x_train``/``y_train`` passed to
:meth:`FeatureSelector.fit` and nothing else. Callers must pass training blocks only (Section
III-C); this module cannot verify that and does not try.

Decisions the paper leaves open, resolved here and flagged per Golden Rule 1:

* **Stage 4 depends on Phase 3, which is gated behind Phase 2 -- a circular dependency in the
  paper's own phase ordering.** Choosing "the knee of the validation macro-F1 curve" requires
  training and evaluating a detector at each candidate F, but the detector is Phase 3 and Phase
  3 is a hard gate that opens only after Phase 2 closes. Resolved by dependency injection rather
  than by starting Phase 3 early: :meth:`FeatureSelector.fit` accepts an optional ``evaluator``
  callable mapping a candidate column list to a validation macro-F1. When it is supplied (Phase
  3 onward) Stage 4 runs for real and the curve is recorded in
  :attr:`SelectionResult.f_sweep_scores`; when it is ``None`` (Phase 2, now) Stage 4 falls back
  to the configured ``f`` and records an empty curve, so no GRU is trained before its gate.
* **The reciprocal-rank-fusion constant is not recoverable from the paper source.** The
  extraction in ``docs/design_paper.md`` dropped every display equation before Eq. (12), which
  includes the RRF formula of Section III-C Stage 3 -- the prose survives ("combined by
  reciprocal rank fusion, [equation] where r_m(j) is the rank of feature j under method m") but
  the equation itself does not. Implemented as the canonical form, ``RRF(j) = sum_m 1/(k +
  r_m(j))`` with 1-based ranks and ``k = 60`` (Cormack et al., 2009), exposed as ``rrf_k``.
  Re-check against the authoritative PDF before quoting Stage 3 numbers in a write-up.
* **"Univariate relevance" (Stage 2's tie-break) is not defined by the paper.** Read here as
  mutual information with the label -- the same statistic Stage 3 ranks by -- so the two stages
  agree about what "relevant" means instead of using two different notions.
* **"Disagreement" (Stage 3) is not defined by the paper either.** Reported here as the
  selection-relevant disagreement: a feature in the top-F of exactly one of the two rankings.
  That is precisely the disagreement reciprocal rank fusion would otherwise bury, which is what
  Section III-C asks to surface.
* **Statistics are computed on a seeded sample of the training rows** (``sample_size``, default
  200,000; ``None`` uses every row). Spearman, mutual information and a random forest over the
  full ~1.24M-row training split cost minutes for estimates that are already stable at this
  size. The sample is drawn once and shared by all three so they see identical rows.
* **Non-finite rows are excluded from the statistics sample only**, never imputed or zero-filled
  -- the raw corpus has genuine ``+inf`` in ``Rate`` (a division by near-zero flow duration,
  confirmed in Phase 1). This mirrors the policy ``data/characterize.py`` already applies to its
  correlation accumulation, and the excluded count is reported in
  :attr:`SelectionResult.n_nonfinite_rows_excluded` rather than passed over silently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# Columns that are ours, not the device's: the label itself and the provenance bookkeeping
# added by data/subsample.py and data/split.py. None of them exists at inference time, so all
# are Stage-1 eliminations (Section III-C, Stage 1).
NON_INFERENCE_COLUMNS = frozenset({"label", "source_file", "block_id"})


@dataclass(frozen=True)
class SelectionResult:
    """Fitted selection outcome; persisted and reused verbatim at inference time."""

    selected_columns: list[str]
    dropped_stage1: list[str]
    dropped_stage2: list[str]
    mi_ranking: list[str]
    rf_ranking: list[str]
    rank_disagreements: list[str]  # reported explicitly, not hidden in an average
    # Added beyond the original scaffold (flagged in CHANGELOG.md): Section III-C requires the
    # fused ranking and the Stage-4 knee curve to be reportable, and the non-finite exclusion
    # count to be visible rather than silent.
    fused_ranking: list[str] = field(default_factory=list)
    f_sweep_scores: dict[int, float] = field(default_factory=dict)
    selected_f: int = 0
    n_nonfinite_rows_excluded: int = 0


def _knee_index(scores: Sequence[float]) -> int:
    """Return the index of the knee: the point furthest from the first-to-last chord.

    Standard maximum-distance-to-chord construction. With fewer than three points there is no
    interior point to be a knee, so the best-scoring point is returned instead.
    """
    n = len(scores)
    if n < 3:
        return int(np.argmax(scores))
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    # Perpendicular distance from each point to the chord, up to a constant factor.
    distance = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    return int(np.argmax(distance))


class FeatureSelector:
    """Stateful four-stage selector, fitted on training blocks and applied everywhere."""

    def __init__(self) -> None:
        self._result: SelectionResult | None = None

    @property
    def result(self) -> SelectionResult:
        """The fitted :class:`SelectionResult`; raises if :meth:`fit` has not run."""
        if self._result is None:
            raise RuntimeError("FeatureSelector is not fitted; call fit() first")
        return self._result

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series[str],
        *,
        tau: float,
        f: int,
        f_sweep: Sequence[int] | None = None,
        evaluator: Callable[[list[str]], float] | None = None,
        rf_n_estimators: int = 200,
        rrf_k: int = 60,
        sample_size: int | None = 200_000,
        seed: int = 0,
    ) -> SelectionResult:
        """Run stages 1-4 on training data only and return the selection result."""
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau}")
        if f <= 0:
            raise ValueError(f"f must be positive, got {f}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
        if len(x_train) != len(y_train):
            raise ValueError(f"x_train has {len(x_train)} rows but y_train has {len(y_train)}")

        # ---- Stage 1: rule-based elimination -------------------------------------------------
        kept: list[str] = []
        dropped_stage1: list[str] = []
        for column in x_train.columns:
            series = x_train[column]
            unavailable = column in NON_INFERENCE_COLUMNS
            non_numeric = not pd.api.types.is_numeric_dtype(series)
            zero_variance = not non_numeric and series.nunique(dropna=True) <= 1
            if unavailable or non_numeric or zero_variance:
                dropped_stage1.append(column)
            else:
                kept.append(column)
        if not kept:
            raise ValueError("Stage 1 eliminated every column; nothing left to select from")

        # ---- Shared statistics sample --------------------------------------------------------
        block = x_train[kept]
        finite_mask = np.isfinite(block.to_numpy(dtype=np.float64)).all(axis=1)
        n_nonfinite = int((~finite_mask).sum())
        x_finite = block.loc[finite_mask]
        y_finite = y_train.loc[finite_mask]
        if x_finite.empty:
            raise ValueError("every training row holds a non-finite value; cannot fit selector")

        if sample_size is not None and len(x_finite) > sample_size:
            rng = np.random.default_rng(seed)
            take = np.sort(rng.choice(len(x_finite), size=sample_size, replace=False))
            x_sample = x_finite.iloc[take]
            y_sample = y_finite.iloc[take]
        else:
            x_sample, y_sample = x_finite, y_finite

        # ---- Stage 2: Spearman correlation pruning -------------------------------------------
        mi_values = mutual_info_classif(x_sample, y_sample, random_state=seed)
        mi_by_column = dict(zip(kept, (float(v) for v in mi_values), strict=True))

        spearman_frame = x_sample.corr(method="spearman").abs()
        # Read the matrix as the float array it is and index it by position. `.at[row, col]`
        # returns a pandas scalar whose declared type is a union spanning str/bytes/datetime,
        # which cannot be compared against a float threshold without lying about the type.
        spearman = spearman_frame.to_numpy(dtype=np.float64)
        at = {column: position for position, column in enumerate(spearman_frame.columns)}
        # Walk columns in descending univariate relevance so the survivor of every correlated
        # pair is always the more relevant member (Section III-C, Stage 2).
        by_relevance = sorted(kept, key=lambda c: (-mi_by_column[c], c))
        survivors: list[str] = []
        dropped_stage2: list[str] = []
        for column in by_relevance:
            if any(spearman[at[column], at[s]] >= tau for s in survivors):
                dropped_stage2.append(column)
            else:
                survivors.append(column)
        # Restore the original column order for everything downstream.
        surviving = [c for c in kept if c in set(survivors)]
        dropped_stage2 = [c for c in kept if c in set(dropped_stage2)]

        # ---- Stage 3: relevance ranking + reciprocal rank fusion -----------------------------
        forest = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            class_weight="balanced",  # imbalance handled by weighting, never by oversampling
            random_state=seed,
            n_jobs=-1,
        )
        forest.fit(x_sample[surviving], y_sample)
        rf_by_column = dict(
            zip(surviving, (float(v) for v in forest.feature_importances_), strict=True)
        )

        mi_ranking = sorted(surviving, key=lambda c: (-mi_by_column[c], c))
        rf_ranking = sorted(surviving, key=lambda c: (-rf_by_column[c], c))
        mi_rank = {c: i + 1 for i, c in enumerate(mi_ranking)}
        rf_rank = {c: i + 1 for i, c in enumerate(rf_ranking)}
        fused_score = {
            c: 1.0 / (rrf_k + mi_rank[c]) + 1.0 / (rrf_k + rf_rank[c]) for c in surviving
        }
        fused_ranking = sorted(surviving, key=lambda c: (-fused_score[c], c))

        # ---- Stage 4: cardinality selection --------------------------------------------------
        f0 = len(surviving)
        candidates = sorted({*(f_sweep or ()), f0} if f_sweep else {f, f0})
        candidates = [c for c in candidates if 0 < c <= f0]
        f_sweep_scores: dict[int, float] = {}
        if evaluator is not None and candidates:
            f_sweep_scores = {c: float(evaluator(fused_ranking[:c])) for c in candidates}
            ordered = sorted(f_sweep_scores)
            selected_f = ordered[_knee_index([f_sweep_scores[c] for c in ordered])]
        else:
            # Phase 2: no detector exists yet to produce a macro-F1 curve (see module docstring).
            selected_f = min(f, f0)

        selected_columns = fused_ranking[:selected_f]

        top_f_mi = set(mi_ranking[:selected_f])
        top_f_rf = set(rf_ranking[:selected_f])
        rank_disagreements = [c for c in surviving if (c in top_f_mi) != (c in top_f_rf)]

        self._result = SelectionResult(
            selected_columns=selected_columns,
            dropped_stage1=dropped_stage1,
            dropped_stage2=dropped_stage2,
            mi_ranking=mi_ranking,
            rf_ranking=rf_ranking,
            rank_disagreements=rank_disagreements,
            fused_ranking=fused_ranking,
            f_sweep_scores=f_sweep_scores,
            selected_f=selected_f,
            n_nonfinite_rows_excluded=n_nonfinite,
        )
        return self._result

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Project any frame onto the persisted selected columns."""
        selected = self.result.selected_columns
        missing = [c for c in selected if c not in x.columns]
        if missing:
            raise ValueError(f"frame is missing selected columns: {missing}")
        return x[selected]
