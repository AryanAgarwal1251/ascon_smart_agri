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

TODO(Phase 2): implement stages 1-4; persist the selected column list + provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SelectionResult:
    """Fitted selection outcome; persisted and reused verbatim at inference time."""

    selected_columns: list[str]
    dropped_stage1: list[str]
    dropped_stage2: list[str]
    mi_ranking: list[str]
    rf_ranking: list[str]
    rank_disagreements: list[str]  # reported explicitly, not hidden in an average


class FeatureSelector:
    """Stateful four-stage selector, fitted on training blocks and applied everywhere."""

    def fit(
        self, x_train: pd.DataFrame, y_train: pd.Series[str], *, tau: float, f: int
    ) -> SelectionResult:
        """Run stages 1-4 on training data only and return the selection result."""
        del x_train, y_train, tau, f
        raise NotImplementedError("Phase 2: feature selection not implemented yet.")

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Project any frame onto the persisted selected columns."""
        del x
        raise NotImplementedError("Phase 2: feature selection not implemented yet.")
