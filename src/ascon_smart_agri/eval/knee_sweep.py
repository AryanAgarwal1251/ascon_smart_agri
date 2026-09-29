"""Stage-4 cardinality selection: the validation macro-F1 curve whose knee picks F.

Section III-C's Stage 4 sweeps ``F in {8, 12, 16, 24, F0}`` and takes "the knee of the
validation macro-F1 curve". Scoring a candidate F means training and evaluating a detector on
just those columns -- but the detector is Phase 3 and Phase 3 is a hard gate that opens only
after Phase 2 closes, so Phase 2 cannot import it. ``features/selection.py`` resolves that
circular dependency by **dependency injection**: :meth:`FeatureSelector.fit` takes an optional
``evaluator`` callable and, without one, falls back to the configured ``f`` and records an empty
curve rather than fabricating one.

This module is the injected side of that seam. It lives under ``eval/`` precisely because it
may import the model, which ``features/selection.py`` must not; the two never import each
other.

LEAKAGE CONTRACT (gate R3, Section III-C). The curve is a *validation* curve, and the
validation set is carved out of the **training blocks only**:

    outer split (the caller's):   deduped -> train blocks        | test blocks  (NEVER here)
    inner split (this module's):  train blocks -> inner-train    | validation

The real test set is not a parameter of this module and cannot be reached from it. Every
per-candidate scaler is fitted on inner-train rows alone, so no validation statistic informs
the training it scores. Choosing F is a modelling decision, which makes any test-set
involvement leakage of exactly the kind Section III-C exists to prevent.

COST. Scoring is a real training run per candidate: five candidates means five GRU fits. The
sweep is therefore opt-in at the driver level and takes its own (typically smaller) epoch
budget, recorded alongside the curve so a reader can see how thoroughly each point was scored.
A curve measured at a reduced budget still locates a knee -- the ranking of candidate F values
is what matters, not the absolute macro-F1 at each point -- but it is not a headline result and
must not be quoted as one.

MEASURE ONCE, THEN PIN (user-approved, 2026-09-29). Choosing F is a single Phase 2 decision for
the whole project, not a per-run one. The intended workflow is:

    1. run ONE sweep -- ``run_phase3.py --knee-sweep`` is the cheap place, since no federation is
       needed to score a candidate;
    2. read the knee off the curve the run prints and stores in its manifest;
    3. set ``features.selected_f`` in the config to it, permanently.

Every driver then inherits that F through the config, and the sweep never needs to run again.
**Only ``run_phase3.py`` and ``run_phase4.py`` take ``--knee-sweep``, and that is deliberate.**
``run_phase3_complete.py``, ``run_phase7.py`` and ``train_federated_model.py`` call
``FeatureSelector.fit`` without an evaluator on purpose: each driver draws its own inner split,
so letting them all sweep independently could select a *different* F per driver and leave the
project internally inconsistent -- the checkpoint trained at one F, the end-to-end run scored at
another. That is a worse failure than not measuring at all, because it is invisible in any single
run's output. Do not "fix" the three unwired drivers by adding the flag to them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .._types import Array
from ..data.scaling import ScalerStats, apply_scaler, fit_scaler
from ..data.split import stratified_block_split
from ..data.taxonomy import to_class_index
from ..model.train import predict, train_centralized
from ..sequences.windowing import build_windows, contiguity_segments
from .metrics import multiclass_metrics

# The inner split is drawn with a seed derived from the caller's rather than equal to it, so the
# validation draw is not correlated with the outer train/test draw that produced its input.
_INNER_SPLIT_SEED_OFFSET = 9973


def _prepare(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    window: int,
    scaler: ScalerStats | None = None,
) -> tuple[Array, Array, ScalerStats]:
    """Frame -> ``(N, W, F)`` sequences, labels, scaler. Non-finite rows are dropped.

    Mirrors the driver pipelines' own ``prepare`` step so a candidate is scored the same way the
    finally-selected columns will be: drop non-finite rows, scale from training statistics,
    window only over contiguous same-label runs within a source.
    """
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    kept = frame.loc[finite]
    values = values[finite]
    if len(values) == 0:
        raise ValueError("every row holds a non-finite value in the candidate columns")
    if scaler is None:
        scaler = fit_scaler(values)  # inner-TRAIN only
    scaled = apply_scaler(values, scaler).astype(np.float32)
    y = to_class_index(kept["label"]).to_numpy()
    # A dropped non-finite row leaves a positional gap; breaking the run here stops a window
    # spanning the hole, exactly as no window may span a held-out block.
    segments = contiguity_segments(kept["source_file"].to_numpy(), kept.index.to_numpy())
    sequences, sequence_labels = build_windows(scaled, y, segments, window)
    return sequences, sequence_labels, scaler


def make_knee_evaluator(
    train_frame: pd.DataFrame,
    train_block_ids: pd.Series[int],
    *,
    window: int,
    class_names: list[str],
    n_classes: int,
    epochs: int,
    hidden_size: int = 96,
    batch_size: int = 1024,
    validation_fraction: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
    max_train_sequences: int | None = None,
    verbose: bool = False,
) -> tuple[Callable[[list[str]], float], dict[str, object]]:
    """Build the ``evaluator`` callable :meth:`FeatureSelector.fit` injects for Stage 4.

    Args:
        train_frame: the TRAINING blocks only, as the outer split produced them. Passing the
            full deduplicated frame, or anything containing test blocks, would leak.
        train_block_ids: block id per row of ``train_frame``, for the inner split.
        window: W, so candidates are scored on the sequences the detector actually consumes.
        epochs: the per-candidate training budget. Deliberately separate from the headline
            training budget -- see this module's docstring on cost.
        max_train_sequences: optional cap on inner-training sequences per candidate, to bound
            sweep cost on a large corpus.

    Returns:
        ``(evaluator, provenance)``. ``evaluator`` maps a candidate column list to its
        validation macro-F1. ``provenance`` records how the curve was measured -- inner split
        sizes, budget, seed -- for the run manifest, so the curve is interpretable later.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if "label" not in train_frame.columns:
        raise ValueError("train_frame must carry a 'label' column")
    if len(train_frame) != len(train_block_ids):
        raise ValueError(
            f"train_frame has {len(train_frame)} rows but train_block_ids has "
            f"{len(train_block_ids)}"
        )

    inner = stratified_block_split(
        train_frame,
        train_block_ids,
        test_fraction=validation_fraction,
        seed=seed + _INNER_SPLIT_SEED_OFFSET,
    )
    inner_train = train_frame.loc[train_block_ids.isin(inner.train_blocks)]
    validation = train_frame.loc[train_block_ids.isin(inner.test_blocks)]
    if inner_train.empty or validation.empty:
        raise ValueError(
            "the inner split left one side empty "
            f"(inner-train {len(inner_train)} rows, validation {len(validation)} rows); "
            "the training set is too small to score a cardinality sweep against"
        )

    provenance: dict[str, object] = {
        "inner_train_rows": len(inner_train),
        "validation_rows": len(validation),
        "validation_fraction": float(validation_fraction),
        "inner_split_seed": int(seed + _INNER_SPLIT_SEED_OFFSET),
        "epochs_per_candidate": int(epochs),
        "window": int(window),
        "hidden_size": int(hidden_size),
        "max_train_sequences": max_train_sequences,
        # Stated in the manifest so the curve is never mistaken for a headline result.
        "note": (
            "Validation macro-F1 per candidate F, measured on a split carved from the TRAINING "
            "blocks only; the test set is never involved. Scored at a reduced epoch budget to "
            "bound sweep cost -- use it to locate the knee, not as a reported result."
        ),
        "scores": {},
    }

    def evaluator(candidate_columns: list[str]) -> float:
        """Train on inner-train restricted to ``candidate_columns``; score on validation."""
        x_inner, y_inner, scaler = _prepare(inner_train, candidate_columns, window=window)
        x_validation, y_validation, _ = _prepare(
            validation, candidate_columns, window=window, scaler=scaler
        )
        if len(x_inner) == 0 or len(x_validation) == 0:
            # No sequences survived windowing at this W; the candidate cannot be scored, and a
            # 0.0 would read as "measured and terrible" rather than "not measurable".
            raise ValueError(
                f"F={len(candidate_columns)}: windowing left "
                f"{len(x_inner)} inner-train and {len(x_validation)} validation sequences at "
                f"W={window}; cannot score this candidate"
            )
        if max_train_sequences is not None and len(x_inner) > max_train_sequences:
            rng = np.random.default_rng(seed)
            take = np.sort(rng.choice(len(x_inner), size=max_train_sequences, replace=False))
            x_inner, y_inner = x_inner[take], y_inner[take]

        model = train_centralized(
            x_inner,
            y_inner,
            hidden_size=hidden_size,
            n_classes=n_classes,
            epochs=epochs,
            seed=seed,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
        y_pred, _ = predict(model, x_validation, device=device)
        macro_f1 = multiclass_metrics(y_validation, y_pred, class_names).macro_f1
        scores = provenance["scores"]
        assert isinstance(scores, dict)
        scores[len(candidate_columns)] = float(macro_f1)
        return float(macro_f1)

    return evaluator, provenance
