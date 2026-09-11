"""Windowing over contiguous same-label runs (Phase 2/3, Section III-D, gap G5).

Rules (stated as assumptions, since the CSVs carry no timestamp/flow id, so a window is a
positional not a temporal construct):

    * Windows are built ONLY over contiguous same-label runs within a single source file.
    * Rows are NEVER shuffled before windowing; only complete sequences are shuffled at
      batch time.
    * Training windows do NOT cross a label boundary (supervision stays unambiguous).
    * A window's label is that of its final record: y_i = y_{i+W-1}.
    * Number of sequences from runs {L_r}: N_seq = sum_r max(0, L_r - W + 1)  (Eq. 12).

A SEPARATE held-out set of mixed windows spanning a benign->attack boundary is reserved for
evaluation only, used to measure detection latency (how many records into an attack the
verdict flips). This set is never used for training.

TODO(Phase 2/3): implement run detection, within-run windowing, and the latency eval set.
"""

from __future__ import annotations

from .._types import Array


def build_windows(
    features: Array, labels: Array, source_ids: Array, window: int
) -> tuple[Array, Array]:
    """Build ``(N, W, F)`` sequences over contiguous same-label runs within each source.

    Returns:
        ``(sequences, sequence_labels)`` where each label is the final record's label.
    """
    del features, labels, source_ids, window
    raise NotImplementedError("Phase 2/3: windowing not implemented yet.")


def build_latency_eval_windows(
    features: Array, labels: Array, source_ids: Array, window: int
) -> tuple[Array, Array]:
    """Build the evaluation-only mixed windows spanning benign->attack boundaries."""
    del features, labels, source_ids, window
    raise NotImplementedError("Phase 2/3: latency eval windows not implemented yet.")


def count_sequences(run_lengths: list[int], window: int) -> int:
    """N_seq = sum_r max(0, L_r - W + 1), Eq. (12)."""
    del run_lengths, window
    raise NotImplementedError("Phase 2/3: sequence counting not implemented yet.")
