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

Implementation notes, and the decisions the paper leaves open (flagged per Golden Rule 1):

* **A "run" is a maximal stretch of adjacent rows sharing BOTH label and source id.** A change
  in either ends the run. ``source_ids`` is the ``source_file`` column ``data/subsample.py``
  attaches; without it, two captures that happen to be adjacent in the pooled frame and share a
  label would be welded into one spurious run. Runs are detected positionally, so the caller
  must pass rows in their original order -- this module cannot verify that and does not try.
* **Sequences are emitted in input order**, never shuffled: Section III-D reserves shuffling for
  complete sequences at batch-formation time, which is the training loop's job, not this
  module's.
* **The latency set is built from benign->attack transitions only**, not attack->benign and not
  attack->attack. Section III-D defines the measurement as "how many records into an attack the
  verdict takes to flip", which presupposes the window enters an attack from benign. Which class
  counts as benign is the caller's to state (``benign_label``), so this module stays independent
  of ``data/taxonomy.py``'s class indices and works on leaf labels or family indices alike.
* **A mixed window is labelled by its final record too**, consistent with y_i = y_{i+W-1}. Each
  transition contributes the windows whose last row sits at offset 1..W-1 past the boundary, so
  the resulting set sweeps the detector across the whole boundary rather than sampling one point
  of it -- that sweep is what a latency curve is measured from. Windows that would need rows
  before the start of the benign run or past the end of the attack run are not emitted, so a
  transition between two short runs contributes fewer than W-1 windows rather than padded ones.
  No padding is ever invented.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import numpy.typing as npt

from .._types import Array

_Indices = npt.NDArray[np.int64]


def _run_start_indices(labels: Array, source_ids: Array) -> _Indices:
    """Return the start offset of every maximal (label, source) run, plus a terminal sentinel."""
    n = len(labels)
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    changed = (labels[1:] != labels[:-1]) | (source_ids[1:] != source_ids[:-1])
    starts = np.flatnonzero(changed) + 1
    return np.concatenate(([0], starts, [n])).astype(np.int64)


def contiguity_segments(source_ids: Array, row_positions: Array) -> Array:
    """Segment ids that break wherever rows stopped being genuinely adjacent in the capture.

    Splitting happens on whole blocks (Section III-B3), so filtering a frame down to the train
    blocks leaves rows that were never neighbours sitting next to each other: block 5 and block 7
    become adjacent when block 6 is held out for test. Windowing over that filtered frame would
    build sequences spanning a gap, fabricating an adjacency the capture never contained -- the
    positional-ordering assumption of Section III-D explicitly does not license that.

    Pass ``row_positions`` as the row's index in the *unfiltered*, deduplicated frame (pandas
    keeps it as ``.index`` after a ``.loc`` filter) and hand the result to
    :func:`build_windows` as its ``source_ids``. The segment id then changes whenever the source
    file changes OR the positions stop being consecutive, so no window can straddle a removed
    block.
    """
    if len(source_ids) != len(row_positions):
        raise ValueError(
            f"source_ids and row_positions lengths disagree: "
            f"{len(source_ids)}, {len(row_positions)}"
        )
    n = len(source_ids)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    positions = np.asarray(row_positions, dtype=np.int64)
    broken = (source_ids[1:] != source_ids[:-1]) | (np.diff(positions) != 1)
    segments: _Indices = np.concatenate(([0], np.cumsum(broken, dtype=np.int64)))
    return segments


def _gather(features: Array, window_starts: _Indices, window: int) -> Array:
    """Expand window start offsets into ``(N, W, F)`` sequences, preserving row order."""
    steps: _Indices = np.arange(window, dtype=np.int64)
    gathered: Array = features[window_starts[:, None] + steps[None, :]]
    return gathered


def _validate(features: Array, labels: Array, source_ids: Array, window: int) -> None:
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (rows, F), got shape {features.shape}")
    if not (len(features) == len(labels) == len(source_ids)):
        raise ValueError(
            f"features/labels/source_ids lengths disagree: "
            f"{len(features)}, {len(labels)}, {len(source_ids)}"
        )


def build_windows(
    features: Array, labels: Array, source_ids: Array, window: int
) -> tuple[Array, Array]:
    """Build ``(N, W, F)`` sequences over contiguous same-label runs within each source.

    Returns:
        ``(sequences, sequence_labels)`` where each label is the final record's label.
    """
    _validate(features, labels, source_ids, window)

    bounds = _run_start_indices(labels, source_ids)
    # Starting offsets of every window that fits entirely inside a single run.
    offsets: list[_Indices] = [
        np.arange(start, stop - window + 1, dtype=np.int64)
        for start, stop in pairwise(bounds)
        if stop - start >= window
    ]

    if not offsets:
        empty_x = np.empty((0, window, features.shape[1]), dtype=features.dtype)
        return empty_x, np.empty(0, dtype=labels.dtype)

    window_starts: _Indices = np.concatenate(offsets)
    # (N, W) gather indices -> (N, W, F) sequences, in input order.
    return _gather(features, window_starts, window), labels[window_starts + window - 1]


def build_latency_eval_windows(
    features: Array,
    labels: Array,
    source_ids: Array,
    window: int,
    *,
    benign_label: object,
) -> tuple[Array, Array]:
    """Build the evaluation-only mixed windows spanning benign->attack boundaries.

    Each window ends between 1 and ``window - 1`` records past a benign->attack transition, so
    the set sweeps the detector across the boundary; the label is still the final record's.
    EVALUATION ONLY -- these windows deliberately straddle a label change and must never be
    used for training (Section III-D).
    """
    _validate(features, labels, source_ids, window)

    bounds = _run_start_indices(labels, source_ids)
    starts: list[int] = []
    for prev_start, boundary, run_end in zip(bounds[:-2], bounds[1:-1], bounds[2:], strict=True):
        # A transition, not a source change: benign run immediately followed by an attack run.
        if labels[boundary - 1] != benign_label or labels[boundary] == benign_label:
            continue
        if source_ids[boundary - 1] != source_ids[boundary]:
            continue
        for past in range(1, window):
            end = boundary + past - 1  # index of the window's final record
            start = end - window + 1
            if start >= prev_start and end < run_end:
                starts.append(start)

    if not starts:
        empty_x = np.empty((0, window, features.shape[1]), dtype=features.dtype)
        return empty_x, np.empty(0, dtype=labels.dtype)

    window_starts: _Indices = np.asarray(starts, dtype=np.int64)
    return _gather(features, window_starts, window), labels[window_starts + window - 1]


def count_sequences(run_lengths: list[int], window: int) -> int:
    """N_seq = sum_r max(0, L_r - W + 1), Eq. (12)."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return sum(max(0, length - window + 1) for length in run_lengths)
