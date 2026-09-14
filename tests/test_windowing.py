"""Unit tests for sequence construction (Section III-D, gap G5).

The invariants under test are the ones CLAUDE.md lists as non-negotiable: windows form only over
contiguous same-label runs, never cross a label boundary, never cross a source file, rows are
never reordered, and the sequence label is the FINAL record's (y_i = y_{i+W-1}).
"""

from __future__ import annotations

import numpy as np
import pytest

from ascon_smart_agri.sequences.windowing import (
    build_latency_eval_windows,
    build_windows,
    contiguity_segments,
    count_sequences,
)


def _features(n: int, n_cols: int = 2) -> np.ndarray:
    """Rows whose values encode their own position, so ordering is checkable."""
    return np.arange(n * n_cols, dtype=np.float64).reshape(n, n_cols)


def test_count_sequences_matches_equation_12() -> None:
    # N_seq = sum_r max(0, L_r - W + 1); short runs contribute nothing once W grows.
    assert count_sequences([10, 5, 2], window=1) == 17
    assert count_sequences([10, 5, 2], window=3) == 8 + 3 + 0
    assert count_sequences([10, 5, 2], window=16) == 0
    assert count_sequences([], window=4) == 0


def test_count_sequences_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window"):
        count_sequences([5], window=0)


def test_windows_do_not_cross_a_label_boundary() -> None:
    x = _features(6)
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    sources = np.array(["f1"] * 6)

    sequences, seq_labels = build_windows(x, labels, sources, window=2)

    # Runs of 3 and 3 at W=2 -> 2 + 2 = 4 windows (Eq. 12), never the pair straddling index 2/3.
    assert len(sequences) == count_sequences([3, 3], window=2) == 4
    assert list(seq_labels) == ["a", "a", "b", "b"]


def test_windows_do_not_cross_a_source_file_even_when_labels_match() -> None:
    x = _features(6)
    labels = np.array(["a"] * 6)  # same label throughout
    sources = np.array(["f1", "f1", "f1", "f2", "f2", "f2"])

    sequences, _ = build_windows(x, labels, sources, window=2)

    # The source change at index 3 must still break the run: 2 + 2, not 5.
    assert len(sequences) == 4


def test_sequence_label_is_the_final_record() -> None:
    x = _features(4)
    labels = np.array(["p", "q", "r", "s"])
    sources = np.array(["f"] * 4)

    _, seq_labels = build_windows(x, labels, sources, window=1)

    # W=1: every row is its own window and its own label.
    assert list(seq_labels) == ["p", "q", "r", "s"]


def test_sequences_preserve_row_order_and_contents() -> None:
    x = _features(4)
    labels = np.array(["a"] * 4)
    sources = np.array(["f"] * 4)

    sequences, _ = build_windows(x, labels, sources, window=3)

    assert sequences.shape == (2, 3, 2)  # (N, W, F)
    # First window is rows 0,1,2 in order; second is rows 1,2,3 -- never reordered.
    np.testing.assert_array_equal(sequences[0], x[0:3])
    np.testing.assert_array_equal(sequences[1], x[1:4])


def test_runs_shorter_than_the_window_contribute_nothing() -> None:
    x = _features(5)
    labels = np.array(["a", "a", "b", "c", "c"])  # runs of 2, 1, 2
    sources = np.array(["f"] * 5)

    sequences, seq_labels = build_windows(x, labels, sources, window=3)

    assert len(sequences) == count_sequences([2, 1, 2], window=3) == 0
    assert sequences.shape == (0, 3, 2)  # still correctly shaped, not a bare empty array
    assert len(seq_labels) == 0


def test_window_larger_than_every_run_yields_empty_not_an_error() -> None:
    x = _features(3)
    labels = np.array(["a", "a", "a"])
    sources = np.array(["f"] * 3)

    sequences, seq_labels = build_windows(x, labels, sources, window=16)

    assert sequences.shape == (0, 16, 2)
    assert len(seq_labels) == 0


def test_build_windows_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="lengths disagree"):
        build_windows(_features(4), np.array(["a", "a"]), np.array(["f"] * 4), window=2)


def test_build_windows_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window"):
        build_windows(_features(4), np.array(["a"] * 4), np.array(["f"] * 4), window=0)


def test_latency_windows_straddle_a_benign_to_attack_boundary() -> None:
    x = _features(8)
    labels = np.array(["benign"] * 4 + ["attack"] * 4)
    sources = np.array(["f"] * 8)

    sequences, seq_labels = build_latency_eval_windows(
        x, labels, sources, window=3, benign_label="benign"
    )

    # Windows ending 1 and 2 records past the boundary at index 4 -> ends at 4 and 5.
    assert len(sequences) == 2
    np.testing.assert_array_equal(sequences[0], x[2:5])  # rows 2,3 benign + row 4 attack
    np.testing.assert_array_equal(sequences[1], x[3:6])
    assert list(seq_labels) == ["attack", "attack"]  # labelled by the final record


def test_latency_windows_ignore_attack_to_benign_transitions() -> None:
    x = _features(8)
    labels = np.array(["attack"] * 4 + ["benign"] * 4)
    sources = np.array(["f"] * 8)

    sequences, _ = build_latency_eval_windows(x, labels, sources, window=3, benign_label="benign")

    # Section III-D measures how many records INTO an attack the verdict flips.
    assert len(sequences) == 0


def test_latency_windows_never_cross_a_source_boundary() -> None:
    x = _features(8)
    labels = np.array(["benign"] * 4 + ["attack"] * 4)
    sources = np.array(["f1"] * 4 + ["f2"] * 4)  # the transition is also a file change

    sequences, _ = build_latency_eval_windows(x, labels, sources, window=3, benign_label="benign")

    assert len(sequences) == 0


def test_latency_windows_are_not_padded_when_runs_are_short() -> None:
    x = _features(4)
    labels = np.array(["benign", "attack", "attack", "attack"])
    sources = np.array(["f"] * 4)

    sequences, _ = build_latency_eval_windows(x, labels, sources, window=3, benign_label="benign")

    # Only one benign row precedes the boundary, so the window ending at offset 1 past it
    # (rows -1,0,1) cannot be formed; only the one ending at offset 2 (rows 0,1,2) can.
    assert len(sequences) == 1
    np.testing.assert_array_equal(sequences[0], x[0:3])


def test_latency_windows_are_disjoint_from_training_windows() -> None:
    """The latency set is evaluation-only: every one of its windows is mixed-label, so none of
    them can appear in the pure-run training set (Section III-D)."""
    x = _features(8)
    labels = np.array(["benign"] * 4 + ["attack"] * 4)
    sources = np.array(["f"] * 8)

    train_seq, _ = build_windows(x, labels, sources, window=3)
    latency_seq, _ = build_latency_eval_windows(x, labels, sources, window=3, benign_label="benign")

    train_rows = {seq.tobytes() for seq in train_seq}
    latency_rows = {seq.tobytes() for seq in latency_seq}
    assert train_rows & latency_rows == set()


def test_contiguity_segments_break_at_a_removed_block() -> None:
    """Rows made adjacent by holding out a block must not be welded into one run."""
    sources = np.array(["f"] * 6)
    # Positions 10,11,12 then 20,21,22: the 13..19 rows went to the test split.
    positions = np.array([10, 11, 12, 20, 21, 22])

    segments = contiguity_segments(sources, positions)

    assert list(segments) == [0, 0, 0, 1, 1, 1]


def test_contiguity_segments_break_at_a_source_change_too() -> None:
    sources = np.array(["f1", "f1", "f2", "f2"])
    positions = np.array([0, 1, 2, 3])  # consecutive, but the file changed

    assert list(contiguity_segments(sources, positions)) == [0, 0, 1, 1]


def test_contiguity_segments_are_one_segment_when_nothing_breaks() -> None:
    sources = np.array(["f"] * 4)
    positions = np.array([7, 8, 9, 10])

    assert list(contiguity_segments(sources, positions)) == [0, 0, 0, 0]


def test_windows_never_span_a_removed_block_when_segments_are_used() -> None:
    x = _features(6)
    labels = np.array(["a"] * 6)  # one label throughout: only the gap may break the run
    sources = np.array(["f"] * 6)
    positions = np.array([0, 1, 2, 50, 51, 52])  # a held-out block sits between 2 and 50

    segments = contiguity_segments(sources, positions)
    sequences, _ = build_windows(x, labels, segments, window=3)

    # Two runs of 3 at W=3 -> 1 + 1 windows. Without the segment ids this would wrongly be 4,
    # two of which would straddle the gap.
    assert len(sequences) == count_sequences([3, 3], window=3) == 2
    np.testing.assert_array_equal(sequences[0], x[0:3])
    np.testing.assert_array_equal(sequences[1], x[3:6])


def test_contiguity_segments_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="lengths disagree"):
        contiguity_segments(np.array(["f"] * 3), np.array([0, 1]))


def test_contiguity_segments_handle_an_empty_frame() -> None:
    assert len(contiguity_segments(np.array([]), np.array([]))) == 0
