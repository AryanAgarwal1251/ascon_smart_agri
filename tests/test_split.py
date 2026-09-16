"""Unit tests for Phase 2 block-level splitting (Section III-B3).

See ``tests/test_leakage.py`` for the end-to-end gating test that chains this module with
``data/dedup.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ascon_smart_agri.data.split import make_blocks, stratified_block_split


def test_make_blocks_splits_by_size_within_one_label() -> None:
    frame = pd.DataFrame({"label": ["a"] * 5})

    block_ids = make_blocks(frame, block_size=2)

    assert list(block_ids) == [0, 0, 1, 1, 2]  # 5 rows, size 2 -> blocks of 2, 2, 1


def test_make_blocks_never_crosses_a_label_change() -> None:
    frame = pd.DataFrame({"label": ["a", "a", "a", "b", "b", "b"]})

    block_ids = make_blocks(frame, block_size=10)  # size alone would never force a new block

    assert list(block_ids) == [0, 0, 0, 1, 1, 1]  # forced to split at the label boundary


def test_make_blocks_rejects_missing_label_column() -> None:
    with pytest.raises(ValueError, match="label"):
        make_blocks(pd.DataFrame({"x": [1, 2, 3]}), block_size=2)


def test_make_blocks_rejects_non_positive_block_size() -> None:
    with pytest.raises(ValueError, match="block_size"):
        make_blocks(pd.DataFrame({"label": ["a"]}), block_size=0)


def test_stratified_block_split_partitions_every_block_exactly_once() -> None:
    frame = pd.DataFrame({"label": ["a"] * 8 + ["b"] * 8})
    block_ids = make_blocks(frame, block_size=2)  # 4 blocks of "a", 4 of "b"

    split = stratified_block_split(frame, block_ids, test_fraction=0.25, seed=0)

    all_blocks = set(block_ids.unique())
    assert set(split.train_blocks) | set(split.test_blocks) == all_blocks
    assert set(split.train_blocks).isdisjoint(split.test_blocks)


def test_stratified_block_split_is_stratified_per_label() -> None:
    # 8 blocks of "a" (common), 2 blocks of "b" (rare) -- test_fraction=0.5 should draw from both.
    frame = pd.DataFrame({"label": ["a"] * 16 + ["b"] * 4})
    block_ids = make_blocks(frame, block_size=2)

    split = stratified_block_split(frame, block_ids, test_fraction=0.5, seed=0)

    label_by_block = frame["label"].groupby(block_ids).first()
    test_labels = {label_by_block[b] for b in split.test_blocks}
    train_labels = {label_by_block[b] for b in split.train_blocks}
    assert test_labels == {"a", "b"}
    assert train_labels == {"a", "b"}


def test_stratified_block_split_is_deterministic_given_a_seed() -> None:
    frame = pd.DataFrame({"label": ["a"] * 12})
    block_ids = make_blocks(frame, block_size=2)

    split1 = stratified_block_split(frame, block_ids, test_fraction=0.3, seed=42)
    split2 = stratified_block_split(frame, block_ids, test_fraction=0.3, seed=42)

    assert split1 == split2


def test_stratified_block_split_rejects_mixed_label_blocks() -> None:
    frame = pd.DataFrame({"label": ["a", "b"]})
    # Force an invalid block assignment (bypassing make_blocks) to check the defensive guard.
    bad_block_ids = pd.Series([0, 0], index=frame.index)

    with pytest.raises(ValueError, match="more than one label"):
        stratified_block_split(frame, bad_block_ids, test_fraction=0.5, seed=0)


def test_stratified_block_split_rejects_out_of_range_test_fraction() -> None:
    frame = pd.DataFrame({"label": ["a", "a"]})
    block_ids = pd.Series([0, 0], index=frame.index)

    with pytest.raises(ValueError, match="test_fraction"):
        stratified_block_split(frame, block_ids, test_fraction=1.5, seed=0)
