"""GATING TEST (R3) --- train/test leakage control (Section III-B2).

Blocking invariant: deduplication happens BEFORE the split, so no record hash may appear in
both the train and test partitions. A failure here inflates results by ~= delta*rho and makes
every downstream number untrustworthy; per Section III-J2 this test gates the whole pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ascon_smart_agri.data.dedup import deduplicate, record_hash
from ascon_smart_agri.data.split import make_blocks, stratified_block_split


def _hashes_for_blocks(
    frame: pd.DataFrame, block_ids: pd.Series[int], blocks: list[int]
) -> set[str]:
    return set(record_hash(frame.loc[block_ids.isin(blocks)]))


@pytest.mark.gating
def test_no_record_hash_in_both_partitions() -> None:
    # Two labels, each containing a value repeated three times -- the exact-duplicate pattern
    # the paper attributes to CICIoT2023 (Section III-B2).
    frame = pd.DataFrame(
        {
            "x": [1, 2, 99, 3, 99, 4, 99, 5, 10, 11, 88, 12, 88, 13, 88, 14],
            "label": ["a"] * 8 + ["b"] * 8,
        }
    )

    deduped = deduplicate(frame)
    assert len(deduped) == len(frame) - 4  # two extra 99's + two extra 88's removed

    block_ids = make_blocks(deduped, block_size=2)
    split = stratified_block_split(deduped, block_ids, test_fraction=0.5, seed=0)

    all_blocks = set(block_ids.unique())
    assert set(split.train_blocks) | set(split.test_blocks) == all_blocks
    assert set(split.train_blocks) & set(split.test_blocks) == set()

    train_hashes = _hashes_for_blocks(deduped, block_ids, split.train_blocks)
    test_hashes = _hashes_for_blocks(deduped, block_ids, split.test_blocks)
    assert train_hashes & test_hashes == set()


def test_leakage_would_occur_without_dedup() -> None:
    """Negative control: proves the gate above is not vacuously true.

    Splitting the SAME duplicate-laden data without deduplicating first reproduces the exact
    failure mode R3 exists to prevent: an identical record ends up on both sides of the split.
    """
    # Six copies of one row plus two distinct ones, all one label, block_size=2 -> 4 blocks;
    # three of the four blocks contain a "99" row. With test_fraction=0.5 exactly two of the
    # four blocks go to test, so by the pigeonhole principle at least one "99" block lands on
    # each side regardless of the random permutation -- this holds for every seed.
    frame = pd.DataFrame({"x": [99] * 6 + [1, 2], "label": ["a"] * 8})

    block_ids = make_blocks(frame, block_size=2)
    split = stratified_block_split(frame, block_ids, test_fraction=0.5, seed=0)

    train_hashes = _hashes_for_blocks(frame, block_ids, split.train_blocks)
    test_hashes = _hashes_for_blocks(frame, block_ids, split.test_blocks)
    assert train_hashes & test_hashes != set()
