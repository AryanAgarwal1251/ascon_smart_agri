"""Block-level splitting (Phase 2, Section III-B3).

Splits operate on contiguous *blocks* of records, not individual rows, so local ordering
survives into sequence construction (III-D). A stratified subset of blocks forms one global
test set shared by every client and every baseline; the remainder is available for
partitioning across clients, and each client splits its own blocks into train/validation.

This module runs only AFTER deduplication (see data/dedup.py and the R3 leakage gate).

Block construction detail (not spelled out by the paper, resolved here): a block never spans a
label change, even within one fixed-size run. This is required for :func:`stratified_block_split`
to stratify meaningfully (a block with more than one label cannot be assigned a single stratum),
and it anticipates Section III-D's later, stricter rule that *windows* may never cross a label
boundary either -- blocks are the coarser unit windows are drawn from, so the two must agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlockSplit:
    """A split expressed as block-id assignments so ordering is preserved."""

    train_blocks: list[int]
    test_blocks: list[int]


def make_blocks(frame: pd.DataFrame, block_size: int) -> pd.Series[int]:
    """Assign each (already-ordered, deduplicated) record to a contiguous block id.

    A new block starts whenever ``block_size`` rows have accumulated in the current block, OR
    the value of ``frame["label"]`` changes from the previous row -- whichever comes first (see
    the module docstring). Requires a ``"label"`` column.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if "label" not in frame.columns:
        raise ValueError("frame must have a 'label' column to assign label-respecting blocks")

    labels = frame["label"].to_numpy()
    n = len(labels)
    block_ids = np.zeros(n, dtype=np.int64)
    current_block = 0
    run_length = 0
    for i in range(n):
        if i > 0 and (labels[i] != labels[i - 1] or run_length == block_size):
            current_block += 1
            run_length = 0
        block_ids[i] = current_block
        run_length += 1
    return pd.Series(block_ids, index=frame.index, name="block_id")


def stratified_block_split(
    frame: pd.DataFrame,
    block_ids: pd.Series[int],
    *,
    test_fraction: float,
    seed: int,
) -> BlockSplit:
    """Draw a stratified subset of blocks as the shared global test set.

    Each block is required to carry exactly one label (guaranteed by :func:`make_blocks`); that
    label is the stratum used to draw ``test_fraction`` of each label's blocks into the test set,
    with the remainder assigned to train. A label with few blocks may round to zero test blocks
    (or zero train blocks) -- a known limitation of block-level, rather than row-level,
    stratification, most visible for the rarest classes.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if "label" not in frame.columns:
        raise ValueError("frame must have a 'label' column for stratified splitting")

    label_per_block = frame["label"].groupby(block_ids).agg(lambda s: s.iloc[0])
    n_labels_per_block = frame["label"].groupby(block_ids).nunique()
    crossing = n_labels_per_block[n_labels_per_block > 1]
    if not crossing.empty:
        raise ValueError(
            f"blocks {crossing.index.tolist()} span more than one label; "
            "stratified_block_split requires single-label blocks (see make_blocks)"
        )

    rng = np.random.default_rng(seed)
    train_blocks: list[int] = []
    test_blocks: list[int] = []
    for label in sorted(label_per_block.unique(), key=str):
        blocks_for_label = np.sort(label_per_block.index[label_per_block == label].to_numpy())
        n_test = round(len(blocks_for_label) * test_fraction)
        permuted = rng.permutation(blocks_for_label)
        test_blocks.extend(int(b) for b in permuted[:n_test])
        train_blocks.extend(int(b) for b in permuted[n_test:])

    return BlockSplit(train_blocks=sorted(train_blocks), test_blocks=sorted(test_blocks))
