"""Block-level splitting (Phase 2, Section III-B3).

Splits operate on contiguous *blocks* of records, not individual rows, so local ordering
survives into sequence construction (III-D). A stratified subset of blocks forms one global
test set shared by every client and every baseline; the remainder is available for
partitioning across clients, and each client splits its own blocks into train/validation.

This module runs only AFTER deduplication (see data/dedup.py and the R3 leakage gate).

TODO(Phase 2): implement contiguous block assembly + stratified block-level split.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BlockSplit:
    """A split expressed as block-id assignments so ordering is preserved."""

    train_blocks: list[int]
    test_blocks: list[int]


def make_blocks(frame: pd.DataFrame, block_size: int) -> pd.Series[int]:
    """Assign each (already-ordered, deduplicated) record to a contiguous block id."""
    del frame, block_size
    raise NotImplementedError("Phase 2: block assembly not implemented yet.")


def stratified_block_split(
    frame: pd.DataFrame,
    block_ids: pd.Series[int],
    *,
    test_fraction: float,
    seed: int,
) -> BlockSplit:
    """Draw a stratified subset of blocks as the shared global test set."""
    del frame, block_ids, test_fraction, seed
    raise NotImplementedError("Phase 2: stratified block split not implemented yet.")
