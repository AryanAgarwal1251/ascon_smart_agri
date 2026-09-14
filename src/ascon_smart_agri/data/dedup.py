"""Deduplication BEFORE splitting --- the leakage control (Phase 2, Section III-B2, risk R3).

CICIoT2023 contains exact duplicate rows. If deduplication runs *after* the train/test split,
identical records land on both sides and the test set measures memorisation, not
generalisation; the observed accuracy is inflated by ~= delta * rho (Section III-B2). The
order is therefore fixed as: DEDUP -> SPLIT, and asserted by ``tests/test_leakage.py`` (a
blocking gate, R3). Nothing downstream is trusted until that test exists and passes.

Ordering note (not stated explicitly by the paper, resolved here): this module's functions
operate on an already-fully-loaded, in-memory ``pd.DataFrame`` -- feasible for the ~1.5-2e6-row
capped subsample of Section III-B1, not for the raw ~46.7M-row corpus (which does not fit in
memory as a single frame; Phase 1's ``characterize_dataset`` handles that scale by streaming).
The paper's R3 invariant only fixes dedup *before split*; it says nothing about dedup's position
relative to subsampling. Given these signatures, the intended pipeline order is therefore
SUBSAMPLE -> DEDUP -> SPLIT, not DEDUP -> SUBSAMPLE -> SPLIT. Flagging this inference rather than
leaving it implicit, since it fixes an ordering the paper left open.
"""

from __future__ import annotations

import pandas as pd


def record_hash(frame: pd.DataFrame) -> pd.Series[str]:
    """Return a stable per-row content hash used to detect exact duplicates and leakage.

    The same hashing is reused by the leakage gate to assert that no hash appears in both
    the train and test partitions.

    Implementation: a single 64-bit hash per row via ``pandas.util.hash_pandas_object`` (over
    every column, order-sensitive), rendered as a fixed-width hex string. At the capped-subsample
    scale this module is meant for (~1.5-2e6 rows, see the module docstring), the birthday-bound
    collision probability is on the order of 1e-10 -- negligible, and consistent with the same
    choice already made and documented for Phase 1's ``characterize.py`` reporting statistic.
    """
    hashed = pd.util.hash_pandas_object(frame, index=False)
    return hashed.map(lambda value: format(value, "016x")).astype(str)


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate records from the pooled frame *before* splitting.

    Keeps the first occurrence of each distinct row (by :func:`record_hash`) and preserves the
    original row order of the survivors -- required, since rows must never be shuffled before
    windowing (Section III-D). The index is reset to a clean contiguous range afterwards; this
    renumbers positions but does not reorder or shuffle the surviving rows.
    """
    hashes = record_hash(frame)
    return frame.loc[~hashes.duplicated(keep="first")].reset_index(drop=True)
