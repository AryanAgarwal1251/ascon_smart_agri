"""Deduplication BEFORE splitting --- the leakage control (Phase 2, Section III-B2, risk R3).

CICIoT2023 contains exact duplicate rows. If deduplication runs *after* the train/test split,
identical records land on both sides and the test set measures memorisation, not
generalisation; the observed accuracy is inflated by ~= delta * rho (Section III-B2). The
order is therefore fixed as: DEDUP -> SPLIT, and asserted by ``tests/test_leakage.py`` (a
blocking gate, R3). Nothing downstream is trusted until that test exists and passes.

TODO(Phase 2): implement record hashing + exact-duplicate removal on the pooled frame,
BEFORE any split is drawn.
"""

from __future__ import annotations

import pandas as pd


def record_hash(frame: pd.DataFrame) -> pd.Series[str]:
    """Return a stable per-row content hash used to detect exact duplicates and leakage.

    The same hashing is reused by the leakage gate to assert that no hash appears in both
    the train and test partitions.
    """
    del frame
    raise NotImplementedError("Phase 2: record hashing not implemented yet.")


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate records from the pooled frame *before* splitting."""
    del frame
    raise NotImplementedError("Phase 2: deduplication not implemented yet.")
