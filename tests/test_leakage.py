"""GATING TEST (R3) --- train/test leakage control (Section III-B2).

Blocking invariant: deduplication happens BEFORE the split, so no record hash may appear in
both the train and test partitions. A failure here inflates results by ~= delta*rho and makes
every downstream number untrustworthy; per Section III-J2 this test gates the whole pipeline.

Activates in Phase 2, once dedup + block splitting land.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.data.dedup import deduplicate, record_hash


@pytest.mark.gating
@pytest.mark.skip(reason="pending Phase 2: dedup/split not implemented yet")
def test_no_record_hash_in_both_partitions() -> None:
    # Phase 2 will: build a frame with known duplicates, dedup THEN split on blocks, and
    # assert set(train_hashes) & set(test_hashes) == empty.
    assert callable(deduplicate)
    assert callable(record_hash)
    raise AssertionError("implement in Phase 2")
