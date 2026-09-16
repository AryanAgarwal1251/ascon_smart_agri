"""Unit tests for Phase 2 deduplication (Section III-B2, risk R3).

See ``tests/test_leakage.py`` for the end-to-end gating test that chains this module with
``data/split.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ascon_smart_agri.data.dedup import deduplicate, record_hash


def test_record_hash_is_stable_and_distinguishes_rows() -> None:
    frame = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    hashes = record_hash(frame)

    assert len(hashes) == 3
    assert hashes.iloc[0] == hashes.iloc[2]  # identical rows -> identical hash
    assert hashes.iloc[0] != hashes.iloc[1]  # different rows -> (almost certainly) different hash
    # Re-hashing the same frame is deterministic.
    assert (record_hash(frame) == hashes).all()


def test_deduplicate_keeps_first_occurrence_and_preserves_order() -> None:
    frame = pd.DataFrame({"a": [1, 2, 1, 3, 2], "b": ["x", "y", "x", "z", "y"]})

    deduped = deduplicate(frame)

    assert len(deduped) == 3
    assert list(deduped["a"]) == [1, 2, 3]  # order of first occurrences preserved
    assert list(deduped.index) == [0, 1, 2]  # contiguous index, not a reordering


def test_deduplicate_is_a_noop_on_already_unique_data() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    deduped = deduplicate(frame)

    pd.testing.assert_frame_equal(deduped, frame)


def test_deduplicate_no_hash_survives_twice() -> None:
    frame = pd.DataFrame({"a": [5, 5, 5, 6], "b": ["p", "p", "p", "q"]})

    deduped = deduplicate(frame)
    hashes = record_hash(deduped)

    assert hashes.is_unique
    assert not hashes.duplicated().any()


@pytest.mark.gating
def test_record_hash_used_by_leakage_gate_is_importable() -> None:
    # A cheap smoke check that the R3 gate's dependency surface hasn't moved.
    assert callable(record_hash)
    assert callable(deduplicate)
