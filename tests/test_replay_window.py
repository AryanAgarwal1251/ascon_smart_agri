"""Unit tests for receiver-side replay enforcement (Section III-G2, near Eq. 27).

Policy confirmed with the user before implementation (per docs/plans/phase6-replay-window.md):
per-(edge_id, device_id) scope, strict-monotonic acceptance, in-memory state.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.routing.replay_guard import ReplayGuard


def test_first_seen_counter_is_accepted() -> None:
    guard = ReplayGuard()

    assert guard.check("edge01", "dev01", 0) is True


def test_increasing_counters_all_accepted() -> None:
    guard = ReplayGuard()

    assert all(guard.check("edge01", "dev01", c) for c in range(5))


def test_repeated_counter_is_rejected_as_replay() -> None:
    guard = ReplayGuard()
    guard.check("edge01", "dev01", 3)

    assert guard.check("edge01", "dev01", 3) is False


def test_any_counter_at_or_below_last_seen_is_rejected() -> None:
    guard = ReplayGuard()
    guard.check("edge01", "dev01", 10)

    assert guard.check("edge01", "dev01", 10) is False  # equal
    assert guard.check("edge01", "dev01", 5) is False  # lower, stale
    assert guard.check("edge01", "dev01", 0) is False


def test_a_rejected_counter_does_not_advance_the_window() -> None:
    """A replay must not silently move the window forward."""
    guard = ReplayGuard()
    guard.check("edge01", "dev01", 10)
    guard.check("edge01", "dev01", 10)  # rejected replay

    assert guard.last_seen("edge01", "dev01") == 10
    assert guard.check("edge01", "dev01", 11) is True  # still accepts the real next message


def test_devices_are_scoped_independently() -> None:
    """A low counter for device B must not be blocked by device A's high counter."""
    guard = ReplayGuard()
    guard.check("edge01", "devA", 100)

    assert guard.check("edge01", "devB", 0) is True


def test_edges_are_scoped_independently_even_with_the_same_device_id() -> None:
    guard = ReplayGuard()
    guard.check("edge01", "dev01", 100)

    assert guard.check("edge02", "dev01", 0) is True


def test_last_seen_reports_none_for_an_unseen_device() -> None:
    guard = ReplayGuard()

    assert guard.last_seen("edge01", "dev01") is None


def test_rejects_a_negative_counter() -> None:
    guard = ReplayGuard()

    with pytest.raises(ValueError, match="non-negative"):
        guard.check("edge01", "dev01", -1)
