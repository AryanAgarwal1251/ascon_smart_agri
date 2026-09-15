"""Unit tests for streaming per-device window assembly (Section III-A/D, Phase 7).

The property that matters: a window is available exactly when W records have arrived for a
device, never before (no padding/fabrication), and devices never cross-contaminate each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from ascon_smart_agri.sequences.streaming import DeviceWindowBuffer


def test_returns_none_until_the_window_is_full() -> None:
    buffer = DeviceWindowBuffer(window=3)

    assert buffer.push("d1", np.array([1.0, 2.0])) is None
    assert buffer.push("d1", np.array([3.0, 4.0])) is None
    window = buffer.push("d1", np.array([5.0, 6.0]))

    assert window is not None
    assert window.shape == (3, 2)


def test_window_contents_are_in_arrival_order_never_shuffled() -> None:
    buffer = DeviceWindowBuffer(window=3)
    rows = [np.array([float(i)]) for i in range(3)]
    for r in rows[:-1]:
        buffer.push("d1", r)

    window = buffer.push("d1", rows[-1])

    np.testing.assert_array_equal(window, np.stack(rows))


def test_buffer_slides_by_one_after_it_first_fills() -> None:
    buffer = DeviceWindowBuffer(window=2)
    buffer.push("d1", np.array([1.0]))
    buffer.push("d1", np.array([2.0]))  # first window: [1, 2]

    second = buffer.push("d1", np.array([3.0]))

    np.testing.assert_array_equal(second, np.array([[2.0], [3.0]]))  # oldest dropped


def test_every_push_once_full_yields_a_window_not_only_every_w_pushes() -> None:
    buffer = DeviceWindowBuffer(window=2)
    buffer.push("d1", np.array([1.0]))
    buffer.push("d1", np.array([2.0]))

    results = [buffer.push("d1", np.array([float(i)])) for i in range(3, 6)]

    assert all(r is not None for r in results)  # not None every-other-push


def test_devices_are_buffered_independently() -> None:
    buffer = DeviceWindowBuffer(window=2)
    buffer.push("d1", np.array([100.0]))

    # d2's first push must not be affected by d1's history.
    assert buffer.push("d2", np.array([1.0])) is None
    window = buffer.push("d2", np.array([2.0]))

    np.testing.assert_array_equal(window, np.array([[1.0], [2.0]]))


def test_no_window_is_fabricated_with_padding() -> None:
    buffer = DeviceWindowBuffer(window=5)

    for _ in range(4):
        assert buffer.push("d1", np.array([1.0])) is None  # never pads to a fake window


def test_depth_reports_current_buffer_fill() -> None:
    buffer = DeviceWindowBuffer(window=3)

    assert buffer.depth("d1") == 0
    buffer.push("d1", np.array([1.0]))
    assert buffer.depth("d1") == 1
    buffer.push("d1", np.array([2.0]))
    buffer.push("d1", np.array([3.0]))
    assert buffer.depth("d1") == 3  # caps at window, never grows past it
    buffer.push("d1", np.array([4.0]))
    assert buffer.depth("d1") == 3


def test_reset_clears_one_device_without_touching_others() -> None:
    buffer = DeviceWindowBuffer(window=2)
    buffer.push("d1", np.array([1.0]))
    buffer.push("d2", np.array([9.0]))
    buffer.push("d2", np.array([9.0]))

    buffer.reset("d1")

    assert buffer.depth("d1") == 0
    assert buffer.depth("d2") == 2  # untouched


def test_reset_with_no_device_id_clears_everything() -> None:
    buffer = DeviceWindowBuffer(window=2)
    buffer.push("d1", np.array([1.0]))
    buffer.push("d2", np.array([1.0]))

    buffer.reset()

    assert buffer.depth("d1") == 0
    assert buffer.depth("d2") == 0


def test_rejects_a_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window"):
        DeviceWindowBuffer(window=0)


def test_rejects_a_non_1d_feature_vector() -> None:
    buffer = DeviceWindowBuffer(window=2)

    with pytest.raises(ValueError, match="1-D"):
        buffer.push("d1", np.ones((2, 2)))


def test_rejects_a_feature_count_that_changes_mid_stream() -> None:
    buffer = DeviceWindowBuffer(window=3)
    buffer.push("d1", np.array([1.0, 2.0]))  # F=2 established

    with pytest.raises(ValueError, match="expected 2 features"):
        buffer.push("d1", np.array([1.0, 2.0, 3.0]))  # F=3, mismatched
