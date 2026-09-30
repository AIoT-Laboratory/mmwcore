from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_tracking_metrics_validate_packed_frame_identity() -> None:
    arrays = (
        np.array([0, 2], dtype=np.int64),
        np.array([4, 4], dtype=np.int64),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.ones(2, dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="repeats track ID 4"):
        _native.summarize_tracking_metrics(arrays, None, 0)


def test_native_tracking_metrics_preserve_empty_sequence_contract() -> None:
    arrays = (
        np.array([0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.uint8),
    )

    header, identity, motion, intervals = _native.summarize_tracking_metrics(arrays, None, 0)

    assert header == (0, 0, 0, 0)
    assert all(array.size == 0 for array in (*identity, *motion))
    np.testing.assert_array_equal(intervals[0], [0])
    assert all(array.size == 0 for array in intervals[1:])
