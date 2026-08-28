from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def _config() -> _native.NativeClusterTrackerConfig:
    return (
        (0.1, (2.0, 2.0), 0.2, 2.0, 0.2),
        (0.5, None, None),
        (1, 0.0, None, None),
        (3, 2, 3),
        ([], 5),
        200,
    )


def _clusters(x_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(((x_m, 1.0, 0.0),), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.int64),
    )


def _measurement_config() -> _native.NativeMeasurementTrackerConfig:
    return (_config(), (0.2, 2, 0.0, False))


def test_native_cluster_tracker_retains_state_and_confirms_lifecycle() -> None:
    tracker = _native.NativeClusterTracker2D(_config())

    first = tracker.step(*_clusters(0.0))
    tracker.step(*_clusters(0.1))
    third = tracker.step(*_clusters(0.2))

    np.testing.assert_array_equal(first[0], [0])
    np.testing.assert_array_equal(first[5], [0])
    np.testing.assert_array_equal(third[0], [0])
    np.testing.assert_array_equal(third[5], [1])
    np.testing.assert_array_equal(third[-1], [0])
    assert third[2][0, 0] > 0.0


def test_native_cluster_tracker_rejects_non_contiguous_or_mismatched_inputs() -> None:
    tracker = _native.NativeClusterTracker2D(_config())
    centers, extents, velocities, point_counts = _clusters(0.0)

    with pytest.raises(ValueError, match="C-contiguous"):
        tracker.step(centers[:, ::-1], extents, velocities, point_counts)
    with pytest.raises(ValueError, match="extents must have shape"):
        tracker.step(centers, extents[:, :2], velocities, point_counts)
    with pytest.raises(ValueError, match="must be non-negative"):
        tracker.step(
            centers,
            np.array(((-1.0, 0.0, 0.0),), dtype=np.float32),
            velocities,
            point_counts,
        )


def test_native_measurement_tracker_partitions_points_between_tracks() -> None:
    tracker = _native.NativePointTracker2D(_measurement_config())
    first_coordinates = np.array(
        ((-1.05, 1.0, 0.0), (-0.95, 1.0, 0.0), (0.95, 1.0, 0.0), (1.05, 1.0, 0.0)),
        dtype=np.float32,
    )
    second_coordinates = np.array(
        ((-0.9, 1.0, 0.0), (-0.8, 1.0, 0.0), (0.8, 1.0, 0.0), (0.9, 1.0, 0.0)),
        dtype=np.float32,
    )
    velocities = np.zeros(4, dtype=np.float32)
    snrs = np.zeros(4, dtype=np.float32)

    tracker.step(first_coordinates, velocities, snrs)
    second = tracker.step(second_coordinates, velocities, snrs)

    np.testing.assert_array_equal(second[0], [0, 1])
    np.testing.assert_array_equal(second[-1], [0, 0, 1, 1])


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
