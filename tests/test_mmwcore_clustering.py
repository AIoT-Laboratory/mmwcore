from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import ClusterFrame, DBSCANSpec, PointCloudFrame
from mmwcore.dsp import cluster_point_cloud


def _cluster_frame(point_counts: np.ndarray, point_labels: np.ndarray) -> ClusterFrame:
    return ClusterFrame(
        centers=np.zeros((1, 3), dtype=np.float32),
        extents=np.zeros((1, 3), dtype=np.float32),
        mean_velocities=np.zeros(1, dtype=np.float32),
        point_counts=point_counts,
        point_labels=point_labels,
    )


def test_cluster_point_cloud_summarizes_each_cluster_velocity() -> None:
    points = np.array(
        [
            [0.0, 1.0, 0.0, 0.2],
            [0.1, 1.1, 0.0, 0.4],
            [3.0, 4.0, 0.5, -1.0],
            [3.1, 4.1, 0.5, -1.2],
            [9.0, 9.0, 9.0, 5.0],
        ],
        dtype=np.float32,
    )
    frame = PointCloudFrame(
        points,
        channels=("x", "y", "z", "velocity"),
        frame_id=7,
        timestamp=0.7,
    )

    clusters = cluster_point_cloud(
        frame,
        DBSCANSpec(eps_m=0.3, min_samples=2, velocity_scale_s=0.2),
    )

    assert clusters.num_clusters == 2
    np.testing.assert_allclose(clusters.centers, [[0.05, 1.05, 0.0], [3.05, 4.05, 0.5]])
    np.testing.assert_allclose(clusters.mean_velocities, [0.3, -1.1])
    np.testing.assert_array_equal(clusters.point_counts, [2, 2])
    np.testing.assert_array_equal(clusters.point_labels, [0, 0, 1, 1, -1])
    assert clusters.frame_id == 7
    assert clusters.timestamp == pytest.approx(0.7)


def test_cluster_point_cloud_handles_empty_frame() -> None:
    frame = PointCloudFrame(np.empty((0, 3), dtype=np.float32))

    clusters = cluster_point_cloud(frame, DBSCANSpec(eps_m=1.0, min_samples=1))

    assert clusters.num_clusters == 0
    assert clusters.point_labels.size == 0


def test_cluster_point_cloud_requires_velocity_only_when_weighted() -> None:
    frame = PointCloudFrame(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="velocity"):
        cluster_point_cloud(
            frame,
            DBSCANSpec(eps_m=1.0, min_samples=1, velocity_scale_s=1.0),
        )


def test_cluster_frame_rejects_counts_that_disagree_with_labels() -> None:
    with pytest.raises(ValueError, match="point_counts"):
        ClusterFrame(
            centers=np.zeros((1, 3), dtype=np.float32),
            extents=np.zeros((1, 3), dtype=np.float32),
            mean_velocities=np.zeros(1, dtype=np.float32),
            point_counts=np.array([2]),
            point_labels=np.array([0, -1]),
        )


def test_cluster_frame_preserves_safe_integer_arrays() -> None:
    frame = _cluster_frame(
        np.array([1], dtype=np.uint8),
        np.array([0], dtype=np.uint64),
    )

    assert frame.point_counts.dtype == np.dtype(np.int64)
    assert frame.point_labels.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(frame.point_counts, [1])
    np.testing.assert_array_equal(frame.point_labels, [0])


@pytest.mark.parametrize(
    ("point_counts", "point_labels"),
    [
        (np.array([1.0]), np.array([0], dtype=np.int64)),
        (np.array([1], dtype=np.int64), np.array([0.0])),
    ],
)
def test_cluster_frame_rejects_float_integer_fields(
    point_counts: np.ndarray,
    point_labels: np.ndarray,
) -> None:
    with pytest.raises(TypeError, match="must contain integer values"):
        _cluster_frame(point_counts, point_labels)


def test_cluster_frame_rejects_integer_values_outside_int64() -> None:
    with pytest.raises(ValueError, match="outside the int64 range"):
        _cluster_frame(
            np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
            np.array([0], dtype=np.int64),
        )
