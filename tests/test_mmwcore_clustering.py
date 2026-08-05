from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import ClusterFrame, DBSCANClusteringSpec, PointCloudFrame
from mmwcore.dsp import cluster_point_cloud


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
        DBSCANClusteringSpec(eps_m=0.3, min_samples=2, velocity_scale_s=0.2),
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

    clusters = cluster_point_cloud(frame, DBSCANClusteringSpec(eps_m=1.0, min_samples=1))

    assert clusters.num_clusters == 0
    assert clusters.point_labels.size == 0


def test_cluster_point_cloud_requires_velocity_only_when_weighted() -> None:
    frame = PointCloudFrame(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="velocity"):
        cluster_point_cloud(
            frame,
            DBSCANClusteringSpec(eps_m=1.0, min_samples=1, velocity_scale_s=1.0),
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
