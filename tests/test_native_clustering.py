from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_dbscan_clusters_velocity_weighted_cartesian_points() -> None:
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

    labels, centers, extents, velocities, counts = _native.cluster_points(
        points,
        (0, 1, 2, 3),
        (0.3, 2, 0.2, True),
    )

    np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, -1], dtype=np.int64))
    np.testing.assert_allclose(
        centers,
        np.array([[0.05, 1.05, 0.0], [3.05, 4.05, 0.5]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        extents,
        np.array([[0.1, 0.1, 0.0], [0.1, 0.1, 0.0]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(velocities, np.array([0.3, -1.1], dtype=np.float32))
    np.testing.assert_array_equal(counts, np.array([2, 2], dtype=np.int64))


def test_native_dbscan_preserves_empty_matrix_shapes() -> None:
    labels, centers, extents, velocities, counts = _native.cluster_points(
        np.empty((0, 3), dtype=np.float32),
        (0, 1, 2, None),
        (1.0, 1, 0.0, True),
    )

    assert labels.shape == (0,)
    assert centers.shape == (0, 3)
    assert extents.shape == (0, 3)
    assert velocities.shape == (0,)
    assert counts.shape == (0,)


def test_native_dbscan_promotes_noise_and_ignores_z_when_configured() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 7.0],
            [2.0, 0.0, -7.0],
        ],
        dtype=np.float32,
    )

    labels, centers, extents, velocities, counts = _native.cluster_points(
        points,
        (0, 1, 2, None),
        (1.1, 3, 0.0, False),
    )

    np.testing.assert_array_equal(labels, np.array([0, 0, 0], dtype=np.int64))
    np.testing.assert_allclose(centers, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(extents, np.array([[2.0, 0.0, 14.0]], dtype=np.float32))
    np.testing.assert_allclose(velocities, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(counts, np.array([3], dtype=np.int64))


def test_native_dbscan_rejects_invalid_direct_boundary_inputs() -> None:
    points = np.ones((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="contiguous"):
        _native.cluster_points(points[:, ::-1], (0, 1, 2, 3), (1.0, 1, 0.0, True))
    with pytest.raises(ValueError, match="velocity column"):
        _native.cluster_points(points, (0, 1, 2, None), (1.0, 1, 0.1, True))
    with pytest.raises(ValueError, match="epsilon"):
        _native.cluster_points(points, (0, 1, 2, 3), (0.0, 1, 0.0, True))
