from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native

type _Columns = tuple[int, int, int, int, int, tuple[int, int] | None, list[int]]
type _Config = tuple[float, float, int, bool, int | None, bool]


def _columns() -> _Columns:
    return (1, 2, 5, 3, 4, (6, 7), [8])


def _config() -> _Config:
    return (0.5, 0.25, 1, False, None, False)


def test_native_detection_point_cloud_projects_3d_rows_and_passthrough() -> None:
    detections = np.array(
        [[0.0, 4.0, 2.0, 3.0, 0.5, 10.0, 0.25, 12.0, 6.0]],
        dtype=np.float32,
    )

    points = _native.project_detection_point_cloud(detections, _columns(), _config())

    expected_range = 2.0
    np.testing.assert_allclose(
        points[:, :4],
        [
            [
                expected_range * np.sin(0.5),
                expected_range * np.sqrt(1.0 - np.sin(0.5) ** 2 - np.sin(0.25) ** 2),
                expected_range * np.sin(0.25),
                0.5,
            ]
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(points[:, 9:], [[0.25, 12.0, 6.0]])


def test_native_detection_point_cloud_centers_unshifted_doppler_bins() -> None:
    detections = np.array([[0.0, 1.0, 7.0, 0.0, 0.0, 3.0]], dtype=np.float32)
    columns: _Columns = (1, 2, 5, 3, 4, None, [])
    config: _Config = (1.0, 0.5, 1, True, 8, False)

    points = _native.project_detection_point_cloud(detections, columns, config)

    assert points[0, 3] == pytest.approx(-0.5)


def test_native_detection_point_cloud_rejects_invalid_input() -> None:
    detections = np.ones((1, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="contiguous"):
        _native.project_detection_point_cloud(detections[:, ::-1], _columns(), _config())
    with pytest.raises(ValueError, match="physical point"):
        _native.project_detection_point_cloud(
            np.array(
                [[0.0, 1.0, 0.0, 0.0, np.pi / 2, 1.0, np.pi / 2, 1.0, 0.0]],
                dtype=np.float32,
            ),
            _columns(),
            _config(),
        )
