# ruff: noqa: UP040

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pytest

from mmwcore import _native

_Config: TypeAlias = tuple[
    tuple[float, int],
    tuple[int, int, int | None, int],
    tuple[float, float, float, bool],
]


def _axes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([-1.0, 1.0], dtype=np.float32),
        np.array([-0.5, 0.5], dtype=np.float32),
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )


def _config() -> _Config:
    return ((-100.0, 256), (0, 0, None, 0), (1.0, 1.0, 0.0, True))


def test_native_cartesian_sparsification_materializes_points_and_diagnostics() -> None:
    volume = np.zeros((2, 2, 3, 3), dtype=np.float32)
    volume[0, 0, 0, 0] = 2.0
    volume[1, 1, 2, 2] = 8.0

    points, noise_floors, counts, status = _native.sparsify_cartesian_volume(
        volume,
        _axes(),
        None,
        None,
        _config(),
    )

    np.testing.assert_allclose(
        points,
        np.array(
            [[1.0, -1.0, -0.5, -1.0, 0.0], [3.0, 1.0, 0.5, 1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assert noise_floors == pytest.approx((2.0, 5.0, 8.0))
    assert counts == (18, 2, 2, 2, 2, 2, 2)
    assert status == (False, 0)


def test_native_cartesian_sparsification_rejects_noncontiguous_volume_and_mask_shape() -> None:
    volume = np.ones((2, 2, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="contiguous"):
        _native.sparsify_cartesian_volume(volume[:, :, :, ::-1], _axes(), None, None, _config())
    with pytest.raises(ValueError, match="spatial_mask_zyx"):
        _native.sparsify_cartesian_volume(
            volume,
            _axes(),
            np.ones((2, 3, 2), dtype=bool),
            None,
            _config(),
        )


def test_native_cartesian_sparsification_excludes_suppressed_doppler_slice() -> None:
    volume = np.zeros((2, 2, 3, 3), dtype=np.float32)
    volume[0, 0, 0, 0] = 2.0
    volume[1, 1, 2, 2] = 8.0

    points, noise_floors, counts, _ = _native.sparsify_cartesian_volume(
        volume,
        _axes(),
        None,
        1,
        _config(),
    )

    np.testing.assert_allclose(
        points,
        np.array([[1.0, -1.0, -0.5, -1.0, 0.0]], dtype=np.float32),
    )
    assert noise_floors == pytest.approx((2.0, 2.0, 2.0))
    assert counts[1:3] == (1, 1)
