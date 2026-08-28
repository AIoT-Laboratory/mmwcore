from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pytest

from mmwcore.core import SparsifySpec
from mmwcore.dsp import sparsify


class _Axes(TypedDict):
    doppler_velocity_mps: np.ndarray
    z_m: np.ndarray
    y_m: np.ndarray
    x_m: np.ndarray


def _axes() -> _Axes:
    return {
        "doppler_velocity_mps": np.array([-1.0, 1.0], dtype=np.float32),
        "z_m": np.array([-0.5, 0.5], dtype=np.float32),
        "y_m": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "x_m": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }


def test_sparsify_maps_physical_axes_and_signed_velocity() -> None:
    volume = np.zeros((2, 2, 3, 3), dtype=np.float32)
    volume[0, 0, 0, 0] = 2.0
    volume[1, 1, 2, 2] = 8.0

    frame = sparsify(
        volume,
        **_axes(),
        spec=SparsifySpec(
            min_snr_db=-100.0,
            spatial_peak_radius=0,
        ),
        frame_id="frame-1",
        coordinate_frame="fixture",
    )

    assert frame.channels == ("x", "y", "z", "velocity", "snr_db")
    np.testing.assert_allclose(
        frame.points[:, :4],
        np.array(
            [
                [1.0, -1.0, -0.5, -1.0],
                [3.0, 1.0, 0.5, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert frame.frame_id == "frame-1"
    assert frame.coordinate_frame == "fixture"


def test_sparsify_excludes_declared_doppler_slice() -> None:
    volume = np.zeros((2, 2, 3, 3), dtype=np.float32)
    volume[0, 0, 0, 0] = 2.0
    volume[1, 1, 2, 2] = 8.0

    frame = sparsify(
        volume,
        **_axes(),
        suppressed_doppler_index=1,
        spec=SparsifySpec(
            min_snr_db=-100.0,
            spatial_peak_radius=0,
        ),
    )

    np.testing.assert_allclose(frame.points[:, :4], [[1.0, -1.0, -0.5, -1.0]])
    extraction = frame.metadata["cartesian_volume_sparsification"]
    assert extraction["suppressed_doppler_index"] == 1
    assert extraction["positive_volume_voxels"] == 1
    assert extraction["noise_floor_max"] == pytest.approx(2.0)


def test_sparsify_applies_spatial_nms_and_deterministic_top_k() -> None:
    volume = np.zeros((2, 2, 3, 3), dtype=np.float32)
    volume[0, 0, 0, 0] = 3.0
    volume[1, 0, 0, 1] = 4.0
    volume[0, 1, 2, 2] = 5.0

    frame = sparsify(
        volume,
        **_axes(),
        spec=SparsifySpec(
            min_snr_db=-100.0,
            max_points=2,
            spatial_peak_radius=1,
        ),
    )

    assert frame.num_points == 2
    np.testing.assert_allclose(frame.points[:, 0], [3.0, 2.0])
    extraction = frame.metadata["cartesian_volume_sparsification"]
    assert extraction["local_peak_voxels"] == 3
    assert extraction["output_points"] == 2
    assert extraction["fallback_used"] is False


def test_sparsify_uses_explicit_nonempty_fallback() -> None:
    volume = np.ones((2, 2, 3, 3), dtype=np.float32)
    volume[1, 1, 2, 2] = 2.0

    frame = sparsify(
        volume,
        **_axes(),
        spec=SparsifySpec(min_snr_db=100.0),
    )

    assert frame.num_points == 1
    extraction = frame.metadata["cartesian_volume_sparsification"]
    assert extraction["fallback_used"] is True
    assert extraction["limited_peak_voxels"] == 0


def test_sparsify_preserves_all_zero_frame_as_empty() -> None:
    frame = sparsify(
        np.zeros((2, 2, 3, 3), dtype=np.float32),
        **_axes(),
    )

    assert frame.points.shape == (0, 5)
    extraction = frame.metadata["cartesian_volume_sparsification"]
    assert extraction["noise_floor_max"] == 0.0
    assert extraction["fallback_used"] is False


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"magnitude_dzyx": np.ones((2, 2, 3))}, "shape"),
        (
            {"magnitude_dzyx": -np.ones((2, 2, 3, 3))},
            "non-negative",
        ),
        (
            {"x_m": np.array([1.0, 1.0, 2.0])},
            "strictly increasing",
        ),
    ],
)
def test_sparsify_rejects_invalid_inputs(
    update: dict[str, np.ndarray],
    message: str,
) -> None:
    kwargs: dict[str, Any] = {
        "magnitude_dzyx": np.ones((2, 2, 3, 3), dtype=np.float32),
        **_axes(),
    }
    kwargs.update(update)
    with pytest.raises(ValueError, match=message):
        sparsify(**kwargs)


def test_cartesian_volume_sparsification_spec_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_points"):
        SparsifySpec(max_points=0)
    with pytest.raises(ValueError, match="noise_floor_scale"):
        SparsifySpec(noise_floor_scale=0.0)
    with pytest.raises(ValueError, match="static_point_capacity_fraction"):
        SparsifySpec(static_point_capacity_fraction=0.0)
    with pytest.raises(ValueError, match="boundary_margin_voxels"):
        SparsifySpec(boundary_margin_voxels=-1)
    with pytest.raises(ValueError, match="doppler_peak_radius"):
        SparsifySpec(doppler_peak_radius=-1)
    with pytest.raises(ValueError, match="max_doppler_peaks_per_spatial"):
        SparsifySpec(max_doppler_peaks_per_spatial=0)
    with pytest.raises(ValueError, match="static_velocity_threshold_mps"):
        SparsifySpec(static_velocity_threshold_mps=-0.1)


def test_sparsify_limits_doppler_sidelobes_per_spatial_voxel() -> None:
    volume = np.ones((5, 1, 1, 3), dtype=np.float32)
    volume[:, 0, 0, 1] = [2.0, 10.0, 3.0, 6.0, 2.0]
    axes = {
        "doppler_velocity_mps": np.arange(5, dtype=np.float32) - 2.0,
        "z_m": np.array([0.0], dtype=np.float32),
        "y_m": np.array([0.0], dtype=np.float32),
        "x_m": np.arange(3, dtype=np.float32),
    }

    strongest = sparsify(
        volume,
        **axes,
        spec=SparsifySpec(
            min_snr_db=5.0,
            spatial_peak_radius=0,
            doppler_peak_radius=1,
            max_doppler_peaks_per_spatial=1,
        ),
    )
    two_peaks = sparsify(
        volume,
        **axes,
        spec=SparsifySpec(
            min_snr_db=5.0,
            spatial_peak_radius=0,
            doppler_peak_radius=1,
            max_doppler_peaks_per_spatial=2,
        ),
    )

    np.testing.assert_allclose(strongest.points[:, :4], [[1.0, 0.0, 0.0, -1.0]])
    np.testing.assert_allclose(
        two_peaks.points[:, :4],
        [[1.0, 0.0, 0.0, -1.0], [1.0, 0.0, 0.0, 1.0]],
    )
    extraction = two_peaks.metadata["cartesian_volume_sparsification"]
    assert extraction["doppler_peak_radius"] == 1
    assert extraction["max_doppler_peaks_per_spatial"] == 2
    assert extraction["limited_peak_voxels"] == 2


def test_sparsify_caps_near_static_velocity_points() -> None:
    volume = np.zeros((4, 1, 1, 6), dtype=np.float32)
    volume[1, 0, 0] = np.arange(20.0, 14.0, -1.0)
    volume[2, 0, 0] = np.arange(14.0, 8.0, -1.0)
    volume[0, 0, 0, :3] = [10.0, 9.0, 8.0]
    volume[3, 0, 0, 3:] = [7.0, 6.0, 5.0]

    frame = sparsify(
        volume,
        doppler_velocity_mps=np.array(
            [-1.0, -0.05, 0.05, 1.0],
            dtype=np.float32,
        ),
        z_m=np.array([0.0], dtype=np.float32),
        y_m=np.array([0.0], dtype=np.float32),
        x_m=np.arange(6, dtype=np.float32),
        spec=SparsifySpec(
            min_snr_db=-100.0,
            max_points=4,
            spatial_peak_radius=0,
            noise_floor_scale=1.0,
            static_point_capacity_fraction=0.5,
            static_velocity_threshold_mps=0.051,
        ),
    )

    assert frame.num_points == 4
    assert np.count_nonzero(np.abs(frame.points[:, 3]) <= 0.051) <= 2


def test_sparsify_keeps_static_points_below_capacity() -> None:
    volume = np.zeros((3, 1, 1, 10), dtype=np.float32)
    volume[1, 0, 0, :8] = np.arange(20.0, 12.0, -1.0)
    volume[0, 0, 0, 8] = 12.0
    volume[2, 0, 0, 9] = 11.0

    frame = sparsify(
        volume,
        doppler_velocity_mps=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        z_m=np.array([0.0], dtype=np.float32),
        y_m=np.array([0.0], dtype=np.float32),
        x_m=np.arange(10, dtype=np.float32),
        spec=SparsifySpec(
            min_snr_db=-100.0,
            max_points=256,
            spatial_peak_radius=0,
            static_point_capacity_fraction=0.5,
            static_velocity_threshold_mps=0.1,
        ),
    )

    static = np.count_nonzero(np.abs(frame.points[:, 3]) <= 0.1)
    assert frame.num_points == 10
    assert static == 8


def test_sparsify_excludes_declared_boundary_margin() -> None:
    volume = np.zeros((1, 3, 3, 3), dtype=np.float32)
    volume[0, 0, 0, 0] = 10.0
    volume[0, 1, 1, 1] = 5.0

    frame = sparsify(
        volume,
        doppler_velocity_mps=np.array([0.0], dtype=np.float32),
        z_m=np.arange(3, dtype=np.float32),
        y_m=np.arange(3, dtype=np.float32),
        x_m=np.arange(3, dtype=np.float32),
        spec=SparsifySpec(
            min_snr_db=-100.0,
            boundary_margin_voxels=1,
        ),
    )

    assert frame.num_points == 1
    np.testing.assert_allclose(frame.points[0, :3], [1.0, 1.0, 1.0])
    extraction = frame.metadata["cartesian_volume_sparsification"]
    assert extraction["excluded_boundary_positive_voxels"] == 1


def test_sparsify_applies_explicit_spatial_mask() -> None:
    volume = np.zeros((1, 1, 1, 3), dtype=np.float32)
    volume[0, 0, 0] = [10.0, 5.0, 20.0]
    spatial_mask = np.zeros((1, 1, 3), dtype=bool)
    spatial_mask[0, 0, :2] = True

    frame = sparsify(
        volume,
        doppler_velocity_mps=np.array([0.0], dtype=np.float32),
        z_m=np.array([0.0], dtype=np.float32),
        y_m=np.array([0.0], dtype=np.float32),
        x_m=np.arange(3, dtype=np.float32),
        spatial_mask_zyx=spatial_mask,
        spec=SparsifySpec(
            min_snr_db=-100.0,
            spatial_peak_radius=0,
        ),
    )

    np.testing.assert_allclose(frame.points[:, 0], [0.0, 1.0])
    extraction = frame.metadata["cartesian_volume_sparsification"]
    assert extraction["spatial_mask_applied"] is True
    assert extraction["valid_spatial_voxels"] == 2


def test_sparsify_suppresses_equal_neighbor_plateau() -> None:
    volume = np.zeros((1, 3, 3, 4), dtype=np.float32)
    volume[0, 1, 1, 1:3] = 5.0

    frame = sparsify(
        volume,
        doppler_velocity_mps=np.array([0.0], dtype=np.float32),
        z_m=np.arange(3, dtype=np.float32),
        y_m=np.arange(3, dtype=np.float32),
        x_m=np.arange(4, dtype=np.float32),
        spec=SparsifySpec(min_snr_db=-100.0),
    )

    assert frame.num_points == 1
    assert frame.points[0, 0] == 1.0


def test_sparsify_falls_back_when_static_cap_removes_only_peak() -> None:
    volume = np.zeros((1, 1, 1, 2), dtype=np.float32)
    volume[0, 0, 0, 0] = 5.0

    frame = sparsify(
        volume,
        doppler_velocity_mps=np.array([0.0], dtype=np.float32),
        z_m=np.array([0.0], dtype=np.float32),
        y_m=np.array([0.0], dtype=np.float32),
        x_m=np.arange(2, dtype=np.float32),
        spec=SparsifySpec(
            min_snr_db=-100.0,
            max_points=1,
            spatial_peak_radius=0,
            static_point_capacity_fraction=0.5,
        ),
    )

    assert frame.num_points == 1
    extraction = frame.metadata["cartesian_volume_sparsification"]
    assert extraction["fallback_used"] is True
    assert extraction["limited_peak_voxels"] == 0
