from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import PlanarApertureLayout, RadarCube
from mmwcore.dsp import PlanarCartesianProjector


def _projector(
    *,
    grid_origin_xyz_m: tuple[float, float, float] = (1.0, 0.0, 0.0),
    target_velocity_mps: float = 0.0,
) -> PlanarCartesianProjector:
    return PlanarCartesianProjector(
        aperture_layout=PlanarApertureLayout(
            ((0, 0), (1, 0), (0, 1), (1, 1)),
            name="fixture",
        ),
        range_resolution_m=0.5,
        source_range_bins=4,
        source_doppler_bins=3,
        source_velocity_start_mps=-1.0,
        source_velocity_step_mps=1.0,
        target_doppler_bins=1,
        target_velocity_start_mps=target_velocity_mps,
        target_velocity_step_mps=1.0,
        grid_shape_zyx=(1, 1, 1),
        grid_origin_xyz_m=grid_origin_xyz_m,
        grid_voxel_size_xyz_m=(0.5, 0.5, 0.5),
        coordinate_frame="forward_lateral_up",
        azimuth_n_fft=4,
        elevation_n_fft=4,
    )


def test_planar_cartesian_projector_maps_broadside_target_to_metric_voxel() -> None:
    source = np.zeros((1, 3, 4, 4), dtype=np.complex64)
    source[0, 1, :, 2] = 1.0 + 0.0j

    projected = _projector().project(
        RadarCube(
            source,
            axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
            frame_id="frame-1",
        )
    )

    assert projected.magnitude_dzyx.dtype == np.float32
    assert projected.magnitude_dzyx.shape == (1, 1, 1, 1)
    assert projected.magnitude_dzyx[0, 0, 0, 0] == pytest.approx(4.0)
    np.testing.assert_allclose(projected.doppler_velocity_mps, [0.0])
    np.testing.assert_allclose(projected.x_m, [1.0])
    np.testing.assert_allclose(projected.y_m, [0.0])
    np.testing.assert_allclose(projected.z_m, [0.0])
    assert projected.coordinate_frame == "forward_lateral_up"
    assert projected.frame_id == "frame-1"
    projection = projected.metadata["planar_cartesian_projection"]
    assert projection["source_selection"] == {
        "doppler_start": 1,
        "doppler_stop": 2,
        "range_start": 2,
        "range_stop": 3,
    }
    assert projection["valid_spatial_voxel_fraction"] == pytest.approx(1.0)
    assert projection["valid_target_doppler_fraction"] == pytest.approx(1.0)


def test_planar_cartesian_projector_preserves_off_axis_direction_cosines() -> None:
    source = np.zeros((1, 3, 4, 4), dtype=np.complex64)
    aperture_indices = ((0, 0), (1, 0), (0, 1), (1, 1))
    source[0, 1, :, 2] = np.asarray(
        [
            np.exp(2j * np.pi * (azimuth + elevation) / 4.0)
            for azimuth, elevation in aperture_indices
        ],
        dtype=np.complex64,
    )
    projected = _projector(
        grid_origin_xyz_m=(np.sqrt(0.5), 0.5, 0.5),
    ).project(
        RadarCube(
            source,
            axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
        )
    )

    assert projected.magnitude_dzyx[0, 0, 0, 0] == pytest.approx(4.0, rel=1e-5)
    np.testing.assert_allclose(projected.x_m, [np.sqrt(0.5)])
    np.testing.assert_allclose(projected.y_m, [0.5])
    np.testing.assert_allclose(projected.z_m, [0.5])


def test_planar_cartesian_projector_interpolates_physical_doppler_magnitude() -> None:
    source = np.zeros((1, 3, 4, 4), dtype=np.complex64)
    source[0, 1, :, 2] = 1.0 + 0.0j
    source[0, 2, :, 2] = 3.0 + 0.0j

    projected = _projector(target_velocity_mps=0.5).project(
        RadarCube(
            source,
            axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
        )
    )

    assert projected.magnitude_dzyx[0, 0, 0, 0] == pytest.approx(8.0)
    np.testing.assert_allclose(projected.doppler_velocity_mps, [0.5])


def test_planar_cartesian_projector_rejects_real_source_contract() -> None:
    source = RadarCube(
        np.ones((1, 3, 4, 4), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    object.__setattr__(source, "data", np.ones(source.data.shape, dtype=np.float32))

    with pytest.raises(TypeError, match="complex antenna samples"):
        _projector().project(source)
