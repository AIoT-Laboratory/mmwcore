from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native
from mmwcore.dsp._cartesian import NativePlanarCartesianConfig, NativePlanarCartesianResult

_APERTURE = ((0, 0), (1, 0), (0, 1), (1, 1))


def _config(
    *,
    target_velocity_start_mps: float = 0.0,
    grid_origin_xyz_m: tuple[float, float, float] = (1.0, 0.0, 1.0),
    mount_pitch_deg: float = 0.0,
) -> NativePlanarCartesianConfig:
    return (
        0.5,
        3,
        -1.0,
        1.0,
        1,
        target_velocity_start_mps,
        1.0,
        (1, 1, 1),
        grid_origin_xyz_m,
        (0.5, 0.5, 0.5),
        1.0,
        mount_pitch_deg,
        4,
        4,
        0.5,
    )


def _broadside_source() -> np.ndarray:
    source = np.zeros((1, 3, 4, 4), dtype=np.complex64)
    source[0, 1, :, 2] = 1.0 + 0.0j
    return source


def _project(
    source: np.ndarray,
    config: NativePlanarCartesianConfig,
) -> NativePlanarCartesianResult:
    (
        range_resolution_m,
        source_doppler_bins,
        source_velocity_start_mps,
        source_velocity_step_mps,
        target_doppler_bins,
        target_velocity_start_mps,
        target_velocity_step_mps,
        grid_shape_zyx,
        grid_origin_xyz_m,
        grid_voxel_size_xyz_m,
        mount_height_m,
        mount_pitch_deg,
        azimuth_n_fft,
        elevation_n_fft,
        aperture_spacing_wavelengths,
    ) = config
    projector = _native.NativeCartesianProjector(
        source.shape[-1],
        _APERTURE,
        (
            range_resolution_m,
            (
                source_doppler_bins,
                source_velocity_start_mps,
                source_velocity_step_mps,
            ),
            (
                target_doppler_bins,
                target_velocity_start_mps,
                target_velocity_step_mps,
            ),
            grid_shape_zyx,
            grid_origin_xyz_m,
            grid_voxel_size_xyz_m,
            (mount_height_m, mount_pitch_deg),
            (azimuth_n_fft, elevation_n_fft, aperture_spacing_wavelengths),
        ),
    )
    return projector.project(source)


def test_native_cartesian_projects_broadside_target_with_selection_diagnostics() -> None:
    (
        magnitude,
        doppler_start,
        doppler_stop,
        range_start,
        range_stop,
        spatial_count,
        doppler_count,
    ) = _project(_broadside_source(), _config())

    assert magnitude.dtype == np.float32
    assert magnitude.shape == (1, 1, 1, 1)
    assert magnitude[0, 0, 0, 0] == pytest.approx(4.0)
    assert (doppler_start, doppler_stop, range_start, range_stop) == (1, 2, 2, 3)
    assert (spatial_count, doppler_count) == (1, 1)


def test_native_cartesian_interpolates_magnitude_on_physical_doppler_axis() -> None:
    source = _broadside_source()
    source[0, 2, :, 2] = 3.0 + 0.0j

    magnitude, *_ = _project(source, _config(target_velocity_start_mps=0.5))

    assert magnitude[0, 0, 0, 0] == pytest.approx(8.0)


def test_native_cartesian_rejects_noncontiguous_input_and_unsupported_grid() -> None:
    source = _broadside_source()

    with pytest.raises(ValueError, match="contiguous"):
        _project(source[:, :, :, ::-1], _config())
    with pytest.raises(ValueError, match="source radar field"):
        _project(
            source,
            _config(grid_origin_xyz_m=(-1.0, 0.0, 1.0)),
        )


def test_native_cartesian_plan_matches_locked_non_axis_reference() -> None:
    rng = np.random.default_rng(2026)
    source = (rng.standard_normal((1, 7, 4, 9)) + 1j * rng.standard_normal((1, 7, 4, 9))).astype(
        np.complex64
    )
    projector = _native.NativeCartesianProjector(
        9,
        _APERTURE,
        (
            0.4,
            (7, -1.5, 0.5),
            (5, -1.25, 0.5),
            (2, 3, 4),
            (0.6, -0.4, 0.8),
            (0.35, 0.3, 0.25),
            (1.0, 0.0),
            (8, 8, 0.5),
        ),
    )

    magnitude, *diagnostics = projector.project(source)

    assert diagnostics == [0, 6, 1, 6, 24, 5]
    sample_indices = np.asarray(
        [0, 1, 5, 11, 23, 24, 37, 48, 59, 71, 83, 95, 107, 119], dtype=np.intp
    )
    np.testing.assert_allclose(
        magnitude.ravel()[sample_indices],
        [
            1.4474211,
            0.93432105,
            1.1210971,
            2.355453,
            2.4334483,
            1.6464275,
            0.94605887,
            2.8977854,
            2.427685,
            2.8213189,
            2.8293383,
            3.4538374,
            2.198183,
            2.0203872,
        ],
        rtol=2e-6,
        atol=2e-6,
    )
    assert float(magnitude.sum()) == pytest.approx(257.60046, rel=2e-6)
