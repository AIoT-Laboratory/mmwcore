"""Native Cartesian radar-volume projection boundary."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native

type NativePlanarCartesianConfig = tuple[
    float,
    int,
    float,
    float,
    int,
    float,
    float,
    tuple[int, int, int],
    tuple[float, float, float],
    tuple[float, float, float],
    int,
    int,
    float,
]
type NativePlanarCartesianResult = tuple[
    NDArray[np.float32],
    int,
    int,
    int,
    int,
    int,
    int,
]


class NativePlanarCartesianProjector:
    """Own one native fixed-geometry projection plan."""

    def __init__(
        self,
        *,
        source_range_bins: int,
        grid_indices: tuple[tuple[int, int], ...],
        config: NativePlanarCartesianConfig,
    ) -> None:
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
            azimuth_n_fft,
            elevation_n_fft,
            aperture_spacing_wavelengths,
        ) = config
        self._projector = _native.NativePlanarCartesianProjector(
            source_range_bins,
            grid_indices,
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
                (azimuth_n_fft, elevation_n_fft, aperture_spacing_wavelengths),
            ),
        )

    def project(self, data: NDArray[np.complex64]) -> NativePlanarCartesianResult:
        return self._projector.project(np.ascontiguousarray(data, dtype=np.complex64))
