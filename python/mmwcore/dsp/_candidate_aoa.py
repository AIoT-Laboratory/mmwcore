"""Native candidate-level angle-of-arrival boundaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import AngleFFTSpec, VirtualAntennaLayout, VirtualSubarraySpec
from mmwcore.dsp._fft import _window_code

type NativeCandidateAzimuthResult = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
]
type NativeCandidateElevationResult = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    tuple[float, float],
]

_ANGLE_AXIS_CODES = {"azimuth": 0, "elevation": 1}


def candidate_azimuth_peaks(
    cube: NDArray[np.complex64],
    *,
    cube_axes: tuple[int, int, int, int],
    candidates: NDArray[np.float32],
    candidate_columns: tuple[int, int, int],
    layout: VirtualAntennaLayout,
    spec: AngleFFTSpec,
) -> NativeCandidateAzimuthResult:
    """Recover calibrated peak azimuths for an entire candidate batch."""

    return _native.candidate_azimuth_peaks(
        np.ascontiguousarray(cube, dtype=np.complex64),
        cube_axes,
        np.ascontiguousarray(candidates, dtype=np.float32),
        candidate_columns,
        np.ascontiguousarray(layout.positions_wavelengths, dtype=np.float32),
        (
            spec.n_fft or layout.num_antennas,
            _window_code(spec.window),
            spec.fftshift,
            _ANGLE_AXIS_CODES[layout.angle_axis],
        ),
    )


def candidate_elevations(
    cube: NDArray[np.complex64],
    *,
    cube_axes: tuple[int, int, int, int],
    candidates: NDArray[np.float32],
    candidate_columns: tuple[int, int, int, int, int],
    azimuth_subarray: VirtualSubarraySpec,
    elevation_subarray: VirtualSubarraySpec,
    spec: AngleFFTSpec,
) -> NativeCandidateElevationResult:
    """Recover paired-row elevations for an entire candidate batch."""

    return _native.candidate_elevations(
        np.ascontiguousarray(cube, dtype=np.complex64),
        cube_axes,
        np.ascontiguousarray(candidates, dtype=np.float32),
        candidate_columns,
        (
            list(azimuth_subarray.antenna_indices),
            list(elevation_subarray.antenna_indices),
            np.ascontiguousarray(
                azimuth_subarray.layout.positions_wavelengths,
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                elevation_subarray.layout.positions_wavelengths,
                dtype=np.float64,
            ),
        ),
        (
            spec.n_fft or azimuth_subarray.layout.num_antennas,
            _window_code(spec.window),
            spec.fftshift,
        ),
    )
