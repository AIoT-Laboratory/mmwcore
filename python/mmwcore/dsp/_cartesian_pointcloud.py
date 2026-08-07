"""Native Cartesian RPC sparsification boundary."""

# ruff: noqa: UP040

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import CartesianVolumeSparsificationSpec

NativeCartesianSparsificationResult: TypeAlias = tuple[
    NDArray[np.float32],
    tuple[float, float, float],
    tuple[int, int, int, int, int, int, int],
    tuple[bool, int],
]


def sparsify_cartesian_volume(
    magnitude_dzyx: NDArray[np.float32],
    *,
    doppler_velocity_mps: NDArray[np.float32],
    z_m: NDArray[np.float32],
    y_m: NDArray[np.float32],
    x_m: NDArray[np.float32],
    spatial_mask_zyx: NDArray[np.bool_] | None,
    suppressed_doppler_index: int | None,
    spec: CartesianVolumeSparsificationSpec,
) -> NativeCartesianSparsificationResult:
    """Extract deterministic Cartesian RPC points through one native call."""

    return _native.sparsify_cartesian_volume(
        np.ascontiguousarray(magnitude_dzyx, dtype=np.float32),
        (
            np.ascontiguousarray(doppler_velocity_mps, dtype=np.float32),
            np.ascontiguousarray(z_m, dtype=np.float32),
            np.ascontiguousarray(y_m, dtype=np.float32),
            np.ascontiguousarray(x_m, dtype=np.float32),
        ),
        (
            None
            if spatial_mask_zyx is None
            else np.ascontiguousarray(spatial_mask_zyx, dtype=np.bool_)
        ),
        suppressed_doppler_index,
        (
            (spec.min_snr_db, spec.max_points),
            (
                spec.spatial_peak_radius,
                spec.doppler_peak_radius,
                spec.max_doppler_peaks_per_spatial,
                spec.boundary_margin_voxels,
            ),
            (
                spec.noise_floor_scale,
                spec.static_point_capacity_fraction,
                spec.static_velocity_threshold_mps,
                spec.strongest_point_fallback,
            ),
        ),
    )
