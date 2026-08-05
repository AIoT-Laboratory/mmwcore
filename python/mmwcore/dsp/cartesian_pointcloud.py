"""Dense Cartesian radar-volume sparsification."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmwcore.core import CartesianVolumeSparsificationSpec, PointCloudFrame

from ._cartesian_pointcloud import sparsify_cartesian_volume as native_sparsify_cartesian_volume

_POINT_CHANNELS = ("x", "y", "z", "velocity", "snr_db")
_SPARSIFICATION_SCHEMA = "mmwcore.cartesian_volume_sparsification.v4"


def sparsify_cartesian_volume(
    magnitude_dzyx: np.ndarray,
    *,
    doppler_velocity_mps: np.ndarray,
    z_m: np.ndarray,
    y_m: np.ndarray,
    x_m: np.ndarray,
    spatial_mask_zyx: np.ndarray | None = None,
    suppressed_doppler_index: int | None = None,
    spec: CartesianVolumeSparsificationSpec | None = None,
    frame_id: str | int | None = None,
    timestamp: float | None = None,
    source: str | None = None,
    coordinate_frame: str = "radar",
    metadata: dict[str, Any] | None = None,
) -> PointCloudFrame:
    """Extract deterministic spatial peaks with signed radial velocity and SNR."""

    policy = spec or CartesianVolumeSparsificationSpec()
    source_volume, axes = _validate_inputs(
        magnitude_dzyx,
        doppler_velocity_mps=doppler_velocity_mps,
        z_m=z_m,
        y_m=y_m,
        x_m=x_m,
    )
    spatial_mask = _spatial_mask(spatial_mask_zyx, shape_zyx=source_volume.shape[1:])
    points, noise_floors, counts, status = native_sparsify_cartesian_volume(
        source_volume,
        doppler_velocity_mps=axes[0],
        z_m=axes[1],
        y_m=axes[2],
        x_m=axes[3],
        spatial_mask_zyx=spatial_mask,
        suppressed_doppler_index=suppressed_doppler_index,
        spec=policy,
    )
    noise_floor_min, noise_floor_median, noise_floor_max = noise_floors
    (
        valid_spatial_voxels,
        positive_volume_voxels,
        valid_positive_volume_voxels,
        local_peak_voxels,
        doppler_peak_voxels,
        threshold_peak_voxels,
        limited_peak_voxels,
    ) = counts
    fallback_used, static_output_points = status
    extraction = {
        "schema": _SPARSIFICATION_SCHEMA,
        "input_axes": ["doppler", "z", "y", "x"],
        "input_shape": list(source_volume.shape),
        "min_snr_db": policy.min_snr_db,
        "max_points": policy.max_points,
        "spatial_peak_radius": policy.spatial_peak_radius,
        "doppler_peak_radius": policy.doppler_peak_radius,
        "max_doppler_peaks_per_spatial": policy.max_doppler_peaks_per_spatial,
        "boundary_margin_voxels": policy.boundary_margin_voxels,
        "spatial_mask_applied": spatial_mask is not None,
        "suppressed_doppler_index": suppressed_doppler_index,
        "valid_spatial_voxels": valid_spatial_voxels,
        "noise_estimator": "per_doppler_valid_positive_geometric_mean",
        "noise_floor_scale": policy.noise_floor_scale,
        "noise_floor_min": noise_floor_min,
        "noise_floor_median": noise_floor_median,
        "noise_floor_max": noise_floor_max,
        "doppler_selection": "doppler_and_spatial_local_peaks",
        "static_point_capacity_fraction": policy.static_point_capacity_fraction,
        "static_velocity_threshold_mps": policy.static_velocity_threshold_mps,
        "static_doppler_definition": "absolute_velocity_threshold",
        "peak_tie_policy": "lexicographic_local_plateau_suppression",
        "strongest_point_fallback": policy.strongest_point_fallback,
        "fallback_used": fallback_used,
        "positive_volume_voxels": positive_volume_voxels,
        "valid_positive_volume_voxels": valid_positive_volume_voxels,
        "excluded_boundary_positive_voxels": (
            positive_volume_voxels - valid_positive_volume_voxels
        ),
        "local_peak_voxels": local_peak_voxels,
        "doppler_peak_voxels": doppler_peak_voxels,
        "threshold_peak_voxels": threshold_peak_voxels,
        "limited_peak_voxels": limited_peak_voxels,
        "output_points": int(points.shape[0]),
        "static_output_points": static_output_points,
        "dynamic_output_points": int(points.shape[0]) - static_output_points,
    }
    return PointCloudFrame(
        points,
        channels=_POINT_CHANNELS,
        frame_id=frame_id,
        timestamp=timestamp,
        source=source,
        coordinate_frame=coordinate_frame,
        units={
            "x": "m",
            "y": "m",
            "z": "m",
            "velocity": "m/s",
            "snr_db": "dB",
        },
        metadata={**(metadata or {}), "cartesian_volume_sparsification": extraction},
    )


def _validate_inputs(
    magnitude_dzyx: np.ndarray,
    *,
    doppler_velocity_mps: np.ndarray,
    z_m: np.ndarray,
    y_m: np.ndarray,
    x_m: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    source = np.asarray(magnitude_dzyx)
    if source.ndim != 4:
        raise ValueError(
            f"Cartesian magnitude volume must have shape (D, Z, Y, X); got {source.shape}."
        )
    if np.iscomplexobj(source):
        raise ValueError("Cartesian magnitude volume must be real-valued.")
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError("Cartesian magnitude volume must be numeric.")
    if not np.isfinite(source).all():
        raise ValueError("Cartesian magnitude volume contains NaN or Inf values.")
    if np.any(source < 0):
        raise ValueError("Cartesian magnitude volume must be non-negative.")
    source = np.asarray(source, dtype=np.float32)
    doppler_velocity, z_axis, y_axis, x_axis = tuple(
        _physical_axis(name, value, expected_size=size)
        for name, value, size in zip(
            ("doppler_velocity_mps", "z_m", "y_m", "x_m"),
            (doppler_velocity_mps, z_m, y_m, x_m),
            source.shape,
            strict=True,
        )
    )
    return source, (doppler_velocity, z_axis, y_axis, x_axis)


def _physical_axis(name: str, value: np.ndarray, *, expected_size: int) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float32)
    if axis.ndim != 1 or axis.size != expected_size:
        raise ValueError(f"{name} must have shape ({expected_size},).")
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    if axis.size > 1 and np.any(np.diff(axis) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return axis


def _spatial_mask(
    spatial_mask_zyx: np.ndarray | None,
    *,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray | None:
    if spatial_mask_zyx is None:
        return None
    spatial_mask = np.asarray(spatial_mask_zyx)
    if spatial_mask.dtype != np.bool_ or spatial_mask.shape != shape_zyx:
        raise ValueError(
            "Cartesian spatial_mask_zyx must be boolean with shape "
            f"{shape_zyx}; got {spatial_mask.shape} and {spatial_mask.dtype}."
        )
    return spatial_mask


__all__ = ["sparsify_cartesian_volume"]
