"""Point-cloud extraction specifications."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CartesianVolumeSparsificationSpec:
    """Deterministic sparsification of a dense Doppler-Cartesian magnitude volume."""

    min_snr_db: float = 0.0
    max_points: int = 256
    spatial_peak_radius: int = 1
    doppler_peak_radius: int = 0
    max_doppler_peaks_per_spatial: int | None = None
    boundary_margin_voxels: int = 0
    noise_floor_scale: float = 1.0
    static_point_capacity_fraction: float = 1.0
    static_velocity_threshold_mps: float = 0.0
    strongest_point_fallback: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_snr_db):
            raise ValueError("Cartesian volume min_snr_db must be finite.")
        if self.max_points <= 0:
            raise ValueError("Cartesian volume max_points must be positive.")
        if self.spatial_peak_radius < 0:
            raise ValueError("Cartesian volume spatial_peak_radius must be non-negative.")
        if self.doppler_peak_radius < 0:
            raise ValueError("Cartesian volume doppler_peak_radius must be non-negative.")
        if (
            self.max_doppler_peaks_per_spatial is not None
            and self.max_doppler_peaks_per_spatial <= 0
        ):
            raise ValueError(
                "Cartesian volume max_doppler_peaks_per_spatial must be positive when provided."
            )
        if self.boundary_margin_voxels < 0:
            raise ValueError("Cartesian volume boundary_margin_voxels must be non-negative.")
        if not math.isfinite(self.noise_floor_scale) or self.noise_floor_scale <= 0:
            raise ValueError("Cartesian volume noise_floor_scale must be finite and positive.")
        if (
            not math.isfinite(self.static_point_capacity_fraction)
            or not 0 < self.static_point_capacity_fraction <= 1
        ):
            raise ValueError(
                "Cartesian volume static_point_capacity_fraction must be within (0, 1]."
            )
        if (
            not math.isfinite(self.static_velocity_threshold_mps)
            or self.static_velocity_threshold_mps < 0
        ):
            raise ValueError(
                "Cartesian volume static_velocity_threshold_mps must be finite and non-negative."
            )


__all__ = ["CartesianVolumeSparsificationSpec"]
