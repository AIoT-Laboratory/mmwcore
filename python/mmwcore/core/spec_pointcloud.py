"""Point-cloud extraction specifications."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from operator import index as integer_index
from sys import maxsize
from typing import SupportsFloat, SupportsIndex, cast


@dataclass(frozen=True)
class SparsifySpec:
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
        object.__setattr__(
            self,
            "min_snr_db",
            _finite_real(self.min_snr_db, name="min_snr_db"),
        )
        object.__setattr__(
            self,
            "max_points",
            _positive_integer(self.max_points, name="max_points"),
        )
        object.__setattr__(
            self,
            "spatial_peak_radius",
            _non_negative_integer(self.spatial_peak_radius, name="spatial_peak_radius"),
        )
        object.__setattr__(
            self,
            "doppler_peak_radius",
            _non_negative_integer(self.doppler_peak_radius, name="doppler_peak_radius"),
        )
        if self.max_doppler_peaks_per_spatial is not None:
            object.__setattr__(
                self,
                "max_doppler_peaks_per_spatial",
                _positive_integer(
                    self.max_doppler_peaks_per_spatial,
                    name="max_doppler_peaks_per_spatial",
                ),
            )
        object.__setattr__(
            self,
            "boundary_margin_voxels",
            _non_negative_integer(self.boundary_margin_voxels, name="boundary_margin_voxels"),
        )
        object.__setattr__(
            self,
            "noise_floor_scale",
            _positive_real(self.noise_floor_scale, name="noise_floor_scale"),
        )
        object.__setattr__(
            self,
            "static_point_capacity_fraction",
            _unit_fraction(
                self.static_point_capacity_fraction,
                name="static_point_capacity_fraction",
            ),
        )
        object.__setattr__(
            self,
            "static_velocity_threshold_mps",
            _non_negative_real(
                self.static_velocity_threshold_mps,
                name="static_velocity_threshold_mps",
            ),
        )
        if type(self.strongest_point_fallback) is not bool:
            raise TypeError("Cartesian volume strongest_point_fallback must be a bool.")


def _platform_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Cartesian volume {name} must be an integer.")
    try:
        normalized = int(integer_index(cast(SupportsIndex, value)))
    except TypeError as exc:
        raise TypeError(f"Cartesian volume {name} must be an integer.") from exc
    if normalized > maxsize:
        raise OverflowError(f"Cartesian volume {name} must fit the platform index range.")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    normalized = _platform_integer(value, name=name)
    if normalized <= 0:
        raise ValueError(f"Cartesian volume {name} must be positive.")
    return normalized


def _non_negative_integer(value: object, *, name: str) -> int:
    normalized = _platform_integer(value, name=name)
    if normalized < 0:
        raise ValueError(f"Cartesian volume {name} must be non-negative.")
    return normalized


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Cartesian volume {name} must be a real number.")
    normalized = float(cast(SupportsFloat, value))
    if not isfinite(normalized):
        raise ValueError(f"Cartesian volume {name} must be finite.")
    return normalized


def _positive_real(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if normalized <= 0:
        raise ValueError(f"Cartesian volume {name} must be finite and positive.")
    return normalized


def _non_negative_real(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if normalized < 0:
        raise ValueError(f"Cartesian volume {name} must be finite and non-negative.")
    return normalized


def _unit_fraction(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if not 0 < normalized <= 1:
        raise ValueError(f"Cartesian volume {name} must be within (0, 1].")
    return normalized


__all__ = ["SparsifySpec"]
