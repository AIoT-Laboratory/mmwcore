"""Clustering and tracking specifications."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from numbers import Real
from operator import index as integer_index
from sys import maxsize as _MAX_PLATFORM_INDEX
from typing import SupportsFloat, cast


class TrackStatus(StrEnum):
    """Lifecycle state for one target track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        normalized = int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if not -_MAX_PLATFORM_INDEX - 1 <= normalized <= _MAX_PLATFORM_INDEX:
        raise OverflowError(f"{name} must fit the platform index range.")
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _require_finite_positive(value: float, *, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _require_finite_non_negative(value: float, *, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _require_finite_unit_smoothing(value: float, *, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be finite and in (0, 1].")


def _positive_acceleration(value: object) -> float:
    name = "Tracker2DSpec.max_acceleration_mps2"
    if isinstance(value, bool):
        raise TypeError(f"{name} values must be real numbers, not bool.")
    if not isinstance(value, Real):
        raise TypeError(f"{name} values must be real numbers.")
    normalized = float(cast(SupportsFloat, value))
    if not isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must contain two finite positive values.")
    return normalized


@dataclass(frozen=True)
class DBSCANClusteringSpec:
    """DBSCAN policy for Cartesian radar points and optional radial velocity."""

    eps_m: float
    min_samples: int
    velocity_scale_s: float = 0.0
    use_z: bool = True

    def __post_init__(self) -> None:
        _require_finite_positive(self.eps_m, name="DBSCANClusteringSpec.eps_m")
        object.__setattr__(
            self,
            "min_samples",
            _positive_integer(self.min_samples, name="DBSCANClusteringSpec.min_samples"),
        )
        _require_finite_non_negative(
            self.velocity_scale_s,
            name="DBSCANClusteringSpec.velocity_scale_s",
        )
        if type(self.use_z) is not bool:
            raise TypeError("DBSCANClusteringSpec.use_z must be a bool.")


@dataclass(frozen=True)
class TrackGatingSpec:
    """Hard limits used before probabilistic track association."""

    max_distance_m: float
    max_radial_velocity_difference_mps: float | None = None
    max_mahalanobis_distance: float | None = None

    def __post_init__(self) -> None:
        _require_finite_positive(self.max_distance_m, name="TrackGatingSpec.max_distance_m")
        if self.max_radial_velocity_difference_mps is not None:
            _require_finite_positive(
                self.max_radial_velocity_difference_mps,
                name="TrackGatingSpec.max_radial_velocity_difference_mps",
            )
        if self.max_mahalanobis_distance is not None:
            _require_finite_positive(
                self.max_mahalanobis_distance,
                name="TrackGatingSpec.max_mahalanobis_distance",
            )


@dataclass(frozen=True)
class TrackAllocationSpec:
    """Minimum cluster support required to allocate a tentative track."""

    min_points: int = 1
    min_abs_radial_velocity_mps: float = 0.0
    min_total_snr: float | None = None
    max_new_tracks_per_frame: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_points",
            _positive_integer(self.min_points, name="TrackAllocationSpec.min_points"),
        )
        _require_finite_non_negative(
            self.min_abs_radial_velocity_mps,
            name="TrackAllocationSpec.min_abs_radial_velocity_mps",
        )
        if self.min_total_snr is not None:
            _require_finite_positive(
                self.min_total_snr,
                name="TrackAllocationSpec.min_total_snr",
            )
        if self.max_new_tracks_per_frame is not None:
            object.__setattr__(
                self,
                "max_new_tracks_per_frame",
                _positive_integer(
                    self.max_new_tracks_per_frame,
                    name="TrackAllocationSpec.max_new_tracks_per_frame",
                ),
            )


@dataclass(frozen=True)
class TrackLifecycleSpec:
    """Explicit hit/miss counts for track confirmation and deletion."""

    confirmation_hits: int = 4
    tentative_max_misses: int = 4
    confirmed_max_misses: int = 11

    def __post_init__(self) -> None:
        for name, value in (
            ("confirmation_hits", self.confirmation_hits),
            ("tentative_max_misses", self.tentative_max_misses),
            ("confirmed_max_misses", self.confirmed_max_misses),
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(value, name=f"TrackLifecycleSpec.{name}"),
            )


@dataclass(frozen=True)
class TrackingBox2D:
    """Inclusive Cartesian tracking region in radar x/y coordinates."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x_min_m", self.x_min_m),
            ("x_max_m", self.x_max_m),
            ("y_min_m", self.y_min_m),
            ("y_max_m", self.y_max_m),
        ):
            if isinstance(value, bool):
                raise TypeError(f"TrackingBox2D.{name} must be a real number, not bool.")
            if not isfinite(value):
                raise ValueError(f"TrackingBox2D.{name} must be finite.")
        if self.x_min_m >= self.x_max_m or self.y_min_m >= self.y_max_m:
            raise ValueError("TrackingBox2D minimum bounds must be below maximum bounds.")

    def contains(self, x_m: float, y_m: float) -> bool:
        return self.x_min_m <= x_m <= self.x_max_m and self.y_min_m <= y_m <= self.y_max_m


@dataclass(frozen=True)
class TrackScenerySpec:
    """Scene regions that constrain association, allocation, and track lifetime."""

    boundary_boxes: tuple[TrackingBox2D, ...] = ()
    outside_max_frames: int = 5

    def __post_init__(self) -> None:
        boxes = tuple(self.boundary_boxes)
        outside_max_frames = _positive_integer(
            self.outside_max_frames,
            name="TrackScenerySpec.outside_max_frames",
        )
        object.__setattr__(self, "boundary_boxes", boxes)
        object.__setattr__(self, "outside_max_frames", outside_max_frames)

    def contains(self, x_m: float, y_m: float) -> bool:
        return not self.boundary_boxes or any(box.contains(x_m, y_m) for box in self.boundary_boxes)


@dataclass(frozen=True)
class Tracker2DSpec:
    """Configuration for stateful two-dimensional target tracking."""

    frame_period_s: float
    gating: TrackGatingSpec
    allocation: TrackAllocationSpec = TrackAllocationSpec()
    lifecycle: TrackLifecycleSpec = TrackLifecycleSpec()
    scenery: TrackScenerySpec = TrackScenerySpec()
    max_tracks: int = 200
    max_acceleration_mps2: tuple[float, float] = (2.0, 2.0)
    measurement_noise_m: float = 0.2
    initial_velocity_std_mps: float = 2.0
    extent_covariance_smoothing: float = 0.2

    def __post_init__(self) -> None:
        _require_finite_positive(self.frame_period_s, name="Tracker2DSpec.frame_period_s")
        object.__setattr__(
            self,
            "max_tracks",
            _positive_integer(self.max_tracks, name="Tracker2DSpec.max_tracks"),
        )
        _require_finite_positive(
            self.measurement_noise_m,
            name="Tracker2DSpec.measurement_noise_m",
        )
        _require_finite_positive(
            self.initial_velocity_std_mps,
            name="Tracker2DSpec.initial_velocity_std_mps",
        )
        _require_finite_unit_smoothing(
            self.extent_covariance_smoothing,
            name="Tracker2DSpec.extent_covariance_smoothing",
        )
        raw_acceleration = tuple(self.max_acceleration_mps2)
        if len(raw_acceleration) != 2:
            raise ValueError(
                "Tracker2DSpec.max_acceleration_mps2 must contain two finite positive values."
            )
        acceleration = tuple(_positive_acceleration(value) for value in raw_acceleration)
        object.__setattr__(self, "max_acceleration_mps2", acceleration)
