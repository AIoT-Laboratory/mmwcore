"""Clustering and tracking specifications."""

from __future__ import annotations

from dataclasses import dataclass

from mmwcore._compat import StrEnum


class TrackStatus(StrEnum):
    """Lifecycle state for one target track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"


@dataclass(frozen=True)
class DBSCANClusteringSpec:
    """DBSCAN policy for Cartesian radar points and optional radial velocity."""

    eps_m: float
    min_samples: int
    velocity_scale_s: float = 0.0
    use_z: bool = True

    def __post_init__(self) -> None:
        if self.eps_m <= 0:
            raise ValueError("DBSCANClusteringSpec.eps_m must be positive.")
        if self.min_samples <= 0:
            raise ValueError("DBSCANClusteringSpec.min_samples must be positive.")
        if self.velocity_scale_s < 0:
            raise ValueError("DBSCANClusteringSpec.velocity_scale_s must be non-negative.")


@dataclass(frozen=True)
class TrackGatingSpec:
    """Hard limits used before probabilistic track association."""

    max_distance_m: float
    max_radial_velocity_difference_mps: float | None = None
    max_mahalanobis_distance: float | None = None

    def __post_init__(self) -> None:
        if self.max_distance_m <= 0:
            raise ValueError("TrackGatingSpec.max_distance_m must be positive.")
        if (
            self.max_radial_velocity_difference_mps is not None
            and self.max_radial_velocity_difference_mps <= 0
        ):
            raise ValueError("TrackGatingSpec.max_radial_velocity_difference_mps must be positive.")
        if self.max_mahalanobis_distance is not None and self.max_mahalanobis_distance <= 0:
            raise ValueError("TrackGatingSpec.max_mahalanobis_distance must be positive.")


@dataclass(frozen=True)
class TrackAllocationSpec:
    """Minimum cluster evidence required to allocate a tentative track."""

    min_points: int = 1
    min_abs_radial_velocity_mps: float = 0.0
    min_total_snr: float | None = None
    max_new_tracks_per_frame: int | None = None

    def __post_init__(self) -> None:
        if self.min_points <= 0:
            raise ValueError("TrackAllocationSpec.min_points must be positive.")
        if self.min_abs_radial_velocity_mps < 0:
            raise ValueError(
                "TrackAllocationSpec.min_abs_radial_velocity_mps must be non-negative."
            )
        if self.min_total_snr is not None and self.min_total_snr <= 0:
            raise ValueError("TrackAllocationSpec.min_total_snr must be positive.")
        if self.max_new_tracks_per_frame is not None and self.max_new_tracks_per_frame <= 0:
            raise ValueError("TrackAllocationSpec.max_new_tracks_per_frame must be positive.")


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
            if value <= 0:
                raise ValueError(f"TrackLifecycleSpec.{name} must be positive.")


@dataclass(frozen=True)
class TrackingBox2D:
    """Inclusive Cartesian tracking region in radar x/y coordinates."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    def __post_init__(self) -> None:
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
        if self.outside_max_frames <= 0:
            raise ValueError("TrackScenerySpec.outside_max_frames must be positive.")
        object.__setattr__(self, "boundary_boxes", boxes)

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
        if self.frame_period_s <= 0:
            raise ValueError("Tracker2DSpec.frame_period_s must be positive.")
        if self.max_tracks <= 0:
            raise ValueError("Tracker2DSpec.max_tracks must be positive.")
        if self.measurement_noise_m <= 0:
            raise ValueError("Tracker2DSpec.measurement_noise_m must be positive.")
        if self.initial_velocity_std_mps <= 0:
            raise ValueError("Tracker2DSpec.initial_velocity_std_mps must be positive.")
        if not 0 < self.extent_covariance_smoothing <= 1:
            raise ValueError("Tracker2DSpec.extent_covariance_smoothing must be in (0, 1].")
        acceleration = tuple(float(value) for value in self.max_acceleration_mps2)
        if len(acceleration) != 2 or any(value <= 0 for value in acceleration):
            raise ValueError(
                "Tracker2DSpec.max_acceleration_mps2 must contain two positive values."
            )
        object.__setattr__(self, "max_acceleration_mps2", acceleration)
