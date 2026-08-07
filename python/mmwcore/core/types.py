"""Small data contracts for the mmwcore radar data-link layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spec_tracking import TrackStatus

_XYZ_CHANNELS = ("x", "y", "z")


def _metadata_copy(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


@dataclass(frozen=True)
class RawADCFrame:
    """Raw ADC frame values captured from hardware or loaded from an ADC file."""

    samples: np.ndarray
    frame_id: str | int | None = None
    timestamp: float | None = None
    source: str | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = self.samples if isinstance(self.samples, np.memmap) else np.asarray(self.samples)
        if samples.ndim != 1:
            raise ValueError(f"RawADCFrame.samples must be one-dimensional; got {samples.shape}.")
        if samples.size == 0:
            raise ValueError("RawADCFrame.samples must not be empty.")
        if samples.dtype != np.int16:
            if not np.issubdtype(samples.dtype, np.integer):
                raise TypeError(
                    "RawADCFrame.samples must contain integer ADC values; "
                    f"got dtype {samples.dtype}."
                )
            limits = np.iinfo(np.int16)
            if np.any(samples < limits.min) or np.any(samples > limits.max):
                raise ValueError("RawADCFrame.samples contains values outside the int16 range.")
            samples = samples.astype(np.int16)

        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "profile", _metadata_copy(self.profile))
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))


@dataclass(frozen=True)
class RadarCube:
    """Complex radar cube with explicit axis names."""

    data: np.ndarray
    axes: tuple[str, ...] = ("frame", "chirp", "rx", "sample")
    frame_id: str | int | None = None
    timestamp: float | None = None
    source: str | None = None
    units: str = "adc"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        axes = tuple(self.axes)
        if not np.iscomplexobj(data):
            data = data.astype(np.complex64)
        if data.ndim != len(axes):
            raise ValueError(
                "RadarCube.data dimensions must match axes; "
                f"got shape {data.shape} and axes {axes}."
            )
        if data.size == 0:
            raise ValueError("RadarCube.data must not be empty.")
        if any(not isinstance(axis, str) or not axis.strip() for axis in axes):
            raise ValueError("RadarCube.axes must contain non-empty string names.")
        if len(set(axes)) != len(axes):
            raise ValueError(f"RadarCube.axes must be unique; got {axes}.")

        object.__setattr__(self, "data", data.astype(np.complex64, copy=False))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))


@dataclass(frozen=True)
class CartesianRadarVolume:
    """Non-negative radar magnitude on physical Doppler and Cartesian axes."""

    magnitude_dzyx: np.ndarray
    doppler_velocity_mps: np.ndarray
    z_m: np.ndarray
    y_m: np.ndarray
    x_m: np.ndarray
    frame_id: str | int | None = None
    timestamp: float | None = None
    source: str | None = None
    coordinate_frame: str = "radar"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        magnitude = np.asarray(self.magnitude_dzyx, dtype=np.float32)
        axes = tuple(
            np.asarray(values, dtype=np.float32)
            for values in (
                self.doppler_velocity_mps,
                self.z_m,
                self.y_m,
                self.x_m,
            )
        )
        expected_shape = tuple(axis.size for axis in axes)
        if magnitude.ndim != 4 or magnitude.shape != expected_shape:
            raise ValueError(
                "CartesianRadarVolume magnitude must have shape (D, Z, Y, X); "
                f"got {magnitude.shape} for axes {expected_shape}."
            )
        if not np.isfinite(magnitude).all() or np.any(magnitude < 0.0):
            raise ValueError("CartesianRadarVolume magnitude must be finite and non-negative.")
        for name, axis in zip(
            ("doppler_velocity_mps", "z_m", "y_m", "x_m"),
            axes,
            strict=True,
        ):
            if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
                raise ValueError(f"CartesianRadarVolume {name} must be a finite 1D axis.")
            if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
                raise ValueError(f"CartesianRadarVolume {name} must be strictly increasing.")
        coordinate_frame = self.coordinate_frame.strip()
        if not coordinate_frame:
            raise ValueError("CartesianRadarVolume coordinate_frame must not be empty.")

        object.__setattr__(self, "magnitude_dzyx", magnitude)
        object.__setattr__(self, "doppler_velocity_mps", axes[0])
        object.__setattr__(self, "z_m", axes[1])
        object.__setattr__(self, "y_m", axes[2])
        object.__setattr__(self, "x_m", axes[3])
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))


@dataclass(frozen=True)
class DetectionFrame:
    """Detected radar targets before conversion to Cartesian point clouds."""

    detections: np.ndarray
    channels: tuple[str, ...] = ("range", "doppler", "azimuth", "snr")
    frame_id: str | int | None = None
    timestamp: float | None = None
    source: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        detections = np.asarray(self.detections, dtype=np.float32)
        channels = tuple(self.channels)
        if detections.ndim != 2:
            raise ValueError(
                f"DetectionFrame.detections must have shape (N, C); got {detections.shape}."
            )
        if detections.shape[1] != len(channels):
            raise ValueError(
                "DetectionFrame.channels length must match the detection dimension; "
                f"got {len(channels)} channels for shape {detections.shape}."
            )
        if not np.isfinite(detections).all():
            raise ValueError("DetectionFrame.detections contains NaN or Inf values.")

        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "units", _metadata_copy(self.units))
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))


@dataclass(frozen=True)
class PointCloudFrame:
    """Cartesian radar point cloud frame produced by mmwcore processing."""

    points: np.ndarray
    channels: tuple[str, ...] = _XYZ_CHANNELS
    frame_id: str | int | None = None
    timestamp: float | None = None
    source: str | None = None
    coordinate_frame: str = "radar"
    units: dict[str, str] = field(default_factory=lambda: {"x": "m", "y": "m", "z": "m"})
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        channels = tuple(self.channels)
        if points.ndim != 2:
            raise ValueError(f"PointCloudFrame.points must have shape (N, C); got {points.shape}.")
        if points.shape[1] != len(channels):
            raise ValueError(
                "PointCloudFrame.channels length must match the point dimension; "
                f"got {len(channels)} channels for shape {points.shape}."
            )
        if len(channels) < 3 or channels[:3] != _XYZ_CHANNELS:
            raise ValueError('PointCloudFrame.channels must start with ("x", "y", "z").')
        if not np.isfinite(points).all():
            raise ValueError("PointCloudFrame.points contains NaN or Inf values.")

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "units", _metadata_copy(self.units))
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self.points.shape[1])

    def xyz(self) -> np.ndarray:
        return self.points[:, :3]


@dataclass(frozen=True)
class ClusterFrame:
    """Per-frame spatial clusters summarized from a Cartesian point cloud."""

    centers: np.ndarray
    extents: np.ndarray
    mean_velocities: np.ndarray
    point_counts: np.ndarray
    point_labels: np.ndarray
    frame_id: str | int | None = None
    timestamp: float | None = None
    source: str | None = None
    coordinate_frame: str = "radar"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float32)
        extents = np.asarray(self.extents, dtype=np.float32)
        velocities = np.asarray(self.mean_velocities, dtype=np.float32)
        counts = np.asarray(self.point_counts, dtype=np.int64)
        labels = np.asarray(self.point_labels, dtype=np.int64)
        _validate_cluster_shapes(centers, extents, velocities, counts, labels)
        _validate_cluster_membership(centers, extents, counts, labels)
        _validate_cluster_values(centers, extents, velocities)

        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "extents", extents)
        object.__setattr__(self, "mean_velocities", velocities)
        object.__setattr__(self, "point_counts", counts)
        object.__setattr__(self, "point_labels", labels)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    @property
    def num_clusters(self) -> int:
        return int(self.centers.shape[0])


@dataclass(frozen=True)
class TrackFrame:
    """Track states, meter-squared covariances, and observation associations."""

    track_ids: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    position_covariances: np.ndarray
    extent_covariances: np.ndarray
    statuses: tuple[TrackStatus, ...]
    ages: np.ndarray
    missed_counts: np.ndarray
    observation_track_ids: np.ndarray
    frame_id: str | int | None = None
    timestamp: float | None = None
    source: str | None = None
    coordinate_frame: str = "radar"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        track_ids = np.asarray(self.track_ids, dtype=np.int64)
        positions = np.asarray(self.positions, dtype=np.float32)
        velocities = np.asarray(self.velocities, dtype=np.float32)
        position_covariances = np.asarray(self.position_covariances, dtype=np.float32)
        extent_covariances = np.asarray(self.extent_covariances, dtype=np.float32)
        statuses = tuple(TrackStatus(status) for status in self.statuses)
        ages = np.asarray(self.ages, dtype=np.int64)
        missed = np.asarray(self.missed_counts, dtype=np.int64)
        associations = np.asarray(self.observation_track_ids, dtype=np.int64)
        count = track_ids.size
        _validate_track_ids(track_ids)
        _validate_track_state_shapes(positions, velocities, count=count)
        for name, covariances in (
            ("position_covariances", position_covariances),
            ("extent_covariances", extent_covariances),
        ):
            _validate_track_covariances(name, covariances, count=count)
        _validate_track_lifecycle(statuses, ages, missed, count=count)
        _validate_track_associations(associations, track_ids)
        _validate_track_state_values(positions, velocities)

        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "position_covariances", position_covariances)
        object.__setattr__(self, "extent_covariances", extent_covariances)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "ages", ages)
        object.__setattr__(self, "missed_counts", missed)
        object.__setattr__(self, "observation_track_ids", associations)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    @property
    def num_tracks(self) -> int:
        return int(self.track_ids.size)


def _validate_cluster_shapes(
    centers: np.ndarray,
    extents: np.ndarray,
    velocities: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
) -> None:
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"ClusterFrame.centers must have shape (N, 3); got {centers.shape}.")
    if extents.shape != centers.shape:
        raise ValueError("ClusterFrame.extents must match centers shape.")
    if velocities.shape != (centers.shape[0],):
        raise ValueError("ClusterFrame.mean_velocities must have shape (N,).")
    if counts.shape != (centers.shape[0],):
        raise ValueError("ClusterFrame.point_counts must have shape (N,).")
    if labels.ndim != 1:
        raise ValueError("ClusterFrame.point_labels must be one-dimensional.")


def _validate_cluster_membership(
    centers: np.ndarray,
    extents: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
) -> None:
    if np.any(counts <= 0):
        raise ValueError("ClusterFrame.point_counts must be positive.")
    if np.any(extents < 0):
        raise ValueError("ClusterFrame.extents must be non-negative.")
    if np.any(labels < -1):
        raise ValueError("ClusterFrame.point_labels may only use -1 for noise.")
    cluster_ids = np.unique(labels[labels >= 0])
    expected_ids = np.arange(centers.shape[0], dtype=np.int64)
    if not np.array_equal(cluster_ids, expected_ids):
        raise ValueError("ClusterFrame.point_labels must use contiguous cluster IDs.")
    if cluster_ids.size:
        label_counts = np.bincount(labels[labels >= 0], minlength=cluster_ids.size)
        if not np.array_equal(label_counts, counts):
            raise ValueError("ClusterFrame.point_counts must match point_labels.")


def _validate_cluster_values(
    centers: np.ndarray, extents: np.ndarray, velocities: np.ndarray
) -> None:
    if not np.isfinite(centers).all() or not np.isfinite(extents).all():
        raise ValueError("ClusterFrame spatial values contain NaN or Inf.")
    if not np.isfinite(velocities).all():
        raise ValueError("ClusterFrame.mean_velocities contains NaN or Inf.")


def _validate_track_ids(track_ids: np.ndarray) -> None:
    if track_ids.ndim != 1 or np.any(track_ids < 0):
        raise ValueError("TrackFrame.track_ids must be one-dimensional and non-negative.")
    if np.unique(track_ids).size != track_ids.size:
        raise ValueError("TrackFrame.track_ids must be unique.")


def _validate_track_state_shapes(
    positions: np.ndarray, velocities: np.ndarray, *, count: int
) -> None:
    if positions.shape != (count, 3) or velocities.shape != (count, 3):
        raise ValueError("TrackFrame positions and velocities must have shape (N, 3).")


def _validate_track_covariances(name: str, covariances: np.ndarray, *, count: int) -> None:
    if covariances.shape != (count, 2, 2):
        raise ValueError(f"TrackFrame.{name} must have shape (N, 2, 2).")
    if not np.isfinite(covariances).all():
        raise ValueError(f"TrackFrame.{name} contains NaN or Inf values.")
    if not np.allclose(covariances, covariances.transpose(0, 2, 1), atol=1e-6):
        raise ValueError(f"TrackFrame.{name} must be symmetric.")
    if count and np.min(np.linalg.eigvalsh(covariances)) < -1e-6:
        raise ValueError(f"TrackFrame.{name} must be positive semidefinite.")


def _validate_track_lifecycle(
    statuses: tuple[TrackStatus, ...],
    ages: np.ndarray,
    missed: np.ndarray,
    *,
    count: int,
) -> None:
    if len(statuses) != count:
        raise ValueError("TrackFrame.statuses length must match track count.")
    if ages.shape != (count,) or missed.shape != (count,):
        raise ValueError("TrackFrame ages and missed_counts must have shape (N,).")
    if np.any(ages <= 0) or np.any(missed < 0):
        raise ValueError("TrackFrame ages must be positive and missed_counts non-negative.")


def _validate_track_associations(associations: np.ndarray, track_ids: np.ndarray) -> None:
    if associations.ndim != 1 or np.any(associations < -1):
        raise ValueError(
            "TrackFrame.observation_track_ids must be one-dimensional and use -1 for unassigned."
        )
    assigned = associations[associations >= 0]
    if assigned.size and not np.isin(assigned, track_ids).all():
        raise ValueError("TrackFrame.observation_track_ids references an unknown track ID.")


def _validate_track_state_values(positions: np.ndarray, velocities: np.ndarray) -> None:
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("TrackFrame state contains NaN or Inf values.")
