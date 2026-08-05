"""Typed Python adapters around the native tracker state machines."""

from __future__ import annotations

from typing import Any

from mmwcore import _native
from mmwcore.core import (
    DBSCANClusteringSpec,
    Tracker2DSpec,
    TrackFrame,
    TrackStatus,
)

_STATUS_BY_CODE = (
    TrackStatus.TENTATIVE,
    TrackStatus.CONFIRMED,
    TrackStatus.COASTING,
)


def native_tracker_config(spec: Tracker2DSpec) -> _native.NativeClusterTrackerConfig:
    """Encode a public tracker specification for the PyO3 boundary."""

    return (
        (
            spec.frame_period_s,
            spec.max_acceleration_mps2,
            spec.measurement_noise_m,
            spec.initial_velocity_std_mps,
            spec.extent_covariance_smoothing,
        ),
        (
            spec.gating.max_distance_m,
            spec.gating.max_radial_velocity_difference_mps,
            spec.gating.max_mahalanobis_distance,
        ),
        (
            spec.allocation.min_points,
            spec.allocation.min_abs_radial_velocity_mps,
            spec.allocation.min_total_snr,
            spec.allocation.max_new_tracks_per_frame,
        ),
        (
            spec.lifecycle.confirmation_hits,
            spec.lifecycle.tentative_max_misses,
            spec.lifecycle.confirmed_max_misses,
        ),
        (
            [
                (box.x_min_m, box.x_max_m, box.y_min_m, box.y_max_m)
                for box in spec.scenery.boundary_boxes
            ],
            spec.scenery.outside_max_frames,
        ),
        spec.max_tracks,
    )


def native_measurement_tracker_config(
    spec: Tracker2DSpec,
    allocation_clustering: DBSCANClusteringSpec,
) -> _native.NativeMeasurementTrackerConfig:
    """Encode the native measurement tracker and its allocation clustering rule."""

    return (
        native_tracker_config(spec),
        (
            allocation_clustering.eps_m,
            allocation_clustering.min_samples,
            allocation_clustering.velocity_scale_s,
            allocation_clustering.use_z,
        ),
    )


def track_frame(
    result: _native.NativeTrackerStepResult,
    *,
    frame_id: str | int | None,
    timestamp: float | None,
    source: str | None,
    coordinate_frame: str,
    metadata: dict[str, Any],
    model: str,
) -> TrackFrame:
    """Attach public frame identity and metadata to a native tracker result."""

    (
        track_ids,
        positions,
        velocities,
        position_covariances,
        extent_covariances,
        status_codes,
        ages,
        missed_counts,
        observation_track_ids,
    ) = result
    return TrackFrame(
        track_ids=track_ids,
        positions=positions,
        velocities=velocities,
        position_covariances=position_covariances,
        extent_covariances=extent_covariances,
        statuses=tuple(_STATUS_BY_CODE[int(code)] for code in status_codes),
        ages=ages,
        missed_counts=missed_counts,
        observation_track_ids=observation_track_ids,
        frame_id=frame_id,
        timestamp=timestamp,
        source=source,
        coordinate_frame=coordinate_frame,
        metadata={**metadata, "tracker": {"model": model}},
    )
