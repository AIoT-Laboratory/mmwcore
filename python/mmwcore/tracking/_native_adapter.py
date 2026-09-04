"""Typed Python adapters around the native tracker state machines."""

from __future__ import annotations

from typing import Any

from mmwcore import _native
from mmwcore.core import (
    AllocationSpec,
    Box2D,
    Box3D,
    DBSCANSpec,
    GatingSpec,
    LifecycleSpec,
    Tracker2DSpec,
    Tracker3DSpec,
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
            spec.angle_noise_rad,
            spec.doppler_noise_mps,
            spec.max_velocity_mps,
        ),
        _gating_config(spec.gating),
        _allocation_config(spec.allocation),
        _lifecycle_config(spec.lifecycle),
        (
            _box_2d_config(spec.scenery.boundary_boxes),
            _box_2d_config(spec.scenery.static_boxes),
            spec.scenery.outside_max_frames,
        ),
        spec.max_tracks,
    )


def native_measurement_tracker_config(
    spec: Tracker2DSpec,
    allocation_clustering: DBSCANSpec,
) -> _native.NativeMeasurementTrackerConfig:
    """Encode the native measurement tracker and its allocation clustering rule."""

    return (
        native_tracker_config(spec),
        _dbscan_config(allocation_clustering),
    )


def native_measurement_tracker_3d_config(
    spec: Tracker3DSpec,
    allocation_clustering: DBSCANSpec,
) -> _native.NativeMeasurementTracker3DConfig:
    """Encode GTrack3D and its allocation clustering rule."""

    tracker = (
        (
            spec.frame_period_s,
            spec.max_acceleration_mps2,
            spec.measurement_noise_m,
            spec.initial_velocity_std_mps,
            spec.extent_covariance_smoothing,
            spec.angle_noise_rad,
            spec.elevation_noise_rad,
            spec.doppler_noise_mps,
            spec.max_velocity_mps,
        ),
        _gating_config(spec.gating),
        _allocation_config(spec.allocation),
        _lifecycle_config(spec.lifecycle),
        (
            _box_3d_config(spec.scenery.boundary_boxes),
            _box_3d_config(spec.scenery.static_boxes),
            spec.scenery.outside_max_frames,
        ),
        spec.max_tracks,
    )
    return tracker, _dbscan_config(allocation_clustering)


def _gating_config(spec: GatingSpec) -> _native.NativeTrackerGatingConfig:
    return (
        spec.max_distance_m,
        spec.max_radial_velocity_difference_mps,
        spec.max_mahalanobis_distance,
    )


def _allocation_config(spec: AllocationSpec) -> _native.NativeTrackerAllocationConfig:
    return (
        spec.min_points,
        spec.min_abs_radial_velocity_mps,
        spec.min_total_snr,
        spec.max_new_tracks_per_frame,
        spec.min_separation_m,
    )


def _lifecycle_config(spec: LifecycleSpec) -> _native.NativeTrackerLifecycleConfig:
    return (
        spec.confirmation_hits,
        spec.tentative_max_misses,
        spec.confirmed_max_misses,
        spec.min_update_points,
        spec.static_max_misses,
        spec.exit_max_misses,
        spec.static_speed_threshold_mps,
    )


def _dbscan_config(spec: DBSCANSpec) -> _native.NativeDbscanConfig:
    return spec.eps_m, spec.min_samples, spec.velocity_scale_s, spec.use_z


def _box_2d_config(values: tuple[Box2D, ...]) -> list[_native.NativeTrackingBox]:
    return [(box.x_min_m, box.x_max_m, box.y_min_m, box.y_max_m) for box in values]


def _box_3d_config(values: tuple[Box3D, ...]) -> list[_native.NativeTrackingBox3D]:
    return [
        (
            box.x_min_m,
            box.x_max_m,
            box.y_min_m,
            box.y_max_m,
            box.z_min_m,
            box.z_max_m,
        )
        for box in values
    ]


def track_frame(
    result: _native.NativeTrackerStepResult,
    *,
    frame_id: str | int | None,
    timestamp: float | None,
    source: str | None,
    coordinate_frame: str,
    metadata: dict[str, Any],
    model: str,
    diagnostics: dict[str, int] | None = None,
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
        metadata={
            **metadata,
            "tracker": {
                "model": model,
                **({} if diagnostics is None else {"diagnostics": diagnostics}),
            },
        },
    )


def gtrack3d_diagnostics(
    values: _native.NativeGTrack3DDiagnostics,
) -> dict[str, int]:
    """Name the cumulative native counters used to diagnose identity churn."""

    (
        (frames, points, outside_points),
        (distance_gate_misses, doppler_gate_misses, mahalanobis_gate_misses),
        (empty_updates, partial_updates),
        (allocations, confirmations, reactivations),
        (tentative_deletions, coasting_deletions, outside_deletions),
    ) = values
    return {
        "frames": frames,
        "points": points,
        "outside_points": outside_points,
        "distance_gate_misses": distance_gate_misses,
        "doppler_gate_misses": doppler_gate_misses,
        "mahalanobis_gate_misses": mahalanobis_gate_misses,
        "empty_updates": empty_updates,
        "partial_updates": partial_updates,
        "allocations": allocations,
        "confirmations": confirmations,
        "reactivations": reactivations,
        "tentative_deletions": tentative_deletions,
        "coasting_deletions": coasting_deletions,
        "outside_deletions": outside_deletions,
    }
