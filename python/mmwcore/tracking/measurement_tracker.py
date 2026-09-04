"""Native measurement-level tracking adapter."""

from __future__ import annotations

import numpy as np

from mmwcore import _native
from mmwcore.core import DBSCANSpec, PointCloudFrame, Tracker2DSpec, Tracker3DSpec, TrackFrame
from mmwcore.tracking._native_adapter import (
    gtrack3d_diagnostics,
    native_measurement_tracker_3d_config,
    native_measurement_tracker_config,
    track_frame,
)


class GTrack2D:
    """Advance one Rust-owned GTRACK 2D unit set per point-cloud frame."""

    def __init__(
        self,
        spec: Tracker2DSpec,
        allocation_clustering: DBSCANSpec,
    ) -> None:
        self.spec = spec
        self.allocation_clustering = allocation_clustering
        self._tracker = _native.NativePointTracker2D(
            native_measurement_tracker_config(spec, allocation_clustering)
        )

    def step(self, point_cloud: PointCloudFrame) -> TrackFrame:
        """Advance native state once and assemble the typed Python report."""

        return _advance(
            point_cloud,
            tracker=self._tracker,
            spec=self.spec,
            allocation_clustering=self.allocation_clustering,
            model="gtrack_2d",
        )


class GTrack3D:
    """Advance one Rust-owned GTRACK 3D unit set per sensor-frame point cloud."""

    def __init__(
        self,
        spec: Tracker3DSpec,
        allocation_clustering: DBSCANSpec,
    ) -> None:
        self.spec = spec
        self.allocation_clustering = allocation_clustering
        self._tracker = _native.NativePointTracker3D(
            native_measurement_tracker_3d_config(spec, allocation_clustering)
        )

    def step(self, point_cloud: PointCloudFrame) -> TrackFrame:
        """Advance native 3D state once and assemble the typed Python report."""

        result, diagnostics = self._tracker.step(
            *_measurement_inputs(point_cloud, self.spec, self.allocation_clustering)
        )
        return _frame(
            point_cloud,
            result,
            model="gtrack_3d",
            diagnostics=gtrack3d_diagnostics(diagnostics),
        )


def _advance(
    point_cloud: PointCloudFrame,
    *,
    tracker: _native.NativePointTracker2D,
    spec: Tracker2DSpec,
    allocation_clustering: DBSCANSpec,
    model: str,
) -> TrackFrame:
    result = tracker.step(*_measurement_inputs(point_cloud, spec, allocation_clustering))
    return _frame(point_cloud, result, model=model)


def _measurement_inputs(
    point_cloud: PointCloudFrame,
    spec: Tracker2DSpec | Tracker3DSpec,
    allocation_clustering: DBSCANSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_required = bool(
        spec.gating.max_radial_velocity_difference_mps is not None
        or spec.allocation.min_abs_radial_velocity_mps > 0
        or allocation_clustering.velocity_scale_s > 0
    )
    return (
        np.ascontiguousarray(point_cloud.xyz(), dtype=np.float32),
        _point_channel(
            point_cloud,
            "velocity",
            required=velocity_required,
            requirement="configured tracking velocity constraints",
        ),
        _linear_snr(point_cloud, required=spec.allocation.min_total_snr is not None),
    )


def _frame(
    point_cloud: PointCloudFrame,
    result: _native.NativeTrackerStepResult,
    *,
    model: str,
    diagnostics: dict[str, int] | None = None,
) -> TrackFrame:
    return track_frame(
        result,
        frame_id=point_cloud.frame_id,
        timestamp=point_cloud.timestamp,
        source=point_cloud.source,
        coordinate_frame=point_cloud.coordinate_frame,
        metadata=point_cloud.metadata,
        model=model,
        diagnostics=diagnostics,
    )


def _linear_snr(point_cloud: PointCloudFrame, *, required: bool) -> np.ndarray:
    """Return linear power ratio; convert the common DSP dB channel at the boundary."""

    if "snr" in point_cloud.channels:
        return _point_channel(
            point_cloud,
            "snr",
            required=required,
            requirement="AllocationSpec.min_total_snr",
        )
    if "snr_db" in point_cloud.channels:
        snr_db = _point_channel(
            point_cloud,
            "snr_db",
            required=True,
            requirement="GTRACK SNR",
        )
        return np.ascontiguousarray(np.power(10.0, snr_db / 10.0), dtype=np.float32)
    if required:
        raise ValueError(
            'PointCloudFrame must include an "snr" or "snr_db" channel for '
            "AllocationSpec.min_total_snr."
        )
    return np.zeros(point_cloud.num_points, dtype=np.float32)


def _point_channel(
    point_cloud: PointCloudFrame,
    channel: str,
    *,
    required: bool,
    requirement: str,
) -> np.ndarray:
    try:
        index = point_cloud.channels.index(channel)
    except ValueError:
        if required:
            raise ValueError(
                f'PointCloudFrame must include a "{channel}" channel for {requirement}.'
            ) from None
        return np.zeros(point_cloud.num_points, dtype=np.float32)
    return np.ascontiguousarray(point_cloud.points[:, index], dtype=np.float32)


PointTracker2D = GTrack2D
