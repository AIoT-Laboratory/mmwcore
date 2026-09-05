"""Complete pinned TI 6843 GTRACK, using a separately built local native plugin.

TI code is licensed for TI devices; it is not included in mmwcore distributions.
The older GTrack3D remains a separate six-state implementation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mmwcore import _native
from mmwcore.core import Box3D, PointCloudFrame, TrackFrame, TrackStatus
from mmwcore.tracking.measurement_tracker import _linear_snr, _point_channel


@dataclass(frozen=True)
class TiGTrackGating:
    """Gating defaults from the pinned ISK_6m_default.cfg tracking layer."""

    gain: float = 3.0
    depth_m: float = 2.0
    width_m: float = 2.0
    height_m: float = 2.0
    velocity_mps: float = 4.0


@dataclass(frozen=True)
class TiGTrackAllocation:
    """Pinned ISK example allocation parameters; not fitted to local point counts."""

    snr_threshold: float = 40.0
    obscured_snr_threshold: float = 100.0
    velocity_threshold_mps: float = 0.1
    points_threshold: int = 20
    max_distance_m: float = 0.5
    max_velocity_difference_mps: float = 20.0


@dataclass(frozen=True)
class TiGTrackLifecycle:
    """TI counters retain their actual > / >= comparisons and reliable-point rules."""

    detection_to_active: int = 3
    detection_to_free: int = 3
    active_to_free: int = 12
    static_to_free: int = 500
    exit_to_free: int = 5
    sleep_to_free: int = 6000


@dataclass(frozen=True)
class TiGTrackScenery:
    """World boxes and installation use workspace forward/right/up coordinates."""

    sensor_position_m: tuple[float, float, float] = (0.0, 0.0, 2.0)
    azimuth_tilt_deg: float = 0.0
    elevation_tilt_deg: float = 0.0
    boundary_boxes: tuple[Box3D, ...] = ()
    static_boxes: tuple[Box3D, ...] = ()
    occupancy_boxes: tuple[Box3D, ...] = ()
    presence_points_threshold: int = 0
    presence_velocity_threshold_mps: float = 0.0
    presence_on_to_off: int = 0


@dataclass(frozen=True)
class TiGTrack3DSpec:
    """Explicit capture timing/velocity and pinned ISK example tracking defaults."""

    frame_period_s: float
    max_radial_velocity_mps: float
    radial_velocity_resolution_mps: float
    max_points: int = 800
    max_tracks: int = 30
    max_acceleration_mps2: tuple[float, float, float] = (0.1, 0.1, 0.1)
    initial_radial_velocity_mps: float = 0.0
    boresight_filtering: bool = False
    gating: TiGTrackGating = field(default_factory=TiGTrackGating)
    allocation: TiGTrackAllocation = field(default_factory=TiGTrackAllocation)
    lifecycle: TiGTrackLifecycle = field(default_factory=TiGTrackLifecycle)
    scenery: TiGTrackScenery = field(default_factory=TiGTrackScenery)

    def native_config(self) -> dict[str, object]:
        """Encode TI ABI fields; authoritative validation runs in Rust before create."""
        g, a, s, scene = self.gating, self.allocation, self.lifecycle, self.scenery
        position = scene.sensor_position_m
        acceleration = self.max_acceleration_mps2
        if len(position) != 3 or len(acceleration) != 3:
            raise ValueError("Sensor position and maximum acceleration must have three axes")
        if not isinstance(self.boresight_filtering, bool):
            raise ValueError("boresight_filtering must be boolean")
        return {
            "max_points": self.max_points,
            "max_tracks": self.max_tracks,
            "delta_t": self.frame_period_s,
            "initial_velocity": self.initial_radial_velocity_mps,
            "max_velocity": self.max_radial_velocity_mps,
            "velocity_resolution": self.radial_velocity_resolution_mps,
            "max_acceleration": [acceleration[1], acceleration[0], acceleration[2]],
            "boresight_filtering": int(self.boresight_filtering),
            "gating_gain": g.gain,
            "gating_limits": [g.depth_m, g.width_m, g.height_m, g.velocity_mps],
            "allocation_snr": a.snr_threshold,
            "allocation_obscured_snr": a.obscured_snr_threshold,
            "allocation_velocity": a.velocity_threshold_mps,
            "allocation_points": a.points_threshold,
            "allocation_distance": a.max_distance_m,
            "allocation_max_velocity": a.max_velocity_difference_mps,
            "state_thresholds": [
                s.detection_to_active,
                s.detection_to_free,
                s.active_to_free,
                s.static_to_free,
                s.exit_to_free,
                s.sleep_to_free,
            ],
            "sensor_position": [position[1], position[0], position[2]],
            "sensor_orientation": [scene.azimuth_tilt_deg, scene.elevation_tilt_deg],
            "boundary_count": len(scene.boundary_boxes),
            "boundary_boxes": _boxes(scene.boundary_boxes),
            "static_count": len(scene.static_boxes),
            "static_boxes": _boxes(scene.static_boxes),
            "occupancy_count": len(scene.occupancy_boxes),
            "occupancy_boxes": _boxes(scene.occupancy_boxes),
            "presence_points": scene.presence_points_threshold,
            "presence_on_to_off": scene.presence_on_to_off,
            "presence_velocity": scene.presence_velocity_threshold_mps,
        }


def _boxes(boxes: tuple[Box3D, ...]) -> list[float]:
    if len(boxes) > 2:
        raise ValueError("TI supports at most two boxes of each kind")
    values = [
        v for b in boxes for v in (b.y_min_m, b.y_max_m, b.x_min_m, b.x_max_m, b.z_min_m, b.z_max_m)
    ]
    return values + [0.0] * (12 - len(values))


class TiGTrack3D:
    """Run the entire original 3DA step; no Python numerical tracker or fallback.

    ``step`` consumes sensor forward/right/up Cartesian point clouds.
    ``step_spherical`` consumes range/azimuth-right/elevation-up/vr/linear-SNR.
    Either call advances exactly one frame. ``last_report`` retains TI axes,
    original point labels, unique/static flags, full states and covariances.
    """

    def __init__(self, spec: TiGTrack3DSpec, *, plugin_manifest: str | Path | None = None) -> None:
        self.spec = spec
        selected = plugin_manifest or os.environ.get("MMWCORE_TI_GTRACK_MANIFEST")
        if not selected:
            raise ValueError(
                "TI GTRACK requires plugin_manifest or MMWCORE_TI_GTRACK_MANIFEST; "
                "build the local TI plugin first"
            )
        self.plugin_manifest = Path(selected).resolve(strict=True)
        self._tracker = self._create_native()
        self.provenance: dict[str, Any] = json.loads(self._tracker.provenance_json())
        self.last_report: dict[str, Any] | None = None

    def step_spherical(
        self, points: np.ndarray, variances: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Advance once; variances are positive (N,4), in m²/rad²/rad²/(m/s)².

        Omit unknown variances; zero is not an unknown-noise placeholder.
        """
        return self._step_native(np.asarray(points, dtype=np.float32), variances, cartesian=False)

    def step(
        self, point_cloud: PointCloudFrame, *, variances: np.ndarray | None = None
    ) -> TrackFrame:
        if point_cloud.coordinate_frame not in {
            "sensor_forward_lateral_up",
            "sensor_forward_right_up",
        }:
            raise ValueError(
                "TI GTRACK Cartesian input must use sensor_forward_right_up "
                "(or the existing lateral alias)"
            )
        velocity = _point_channel(
            point_cloud, "velocity", required=True, requirement="TI GTRACK spherical measurements"
        )
        snr = _linear_snr(point_cloud, required=True)
        points = np.column_stack((point_cloud.xyz(), velocity, snr)).astype(np.float32)
        raw = self._step_native(points, variances, cartesian=True)
        targets, views = raw["targets"], raw["sensor_targets"]
        count = len(targets)
        state = np.asarray([v["state_vector"] for v in views], np.float32).reshape(count, 9)
        lifecycle = tuple(_lifecycle(t) for t in targets)
        metadata = dict(point_cloud.metadata)
        metadata["tracker"] = {
            "model": "ti_gtrack_6843_3da",
            "configuration": asdict(self.spec),
            "source_version": self.provenance["source_version"],
            "library_sha256": self.provenance["library_sha256"],
            "ti_report_frame": "sensor_right_forward_up",
            "ti_report": raw,
            "extent_covariance": "spherical_group_dispersion_projected_to_sensor_cartesian",
        }
        return TrackFrame(
            track_ids=np.asarray([t["tid"] for t in targets], np.int64),
            positions=state[:, :3],
            velocities=state[:, 3:6],
            position_covariances=np.asarray(
                [v["position_covariance"] for v in views], np.float32
            ).reshape(count, 3, 3),
            extent_covariances=np.asarray(
                [v["extent_covariance"] for v in views], np.float32
            ).reshape(count, 3, 3),
            statuses=tuple(status for status, _ in lifecycle),
            ages=np.asarray([t["age"] for t in targets], np.int64),
            missed_counts=np.asarray(
                [missed for _, missed in lifecycle],
                np.int64,
            ),
            observation_track_ids=np.asarray(raw["point_tid"], np.int64),
            frame_id=point_cloud.frame_id,
            timestamp=point_cloud.timestamp,
            source=point_cloud.source,
            coordinate_frame=point_cloud.coordinate_frame,
            metadata=metadata,
        )

    def close(self) -> None:
        self._tracker.close()

    def reset(self) -> None:
        replacement = self._create_native()
        self.close()
        self._tracker = replacement
        self.provenance = json.loads(replacement.provenance_json())
        self.last_report = None

    def _create_native(self) -> _native.NativeTiGTrack3D:
        return _native.NativeTiGTrack3D(
            str(self.plugin_manifest), json.dumps(self.spec.native_config(), allow_nan=False)
        )

    def _step_native(
        self, points: np.ndarray, variances: np.ndarray | None, *, cartesian: bool
    ) -> dict[str, Any]:
        report: dict[str, Any] = json.loads(
            self._tracker.step(points, _variances(variances), cartesian=cartesian)
        )
        self.last_report = report
        return report

    def __enter__(self) -> TiGTrack3D:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _variances(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else np.asarray(values, dtype=np.float32)


def _lifecycle(target: dict[str, Any]) -> tuple[TrackStatus, int]:
    if target["state"] == 2:  # Native DETECTION
        return TrackStatus.TENTATIVE, target["counters"][1]
    missed = target["counters"][2]  # Native ACTIVE's active2freeCount
    status = TrackStatus.COASTING if missed else TrackStatus.CONFIRMED
    return status, missed
