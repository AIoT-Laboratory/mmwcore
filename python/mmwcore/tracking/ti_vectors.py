"""Readers for the documented TI mmWave SDK GTRACK 2D test-vector format."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from struct import Struct

import numpy as np

from mmwcore.core import (
    AllocationSpec,
    Box2D,
    DBSCANSpec,
    GatingSpec,
    LifecycleSpec,
    PointCloudFrame,
    ScenerySpec,
    Tracker2DSpec,
    TrackFrame,
)
from mmwcore.dsp import cluster_point_cloud
from mmwcore.tracking.benchmark import (
    TrackingBenchmarkSummary,
    TrackingGroundTruthFrame,
    evaluate_track_frames,
)
from mmwcore.tracking.measurement_tracker import PointTracker2D
from mmwcore.tracking.tracker import ClusterTracker2D

_HEADER = Struct("<II")
_MEASUREMENT = np.dtype(("<f4", 4))
_REFERENCE = np.dtype([("track_id", "<u4"), ("state", "<f4", (6,))])
_POINT_CLOUD_TLV = 6
_TARGET_LIST_TLV = 7


@dataclass(frozen=True)
class TiGTrack2DVectorFrame:
    """One SDK vector frame with spherical measurements and target references."""

    frame_number: int
    measurements: np.ndarray
    ground_truth: TrackingGroundTruthFrame

    def __post_init__(self) -> None:
        measurements = np.asarray(self.measurements, dtype=np.float32)
        if self.frame_number < 0:
            raise ValueError("TI GTRACK frame number must be non-negative.")
        if measurements.ndim != 2 or measurements.shape[1] != 4:
            raise ValueError("TI GTRACK measurements must have shape (N, 4).")
        if not np.isfinite(measurements).all():
            raise ValueError("TI GTRACK measurements contain NaN or Inf.")
        object.__setattr__(self, "measurements", measurements)

    def cartesian_points(self) -> np.ndarray:
        """Return x, y, z, radial_velocity, and SNR channels."""

        points = np.zeros((self.measurements.shape[0], 5), dtype=np.float32)
        ranges = self.measurements[:, 0]
        angles = self.measurements[:, 1]
        points[:, 0] = ranges * np.cos(angles)
        points[:, 1] = ranges * np.sin(angles)
        points[:, 3:] = self.measurements[:, 2:]
        return points


@dataclass(frozen=True)
class TiGTrackComparison:
    """Cluster-tracker results against one TI GTRACK vector sequence."""

    all_tracks: TrackingBenchmarkSummary
    confirmed_tracks: TrackingBenchmarkSummary

    def to_record(self) -> dict[str, object]:
        return {
            "all_tracks": self.all_tracks.to_record(),
            "confirmed_tracks": self.confirmed_tracks.to_record(),
        }


@dataclass(frozen=True)
class TiGTrackStrategyRun:
    """Per-frame predictions and their comparison against TI references."""

    frames: tuple[TrackFrame, ...]
    comparison: TiGTrackComparison


@dataclass(frozen=True)
class TiGTrack2DBenchmarkSpec:
    """Reproducible mmwcore configuration for one TI 2D vector scenario."""

    clustering: DBSCANSpec
    tracker: Tracker2DSpec
    match_distance_m: float


@dataclass(frozen=True)
class TiGTrack2DBenchmarkReport:
    """Paired strategy results with one shared vector and configuration contract."""

    scenario: str
    num_frames: int
    first_frame_number: int | None
    last_frame_number: int | None
    spec: TiGTrack2DBenchmarkSpec
    cluster: TiGTrackComparison
    measurement: TiGTrackComparison

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable comparison record."""

        return {
            "schema_version": 1,
            "scenario": self.scenario,
            "num_frames": self.num_frames,
            "first_frame_number": self.first_frame_number,
            "last_frame_number": self.last_frame_number,
            "configuration": asdict(self.spec),
            "results": {
                "cluster": self.cluster.to_record(),
                "measurement": self.measurement.to_record(),
            },
        }


@dataclass(frozen=True)
class TiGTrack2DBenchmarkRun:
    """Paired tracker runs retaining both reports and per-frame predictions."""

    report: TiGTrack2DBenchmarkReport
    cluster: TiGTrackStrategyRun
    measurement: TiGTrackStrategyRun


def ti_people_counting_2d_benchmark_spec() -> TiGTrack2DBenchmarkSpec:
    """Map representable SDK people-counting parameters to mmwcore primitives."""

    return TiGTrack2DBenchmarkSpec(
        clustering=DBSCANSpec(
            eps_m=1.5,
            min_samples=6,
            velocity_scale_s=0.75,
            use_z=False,
        ),
        tracker=Tracker2DSpec(
            frame_period_s=0.05,
            gating=GatingSpec(
                max_distance_m=1.5,
                max_radial_velocity_difference_mps=2.0,
            ),
            allocation=AllocationSpec(
                min_points=6,
                min_abs_radial_velocity_mps=0.1,
                max_new_tracks_per_frame=1,
            ),
            lifecycle=LifecycleSpec(
                confirmation_hits=10,
                tentative_max_misses=5,
                confirmed_max_misses=50,
            ),
            scenery=ScenerySpec(
                boundary_boxes=(Box2D(0.5, 7.5, -4.0, 4.0),),
                outside_max_frames=5,
            ),
            max_tracks=20,
            max_acceleration_mps2=(0.1, 0.1),
        ),
        match_distance_m=1.0,
    )


def benchmark_ti_people_counting_2d(
    frames: tuple[TiGTrack2DVectorFrame, ...],
    spec: TiGTrack2DBenchmarkSpec | None = None,
) -> TiGTrack2DBenchmarkReport:
    """Evaluate both mmwcore strategies under one people-counting contract."""

    return run_ti_people_counting_2d(frames, spec).report


def run_ti_people_counting_2d(
    frames: tuple[TiGTrack2DVectorFrame, ...],
    spec: TiGTrack2DBenchmarkSpec | None = None,
) -> TiGTrack2DBenchmarkRun:
    """Run both strategies while retaining their frame-level predictions."""

    selected_spec = spec or ti_people_counting_2d_benchmark_spec()
    cluster = run_cluster_tracker_on_ti_vectors(
        frames,
        selected_spec.clustering,
        selected_spec.tracker,
        match_distance_m=selected_spec.match_distance_m,
    )
    measurement = run_measurement_tracker_on_ti_vectors(
        frames,
        selected_spec.clustering,
        selected_spec.tracker,
        match_distance_m=selected_spec.match_distance_m,
    )
    report = TiGTrack2DBenchmarkReport(
        scenario="ti_gtrack_people_counting_2d",
        num_frames=len(frames),
        first_frame_number=frames[0].frame_number if frames else None,
        last_frame_number=frames[-1].frame_number if frames else None,
        spec=selected_spec,
        cluster=cluster.comparison,
        measurement=measurement.comparison,
    )
    return TiGTrack2DBenchmarkRun(report=report, cluster=cluster, measurement=measurement)


def benchmark_cluster_tracker_on_ti_vectors(
    frames: tuple[TiGTrack2DVectorFrame, ...],
    clustering: DBSCANSpec,
    tracker: Tracker2DSpec,
    *,
    match_distance_m: float,
) -> TiGTrackComparison:
    """Run the cluster-level baseline against parsed TI reference targets."""

    return run_cluster_tracker_on_ti_vectors(
        frames, clustering, tracker, match_distance_m=match_distance_m
    ).comparison


def run_cluster_tracker_on_ti_vectors(
    frames: tuple[TiGTrack2DVectorFrame, ...],
    clustering: DBSCANSpec,
    tracker: Tracker2DSpec,
    *,
    match_distance_m: float,
) -> TiGTrackStrategyRun:
    """Run the cluster-level tracker and retain its frame predictions."""

    stateful_tracker = ClusterTracker2D(tracker)
    predictions: list[TrackFrame] = []
    for frame in frames:
        point_cloud = _point_cloud(frame, tracker.frame_period_s)
        clusters = cluster_point_cloud(point_cloud, clustering)
        predictions.append(stateful_tracker.step(clusters))
    return TiGTrackStrategyRun(
        frames=tuple(predictions),
        comparison=_compare(predictions, frames, match_distance_m),
    )


def benchmark_measurement_tracker_on_ti_vectors(
    frames: tuple[TiGTrack2DVectorFrame, ...],
    allocation_clustering: DBSCANSpec,
    tracker: Tracker2DSpec,
    *,
    match_distance_m: float,
) -> TiGTrackComparison:
    """Run the measurement-level tracker against parsed TI reference targets."""

    return run_measurement_tracker_on_ti_vectors(
        frames, allocation_clustering, tracker, match_distance_m=match_distance_m
    ).comparison


def run_measurement_tracker_on_ti_vectors(
    frames: tuple[TiGTrack2DVectorFrame, ...],
    allocation_clustering: DBSCANSpec,
    tracker: Tracker2DSpec,
    *,
    match_distance_m: float,
) -> TiGTrackStrategyRun:
    """Run the measurement-level tracker and retain its frame predictions."""

    stateful_tracker = PointTracker2D(tracker, allocation_clustering)
    predictions = [
        stateful_tracker.step(_point_cloud(frame, tracker.frame_period_s)) for frame in frames
    ]
    return TiGTrackStrategyRun(
        frames=tuple(predictions),
        comparison=_compare(predictions, frames, match_distance_m),
    )


def _point_cloud(frame: TiGTrack2DVectorFrame, frame_period_s: float) -> PointCloudFrame:
    return PointCloudFrame(
        frame.cartesian_points(),
        channels=("x", "y", "z", "velocity", "snr"),
        frame_id=frame.frame_number,
        timestamp=frame.frame_number * frame_period_s,
        source="ti_gtrack_2d_vector",
    )


def _compare(
    predictions: list[TrackFrame],
    frames: tuple[TiGTrack2DVectorFrame, ...],
    match_distance_m: float,
) -> TiGTrackComparison:
    ground_truth = [frame.ground_truth for frame in frames]
    return TiGTrackComparison(
        all_tracks=evaluate_track_frames(
            predictions,
            ground_truth,
            match_distance_m=match_distance_m,
            confirmed_only=False,
        ),
        confirmed_tracks=evaluate_track_frames(
            predictions,
            ground_truth,
            match_distance_m=match_distance_m,
            confirmed_only=True,
        ),
    )


def read_ti_gtrack_2d_vectors(path: str | Path) -> tuple[TiGTrack2DVectorFrame, ...]:
    """Read little-endian frames produced by the SDK GTRACK 2D test harness."""

    data = memoryview(Path(path).read_bytes())
    offset = 0
    frames: list[TiGTrack2DVectorFrame] = []
    while offset < len(data):
        frame_number, num_tlvs, offset = _unpack_header(data, offset, "frame header")
        measurements = np.empty((0, 4), dtype=np.float32)
        ground_truth = TrackingGroundTruthFrame(
            track_ids=np.empty(0, dtype=np.int64),
            positions=np.empty((0, 3), dtype=np.float32),
        )
        seen_types: set[int] = set()
        for _ in range(num_tlvs):
            tlv_type, length, offset = _unpack_header(data, offset, "TLV header")
            payload, offset = _read_payload(data, offset, length)
            if tlv_type in seen_types:
                raise ValueError(f"TI GTRACK frame {frame_number} repeats TLV type {tlv_type}.")
            seen_types.add(tlv_type)
            if tlv_type == _POINT_CLOUD_TLV:
                measurements = _parse_measurements(payload)
            elif tlv_type == _TARGET_LIST_TLV:
                ground_truth = _parse_ground_truth(payload)
        frames.append(
            TiGTrack2DVectorFrame(
                frame_number=frame_number,
                measurements=measurements,
                ground_truth=ground_truth,
            )
        )
    return tuple(frames)


def _unpack_header(
    data: memoryview,
    offset: int,
    label: str,
) -> tuple[int, int, int]:
    end = offset + _HEADER.size
    if end > len(data):
        raise ValueError(f"TI GTRACK vector has a truncated {label}.")
    first, second = _HEADER.unpack(data[offset:end])
    return first, second, end


def _read_payload(data: memoryview, offset: int, length: int) -> tuple[memoryview, int]:
    end = offset + length
    if end > len(data):
        raise ValueError("TI GTRACK vector has a truncated TLV payload.")
    return data[offset:end], end


def _parse_measurements(payload: memoryview) -> np.ndarray:
    if len(payload) % _MEASUREMENT.itemsize:
        raise ValueError("TI GTRACK measurement TLV length is not a whole record count.")
    return np.frombuffer(payload, dtype=_MEASUREMENT).reshape(-1, 4).copy()


def _parse_ground_truth(payload: memoryview) -> TrackingGroundTruthFrame:
    if len(payload) % _REFERENCE.itemsize:
        raise ValueError("TI GTRACK target TLV length is not a whole record count.")
    records = np.frombuffer(payload, dtype=_REFERENCE)
    positions = np.zeros((records.size, 3), dtype=np.float32)
    positions[:, 0] = records["state"][:, 1]
    positions[:, 1] = records["state"][:, 0]
    return TrackingGroundTruthFrame(
        track_ids=records["track_id"].astype(np.int64),
        positions=positions,
    )
