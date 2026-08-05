from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import (
    DBSCANClusteringSpec,
    PointCloudFrame,
    TrackAllocationSpec,
    Tracker2DSpec,
    TrackGatingSpec,
    TrackLifecycleSpec,
)
from mmwcore.tracking import MeasurementTracker2D


def _points(*points: tuple[float, ...], frame_id: int = 0) -> PointCloudFrame:
    channel_options = {
        3: ("x", "y", "z"),
        4: ("x", "y", "z", "velocity"),
        5: ("x", "y", "z", "velocity", "snr"),
    }
    channels = channel_options[len(points[0]) if points else 3]
    return PointCloudFrame(
        points=np.asarray(points, dtype=np.float32).reshape(len(points), len(channels)),
        channels=channels,
        frame_id=frame_id,
        timestamp=frame_id * 0.1,
    )


def _tracker(*, velocity_gate: float | None = None) -> MeasurementTracker2D:
    return MeasurementTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=TrackGatingSpec(
                max_distance_m=0.5,
                max_radial_velocity_difference_mps=velocity_gate,
            ),
            lifecycle=TrackLifecycleSpec(confirmation_hits=1),
        ),
        DBSCANClusteringSpec(eps_m=0.2, min_samples=2, use_z=False),
    )


def test_measurement_tracker_associates_multiple_points_to_one_track() -> None:
    tracker = _tracker()
    first = tracker.step(_points((-0.05, 1.0, 0.0), (0.0, 1.0, 0.0), (0.05, 1.0, 0.0)))
    second = tracker.step(_points((0.05, 1.0, 0.0), (0.1, 1.0, 0.0), (0.15, 1.0, 0.0), frame_id=1))

    assert first.track_ids.tolist() == [0]
    assert first.observation_track_ids.tolist() == [0, 0, 0]
    assert second.track_ids.tolist() == [0]
    assert second.observation_track_ids.tolist() == [0, 0, 0]
    assert second.positions[0, 0] > first.positions[0, 0]
    assert second.position_covariances.shape == (1, 2, 2)
    assert second.extent_covariances[0, 0, 0] > 0


def test_measurement_tracker_partitions_points_between_existing_tracks() -> None:
    tracker = _tracker()
    tracker.step(
        _points(
            (-1.05, 1.0, 0.0),
            (-0.95, 1.0, 0.0),
            (0.95, 1.0, 0.0),
            (1.05, 1.0, 0.0),
        )
    )

    frame = tracker.step(
        _points(
            (-0.9, 1.0, 0.0),
            (-0.8, 1.0, 0.0),
            (0.8, 1.0, 0.0),
            (0.9, 1.0, 0.0),
            frame_id=1,
        )
    )

    assert frame.track_ids.tolist() == [0, 1]
    assert frame.observation_track_ids.tolist() == [0, 0, 1, 1]


def test_measurement_tracker_requires_velocity_for_velocity_gate() -> None:
    tracker = _tracker(velocity_gate=0.5)

    with pytest.raises(ValueError, match="velocity constraints"):
        tracker.step(_points((0.0, 1.0, 0.0), (0.1, 1.0, 0.0)))


def test_measurement_tracker_limits_births_to_strongest_candidate() -> None:
    tracker = MeasurementTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=TrackGatingSpec(max_distance_m=0.5),
            allocation=TrackAllocationSpec(max_new_tracks_per_frame=1),
        ),
        DBSCANClusteringSpec(eps_m=0.2, min_samples=2, use_z=False),
    )

    frame = tracker.step(
        _points(
            (-1.05, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-0.95, 1.0, 0.0),
            (0.95, 1.0, 0.0),
            (1.05, 1.0, 0.0),
        )
    )

    assert frame.num_tracks == 1
    assert frame.observation_track_ids.tolist() == [0, 0, 0, -1, -1]


def test_measurement_tracker_adapts_gate_to_target_spread() -> None:
    def adaptive_tracker() -> MeasurementTracker2D:
        return MeasurementTracker2D(
            Tracker2DSpec(
                frame_period_s=0.1,
                gating=TrackGatingSpec(
                    max_distance_m=0.8,
                    max_mahalanobis_distance=1.1,
                ),
                lifecycle=TrackLifecycleSpec(confirmation_hits=1),
            ),
            DBSCANClusteringSpec(eps_m=1.1, min_samples=2, use_z=False),
        )

    narrow = adaptive_tracker()
    narrow.step(_points((-0.02, 1.0, 0.0), (0.02, 1.0, 0.0)))
    narrow_frame = narrow.step(_points((0.48, 1.0, 0.0), (0.52, 1.0, 0.0), frame_id=1))

    wide = adaptive_tracker()
    wide.step(_points((-0.5, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 1.0, 0.0)))
    wide_frame = wide.step(_points((0.48, 1.0, 0.0), (0.52, 1.0, 0.0), frame_id=1))

    assert narrow_frame.observation_track_ids.tolist() == [1, 1]
    assert wide_frame.observation_track_ids.tolist() == [0, 0]


def test_measurement_tracker_applies_total_snr_allocation_threshold() -> None:
    tracker = MeasurementTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=TrackGatingSpec(max_distance_m=0.5),
            allocation=TrackAllocationSpec(min_total_snr=10.0),
        ),
        DBSCANClusteringSpec(eps_m=0.2, min_samples=2, use_z=False),
    )

    weak = tracker.step(_points((0.0, 1.0, 0.0, 0.1, 4.0), (0.1, 1.0, 0.0, 0.1, 5.0)))
    strong = tracker.step(_points((0.0, 1.0, 0.0, 0.1, 5.0), (0.1, 1.0, 0.0, 0.1, 5.0), frame_id=1))

    assert weak.num_tracks == 0
    assert strong.track_ids.tolist() == [0]
    assert strong.observation_track_ids.tolist() == [0, 0]


def test_measurement_tracker_requires_snr_channel_for_snr_allocation() -> None:
    tracker = MeasurementTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=TrackGatingSpec(max_distance_m=0.5),
            allocation=TrackAllocationSpec(min_total_snr=10.0),
        ),
        DBSCANClusteringSpec(eps_m=0.2, min_samples=2, use_z=False),
    )

    with pytest.raises(ValueError, match="min_total_snr"):
        tracker.step(_points((0.0, 1.0, 0.0), (0.1, 1.0, 0.0)))
