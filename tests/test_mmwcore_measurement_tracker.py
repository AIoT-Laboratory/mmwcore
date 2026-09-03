from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import (
    AllocationSpec,
    DBSCANSpec,
    GatingSpec,
    LifecycleSpec,
    PointCloudFrame,
    Tracker2DSpec,
)
from mmwcore.tracking import PointTracker2D


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


def _tracker(*, velocity_gate: float | None = None) -> PointTracker2D:
    return PointTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(
                max_distance_m=0.5,
                max_radial_velocity_difference_mps=velocity_gate,
            ),
            lifecycle=LifecycleSpec(confirmation_hits=1),
        ),
        DBSCANSpec(eps_m=0.2, min_samples=2, use_z=False),
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
    tracker = PointTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            allocation=AllocationSpec(max_new_tracks_per_frame=1),
        ),
        DBSCANSpec(eps_m=0.2, min_samples=2, use_z=False),
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
    def adaptive_tracker() -> PointTracker2D:
        return PointTracker2D(
            Tracker2DSpec(
                frame_period_s=0.1,
                gating=GatingSpec(
                    max_distance_m=0.8,
                    max_mahalanobis_distance=1.1,
                ),
                lifecycle=LifecycleSpec(confirmation_hits=1),
            ),
            DBSCANSpec(eps_m=1.1, min_samples=2, use_z=False),
        )

    narrow = adaptive_tracker()
    narrow.step(_points((-0.02, 1.0, 0.0), (0.02, 1.0, 0.0)))
    narrow_frame = narrow.step(_points((0.48, 1.0, 0.0), (0.52, 1.0, 0.0), frame_id=1))

    wide = adaptive_tracker()
    wide.step(_points((-0.5, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 1.0, 0.0)))
    wide_frame = wide.step(_points((0.48, 1.0, 0.0), (0.52, 1.0, 0.0), frame_id=1))

    assert narrow_frame.observation_track_ids.tolist() == [1, 1]
    assert wide_frame.observation_track_ids.tolist().count(0) > 0


def test_measurement_tracker_uses_centroid_uncertainty_without_mahalanobis_gate() -> None:
    def tracker() -> PointTracker2D:
        return PointTracker2D(
            Tracker2DSpec(
                frame_period_s=0.1,
                gating=GatingSpec(max_distance_m=1.5),
                lifecycle=LifecycleSpec(confirmation_hits=1),
            ),
            DBSCANSpec(eps_m=0.2, min_samples=2, use_z=False),
        )

    narrow = tracker()
    narrow.step(_points((-0.05, 1.0, 0.0), (0.05, 1.0, 0.0)))
    narrow_frame = narrow.step(_points((0.49, 1.0, 0.0), (0.51, 1.0, 0.0), frame_id=1))

    wide = tracker()
    wide.step(_points((-0.05, 1.0, 0.0), (0.05, 1.0, 0.0)))
    wide_frame = wide.step(_points((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), frame_id=1))

    assert narrow_frame.positions[0, 0] > wide_frame.positions[0, 0]
    assert narrow_frame.position_covariances[0, 0, 0] < wide_frame.position_covariances[0, 0, 0]


def test_measurement_tracker_does_not_collapse_extent_on_one_point_update() -> None:
    tracker = _tracker()
    allocated = tracker.step(_points((-0.1, 1.0, 0.0), (0.0, 1.0, 0.0), (0.1, 1.0, 0.0)))

    updated = tracker.step(_points((0.0, 1.0, 0.0), frame_id=1))

    np.testing.assert_allclose(updated.extent_covariances, allocated.extent_covariances)


def test_measurement_tracker_applies_total_snr_allocation_threshold() -> None:
    tracker = PointTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            allocation=AllocationSpec(min_total_snr=10.0),
        ),
        DBSCANSpec(eps_m=0.2, min_samples=2, use_z=False),
    )

    weak = tracker.step(_points((0.0, 1.0, 0.0, 0.1, 4.0), (0.1, 1.0, 0.0, 0.1, 5.0)))
    strong = tracker.step(_points((0.0, 1.0, 0.0, 0.1, 5.0), (0.1, 1.0, 0.0, 0.1, 5.0), frame_id=1))

    assert weak.num_tracks == 0
    assert strong.track_ids.tolist() == [0]
    assert strong.observation_track_ids.tolist() == [0, 0]


def test_measurement_tracker_requires_snr_channel_for_snr_allocation() -> None:
    tracker = PointTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            allocation=AllocationSpec(min_total_snr=10.0),
        ),
        DBSCANSpec(eps_m=0.2, min_samples=2, use_z=False),
    )

    with pytest.raises(ValueError, match="min_total_snr"):
        tracker.step(_points((0.0, 1.0, 0.0), (0.1, 1.0, 0.0)))


def test_gtrack_converts_db_snr_to_linear_power_for_allocation() -> None:
    tracker = PointTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            allocation=AllocationSpec(min_total_snr=19.0),
        ),
        DBSCANSpec(eps_m=0.2, min_samples=2, use_z=False),
    )
    cloud = PointCloudFrame(
        points=np.asarray(
            [[1.0, -0.05, 0.0, 0.1, 10.0], [1.0, 0.05, 0.0, 0.1, 10.0]],
            dtype=np.float32,
        ),
        channels=("x", "y", "z", "velocity", "snr_db"),
    )

    frame = tracker.step(cloud)

    assert frame.track_ids.tolist() == [0]
