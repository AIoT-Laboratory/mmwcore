from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import (
    AllocationSpec,
    Box2D,
    ClusterFrame,
    GatingSpec,
    LifecycleSpec,
    ScenerySpec,
    Tracker2DSpec,
    TrackStatus,
)
from mmwcore.tracking import (
    ClusterTracker2D,
    TrackingGroundTruthFrame,
    evaluate_track_frames,
)


def _clusters(*centers: tuple[float, float, float], frame_id: int = 0) -> ClusterFrame:
    count = len(centers)
    return ClusterFrame(
        centers=np.asarray(centers, dtype=np.float32).reshape(count, 3),
        extents=np.zeros((count, 3), dtype=np.float32),
        mean_velocities=np.zeros(count, dtype=np.float32),
        point_counts=np.ones(count, dtype=np.int64),
        point_labels=np.arange(count, dtype=np.int64),
        frame_id=frame_id,
        timestamp=frame_id * 0.1,
    )


def test_cluster_tracker_confirms_and_estimates_motion() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            lifecycle=LifecycleSpec(
                confirmation_hits=3,
                tentative_max_misses=2,
                confirmed_max_misses=3,
            ),
        )
    )

    first = tracker.step(_clusters((0.0, 1.0, 0.0), frame_id=0))
    tracker.step(_clusters((0.1, 1.0, 0.0), frame_id=1))
    third = tracker.step(_clusters((0.2, 1.0, 0.0), frame_id=2))

    assert first.statuses == (TrackStatus.TENTATIVE,)
    assert third.statuses == (TrackStatus.CONFIRMED,)
    assert third.track_ids.tolist() == [0]
    assert third.observation_track_ids.tolist() == [0]
    assert third.velocities[0, 0] > 0


def test_cluster_tracker_rejects_snr_allocation_policy() -> None:
    with pytest.raises(ValueError, match="PointTracker2D"):
        ClusterTracker2D(
            Tracker2DSpec(
                frame_period_s=0.1,
                gating=GatingSpec(max_distance_m=0.5),
                allocation=AllocationSpec(min_total_snr=10.0),
            )
        )


def test_cluster_tracker_coasts_and_deletes_confirmed_track() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=1.0),
            lifecycle=LifecycleSpec(
                confirmation_hits=1,
                tentative_max_misses=1,
                confirmed_max_misses=2,
            ),
        )
    )
    tracker.step(_clusters((0.0, 1.0, 0.0)))

    coasting = tracker.step(_clusters(frame_id=1))
    deleted = tracker.step(_clusters(frame_id=2))

    assert coasting.statuses == (TrackStatus.COASTING,)
    assert coasting.missed_counts.tolist() == [1]
    assert deleted.num_tracks == 0


def test_cluster_tracker_deletes_unconfirmed_track_after_misses() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=1.0),
            lifecycle=LifecycleSpec(
                confirmation_hits=3,
                tentative_max_misses=1,
                confirmed_max_misses=3,
            ),
        )
    )
    tracker.step(_clusters((0.0, 1.0, 0.0)))

    frame = tracker.step(_clusters(frame_id=1))

    assert frame.num_tracks == 0


def test_cluster_tracker_requires_consecutive_hits_for_confirmation() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=1.0),
            lifecycle=LifecycleSpec(
                confirmation_hits=3,
                tentative_max_misses=2,
                confirmed_max_misses=3,
            ),
        )
    )

    tracker.step(_clusters((0.0, 1.0, 0.0), frame_id=0))
    tracker.step(_clusters(frame_id=1))
    tracker.step(_clusters((0.0, 1.0, 0.0), frame_id=2))
    before_confirmation = tracker.step(_clusters((0.0, 1.0, 0.0), frame_id=3))
    confirmed = tracker.step(_clusters((0.0, 1.0, 0.0), frame_id=4))

    assert before_confirmation.statuses == (TrackStatus.TENTATIVE,)
    assert confirmed.statuses == (TrackStatus.CONFIRMED,)


def test_cluster_tracker_reconfirms_coasting_track_after_one_hit() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=1.0),
            lifecycle=LifecycleSpec(
                confirmation_hits=3,
                tentative_max_misses=2,
                confirmed_max_misses=3,
            ),
        )
    )
    for frame_id in range(3):
        tracker.step(_clusters((0.0, 1.0, 0.0), frame_id=frame_id))

    coasting = tracker.step(_clusters(frame_id=3))
    recovered = tracker.step(_clusters((0.0, 1.0, 0.0), frame_id=4))

    assert coasting.statuses == (TrackStatus.COASTING,)
    assert recovered.statuses == (TrackStatus.CONFIRMED,)


def test_cluster_tracker_uses_global_one_to_one_association() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=1.0),
            lifecycle=LifecycleSpec(
                confirmation_hits=1,
                tentative_max_misses=1,
                confirmed_max_misses=2,
            ),
        )
    )
    tracker.step(_clusters((0.0, 1.0, 0.0), (2.0, 1.0, 0.0)))

    frame = tracker.step(_clusters((1.8, 1.0, 0.0), (0.2, 1.0, 0.0), frame_id=1))

    assert frame.observation_track_ids.tolist() == [1, 0]


def test_cluster_tracker_rejects_allocations_outside_scenery() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=1.0),
            scenery=ScenerySpec(
                boundary_boxes=(Box2D(-1.0, 1.0, 0.0, 2.0),),
            ),
        )
    )

    frame = tracker.step(_clusters((0.0, 1.0, 0.0), (0.0, 3.0, 0.0)))

    assert frame.num_tracks == 1
    assert frame.observation_track_ids.tolist() == [0, -1]


def test_cluster_tracker_deletes_track_after_repeated_predictions_outside_scenery() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=1.0,
            gating=GatingSpec(max_distance_m=2.0),
            lifecycle=LifecycleSpec(confirmation_hits=1, confirmed_max_misses=10),
            scenery=ScenerySpec(
                boundary_boxes=(Box2D(-1.0, 1.0, 0.0, 2.0),),
                outside_max_frames=2,
            ),
        )
    )
    tracker.step(_clusters((0.0, 1.0, 0.0)))
    tracker.step(_clusters((0.8, 1.0, 0.0), frame_id=1))
    tracker.step(_clusters(frame_id=2))

    frame = tracker.step(_clusters(frame_id=3))

    assert frame.num_tracks == 0


def test_cluster_tracker_preserves_ids_through_two_target_crossing() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=1.0,
            gating=GatingSpec(max_distance_m=0.6),
            lifecycle=LifecycleSpec(confirmation_hits=1),
        )
    )
    predictions = []
    truth = []
    for frame_index in range(11):
        left_to_right = -1.0 + 0.2 * frame_index
        right_to_left = 1.0 - 0.2 * frame_index
        centers = (
            (left_to_right, 1.0, 0.0),
            (right_to_left, 1.0, 0.0),
        )
        predictions.append(tracker.step(_clusters(*centers, frame_id=frame_index)))
        truth.append(
            TrackingGroundTruthFrame(
                track_ids=np.array([100, 200]),
                positions=np.array(centers, dtype=np.float32),
            )
        )

    summary = evaluate_track_frames(predictions, truth, match_distance_m=0.5)

    assert summary.id_switches == 0
    assert summary.missed_observations == 0


def test_cluster_tracker_recovers_same_id_after_temporary_occlusion() -> None:
    tracker = ClusterTracker2D(
        Tracker2DSpec(
            frame_period_s=1.0,
            gating=GatingSpec(max_distance_m=0.4),
            lifecycle=LifecycleSpec(
                confirmation_hits=1,
                confirmed_max_misses=3,
            ),
        )
    )
    observations = (0.0, 0.1, 0.2, None, None, 0.5)
    predictions = []
    truth = []
    for frame_index, x_position in enumerate(observations):
        clusters = (
            _clusters(frame_id=frame_index)
            if x_position is None
            else _clusters((x_position, 1.0, 0.0), frame_id=frame_index)
        )
        predictions.append(tracker.step(clusters))
        truth.append(
            TrackingGroundTruthFrame(
                track_ids=np.array([100]),
                positions=np.array([[0.1 * frame_index, 1.0, 0.0]], dtype=np.float32),
            )
        )

    summary = evaluate_track_frames(predictions, truth, match_distance_m=0.4)

    assert predictions[-1].track_ids.tolist() == [0]
    assert predictions[-1].statuses == (TrackStatus.CONFIRMED,)
    assert summary.id_switches == 0
    assert summary.missed_observations == 2
