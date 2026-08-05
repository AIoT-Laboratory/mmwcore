from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import (
    TrackAllocationSpec,
    Tracker2DSpec,
    TrackFrame,
    TrackGatingSpec,
    TrackingBox2D,
    TrackLifecycleSpec,
    TrackScenerySpec,
    TrackStatus,
)


def test_cluster_tracker_spec_keeps_explicit_timing_and_lifecycle() -> None:
    spec = Tracker2DSpec(
        frame_period_s=0.1,
        gating=TrackGatingSpec(
            max_distance_m=0.8,
            max_radial_velocity_difference_mps=1.5,
        ),
        allocation=TrackAllocationSpec(min_points=3),
        lifecycle=TrackLifecycleSpec(
            confirmation_hits=4,
            tentative_max_misses=2,
            confirmed_max_misses=10,
        ),
    )

    assert spec.frame_period_s == pytest.approx(0.1)
    assert spec.gating.max_distance_m == pytest.approx(0.8)
    assert spec.allocation.min_points == 3
    assert spec.lifecycle.confirmation_hits == 4


def test_track_allocation_spec_rejects_non_positive_per_frame_limit() -> None:
    with pytest.raises(ValueError, match="max_new_tracks_per_frame"):
        TrackAllocationSpec(max_new_tracks_per_frame=0)


def test_track_allocation_spec_rejects_non_positive_snr_threshold() -> None:
    with pytest.raises(ValueError, match="min_total_snr"):
        TrackAllocationSpec(min_total_snr=0.0)


def test_track_gating_spec_rejects_non_positive_mahalanobis_limit() -> None:
    with pytest.raises(ValueError, match="max_mahalanobis_distance"):
        TrackGatingSpec(max_distance_m=1.0, max_mahalanobis_distance=0.0)


def test_track_frame_normalizes_state_and_associations() -> None:
    frame = TrackFrame(
        track_ids=np.array([3, 8]),
        positions=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
        velocities=np.array([[0.1, 0.2, 0.0], [0.0, -0.1, 0.0]]),
        position_covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        extent_covariances=np.zeros((2, 2, 2)),
        statuses=(TrackStatus.CONFIRMED, TrackStatus.COASTING),
        ages=np.array([10, 4]),
        missed_counts=np.array([0, 1]),
        observation_track_ids=np.array([3, -1, 8]),
        frame_id=5,
        timestamp=0.5,
    )

    assert frame.num_tracks == 2
    assert frame.statuses == (TrackStatus.CONFIRMED, TrackStatus.COASTING)
    assert frame.observation_track_ids.tolist() == [3, -1, 8]


def test_track_frame_rejects_unknown_associated_track() -> None:
    with pytest.raises(ValueError, match="unknown track"):
        TrackFrame(
            track_ids=np.array([1]),
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
            position_covariances=np.zeros((1, 2, 2)),
            extent_covariances=np.zeros((1, 2, 2)),
            statuses=(TrackStatus.TENTATIVE,),
            ages=np.array([1]),
            missed_counts=np.array([0]),
            observation_track_ids=np.array([2]),
        )


def test_track_frame_rejects_non_positive_semidefinite_covariance() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        TrackFrame(
            track_ids=np.array([1]),
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
            position_covariances=np.array([[[1.0, 0.0], [0.0, -1.0]]]),
            extent_covariances=np.zeros((1, 2, 2)),
            statuses=(TrackStatus.TENTATIVE,),
            ages=np.array([1]),
            missed_counts=np.array([0]),
            observation_track_ids=np.array([1]),
        )


def test_tracking_scenery_accepts_any_configured_boundary_box() -> None:
    scenery = TrackScenerySpec(
        boundary_boxes=(
            TrackingBox2D(-1.0, 1.0, 0.0, 2.0),
            TrackingBox2D(2.0, 3.0, 4.0, 5.0),
        ),
        outside_max_frames=3,
    )

    assert scenery.contains(0.0, 1.0)
    assert scenery.contains(2.5, 4.5)
    assert not scenery.contains(0.0, 3.0)
