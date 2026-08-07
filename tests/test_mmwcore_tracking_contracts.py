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


def _single_track_frame(**integer_fields: np.ndarray) -> TrackFrame:
    fields = {
        "track_ids": np.array([1], dtype=np.int64),
        "ages": np.array([1], dtype=np.int64),
        "missed_counts": np.array([0], dtype=np.int64),
        "observation_track_ids": np.array([1], dtype=np.int64),
    }
    fields.update(integer_fields)
    return TrackFrame(
        track_ids=fields["track_ids"],
        positions=np.zeros((1, 3)),
        velocities=np.zeros((1, 3)),
        position_covariances=np.zeros((1, 2, 2)),
        extent_covariances=np.zeros((1, 2, 2)),
        statuses=(TrackStatus.TENTATIVE,),
        ages=fields["ages"],
        missed_counts=fields["missed_counts"],
        observation_track_ids=fields["observation_track_ids"],
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
        track_ids=np.array([3, 8], dtype=np.uint8),
        positions=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
        velocities=np.array([[0.1, 0.2, 0.0], [0.0, -0.1, 0.0]]),
        position_covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        extent_covariances=np.zeros((2, 2, 2)),
        statuses=(TrackStatus.CONFIRMED, TrackStatus.COASTING),
        ages=np.array([10, 4], dtype=np.uint16),
        missed_counts=np.array([0, 1], dtype=np.uint32),
        observation_track_ids=np.array([3, -1, 8], dtype=np.int16),
        frame_id=5,
        timestamp=0.5,
    )

    assert frame.num_tracks == 2
    assert frame.statuses == (TrackStatus.CONFIRMED, TrackStatus.COASTING)
    assert frame.observation_track_ids.tolist() == [3, -1, 8]
    assert frame.track_ids.dtype == np.dtype(np.int64)
    assert frame.ages.dtype == np.dtype(np.int64)
    assert frame.missed_counts.dtype == np.dtype(np.int64)
    assert frame.observation_track_ids.dtype == np.dtype(np.int64)


@pytest.mark.parametrize(
    ("field_name", "values"),
    [
        (field_name, values)
        for field_name in (
            "track_ids",
            "ages",
            "missed_counts",
            "observation_track_ids",
        )
        for values in (np.array([1.0]), np.array([True]))
    ],
)
def test_track_frame_rejects_non_integer_semantic_fields(
    field_name: str,
    values: np.ndarray,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"TrackFrame\.{field_name} must contain integer values",
    ):
        _single_track_frame(**{field_name: values})


@pytest.mark.parametrize(
    "field_name",
    ["track_ids", "ages", "missed_counts", "observation_track_ids"],
)
def test_track_frame_rejects_integer_values_outside_int64(field_name: str) -> None:
    values = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)

    with pytest.raises(
        ValueError,
        match=rf"TrackFrame\.{field_name} contains values outside the int64 range",
    ):
        _single_track_frame(**{field_name: values})


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
