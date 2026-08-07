from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import TrackFrame, TrackStatus
from mmwcore.tracking import TrackingGroundTruthFrame, evaluate_track_frames


def _prediction(ids: tuple[int, ...], positions: list[tuple[float, float, float]]) -> TrackFrame:
    count = len(ids)
    return TrackFrame(
        track_ids=np.array(ids, dtype=np.int64),
        positions=np.array(positions, dtype=np.float32).reshape(count, 3),
        velocities=np.zeros((count, 3)),
        position_covariances=np.zeros((count, 2, 2)),
        extent_covariances=np.zeros((count, 2, 2)),
        statuses=(TrackStatus.CONFIRMED,) * count,
        ages=np.ones(count, dtype=np.int64),
        missed_counts=np.zeros(count, dtype=np.int64),
        observation_track_ids=np.empty(0, dtype=np.int64),
    )


def _truth(
    ids: tuple[int, ...],
    positions: list[tuple[float, float, float]],
) -> TrackingGroundTruthFrame:
    return TrackingGroundTruthFrame(
        track_ids=np.array(ids),
        positions=np.array(positions, dtype=np.float32).reshape(len(ids), 3),
    )


def test_evaluate_track_frames_reports_perfect_sequence() -> None:
    predictions = [
        _prediction((10,), [(0.0, 1.0, 0.0)]),
        _prediction((10,), [(0.1, 1.0, 0.0)]),
    ]
    truth = [
        _truth((1,), [(0.0, 1.0, 0.0)]),
        _truth((1,), [(0.1, 1.0, 0.0)]),
    ]

    summary = evaluate_track_frames(predictions, truth, match_distance_m=0.5)

    assert summary.matched_observations == 2
    assert summary.missed_observations == 0
    assert summary.false_track_observations == 0
    assert summary.id_switches == 0
    assert summary.position_rmse_m == pytest.approx(0.0)
    assert summary.recall == pytest.approx(1.0)


def test_evaluate_track_frames_detects_identity_switches() -> None:
    predictions = [
        _prediction((10, 20), [(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]),
        _prediction((20, 10), [(-0.5, 1.0, 0.0), (0.5, 1.0, 0.0)]),
    ]
    truth = [
        _truth((1, 2), [(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]),
        _truth((1, 2), [(-0.5, 1.0, 0.0), (0.5, 1.0, 0.0)]),
    ]

    summary = evaluate_track_frames(predictions, truth, match_distance_m=0.25)

    assert summary.id_switches == 2
    assert [event.frame_index for event in summary.identity_switches] == [1, 1]
    assert {event.ground_truth_id for event in summary.identity_switches} == {1, 2}


def test_evaluate_track_frames_counts_misses_and_false_tracks() -> None:
    summary = evaluate_track_frames(
        [_prediction((10,), [(4.0, 4.0, 0.0)])],
        [_truth((1,), [(0.0, 1.0, 0.0)])],
        match_distance_m=0.5,
    )

    assert summary.matched_observations == 0
    assert summary.missed_observations == 1
    assert summary.false_track_observations == 1
    assert summary.position_rmse_m is None
