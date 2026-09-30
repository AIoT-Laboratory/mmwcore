from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mmwcore.core import TrackFrame, TrackStatus
from mmwcore.tracking import (
    ScatterBodyTracker,
    TrackingGroundTruthFrame,
    TrackingPredictionFrame,
    evaluate_track_frames,
)


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


def test_matching_maximizes_valid_pairs_before_minimizing_distance() -> None:
    # Distances [[0.1, 0.9], [0.9, 1.1]]: unconstrained assignment loses a valid pair.
    summary = evaluate_track_frames(
        [_prediction((10, 20), [(2.1, 0, 0), (1.1, 0, 0)])],
        [_truth((1, 2), [(2, 0, 0), (1.8, np.sqrt(0.72), 0)])],
        match_distance_m=1,
    )
    assert summary.matched_observations == 2
    assert summary.missed_observations == summary.false_track_observations == 0
    assert summary.position_rmse_m == pytest.approx(0.9)


@pytest.mark.parametrize("extra_prediction", [False, True])
def test_matching_handles_rectangular_inputs_and_minimizes_valid_distance(
    extra_prediction: bool,
) -> None:
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    extra = [*positions, (10.0, 0.0, 0.0)]
    predictions, truth = (extra, positions) if extra_prediction else (positions, extra)
    summary = evaluate_track_frames(
        [_prediction(tuple(range(len(predictions))), predictions)],
        [_truth(tuple(range(len(truth))), truth)],
        match_distance_m=2,
    )
    assert summary.matched_observations == 2
    assert summary.position_rmse_m == 0
    assert summary.false_track_observations == int(extra_prediction)
    assert summary.missed_observations == int(not extra_prediction)


@pytest.mark.parametrize("predicted,actual", [(0, 0), (1, 0), (0, 1)])
def test_matching_handles_empty_frames(predicted: int, actual: int) -> None:
    summary = evaluate_track_frames(
        [_prediction((1,) * predicted, [(0, 0, 0)] * predicted)],
        [_truth((1,) * actual, [(0, 0, 0)] * actual)],
        match_distance_m=1,
    )
    assert summary.matched_observations == 0
    assert summary.false_track_observations == predicted
    assert summary.missed_observations == actual


@pytest.mark.parametrize("distance", [0, -1, float("nan"), float("inf")])
def test_matching_rejects_invalid_threshold(distance: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_track_frames([], [], match_distance_m=distance)


def test_xy_comparison_does_not_score_display_height_as_filtered_state():
    truth = [_truth((1,), [(2, 0, 1)])]
    prediction = _prediction((10,), [(2, 0, 7)])
    xy = evaluate_track_frames([prediction], truth, match_distance_m=0.5, dimensions=2)
    xyz = evaluate_track_frames([prediction], truth, match_distance_m=0.5, dimensions=3)
    assert xy.matched_observations == 1 and xy.position_rmse_m == 0
    assert xyz.matched_observations == 0
    assert xy.to_record()["dimensions"] == 2
    assert xy.to_record()["coordinate_frame"] == "radar"


def test_published_continuity_includes_coasting_but_not_tentative():
    prediction = replace(
        _prediction((10, 20, 30), [(1, 0, 0), (2, 0, 0), (3, 0, 0)]),
        statuses=(TrackStatus.CONFIRMED, TrackStatus.COASTING, TrackStatus.TENTATIVE),
    )
    truth = [_truth((1, 2, 3), [(1, 0, 0), (2, 0, 0), (3, 0, 0)])]
    published = evaluate_track_frames([prediction], truth, match_distance_m=0.2)
    active = evaluate_track_frames(
        [prediction], truth, match_distance_m=0.2, include_statuses=(TrackStatus.CONFIRMED,)
    )
    assert published.matched_observations == 2 and published.missed_observations == 1
    assert active.matched_observations == 1 and active.missed_observations == 2
    assert published.to_record()["included_statuses"] == ["confirmed", "coasting"]


def test_multiscale_adapter_scores_body_ids_without_height_or_covariance():
    tracker = ScatterBodyTracker(max_bodies=1)
    predictions = []
    truths = []
    for frame in range(8):
        # Known target centre, independent of the tracker's own predicted state.
        points = np.array([[2, y, 1, 0, 20] for y in (-0.03, 0, 0.03)])
        bodies = tracker.step_points(points if frame < 7 else np.empty((0, 5)))[-1]
        predictions.append(
            TrackingPredictionFrame.from_multiscale(
                bodies,
                coordinate_frame="level_forward_lateral_up",
                frame_id=frame,
                timestamp=frame * 0.1,
            )
        )
        truths.append(
            TrackingGroundTruthFrame(
                np.array([1]),
                np.array([[2, 0]]),
                frame_id=frame,
                timestamp=frame * 0.1,
                coordinate_frame="level_forward_lateral_up",
            )
        )
    summary = evaluate_track_frames(predictions, truths, match_distance_m=0.1, dimensions=2)
    assert summary.matched_observations == 6 and summary.missed_observations == 2
    assert summary.id_switches == summary.false_track_observations == 0
    assert summary.position_rmse_m == pytest.approx(0)
    assert predictions[-1].statuses == (TrackStatus.COASTING,)
    with pytest.raises(ValueError, match="dimensions"):
        evaluate_track_frames(predictions, truths, match_distance_m=0.1, dimensions=3)


@pytest.mark.parametrize(
    "prediction_metadata,truth_metadata,error",
    [
        ({"coordinate_frame": "sensor"}, {"coordinate_frame": "world"}, "coordinate_frame"),
        ({"frame_id": 2}, {"frame_id": 1}, "frame_id"),
        ({"frame_id": 1}, {}, "frame_id"),
        ({"timestamp": 0.2}, {"timestamp": 0.1}, "timestamp"),
        ({"timestamp": 0.1}, {}, "timestamp"),
        ({"timestamp": float("nan")}, {"timestamp": float("nan")}, "timestamp"),
        ({"timestamp": float("inf")}, {"timestamp": float("inf")}, "timestamp"),
    ],
)
def test_comparison_rejects_misalignment_even_on_empty_frames(
    prediction_metadata, truth_metadata, error
):
    with pytest.raises(ValueError, match=error):
        evaluate_track_frames(
            [replace(_prediction((), []), **prediction_metadata)],
            [replace(_truth((), []), **truth_metadata)],
            match_distance_m=0.5,
        )


@pytest.mark.parametrize(
    "ids,positions,error",
    [
        ([0.5], [[0, 0]], "integers"),
        ([-1], [[0, 0]], "integers"),
        ([1, 1], [[0, 0], [0, 1]], "unique"),
        ([1], [[0]], "shape"),
        ([1], [[float("nan"), 0]], "NaN"),
    ],
)
def test_comparison_labels_reject_invalid_geometry_or_ids(ids, positions, error):
    with pytest.raises(ValueError, match=error):
        TrackingGroundTruthFrame(np.array(ids), np.array(positions))


def test_prediction_statuses_must_match_positions():
    with pytest.raises(ValueError, match="statuses"):
        TrackingPredictionFrame(
            track_ids=np.array([1]),
            positions=np.array([[0, 0]]),
            statuses=(),
            coordinate_frame="world",
        )
