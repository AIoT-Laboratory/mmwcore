from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import TrackFrame, TrackingBox2D, TrackScenerySpec, TrackStatus
from mmwcore.tracking import summarize_track_frames


def _frame(
    track_ids: tuple[int, ...],
    statuses: tuple[TrackStatus, ...],
    *,
    positions: tuple[tuple[float, float, float], ...] | None = None,
    velocities: tuple[tuple[float, float, float], ...] | None = None,
) -> TrackFrame:
    count = len(track_ids)
    return TrackFrame(
        track_ids=np.array(track_ids, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float32)
        if positions is not None
        else np.zeros((count, 3)),
        velocities=np.asarray(velocities, dtype=np.float32)
        if velocities is not None
        else np.zeros((count, 3)),
        position_covariances=np.zeros((count, 2, 2)),
        extent_covariances=np.zeros((count, 2, 2)),
        statuses=statuses,
        ages=np.ones(count, dtype=np.int64),
        missed_counts=np.zeros(count, dtype=np.int64),
        observation_track_ids=np.empty(0, dtype=np.int64),
    )


def test_summarize_track_frames_reports_identity_lifetimes_and_coverage() -> None:
    summary = summarize_track_frames(
        [
            _frame((2,), (TrackStatus.TENTATIVE,)),
            _frame((2, 7), (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE)),
            _frame((2,), (TrackStatus.COASTING,)),
            _frame((), ()),
        ]
    )

    assert summary.num_frames == 4
    assert summary.frames_with_tracks == 3
    assert summary.frames_with_confirmed_tracks == 1
    assert summary.confirmed_frame_coverage == 0.25
    assert summary.max_concurrent_tracks == 2
    assert summary.observed_track_ids == 2
    assert summary.confirmed_track_ids == 1
    assert summary.tracks[0].track_id == 2
    assert summary.tracks[0].observed_frames == 3
    assert summary.tracks[0].confirmed_frames == 1
    assert summary.tracks[0].first_frame_index == 0
    assert summary.tracks[0].last_frame_index == 2
    assert summary.tracks[0].confirmed_intervals == ((1, 1),)
    assert summary.tracks[0].in_scenery_frames is None
    assert summary.tracks[0].outside_scenery_frames is None


def test_summarize_track_frames_reports_spatial_motion_and_scenery() -> None:
    summary = summarize_track_frames(
        [
            _frame(
                (4,),
                (TrackStatus.CONFIRMED,),
                positions=((0.0, 1.0, 0.0),),
                velocities=((0.0, 0.0, 0.0),),
            ),
            _frame(
                (4,),
                (TrackStatus.CONFIRMED,),
                positions=((0.3, 1.4, 0.0),),
                velocities=((0.3, 0.4, 0.0),),
            ),
            _frame(
                (4,),
                (TrackStatus.COASTING,),
                positions=((0.6, 1.8, 0.0),),
                velocities=((0.0, 1.0, 0.0),),
            ),
            _frame(
                (4,),
                (TrackStatus.CONFIRMED,),
                positions=((1.2, 1.8, 0.0),),
                velocities=((0.0, 2.0, 0.0),),
            ),
        ],
        scenery=TrackScenerySpec((TrackingBox2D(-0.5, 1.0, 0.5, 2.0),)),
    )

    track = summary.tracks[0]
    assert track.first_position_m == (0.0, 1.0, 0.0)
    assert track.last_position_m == pytest.approx((1.2, 1.8, 0.0))
    assert track.median_position_m == pytest.approx((0.45, 1.6, 0.0))
    assert track.displacement_m == pytest.approx(np.hypot(1.2, 0.8))
    assert track.path_length_m == pytest.approx(1.6)
    assert track.median_speed_mps == pytest.approx(0.75)
    assert track.max_speed_mps == pytest.approx(2.0)
    assert track.confirmed_intervals == ((0, 1), (3, 3))
    assert track.in_scenery_frames == 3
    assert track.outside_scenery_frames == 1
    records = summary.to_record()["tracks"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    assert record["confirmed_intervals"] == [[0, 1], [3, 3]]
    assert record["median_position_m"] == pytest.approx([0.45, 1.6, 0.0])


def test_summarize_track_frames_handles_empty_sequence() -> None:
    summary = summarize_track_frames([])

    assert summary.num_frames == 0
    assert summary.confirmed_frame_coverage == 0.0
    assert summary.to_record()["tracks"] == []


def test_summarize_track_frames_applies_explicit_frame_index_offset() -> None:
    summary = summarize_track_frames(
        [_frame((1,), (TrackStatus.CONFIRMED,))], frame_index_offset=50
    )

    assert summary.tracks[0].first_frame_index == 50
    assert summary.tracks[0].last_frame_index == 50
    assert summary.tracks[0].confirmed_intervals == ((50, 50),)


def test_summarize_track_frames_rejects_negative_frame_index_offset() -> None:
    with pytest.raises(ValueError, match="frame_index_offset"):
        summarize_track_frames([], frame_index_offset=-1)
