from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest

from mmwcore.core import TrackFrame, TrackStatus
from mmwcore.tracking import track_frame_from_record, track_frame_to_record


def _frame(metadata: dict[str, object] | None = None) -> TrackFrame:
    return TrackFrame(
        track_ids=np.array([7]),
        positions=np.array([[1.0, 2.0, 0.0]]),
        velocities=np.array([[0.1, -0.2, 0.0]]),
        position_covariances=np.array([[[0.2, 0.0], [0.0, 0.3]]]),
        extent_covariances=np.array([[[0.4, 0.1], [0.1, 0.5]]]),
        statuses=(TrackStatus.CONFIRMED,),
        ages=np.array([4]),
        missed_counts=np.array([0]),
        observation_track_ids=np.array([7, -1]),
        frame_id=12,
        timestamp=0.6,
        source="test",
        metadata=metadata or {},
    )


def test_track_frame_record_is_versioned_and_json_serializable() -> None:
    record = track_frame_to_record(_frame({"array": np.array([1, 2]), "score": np.float32(0.5)}))

    assert record["schema"] == "mmwcore.track_frame"
    assert record["schema_version"] == 1
    assert record["observation_track_ids"] == [7, -1]
    tracks = cast(list[dict[str, Any]], record["tracks"])
    assert tracks[0]["status"] == "confirmed"
    assert record["metadata"] == {"array": [1, 2], "score": 0.5}
    json.dumps(record, allow_nan=False)

    restored = track_frame_from_record(record)
    np.testing.assert_array_equal(restored.track_ids, [7])
    np.testing.assert_allclose(restored.positions, [[1.0, 2.0, 0.0]])
    assert restored.statuses == (TrackStatus.CONFIRMED,)
    assert restored.metadata == {"array": [1, 2], "score": 0.5}


def test_track_frame_record_rejects_unknown_schema() -> None:
    record = track_frame_to_record(_frame())
    record["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        track_frame_from_record(record)


def test_track_frame_record_rejects_fractional_track_id() -> None:
    record = track_frame_to_record(_frame())
    tracks = cast(list[dict[str, Any]], record["tracks"])
    tracks[0]["track_id"] = 7.5

    with pytest.raises(TypeError, match="track_id must be an integer"):
        track_frame_from_record(record)


def test_track_frame_record_rejects_fractional_observation_id() -> None:
    record = track_frame_to_record(_frame())
    record["observation_track_ids"] = [7.5]

    with pytest.raises(TypeError, match="observation_track_ids values must be integers"):
        track_frame_from_record(record)


@pytest.mark.parametrize(
    ("metadata", "error"),
    [({"bad": float("nan")}, ValueError), ({"bad": object()}, TypeError)],
)
def test_track_frame_record_rejects_non_json_metadata(
    metadata: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error, match="metadata.bad"):
        track_frame_to_record(_frame(metadata))
