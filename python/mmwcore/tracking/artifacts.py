"""Pure serialization contracts for tracking artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mmwcore.core import TrackFrame, TrackStatus


def track_frame_to_record(frame: TrackFrame) -> dict[str, object]:
    """Convert one track frame to the versioned, JSON-compatible record contract."""

    tracks = [
        {
            "track_id": int(frame.track_ids[index]),
            "position_m": frame.positions[index].tolist(),
            "velocity_mps": frame.velocities[index].tolist(),
            "position_covariance_m2": frame.position_covariances[index].tolist(),
            "extent_covariance_m2": frame.extent_covariances[index].tolist(),
            "status": frame.statuses[index].value,
            "age_frames": int(frame.ages[index]),
            "missed_frames": int(frame.missed_counts[index]),
        }
        for index in range(frame.num_tracks)
    ]
    return {
        "schema": "mmwcore.track_frame",
        "schema_version": 1,
        "frame_id": frame.frame_id,
        "timestamp_s": frame.timestamp,
        "source": frame.source,
        "coordinate_frame": frame.coordinate_frame,
        "tracks": tracks,
        "observation_track_ids": frame.observation_track_ids.tolist(),
        "metadata": _json_value(frame.metadata, path="metadata"),
    }


def track_frame_from_record(record: Mapping[str, Any]) -> TrackFrame:
    """Validate and reconstruct one frame from the versioned record contract."""

    if record.get("schema") != "mmwcore.track_frame":
        raise ValueError("Track artifact schema must be 'mmwcore.track_frame'.")
    schema_version = record.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("Unsupported track artifact schema_version; expected 1.")
    tracks = _sequence(record.get("tracks"), path="tracks")
    track_records = [_mapping(item, path=f"tracks[{index}]") for index, item in enumerate(tracks)]
    count = len(track_records)
    metadata = _mapping(record.get("metadata", {}), path="metadata")
    frame_id = record.get("frame_id")
    if frame_id is not None and (
        isinstance(frame_id, bool) or not isinstance(frame_id, (str, int))
    ):
        raise TypeError("Track artifact frame_id must be a string, integer, or null.")
    timestamp = record.get("timestamp_s")
    if timestamp is not None:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TypeError("Track artifact timestamp_s must be a finite number or null.")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("Track artifact timestamp_s must be finite.")
    source = _optional_string(record.get("source"), path="source")
    coordinate_frame = _required_string(
        record.get("coordinate_frame", "radar"), path="coordinate_frame"
    )
    return TrackFrame(
        track_ids=np.asarray(
            [_integer_field(item, "track_id", index) for index, item in enumerate(track_records)],
            dtype=np.int64,
        ),
        positions=_matrix_field(track_records, "position_m", count, (3,)),
        velocities=_matrix_field(track_records, "velocity_mps", count, (3,)),
        position_covariances=_matrix_field(track_records, "position_covariance_m2", count, (2, 2)),
        extent_covariances=_matrix_field(track_records, "extent_covariance_m2", count, (2, 2)),
        statuses=tuple(
            TrackStatus(_string_field(item, "status", index))
            for index, item in enumerate(track_records)
        ),
        ages=np.asarray(
            [_integer_field(item, "age_frames", index) for index, item in enumerate(track_records)],
            dtype=np.int64,
        ),
        missed_counts=np.asarray(
            [
                _integer_field(item, "missed_frames", index)
                for index, item in enumerate(track_records)
            ],
            dtype=np.int64,
        ),
        observation_track_ids=np.asarray(
            _integer_sequence(record.get("observation_track_ids"), path="observation_track_ids"),
            dtype=np.int64,
        ),
        frame_id=frame_id,
        timestamp=timestamp,
        source=source,
        coordinate_frame=coordinate_frame,
        metadata=dict(metadata),
    )


def _matrix_field(
    tracks: list[Mapping[str, Any]], name: str, count: int, item_shape: tuple[int, ...]
) -> np.ndarray:
    values = [_required(track, name, index) for index, track in enumerate(tracks)]
    array = np.asarray(values, dtype=np.float32)
    expected = (count, *item_shape)
    if array.size == 0:
        return np.empty(expected, dtype=np.float32)
    if array.shape != expected:
        raise ValueError(f"Track artifact {name} values must have shape {expected}.")
    return array


def _required(record: Mapping[str, Any], name: str, index: int) -> Any:
    if name not in record:
        raise ValueError(f"Track artifact tracks[{index}] is missing {name}.")
    return record[name]


def _integer_field(record: Mapping[str, Any], name: str, index: int) -> int:
    value = _required(record, name, index)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Track artifact tracks[{index}].{name} must be an integer.")
    return value


def _string_field(record: Mapping[str, Any], name: str, index: int) -> str:
    return _required_string(_required(record, name, index), path=f"tracks[{index}].{name}")


def _required_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"Track artifact {path} must be a non-empty string.")
    return value


def _optional_string(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, path=path)


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"Track artifact {path} must be a mapping with string keys.")
    return value


def _sequence(value: Any, *, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"Track artifact {path} must be a sequence.")
    return value


def _integer_sequence(value: Any, *, path: str) -> list[int]:
    sequence = _sequence(value, path=path)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in sequence):
        raise TypeError(f"Track artifact {path} values must be integers.")
    return list(sequence)


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Track artifact {path} contains a non-finite number.")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path=path)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"Track artifact {path} mapping keys must be strings.")
        return {key: _json_value(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"Track artifact {path} has unsupported type {type(value).__name__}.")
