"""Sequence-level summaries for tracker validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from mmwcore import _native
from mmwcore.core import ScenerySpec, TrackFrame, TrackStatus

_STATUS_CODES = {
    TrackStatus.TENTATIVE: 0,
    TrackStatus.CONFIRMED: 1,
    TrackStatus.COASTING: 2,
}


@dataclass(frozen=True)
class TrackObservationSummary:
    """Observed lifetime of one track ID within a sequence."""

    track_id: int
    observed_frames: int
    confirmed_frames: int
    first_frame_index: int
    last_frame_index: int
    first_position_m: tuple[float, float, float]
    last_position_m: tuple[float, float, float]
    median_position_m: tuple[float, float, float]
    displacement_m: float
    path_length_m: float
    median_speed_mps: float
    max_speed_mps: float
    confirmed_intervals: tuple[tuple[int, int], ...]
    in_scenery_frames: int | None
    outside_scenery_frames: int | None


@dataclass(frozen=True)
class TrackingSequenceSummary:
    """Identity and coverage statistics over an ordered frame sequence."""

    num_frames: int
    frames_with_tracks: int
    frames_with_confirmed_tracks: int
    max_concurrent_tracks: int
    tracks: tuple[TrackObservationSummary, ...]

    @property
    def observed_track_ids(self) -> int:
        return len(self.tracks)

    @property
    def confirmed_track_ids(self) -> int:
        return sum(track.confirmed_frames > 0 for track in self.tracks)

    @property
    def confirmed_frame_coverage(self) -> float:
        if self.num_frames == 0:
            return 0.0
        return self.frames_with_confirmed_tracks / self.num_frames

    def to_record(self) -> dict[str, object]:
        return {
            "num_frames": self.num_frames,
            "frames_with_tracks": self.frames_with_tracks,
            "frames_with_confirmed_tracks": self.frames_with_confirmed_tracks,
            "confirmed_frame_coverage": self.confirmed_frame_coverage,
            "max_concurrent_tracks": self.max_concurrent_tracks,
            "observed_track_ids": self.observed_track_ids,
            "confirmed_track_ids": self.confirmed_track_ids,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "observed_frames": track.observed_frames,
                    "confirmed_frames": track.confirmed_frames,
                    "first_frame_index": track.first_frame_index,
                    "last_frame_index": track.last_frame_index,
                    "first_position_m": list(track.first_position_m),
                    "last_position_m": list(track.last_position_m),
                    "median_position_m": list(track.median_position_m),
                    "displacement_m": track.displacement_m,
                    "path_length_m": track.path_length_m,
                    "median_speed_mps": track.median_speed_mps,
                    "max_speed_mps": track.max_speed_mps,
                    "confirmed_intervals": [
                        [start, stop] for start, stop in track.confirmed_intervals
                    ],
                    "in_scenery_frames": track.in_scenery_frames,
                    "outside_scenery_frames": track.outside_scenery_frames,
                }
                for track in self.tracks
            ],
        }


def summarize_track_frames(
    frames: Iterable[TrackFrame],
    *,
    scenery: ScenerySpec | None = None,
    frame_index_offset: int = 0,
) -> TrackingSequenceSummary:
    """Summarize ordered tracker reports without assuming ground-truth identities."""

    if frame_index_offset < 0:
        raise ValueError("Tracking summary frame_index_offset must be non-negative.")
    packed = _pack_track_frames(tuple(frames))
    scenery_boxes = (
        [(box.x_min_m, box.x_max_m, box.y_min_m, box.y_max_m) for box in scenery.boundary_boxes]
        if scenery is not None
        else None
    )
    result = _native.summarize_tracking_metrics(
        packed,
        scenery_boxes,
        frame_index_offset,
    )
    header, identity, motion, interval_data = result
    (
        num_frames,
        frames_with_tracks,
        frames_with_confirmed,
        max_concurrent,
    ) = header
    (
        track_ids,
        observed_frames,
        confirmed_frames,
        first_frame_indices,
        last_frame_indices,
        first_positions,
        last_positions,
        median_positions,
    ) = identity
    displacement, path_length, median_speed, max_speed = motion
    interval_offsets, intervals, in_scenery_frames, outside_scenery_frames = interval_data
    tracks = tuple(
        _track_summary_from_native(
            index,
            track_ids=track_ids,
            observed_frames=observed_frames,
            confirmed_frames=confirmed_frames,
            first_frame_indices=first_frame_indices,
            last_frame_indices=last_frame_indices,
            first_positions=first_positions,
            last_positions=last_positions,
            median_positions=median_positions,
            displacement=displacement,
            path_length=path_length,
            median_speed=median_speed,
            max_speed=max_speed,
            interval_offsets=interval_offsets,
            intervals=intervals,
            in_scenery_frames=in_scenery_frames,
            outside_scenery_frames=outside_scenery_frames,
        )
        for index in range(track_ids.size)
    )
    return TrackingSequenceSummary(
        num_frames=num_frames,
        frames_with_tracks=frames_with_tracks,
        frames_with_confirmed_tracks=frames_with_confirmed,
        max_concurrent_tracks=max_concurrent,
        tracks=tracks,
    )


def _pack_track_frames(
    frames: tuple[TrackFrame, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observation_count = sum(frame.num_tracks for frame in frames)
    frame_offsets = np.empty(len(frames) + 1, dtype=np.int64)
    track_ids = np.empty(observation_count, dtype=np.int64)
    positions = np.empty((observation_count, 3), dtype=np.float32)
    velocities = np.empty((observation_count, 3), dtype=np.float32)
    status_codes = np.empty(observation_count, dtype=np.uint8)
    frame_offsets[0] = 0
    cursor = 0
    for frame_index, frame in enumerate(frames):
        stop = cursor + frame.num_tracks
        track_ids[cursor:stop] = frame.track_ids
        positions[cursor:stop] = frame.positions
        velocities[cursor:stop] = frame.velocities
        status_codes[cursor:stop] = [_STATUS_CODES[status] for status in frame.statuses]
        frame_offsets[frame_index + 1] = stop
        cursor = stop
    return (
        frame_offsets,
        track_ids,
        positions,
        velocities,
        status_codes,
    )


def _track_summary_from_native(
    index: int,
    *,
    track_ids: np.ndarray,
    observed_frames: np.ndarray,
    confirmed_frames: np.ndarray,
    first_frame_indices: np.ndarray,
    last_frame_indices: np.ndarray,
    first_positions: np.ndarray,
    last_positions: np.ndarray,
    median_positions: np.ndarray,
    displacement: np.ndarray,
    path_length: np.ndarray,
    median_speed: np.ndarray,
    max_speed: np.ndarray,
    interval_offsets: np.ndarray,
    intervals: np.ndarray,
    in_scenery_frames: np.ndarray,
    outside_scenery_frames: np.ndarray,
) -> TrackObservationSummary:
    interval_start = int(interval_offsets[index])
    interval_stop = int(interval_offsets[index + 1])
    in_scenery = int(in_scenery_frames[index])
    outside_scenery = int(outside_scenery_frames[index])
    return TrackObservationSummary(
        track_id=int(track_ids[index]),
        observed_frames=int(observed_frames[index]),
        confirmed_frames=int(confirmed_frames[index]),
        first_frame_index=int(first_frame_indices[index]),
        last_frame_index=int(last_frame_indices[index]),
        first_position_m=_vector3(first_positions[index]),
        last_position_m=_vector3(last_positions[index]),
        median_position_m=_vector3(median_positions[index]),
        displacement_m=float(displacement[index]),
        path_length_m=float(path_length[index]),
        median_speed_mps=float(median_speed[index]),
        max_speed_mps=float(max_speed[index]),
        confirmed_intervals=tuple(
            (int(start), int(stop)) for start, stop in intervals[interval_start:interval_stop]
        ),
        in_scenery_frames=None if in_scenery < 0 else in_scenery,
        outside_scenery_frames=None if outside_scenery < 0 else outside_scenery,
    )


def _vector3(values: np.ndarray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))
