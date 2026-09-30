"""Ground-truth benchmark helpers for deterministic tracking scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, sqrt
from typing import Literal

import numpy as np

from mmwcore import _native
from mmwcore.core import TrackFrame, TrackStatus


@dataclass(frozen=True)
class TrackingGroundTruthFrame:
    """Independent Cartesian target labels in metres, with 2D or 3D positions."""

    track_ids: np.ndarray
    positions: np.ndarray
    frame_id: str | int | None = None
    timestamp: float | None = None
    coordinate_frame: str = "radar"

    def __post_init__(self) -> None:
        track_ids, positions = _positions(self.track_ids, self.positions)
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True, kw_only=True)
class TrackingPredictionFrame:
    """Positions used for scoring, without fabricated height or covariance.

    TI's full TrackFrame is accepted directly by the evaluator. This smaller
    contract admits multiscale body outputs with genuine horizontal state only.
    """

    track_ids: np.ndarray
    positions: np.ndarray
    statuses: tuple[TrackStatus, ...]
    coordinate_frame: str
    frame_id: str | int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        track_ids, positions = _positions(self.track_ids, self.positions)
        statuses = tuple(TrackStatus(status) for status in self.statuses)
        if len(statuses) != len(track_ids):
            raise ValueError("statuses must match track_ids.")
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "statuses", statuses)

    @classmethod
    def from_multiscale(
        cls,
        bodies: list[dict],
        *,
        coordinate_frame: str,
        frame_id: str | int | None = None,
        timestamp: float | None = None,
    ) -> TrackingPredictionFrame:
        """Adapt step_points' body reports, never its internal scatter tracks."""
        return cls(
            track_ids=np.asarray([body["id"] for body in bodies], dtype=np.int64),
            positions=np.asarray([body["xy"] for body in bodies]).reshape(-1, 2),
            statuses=tuple(
                TrackStatus.COASTING if body["coasting"] else TrackStatus.CONFIRMED
                for body in bodies
            ),
            coordinate_frame=coordinate_frame,
            frame_id=frame_id,
            timestamp=timestamp,
        )


def _positions(ids: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids)
    if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu") or np.any(ids < 0):
        raise ValueError("track_ids must be one-dimensional non-negative integers.")
    if ids.size and np.max(ids) > np.iinfo(np.int64).max:
        raise ValueError("track_ids must fit int64.")
    ids = ids.astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError("track_ids must be unique within a frame.")
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape not in {(ids.size, 2), (ids.size, 3)}:
        raise ValueError("positions must have shape (N, 2) or (N, 3).")
    if not np.isfinite(positions).all():
        raise ValueError("positions contain NaN or Inf.")
    return ids, positions


def _validate_alignment(
    prediction: TrackFrame | TrackingPredictionFrame,
    truth: TrackingGroundTruthFrame,
    dimensions: int,
) -> None:
    if not truth.coordinate_frame or prediction.coordinate_frame != truth.coordinate_frame:
        raise ValueError("Prediction and ground truth must use the same coordinate_frame.")
    if prediction.frame_id != truth.frame_id:
        raise ValueError("Prediction and ground truth frame_id must match.")
    a, b = prediction.timestamp, truth.timestamp
    if a is None or b is None:
        if a != b:
            raise ValueError("Both frames must supply timestamp, or both omit it.")
    elif not (isfinite(a) and isfinite(b) and isclose(a, b, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Prediction and ground truth timestamp must match and be finite.")
    if min(prediction.positions.shape[1], truth.positions.shape[1]) < dimensions:
        raise ValueError("Requested dimensions exceed the available position state.")


@dataclass(frozen=True)
class IdentitySwitchEvent:
    """One ground-truth identity changing its associated predicted track ID."""

    frame_index: int
    ground_truth_id: int
    previous_track_id: int
    new_track_id: int


@dataclass(frozen=True)
class TrackingBenchmarkSummary:
    """Basic identity and position errors against labeled target frames."""

    num_frames: int
    ground_truth_observations: int
    matched_observations: int
    missed_observations: int
    false_track_observations: int
    identity_switches: tuple[IdentitySwitchEvent, ...]
    position_rmse_m: float | None
    dimensions: int
    included_statuses: tuple[TrackStatus, ...]
    coordinate_frame: str | None

    @property
    def id_switches(self) -> int:
        return len(self.identity_switches)

    @property
    def recall(self) -> float:
        if self.ground_truth_observations == 0:
            return 0.0
        return self.matched_observations / self.ground_truth_observations

    def to_record(self) -> dict[str, object]:
        return {
            "num_frames": self.num_frames,
            "ground_truth_observations": self.ground_truth_observations,
            "matched_observations": self.matched_observations,
            "missed_observations": self.missed_observations,
            "false_track_observations": self.false_track_observations,
            "id_switches": self.id_switches,
            "identity_switches": [
                {
                    "frame_index": event.frame_index,
                    "ground_truth_id": event.ground_truth_id,
                    "previous_track_id": event.previous_track_id,
                    "new_track_id": event.new_track_id,
                }
                for event in self.identity_switches
            ],
            "position_rmse_m": self.position_rmse_m,
            "recall": self.recall,
            "dimensions": self.dimensions,
            "included_statuses": list(self.included_statuses),
            "coordinate_frame": self.coordinate_frame,
        }


def evaluate_track_frames(
    predictions: list[TrackFrame] | list[TrackingPredictionFrame],
    ground_truth: list[TrackingGroundTruthFrame],
    *,
    match_distance_m: float,
    dimensions: Literal[2, 3] = 3,
    include_statuses: tuple[TrackStatus, ...] = (TrackStatus.CONFIRMED, TrackStatus.COASTING),
) -> TrackingBenchmarkSummary:
    """Score body IDs with gated matching in XY (2) or XYZ (3).

    Published continuity includes confirmed and coasting states by default.
    If both sides omit frame IDs/timestamps, the caller guarantees alignment by
    list index. Supplied metadata must agree; this function never aligns streams
    or transforms coordinates. Use dimensions=2 for TI/multiscale comparison.
    """

    if len(predictions) != len(ground_truth):
        raise ValueError("Prediction and ground-truth sequences must have equal length.")
    if not isfinite(match_distance_m) or match_distance_m <= 0:
        raise ValueError("match_distance_m must be finite and positive.")
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3.")
    statuses = tuple(TrackStatus(status) for status in include_statuses)
    if not statuses or len(set(statuses)) != len(statuses):
        raise ValueError("include_statuses must be non-empty and unique.")

    previous_matches: dict[int, int] = {}
    ground_truth_observations = 0
    matches = 0
    misses = 0
    false_tracks = 0
    identity_switches: list[IdentitySwitchEvent] = []
    squared_errors: list[float] = []
    for frame_index, (prediction, truth) in enumerate(zip(predictions, ground_truth, strict=True)):
        _validate_alignment(prediction, truth, dimensions)
        if truth.coordinate_frame != ground_truth[0].coordinate_frame:
            raise ValueError("coordinate_frame must remain fixed throughout the sequence.")
        selected = np.array(
            [index for index, status in enumerate(prediction.statuses) if status in statuses],
            dtype=np.int64,
        )
        predicted_positions = prediction.positions[selected, :dimensions]
        predicted_ids = prediction.track_ids[selected]
        ground_truth_observations += truth.track_ids.size
        if truth.track_ids.size == 0 or predicted_ids.size == 0:
            misses += truth.track_ids.size
            false_tracks += predicted_ids.size
            continue

        distances = np.linalg.norm(
            truth.positions[:, None, :dimensions]
            - predicted_positions.astype(np.float64)[None, :, :],
            axis=2,
        )
        valid = distances <= match_distance_m
        # Each valid edge costs at most 1. One invalid edge must cost more than
        # every valid edge combined: maximize matches first, minimize distance second.
        cost = np.full(distances.shape, min(distances.shape) + 1.0)
        cost[valid] = distances[valid] / match_distance_m
        truth_indices, prediction_indices = _native.linear_sum_assignment(
            np.ascontiguousarray(cost)
        )
        accepted = [
            (int(truth_index), int(prediction_index))
            for truth_index, prediction_index in zip(truth_indices, prediction_indices, strict=True)
            if valid[truth_index, prediction_index]
        ]
        matches += len(accepted)
        misses += truth.track_ids.size - len(accepted)
        false_tracks += predicted_ids.size - len(accepted)
        for truth_index, prediction_index in accepted:
            truth_id = int(truth.track_ids[truth_index])
            prediction_id = int(predicted_ids[prediction_index])
            previous_id = previous_matches.get(truth_id)
            if previous_id is not None and previous_id != prediction_id:
                identity_switches.append(
                    IdentitySwitchEvent(
                        frame_index=frame_index,
                        ground_truth_id=truth_id,
                        previous_track_id=previous_id,
                        new_track_id=prediction_id,
                    )
                )
            previous_matches[truth_id] = prediction_id
            squared_errors.append(float(distances[truth_index, prediction_index] ** 2))

    rmse = sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else None
    return TrackingBenchmarkSummary(
        num_frames=len(predictions),
        ground_truth_observations=ground_truth_observations,
        matched_observations=matches,
        missed_observations=misses,
        false_track_observations=false_tracks,
        identity_switches=tuple(identity_switches),
        position_rmse_m=rmse,
        dimensions=dimensions,
        included_statuses=statuses,
        coordinate_frame=ground_truth[0].coordinate_frame if ground_truth else None,
    )
