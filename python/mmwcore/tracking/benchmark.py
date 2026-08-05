"""Ground-truth benchmark helpers for deterministic tracking scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from mmwcore import _native
from mmwcore.core import TrackFrame, TrackStatus


@dataclass(frozen=True)
class TrackingGroundTruthFrame:
    """Ground-truth Cartesian target identities for one frame."""

    track_ids: np.ndarray
    positions: np.ndarray

    def __post_init__(self) -> None:
        track_ids = np.asarray(self.track_ids, dtype=np.int64)
        positions = np.asarray(self.positions, dtype=np.float32)
        if track_ids.ndim != 1 or np.any(track_ids < 0):
            raise ValueError("Ground-truth track_ids must be one-dimensional and non-negative.")
        if np.unique(track_ids).size != track_ids.size:
            raise ValueError("Ground-truth track_ids must be unique within a frame.")
        if positions.shape != (track_ids.size, 3):
            raise ValueError("Ground-truth positions must have shape (N, 3).")
        if not np.isfinite(positions).all():
            raise ValueError("Ground-truth positions contain NaN or Inf.")
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "positions", positions)


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
        }


def evaluate_track_frames(
    predictions: list[TrackFrame],
    ground_truth: list[TrackingGroundTruthFrame],
    *,
    match_distance_m: float,
    confirmed_only: bool = True,
) -> TrackingBenchmarkSummary:
    """Evaluate frame-aligned predictions with global distance matching."""

    if len(predictions) != len(ground_truth):
        raise ValueError("Prediction and ground-truth sequences must have equal length.")
    if match_distance_m <= 0:
        raise ValueError("match_distance_m must be positive.")

    previous_matches: dict[int, int] = {}
    ground_truth_observations = 0
    matches = 0
    misses = 0
    false_tracks = 0
    identity_switches: list[IdentitySwitchEvent] = []
    squared_errors: list[float] = []
    for frame_index, (prediction, truth) in enumerate(zip(predictions, ground_truth, strict=True)):
        selected = np.array(
            [
                index
                for index, status in enumerate(prediction.statuses)
                if not confirmed_only or status is TrackStatus.CONFIRMED
            ],
            dtype=np.int64,
        )
        predicted_positions = prediction.positions[selected]
        predicted_ids = prediction.track_ids[selected]
        ground_truth_observations += truth.track_ids.size
        if truth.track_ids.size == 0 or predicted_ids.size == 0:
            misses += truth.track_ids.size
            false_tracks += predicted_ids.size
            continue

        distances = np.linalg.norm(
            truth.positions[:, None, :] - predicted_positions[None, :, :],
            axis=2,
        )
        truth_indices, prediction_indices = _native.linear_sum_assignment(
            np.ascontiguousarray(distances, dtype=np.float64)
        )
        accepted = [
            (int(truth_index), int(prediction_index))
            for truth_index, prediction_index in zip(truth_indices, prediction_indices, strict=True)
            if distances[truth_index, prediction_index] <= match_distance_m
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
    )
