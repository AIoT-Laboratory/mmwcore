"""TI GTRACK and multiscale tracking, with shared comparison metrics."""

from __future__ import annotations

from .benchmark import (
    IdentitySwitchEvent,
    TrackingBenchmarkSummary,
    TrackingGroundTruthFrame,
    TrackingPredictionFrame,
    evaluate_track_frames,
)
from .metrics import TrackingSequenceSummary, TrackObservationSummary, summarize_track_frames
from .multiscale import ClusterConfig, ScatterBodyTracker
from .ti_gtrack import (
    TiGTrack3D,
    TiGTrack3DSpec,
    TiGTrackAllocation,
    TiGTrackGating,
    TiGTrackLifecycle,
    TiGTrackScenery,
)

__all__ = [
    "TiGTrack3D",
    "TiGTrack3DSpec",
    "TiGTrackAllocation",
    "TiGTrackGating",
    "TiGTrackLifecycle",
    "TiGTrackScenery",
    "ClusterConfig",
    "ScatterBodyTracker",
    "IdentitySwitchEvent",
    "TrackObservationSummary",
    "TrackingSequenceSummary",
    "TrackingBenchmarkSummary",
    "TrackingGroundTruthFrame",
    "TrackingPredictionFrame",
    "evaluate_track_frames",
    "summarize_track_frames",
]
