"""Stateful target tracking over typed mmwcore cluster frames."""

from __future__ import annotations

from .benchmark import (
    IdentitySwitchEvent,
    TrackingBenchmarkSummary,
    TrackingGroundTruthFrame,
    evaluate_track_frames,
)
from .measurement_tracker import GTrack2D, PointTracker2D
from .metrics import TrackingSequenceSummary, TrackObservationSummary, summarize_track_frames
from .runners import iter_adc_cluster_track_frames, iter_adc_measurement_track_frames
from .ti_vectors import (
    TiGTrack2DBenchmarkReport,
    TiGTrack2DBenchmarkRun,
    TiGTrack2DBenchmarkSpec,
    TiGTrack2DVectorFrame,
    TiGTrackComparison,
    TiGTrackStrategyRun,
    benchmark_cluster_tracker_on_ti_vectors,
    benchmark_measurement_tracker_on_ti_vectors,
    benchmark_ti_people_counting_2d,
    read_ti_gtrack_2d_vectors,
    run_cluster_tracker_on_ti_vectors,
    run_measurement_tracker_on_ti_vectors,
    run_ti_people_counting_2d,
    ti_people_counting_2d_benchmark_spec,
)
from .tracker import ClusterTracker2D

__all__ = [
    "ClusterTracker2D",
    "GTrack2D",
    "IdentitySwitchEvent",
    "PointTracker2D",
    "TrackObservationSummary",
    "TrackingSequenceSummary",
    "TiGTrack2DVectorFrame",
    "TiGTrack2DBenchmarkSpec",
    "TiGTrack2DBenchmarkReport",
    "TiGTrack2DBenchmarkRun",
    "TiGTrackComparison",
    "TiGTrackStrategyRun",
    "benchmark_cluster_tracker_on_ti_vectors",
    "benchmark_measurement_tracker_on_ti_vectors",
    "benchmark_ti_people_counting_2d",
    "TrackingBenchmarkSummary",
    "TrackingGroundTruthFrame",
    "evaluate_track_frames",
    "iter_adc_cluster_track_frames",
    "iter_adc_measurement_track_frames",
    "read_ti_gtrack_2d_vectors",
    "run_cluster_tracker_on_ti_vectors",
    "run_measurement_tracker_on_ti_vectors",
    "run_ti_people_counting_2d",
    "ti_people_counting_2d_benchmark_spec",
    "summarize_track_frames",
]
