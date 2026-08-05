"""Lazy composition runners for offline tracking sequences."""

from __future__ import annotations

from collections.abc import Iterator
from math import isclose

from mmwcore.core import (
    DBSCANClusteringSpec,
    PointCloudFrame,
    PointCloudRecipe,
    Tracker2DSpec,
    TrackFrame,
)
from mmwcore.dsp import cluster_point_cloud, process_adc_to_calibrated_point_cloud
from mmwcore.io import ADCFileFrameReader
from mmwcore.tracking.measurement_tracker import MeasurementTracker2D
from mmwcore.tracking.tracker import ClusterTracker2D


def iter_adc_cluster_track_frames(
    reader: ADCFileFrameReader,
    point_cloud_recipe: PointCloudRecipe,
    clustering: DBSCANClusteringSpec,
    tracker: Tracker2DSpec,
    *,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[TrackFrame]:
    """Track DBSCAN cluster centers over a contiguous ADC file interval."""

    stateful_tracker = ClusterTracker2D(tracker)
    for point_cloud in _iter_adc_point_clouds(
        reader,
        point_cloud_recipe,
        tracker,
        start=start,
        stop=stop,
    ):
        yield stateful_tracker.step(cluster_point_cloud(point_cloud, clustering))


def iter_adc_measurement_track_frames(
    reader: ADCFileFrameReader,
    point_cloud_recipe: PointCloudRecipe,
    allocation_clustering: DBSCANClusteringSpec,
    tracker: Tracker2DSpec,
    *,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[TrackFrame]:
    """Track raw measurements over a contiguous ADC file interval."""

    stateful_tracker = MeasurementTracker2D(tracker, allocation_clustering)
    for point_cloud in _iter_adc_point_clouds(
        reader,
        point_cloud_recipe,
        tracker,
        start=start,
        stop=stop,
    ):
        yield stateful_tracker.step(point_cloud)


def _iter_adc_point_clouds(
    reader: ADCFileFrameReader,
    point_cloud_recipe: PointCloudRecipe,
    tracker: Tracker2DSpec,
    *,
    start: int,
    stop: int | None,
) -> Iterator[PointCloudFrame]:
    """Validate sequence contracts and lazily produce calibrated point clouds."""

    recipe_adc = point_cloud_recipe.detection.transform.decode.adc
    if reader.spec != recipe_adc:
        raise ValueError("ADC reader spec must match the point-cloud recipe decode spec.")
    if reader.frame_periodicity_s is not None and not isclose(
        reader.frame_periodicity_s,
        tracker.frame_period_s,
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        raise ValueError("ADC reader frame periodicity must match tracker frame_period_s.")
    end = reader.num_frames if stop is None else stop
    if start < 0 or end < start or end > reader.num_frames:
        raise ValueError(
            f"Tracking frame interval [{start}, {end}) is outside [0, {reader.num_frames})."
        )

    for frame_index in range(start, end):
        yield process_adc_to_calibrated_point_cloud(
            reader.read_frame(frame_index),
            point_cloud_recipe,
        )
