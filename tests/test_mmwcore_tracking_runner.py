from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import (
    ADCDecodeRecipe,
    ADCFrameSpec,
    AngleFFTSpec,
    DBSCANClusteringSpec,
    DetectionRecipe,
    DopplerFFTSpec,
    PeakDetectionSpec,
    PointCloudProjectionSpec,
    PointCloudRecipe,
    RangeDopplerRecipe,
    RangeFFTSpec,
    Tracker2DSpec,
    TrackGatingSpec,
    TrackLifecycleSpec,
    VirtualAntennaLayout,
)
from mmwcore.io import ADCFileFrameReader
from mmwcore.tracking import (
    iter_adc_cluster_track_frames,
    iter_adc_measurement_track_frames,
    summarize_track_frames,
)


def _point_cloud_recipe(adc: ADCFrameSpec) -> PointCloudRecipe:
    layout = VirtualAntennaLayout.uniform_linear(2)
    return PointCloudRecipe(
        detection=DetectionRecipe(
            transform=RangeDopplerRecipe(
                decode=ADCDecodeRecipe(adc),
                range_fft=RangeFFTSpec(n_fft=2, one_sided=True),
                doppler_fft=DopplerFFTSpec(n_fft=2),
            ),
            peak_detection=PeakDetectionSpec(threshold=0.0),
            angle_fft=AngleFFTSpec(
                n_fft=2,
                input_axis="rx",
                virtual_layout=layout,
            ),
        ),
        projection=PointCloudProjectionSpec(
            range_resolution_m=0.1,
            doppler_resolution_mps=0.1,
        ),
    )


def test_iter_adc_cluster_track_frames_composes_contiguous_sequence(tmp_path) -> None:
    adc = ADCFrameSpec(num_chirps=2, num_rx=2, num_samples=2)
    adc_path = tmp_path / "adc.bin"
    np.arange(adc.raw_values_per_frame * 3, dtype=np.int16).tofile(adc_path)
    reader = ADCFileFrameReader(adc_path, adc, frame_periodicity_s=0.1)
    tracker = Tracker2DSpec(
        frame_period_s=0.1,
        gating=TrackGatingSpec(max_distance_m=1.0),
        lifecycle=TrackLifecycleSpec(confirmation_hits=1),
    )

    frames = list(
        iter_adc_cluster_track_frames(
            reader,
            _point_cloud_recipe(adc),
            DBSCANClusteringSpec(eps_m=1.0, min_samples=1),
            tracker,
            start=1,
        )
    )
    summary = summarize_track_frames(frames)

    assert len(frames) == 2
    assert frames[0].frame_id == 1
    assert frames[1].timestamp == pytest.approx(0.2)
    assert summary.num_frames == 2
    assert summary.frames_with_confirmed_tracks == 2


def test_iter_adc_cluster_track_frames_rejects_timing_mismatch(tmp_path) -> None:
    adc = ADCFrameSpec(num_chirps=2, num_rx=2, num_samples=2)
    adc_path = tmp_path / "adc.bin"
    np.arange(adc.raw_values_per_frame, dtype=np.int16).tofile(adc_path)
    reader = ADCFileFrameReader(adc_path, adc, frame_periodicity_s=0.2)
    frames = iter_adc_cluster_track_frames(
        reader,
        _point_cloud_recipe(adc),
        DBSCANClusteringSpec(eps_m=1.0, min_samples=1),
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=TrackGatingSpec(max_distance_m=1.0),
        ),
    )

    with pytest.raises(ValueError, match="periodicity"):
        next(frames)


def test_iter_adc_measurement_track_frames_reports_point_associations(tmp_path) -> None:
    adc = ADCFrameSpec(num_chirps=2, num_rx=2, num_samples=2)
    adc_path = tmp_path / "adc.bin"
    np.arange(adc.raw_values_per_frame * 2, dtype=np.int16).tofile(adc_path)
    reader = ADCFileFrameReader(adc_path, adc, frame_periodicity_s=0.1)

    frames = list(
        iter_adc_measurement_track_frames(
            reader,
            _point_cloud_recipe(adc),
            DBSCANClusteringSpec(eps_m=1.0, min_samples=1),
            Tracker2DSpec(
                frame_period_s=0.1,
                gating=TrackGatingSpec(max_distance_m=1.0),
                lifecycle=TrackLifecycleSpec(confirmation_hits=1),
            ),
        )
    )

    assert len(frames) == 2
    assert frames[0].metadata["tracker"]["model"] == "constant_velocity_2d_measurement"
    assert frames[0].observation_track_ids.size > 0
