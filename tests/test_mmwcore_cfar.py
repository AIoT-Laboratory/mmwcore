from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import CFAR1DSpec, CFARDetectionSpec, PeakDetectionSpec, RadarCube
from mmwcore.dsp import detect_cfar


def test_cfar_detection_spec_validates_window_parameters() -> None:
    with pytest.raises(ValueError, match="training_cells"):
        CFARDetectionSpec(training_cells=0, guard_cells=1, threshold_scale=2.0)

    with pytest.raises(ValueError, match="guard_cells"):
        CFARDetectionSpec(training_cells=1, guard_cells=-1, threshold_scale=2.0)

    with pytest.raises(ValueError, match="threshold_scale"):
        CFARDetectionSpec(training_cells=1, guard_cells=1, threshold_scale=-1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_detection_thresholds_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        PeakDetectionSpec(threshold=value)
    with pytest.raises(ValueError, match="finite and non-negative"):
        CFARDetectionSpec(training_cells=1, guard_cells=0, threshold_scale=value)
    with pytest.raises(ValueError, match="finite and non-negative"):
        CFAR1DSpec(training_cells=1, guard_cells=0, threshold_scale=value)


def test_detect_cfar_returns_detection_frame_for_local_peak() -> None:
    data = np.ones((1, 5, 1, 5), dtype=np.complex64)
    data[0, 2, 0, 2] = 8
    cube = RadarCube(
        data,
        axes=("frame", "doppler_bin", "rx", "range_bin"),
        frame_id="cfar-0",
        units="doppler_fft",
        metadata={"source": "unit"},
    )

    detections = detect_cfar(
        cube,
        CFARDetectionSpec(training_cells=1, guard_cells=0, threshold_scale=4.0),
    )

    assert detections.channels == (
        "frame",
        "range_bin",
        "doppler_bin",
        "magnitude",
        "noise",
        "snr",
    )
    assert detections.frame_id == "cfar-0"
    assert detections.units == {
        "magnitude": "doppler_fft",
        "noise": "magnitude",
        "snr": "linear_ratio",
    }
    assert detections.metadata["source"] == "unit"
    assert detections.metadata["cfar_detection"] == {
        "training_cells": 1,
        "guard_cells": 0,
        "threshold_scale": 4.0,
        "aggregate_rx": "max",
        "output_candidates": 1,
    }
    np.testing.assert_array_equal(
        detections.detections,
        np.array([[0, 2, 2, 8, 1, 8]], dtype=np.float32),
    )
    assert detections.units["snr"] == "linear_ratio"


def test_detect_cfar_skips_cells_without_full_window() -> None:
    data = np.ones((1, 5, 1, 5), dtype=np.complex64)
    data[0, 0, 0, 0] = 100
    cube = RadarCube(data, axes=("frame", "doppler_bin", "rx", "range_bin"))

    detections = detect_cfar(
        cube,
        CFARDetectionSpec(training_cells=1, guard_cells=1, threshold_scale=2.0),
    )

    assert detections.detections.shape == (0, 6)


def test_detect_cfar_returns_empty_frame_when_map_is_too_small() -> None:
    cube = RadarCube(
        np.ones((1, 3, 1, 3), dtype=np.complex64),
        axes=("frame", "doppler_bin", "rx", "range_bin"),
    )

    detections = detect_cfar(
        cube,
        CFARDetectionSpec(training_cells=1, guard_cells=1, threshold_scale=2.0),
    )

    assert detections.detections.shape == (0, 6)


def test_detect_cfar_requires_range_doppler_axes() -> None:
    cube = RadarCube(np.ones((1, 5, 1, 5), dtype=np.complex64))

    with pytest.raises(ValueError, match="doppler_bin"):
        detect_cfar(cube, CFARDetectionSpec(training_cells=1, guard_cells=1, threshold_scale=2.0))
