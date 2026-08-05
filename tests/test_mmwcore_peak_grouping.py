from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import DetectionFrame, PeakGroupingSpec, RadarCube
from mmwcore.dsp import group_detection_peaks


def _cube(values: np.ndarray) -> RadarCube:
    return RadarCube(
        values[:, :, None, :].astype(np.complex64),
        axes=("frame", "doppler_bin", "rx", "range_bin"),
        units="doppler_fft",
    )


def _detections(rows: list[list[float]]) -> DetectionFrame:
    return DetectionFrame(
        np.asarray(rows, dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "magnitude"),
        metadata={"detector": "fixture"},
    )


def test_peak_grouping_keeps_only_local_maximum() -> None:
    values = np.ones((1, 5, 5), dtype=np.float32)
    values[0, 2, 2] = 10
    values[0, 2, 3] = 7
    grouped = group_detection_peaks(
        _cube(values),
        _detections([[0, 2, 2, 10], [0, 3, 2, 7]]),
        PeakGroupingSpec(aggregate_rx="sum"),
    )

    np.testing.assert_array_equal(
        grouped.detections,
        np.array([[0, 2, 2, 10]], dtype=np.float32),
    )
    assert grouped.metadata["detector"] == "fixture"
    assert grouped.metadata["peak_grouping"]["input_candidates"] == 2
    assert grouped.metadata["peak_grouping"]["output_peaks"] == 1


def test_peak_grouping_preserves_detector_quality_channels() -> None:
    values = np.ones((1, 3, 3), dtype=np.float32)
    values[0, 1, 1] = 10
    candidates = DetectionFrame(
        np.array([[0, 1, 1, 10, 2, 5]], dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "magnitude", "noise", "snr"),
        units={"noise": "power", "snr": "linear_ratio"},
    )

    grouped = group_detection_peaks(_cube(values), candidates)

    assert grouped.channels[-2:] == ("noise", "snr")
    np.testing.assert_allclose(grouped.detections[0, -2:], [2, 5])
    assert grouped.units["snr"] == "linear_ratio"


def test_peak_grouping_can_wrap_doppler_edge() -> None:
    values = np.ones((1, 5, 3), dtype=np.float32)
    values[0, 0, 1] = 8
    values[0, 4, 1] = 10
    candidates = _detections([[0, 1, 0, 8], [0, 1, 4, 10]])

    cyclic = group_detection_peaks(
        _cube(values),
        candidates,
        PeakGroupingSpec(range_radius=0, cyclic_doppler=True),
    )
    noncyclic = group_detection_peaks(
        _cube(values),
        candidates,
        PeakGroupingSpec(range_radius=0, cyclic_doppler=False),
    )

    np.testing.assert_array_equal(cyclic.detections[:, 2], np.array([4]))
    np.testing.assert_array_equal(noncyclic.detections[:, 2], np.array([0, 4]))


def test_peak_grouping_plateau_policy_is_explicit() -> None:
    values = np.ones((1, 3, 3), dtype=np.float32)
    values[0, 1, 1:3] = 5
    candidates = _detections([[0, 1, 1, 5], [0, 2, 1, 5]])

    strict = group_detection_peaks(_cube(values), candidates, PeakGroupingSpec(strict=True))
    plateau = group_detection_peaks(_cube(values), candidates, PeakGroupingSpec(strict=False))

    assert strict.detections.shape == (0, 4)
    assert plateau.detections.shape == (2, 4)


def test_peak_grouping_rejects_missing_or_invalid_indices() -> None:
    cube = _cube(np.ones((1, 3, 3), dtype=np.float32))
    missing = DetectionFrame(np.ones((1, 2)), channels=("range_bin", "magnitude"))
    with pytest.raises(ValueError, match="missing grouping channels"):
        group_detection_peaks(cube, missing)

    with pytest.raises(ValueError, match="outside shape"):
        group_detection_peaks(cube, _detections([[0, 8, 1, 2]]))
