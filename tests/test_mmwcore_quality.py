from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import DetectionFrame, DetectionQualitySpec
from mmwcore.dsp import filter_detection_quality


def _detections(*, units: dict[str, str] | None = None) -> DetectionFrame:
    return DetectionFrame(
        np.array(
            [
                [0, 1, 2, 10, 2, 5],
                [0, 2, 2, 20, 2, 10],
            ],
            dtype=np.float32,
        ),
        channels=("frame", "range_bin", "doppler_bin", "magnitude", "noise", "snr"),
        units=units or {"snr": "linear_ratio"},
        metadata={"detector": "fixture"},
    )


def test_detection_quality_spec_requires_positive_finite_snr() -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            DetectionQualitySpec(value)


def test_filter_detection_quality_keeps_threshold_boundary() -> None:
    filtered = filter_detection_quality(_detections(), DetectionQualitySpec(10.0))

    np.testing.assert_array_equal(filtered.detections, [[0, 2, 2, 20, 2, 10]])
    assert filtered.channels[-2:] == ("noise", "snr")
    assert filtered.metadata["detector"] == "fixture"
    assert filtered.metadata["quality_filter"] == {
        "min_snr": 10.0,
        "snr_unit": "linear_ratio",
        "input_detections": 2,
        "output_detections": 1,
    }


def test_filter_detection_quality_requires_linear_snr_channel() -> None:
    missing = DetectionFrame(np.ones((1, 1)), channels=("magnitude",))
    with pytest.raises(ValueError, match='requires an "snr" channel'):
        filter_detection_quality(missing, DetectionQualitySpec(2.0))

    with pytest.raises(ValueError, match="linear_ratio"):
        filter_detection_quality(_detections(units={"snr": "dB"}), DetectionQualitySpec(2.0))
