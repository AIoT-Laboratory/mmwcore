from __future__ import annotations

import numpy as np

from mmwcore.core import (
    CFAR1DSpec,
    CFARInputScale,
    RadarCube,
    RangeDopplerCFARSpec,
)
from mmwcore.dsp import detect_range_doppler_cfar


def _cube_with_peak(*, edge_doppler: bool = False) -> RadarCube:
    data = np.ones((1, 7, 1, 7), dtype=np.complex64)
    doppler_idx = 0 if edge_doppler else 3
    data[0, doppler_idx, 0, 3] = 10
    return RadarCube(
        data,
        axes=("frame", "doppler_bin", "rx", "range_bin"),
        frame_id="rd-0",
        units="doppler_fft",
    )


def test_range_doppler_cfar_can_apply_range_pass_only() -> None:
    detections = detect_range_doppler_cfar(
        _cube_with_peak(),
        RangeDopplerCFARSpec(
            range=CFAR1DSpec(training_cells=1, guard_cells=1, threshold_scale=4.0),
            input_scale=CFARInputScale.MAGNITUDE,
        ),
    )

    np.testing.assert_array_equal(
        detections.detections,
        np.array([[0, 3, 3, 10, 1, 10]], dtype=np.float32),
    )
    assert detections.metadata["range_doppler_cfar"]["doppler"] is None
    assert detections.metadata["range_doppler_cfar"]["output_candidates"] == 1
    assert detections.units["noise"] == "magnitude"
    assert detections.units["snr"] == "linear_ratio"


def test_range_doppler_cfar_intersects_range_and_cyclic_doppler_passes() -> None:
    detections = detect_range_doppler_cfar(
        _cube_with_peak(edge_doppler=True),
        RangeDopplerCFARSpec(
            range=CFAR1DSpec(training_cells=1, guard_cells=1, threshold_scale=4.0),
            doppler=CFAR1DSpec(
                training_cells=1,
                guard_cells=1,
                threshold_scale=4.0,
                cyclic=True,
            ),
            input_scale=CFARInputScale.MAGNITUDE,
        ),
    )

    np.testing.assert_array_equal(
        detections.detections,
        np.array([[0, 3, 0, 10, 1, 10]], dtype=np.float32),
    )
    metadata = detections.metadata["range_doppler_cfar"]
    assert metadata["input_scale"] == "magnitude"
    assert metadata["doppler"]["cyclic"] is True


def test_range_doppler_cfar_power_scale_changes_threshold_domain() -> None:
    cube = _cube_with_peak()
    magnitude = detect_range_doppler_cfar(
        cube,
        RangeDopplerCFARSpec(
            range=CFAR1DSpec(training_cells=1, guard_cells=1, threshold_scale=20.0),
            input_scale=CFARInputScale.MAGNITUDE,
        ),
    )
    power = detect_range_doppler_cfar(
        cube,
        RangeDopplerCFARSpec(
            range=CFAR1DSpec(training_cells=1, guard_cells=1, threshold_scale=20.0),
            input_scale=CFARInputScale.POWER,
        ),
    )

    assert magnitude.detections.shape == (0, 6)
    assert power.detections.shape == (1, 6)
    assert power.detections[0, power.channels.index("noise")] == 1.0
    assert power.detections[0, power.channels.index("snr")] == 100.0
