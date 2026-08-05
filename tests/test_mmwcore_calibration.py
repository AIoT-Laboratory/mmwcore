from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import RadarCube, TimeDomainChannelCalibration
from mmwcore.dsp import apply_time_domain_channel_calibration


def test_time_domain_channel_calibration_applies_phase_ramp_and_correction() -> None:
    cube = RadarCube(
        np.ones((1, 1, 2, 2, 3), dtype=np.complex64),
        axes=("frame", "loop", "tx", "rx", "sample"),
        metadata={"calibration_applied": False},
    )
    calibration = TimeDomainChannelCalibration(
        frequency_rad_per_sample=((0.0, np.pi / 2), (np.pi, 0.0)),
        complex_corrections=((1 + 0j, 0 + 1j), (0.5 + 0j, 0 - 1j)),
        source="fixture",
        version="v1",
    )

    corrected = apply_time_domain_channel_calibration(cube, calibration)

    sample = np.arange(3)
    expected = np.empty((2, 2, 3), dtype=np.complex64)
    for tx in range(2):
        for rx in range(2):
            expected[tx, rx] = calibration.complex_corrections[tx][rx] * np.exp(
                1j * calibration.frequency_rad_per_sample[tx][rx] * sample
            )
    np.testing.assert_allclose(corrected.data[0, 0], expected, atol=1e-6)
    assert corrected.metadata["calibration_applied"] is True
    assert corrected.metadata["time_domain_channel_calibration"] == {
        "num_tx": 2,
        "num_rx": 2,
        "source": "fixture",
        "version": "v1",
    }


def test_time_domain_channel_calibration_validates_cube_shape() -> None:
    cube = RadarCube(
        np.ones((1, 1, 1, 2), dtype=np.complex64),
        axes=("frame", "tx", "rx", "sample"),
    )
    calibration = TimeDomainChannelCalibration(((0.0,),), ((1 + 0j,),))

    with pytest.raises(ValueError, match="Calibration shape"):
        apply_time_domain_channel_calibration(
            cube,
            TimeDomainChannelCalibration(((0.0, 0.0),), ((1 + 0j, 1 + 0j),)),
        )

    corrected = apply_time_domain_channel_calibration(cube, calibration)
    np.testing.assert_array_equal(corrected.data, cube.data)


def test_time_domain_channel_calibration_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        TimeDomainChannelCalibration(((float("nan"),),), ((1 + 0j,),))
