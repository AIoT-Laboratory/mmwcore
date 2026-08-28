from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import (
    ADCDecodeSpec,
    ADCFrameSpec,
    DopplerFFTSpec,
    RadarCube,
    RangeDopplerPipeline,
)
from mmwcore.dsp import range_doppler, remove_static_clutter


def test_remove_static_clutter_uses_named_axis_and_preserves_metadata() -> None:
    data = np.array([[[[1 + 1j]], [[3 + 3j]]]], dtype=np.complex64)
    cube = RadarCube(
        data,
        axes=("frame", "chirp", "rx", "range_bin"),
        frame_id="f0",
        units="range_fft",
        metadata={"source": "fixture"},
    )

    filtered = remove_static_clutter(cube)

    np.testing.assert_allclose(
        filtered.data,
        np.array([[[[-1 - 1j]], [[1 + 1j]]]], dtype=np.complex64),
    )
    assert filtered.axes == cube.axes
    assert filtered.frame_id == "f0"
    assert filtered.units == "range_fft"
    assert filtered.metadata == {
        "source": "fixture",
        "static_clutter_removal": {"axis": "chirp"},
    }


def test_remove_static_clutter_rejects_missing_axis() -> None:
    cube = RadarCube(
        np.ones((1, 2), dtype=np.complex64),
        axes=("frame", "range_bin"),
    )

    with pytest.raises(ValueError, match="chirp"):
        remove_static_clutter(cube)


def test_range_doppler_recipe_can_remove_stationary_component() -> None:
    raw = np.array([1, 0, 1, 0], dtype=np.int16)
    recipe = RangeDopplerPipeline(
        decode=ADCDecodeSpec(ADCFrameSpec(num_chirps=2, num_rx=1, num_samples=1)),
        doppler_fft=DopplerFFTSpec(fftshift=False),
        remove_static_clutter=True,
    )

    cube = range_doppler(raw, recipe)

    np.testing.assert_array_equal(cube.data, np.zeros((1, 2, 1, 1), dtype=np.complex64))
    assert cube.metadata["static_clutter_removal"] == {"axis": "chirp"}
