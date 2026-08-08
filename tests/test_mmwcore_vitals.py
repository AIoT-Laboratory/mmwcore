from __future__ import annotations

from dataclasses import replace
from sys import maxsize

import matplotlib
import numpy as np
import pytest

from mmwcore.core import RadarCube, VitalSignQuantity, VitalSignWaveform
from mmwcore.dsp import extract_vital_sign_phase, phase_to_displacement
from mmwcore.plot import plot_vital_sign_waveform

matplotlib.use("Agg")


def test_vital_sign_waveform_exposes_uniform_time_contract() -> None:
    waveform = VitalSignWaveform(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        sample_rate_hz=2.0,
        start_time_s=1.5,
        range_bin=4,
    )

    assert waveform.quantity is VitalSignQuantity.PHASE_RAD
    assert waveform.units == "rad"
    assert waveform.num_samples == 3
    assert waveform.duration_s == pytest.approx(1.0)
    np.testing.assert_allclose(waveform.time_axis_s(), [1.5, 2.0, 2.5])


def test_extract_vital_sign_phase_requires_explicit_channel_selection() -> None:
    cube = RadarCube(
        np.ones((8, 2, 4), dtype=np.complex64),
        axes=("frame", "rx", "range_bin"),
    )

    with pytest.raises(ValueError, match="explicit index.*rx"):
        extract_vital_sign_phase(cube, range_bin=2, sample_rate_hz=10.0)


def test_extract_vital_sign_phase_unwraps_selected_range_sequence() -> None:
    sample_rate_hz = 10.0
    times = np.arange(80, dtype=np.float32) / sample_rate_hz
    expected_phase = 0.4 * np.sin(2.0 * np.pi * 0.25 * times)
    data = np.ones((times.size, 2, 4), dtype=np.complex64)
    data[:, 1, 2] = np.exp(1j * expected_phase)
    cube = RadarCube(
        data,
        axes=("frame", "rx", "range_bin"),
        frame_id="range-window-0",
        timestamp=12.5,
        source="synthetic-vital",
        units="range_fft",
    )

    waveform = extract_vital_sign_phase(
        cube,
        range_bin=2,
        sample_rate_hz=sample_rate_hz,
        selectors={"rx": 1},
    )

    np.testing.assert_allclose(waveform.values, expected_phase, atol=1e-6)
    assert waveform.start_time_s == 12.5
    assert waveform.source == "synthetic-vital"
    assert waveform.metadata["source_frame_id"] == "range-window-0"
    assert waveform.metadata["selectors"] == {"rx": 1}
    assert waveform.metadata["phase_unwrapped"] is True


def test_phase_to_displacement_uses_monostatic_round_trip_geometry() -> None:
    phase = VitalSignWaveform(
        np.array([-np.pi, np.pi], dtype=np.float32),
        sample_rate_hz=5.0,
        range_bin=3,
    )

    displacement = phase_to_displacement(phase, wavelength_m=0.004)

    assert displacement.quantity is VitalSignQuantity.DISPLACEMENT_M
    assert displacement.units == "m"
    np.testing.assert_allclose(displacement.values, [-0.001, 0.001], atol=1e-8)
    assert displacement.metadata["phase_to_displacement"]["geometry"] == ("monostatic_round_trip")


def test_plot_vital_sign_waveform_shows_time_and_frequency_views() -> None:
    sample_rate_hz = 10.0
    times = np.arange(200, dtype=np.float32) / sample_rate_hz
    waveform = VitalSignWaveform(
        0.002 * np.sin(2.0 * np.pi * 0.25 * times),
        sample_rate_hz=sample_rate_hz,
        quantity=VitalSignQuantity.DISPLACEMENT_M,
    )

    figure = plot_vital_sign_waveform(
        waveform,
        frequency_band_hz=(0.1, 0.6),
        title="Exploratory respiration motion",
    )
    spectrum_line = figure.axes[1].lines[0]
    frequencies = np.asarray(spectrum_line.get_xdata())
    amplitudes = np.asarray(spectrum_line.get_ydata())
    peak_frequency = frequencies[1:][np.argmax(amplitudes[1:])]

    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylabel() == "displacement_m (m)"
    assert figure.axes[1].get_xlabel() == "frequency (Hz)"
    assert peak_frequency == pytest.approx(0.25)
    assert len(figure.axes[1].patches) == 1


def test_vital_sign_spectrum_does_not_double_nyquist_amplitude() -> None:
    values = (-1.0) ** np.arange(32)
    waveform = VitalSignWaveform(values, sample_rate_hz=4.0)

    figure = plot_vital_sign_waveform(waveform)
    frequencies = np.asarray(figure.axes[1].lines[0].get_xdata())
    amplitudes = np.asarray(figure.axes[1].lines[0].get_ydata())

    assert frequencies[-1] == pytest.approx(2.0)
    assert amplitudes[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("values", "sample_rate_hz", "message"),
    [
        (np.array([1.0]), 10.0, "at least two"),
        (np.array([0.0, np.nan]), 10.0, "NaN or Inf"),
        (np.array([0.0, 1.0]), 0.0, "finite and positive"),
    ],
)
def test_vital_sign_waveform_rejects_invalid_samples(
    values: np.ndarray, sample_rate_hz: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        VitalSignWaveform(values, sample_rate_hz=sample_rate_hz)


@pytest.mark.parametrize("field_name", ["sample_rate_hz", "start_time_s"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "1.0"])
def test_vital_sign_waveform_rejects_non_real_physical_scalars(
    field_name: str,
    value: object,
) -> None:
    waveform = VitalSignWaveform(np.array([0.0, 1.0]), sample_rate_hz=2.0)

    with pytest.raises(TypeError, match=field_name):
        replace(waveform, **{field_name: value})


@pytest.mark.parametrize("field_name", ["sample_rate_hz", "start_time_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_vital_sign_waveform_rejects_nonfinite_physical_scalars(
    field_name: str,
    value: float,
) -> None:
    waveform = VitalSignWaveform(np.array([0.0, 1.0]), sample_rate_hz=2.0)

    with pytest.raises(ValueError, match=field_name):
        replace(waveform, **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    [
        ("sample_rate_hz", np.float32(2.5), 2.5),
        ("start_time_s", np.float64(-1.25), -1.25),
    ],
)
def test_vital_sign_waveform_normalizes_numpy_physical_scalars(
    field_name: str,
    value: object,
    expected: float,
) -> None:
    waveform = VitalSignWaveform(np.array([0.0, 1.0]), sample_rate_hz=2.0)
    normalized = getattr(replace(waveform, **{field_name: value}), field_name)

    assert normalized == pytest.approx(expected)
    assert type(normalized) is float


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5])
def test_vital_sign_waveform_rejects_nonintegral_range_bin(value: object) -> None:
    with pytest.raises(TypeError, match="range_bin"):
        VitalSignWaveform(
            np.array([0.0, 1.0]),
            sample_rate_hz=2.0,
            range_bin=value,  # type: ignore[arg-type]
        )


def test_vital_sign_waveform_normalizes_numpy_range_bin() -> None:
    waveform = replace(
        VitalSignWaveform(np.array([0.0, 1.0]), sample_rate_hz=2.0),
        range_bin=np.int64(4),
    )

    assert waveform.range_bin == 4
    assert type(waveform.range_bin) is int


def test_vital_sign_waveform_rejects_invalid_range_bin_domain() -> None:
    with pytest.raises(ValueError, match="range_bin"):
        VitalSignWaveform(np.array([0.0, 1.0]), sample_rate_hz=2.0, range_bin=-1)
    with pytest.raises(OverflowError, match="range_bin.*platform index"):
        VitalSignWaveform(
            np.array([0.0, 1.0]),
            sample_rate_hz=2.0,
            range_bin=maxsize + 1,
        )


@pytest.mark.parametrize(
    "values",
    [np.array([True, False]), np.array([True, False], dtype=np.bool_)],
)
def test_vital_sign_waveform_rejects_boolean_values_dtype(values: np.ndarray) -> None:
    with pytest.raises(TypeError, match="values.*boolean dtype"):
        VitalSignWaveform(values, sample_rate_hz=2.0)


@pytest.mark.parametrize(
    "values",
    [np.array([1, 2], dtype=np.int16), np.array([1.0, 2.0], dtype=np.float64)],
)
def test_vital_sign_waveform_normalizes_numeric_values(values: np.ndarray) -> None:
    waveform = VitalSignWaveform(values, sample_rate_hz=2.0)

    assert waveform.values.dtype == np.float32
    np.testing.assert_allclose(waveform.values, [1.0, 2.0])
