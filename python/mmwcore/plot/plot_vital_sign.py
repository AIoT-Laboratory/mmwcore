"""Exploratory vital-sign waveform visualization."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mmwcore.core import VitalSignWaveform


def plot_vital_sign_waveform(
    waveform: VitalSignWaveform,
    *,
    frequency_band_hz: tuple[float, float] | None = None,
    title: str | None = None,
) -> Any:
    """Plot a radar micro-motion waveform and its one-sided spectrum."""

    frequencies, amplitudes = _amplitude_spectrum(waveform)
    if frequency_band_hz is not None:
        _validate_frequency_band(
            frequency_band_hz,
            nyquist_hz=waveform.sample_rate_hz / 2.0,
        )

    colors = plt.get_cmap("viridis")
    figure, (time_axis, spectrum_axis) = plt.subplots(2, 1, figsize=(8, 6))
    time_axis.plot(
        waveform.time_axis_s(),
        waveform.values,
        color=colors(0.68),
        linewidth=1.4,
    )
    time_axis.set_xlabel("time (s)")
    time_axis.set_ylabel(f"{waveform.quantity.value} ({waveform.units})")
    time_axis.grid(alpha=0.2)

    spectrum_axis.plot(frequencies, amplitudes, color=colors(0.88), linewidth=1.4)
    if frequency_band_hz is not None:
        spectrum_axis.axvspan(
            *frequency_band_hz,
            color=colors(0.25),
            alpha=0.18,
            label="selected research band",
        )
        spectrum_axis.legend(loc="best")
    spectrum_axis.set_xlim(0.0, waveform.sample_rate_hz / 2.0)
    spectrum_axis.set_xlabel("frequency (Hz)")
    spectrum_axis.set_ylabel(f"amplitude ({waveform.units})")
    spectrum_axis.grid(alpha=0.2)

    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def _amplitude_spectrum(waveform: VitalSignWaveform) -> tuple[np.ndarray, np.ndarray]:
    centered = waveform.values.astype(np.float64) - float(np.mean(waveform.values))
    window = np.hanning(waveform.num_samples)
    if not np.any(window):
        window = np.ones(waveform.num_samples, dtype=np.float64)
    spectrum = np.fft.rfft(centered * window)
    amplitudes = np.abs(spectrum) * (2.0 / float(window.sum()))
    if amplitudes.size:
        amplitudes[0] *= 0.5
        if waveform.num_samples % 2 == 0:
            amplitudes[-1] *= 0.5
    frequencies = np.fft.rfftfreq(waveform.num_samples, d=1.0 / waveform.sample_rate_hz)
    return frequencies, amplitudes


def _validate_frequency_band(frequency_band_hz: tuple[float, float], *, nyquist_hz: float) -> None:
    low, high = frequency_band_hz
    if not np.isfinite((low, high)).all() or low < 0 or high <= low:
        raise ValueError("frequency_band_hz must be finite and strictly increasing.")
    if high > nyquist_hz:
        raise ValueError("frequency_band_hz must not exceed the waveform Nyquist frequency.")


__all__ = ["plot_vital_sign_waveform"]
