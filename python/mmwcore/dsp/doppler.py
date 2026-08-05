"""Doppler-domain transforms for offline mmwcore processing."""

from __future__ import annotations

from mmwcore.core import DopplerFFTSpec, RadarCube
from mmwcore.dsp._fft import fft_complex_axis


def doppler_fft(cube: RadarCube, spec: DopplerFFTSpec | None = None) -> RadarCube:
    """Run an FFT over a named slow-time axis and return a Doppler-bin cube."""

    fft_spec = spec or DopplerFFTSpec()
    try:
        chirp_axis = cube.axes.index(fft_spec.input_axis)
    except ValueError as exc:
        raise ValueError(
            f'RadarCube axes must include "{fft_spec.input_axis}"; got {cube.axes}.'
        ) from exc

    n_fft = fft_spec.n_fft or cube.data.shape[chirp_axis]
    transformed = fft_complex_axis(
        cube.data,
        axis=chirp_axis,
        n_fft=n_fft,
        window=fft_spec.window,
        remove_dc=False,
        fftshift=fft_spec.fftshift,
        one_sided=False,
    )

    axes = tuple("doppler_bin" if axis == fft_spec.input_axis else axis for axis in cube.axes)
    return RadarCube(
        transformed,
        axes=axes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units="doppler_fft",
        metadata={
            **cube.metadata,
            "doppler_fft": {
                "n_fft": n_fft,
                "window": fft_spec.window.value,
                "fftshift": fft_spec.fftshift,
                "input_axis": fft_spec.input_axis,
            },
        },
    )
