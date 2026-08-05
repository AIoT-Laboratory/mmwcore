"""Range-domain transforms for offline mmwcore processing."""

from __future__ import annotations

from mmwcore.core import RadarCube, RangeFFTSpec
from mmwcore.dsp._fft import fft_complex_axis


def range_fft(cube: RadarCube, spec: RangeFFTSpec | None = None) -> RadarCube:
    """Run an FFT over the ``sample`` axis and return a range-bin cube."""

    fft_spec = spec or RangeFFTSpec()
    try:
        sample_axis = cube.axes.index("sample")
    except ValueError as exc:
        raise ValueError(f'RadarCube axes must include "sample"; got {cube.axes}.') from exc

    n_fft = fft_spec.n_fft or cube.data.shape[sample_axis]
    transformed = fft_complex_axis(
        cube.data,
        axis=sample_axis,
        n_fft=n_fft,
        window=fft_spec.window,
        remove_dc=fft_spec.remove_dc,
        fftshift=False,
        one_sided=fft_spec.one_sided,
    )

    axes = tuple("range_bin" if axis == "sample" else axis for axis in cube.axes)
    return RadarCube(
        transformed,
        axes=axes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units="range_fft",
        metadata={
            **cube.metadata,
            "range_fft": {
                "n_fft": n_fft,
                "window": fft_spec.window.value,
                "one_sided": fft_spec.one_sided,
                "remove_dc": fft_spec.remove_dc,
            },
        },
    )
