"""Native complex FFT boundary for range and Doppler transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import FFTWindow


def fft_complex_axis(
    data: NDArray[np.complex64],
    *,
    axis: int,
    n_fft: int,
    window: FFTWindow,
    remove_dc: bool,
    fftshift: bool,
    one_sided: bool,
) -> NDArray[np.complex64]:
    """Run the maintained native complex FFT contract on one axis."""

    return _native.fft_complex_axis(
        np.ascontiguousarray(data, dtype=np.complex64),
        axis,
        n_fft,
        _window_code(window),
        _flags(remove_dc, fftshift, one_sided),
    )


def _window_code(window: FFTWindow) -> int:
    if window is FFTWindow.NONE:
        return 0
    if window is FFTWindow.HANN:
        return 1
    if window is FFTWindow.HAMMING:
        return 2
    raise ValueError(f"Unsupported FFT window: {window}")


def _flags(remove_dc: bool, fftshift: bool, one_sided: bool) -> int:
    return int(remove_dc) | int(fftshift) << 1 | int(one_sided) << 2
