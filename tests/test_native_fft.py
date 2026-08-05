from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_fft_matches_numpy_for_nonleading_range_axis() -> None:
    random = np.random.default_rng(17)
    data = _complex_data(random, (2, 3, 4, 5))
    axis = 2
    n_fft = 7

    expected = data - data.mean(axis=axis, keepdims=True)
    window_shape = [1] * data.ndim
    window_shape[axis] = data.shape[axis]
    expected = expected * np.hanning(data.shape[axis]).astype(np.float32).reshape(window_shape)
    expected = np.fft.fft(expected, n=n_fft, axis=axis)
    expected = np.take(expected, np.arange(n_fft // 2 + 1), axis=axis).astype(np.complex64)

    actual = _native.fft_complex_axis(data, axis, n_fft, 1, 0b101)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_native_fft_matches_numpy_for_nonleading_shifted_doppler_axis() -> None:
    random = np.random.default_rng(19)
    data = _complex_data(random, (2, 5, 3, 4))
    axis = 1
    n_fft = 3

    window_shape = [1] * data.ndim
    window_shape[axis] = data.shape[axis]
    expected = data * np.hamming(data.shape[axis]).astype(np.float32).reshape(window_shape)
    expected = np.fft.fft(expected, n=n_fft, axis=axis)
    expected = np.fft.fftshift(expected, axes=axis).astype(np.complex64)

    actual = _native.fft_complex_axis(data, axis, n_fft, 2, 0b010)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_native_fft_rejects_invalid_direct_boundary_inputs() -> None:
    data = np.ones((2, 2), dtype=np.complex64)

    with pytest.raises(ValueError, match="contiguous"):
        _native.fft_complex_axis(data.T, 0, 2, 0, 0)
    with pytest.raises(ValueError, match="positive"):
        _native.fft_complex_axis(data, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="Unsupported native FFT window"):
        _native.fft_complex_axis(data, 0, 2, 3, 0)
    with pytest.raises(ValueError, match="Unsupported native FFT flags"):
        _native.fft_complex_axis(data, 0, 2, 0, 0b1000)


def test_native_fft_preserves_empty_nontransform_batch_axes() -> None:
    actual = _native.fft_complex_axis(
        np.empty((0, 4), dtype=np.complex64),
        1,
        4,
        0,
        0b010,
    )

    assert actual.shape == (0, 4)
    assert actual.dtype == np.complex64


def _complex_data(random: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (
        random.normal(size=shape).astype(np.float32)
        + 1j * random.normal(size=shape).astype(np.float32)
    ).astype(np.complex64)
