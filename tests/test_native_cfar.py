from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


@pytest.mark.parametrize(
    ("mode", "expected_noise"),
    [(0, 1.5), (1, 2.5), (2, 0.5), (3, 6.0)],
)
def test_native_cfar_1d_preserves_window_reduction_modes(
    mode: int,
    expected_noise: float,
) -> None:
    indices, noise = _native.detect_cfar_1d(
        np.array([1, 1, 0, 20, 0, 5, 5], dtype=np.float32),
        (2, 0, 1.1, mode, False, 0, 0),
    )

    np.testing.assert_array_equal(indices, np.array([3], dtype=np.int64))
    np.testing.assert_allclose(noise, np.array([expected_noise], dtype=np.float32))


def test_native_composed_range_doppler_cfar_preserves_axes_and_power_domain() -> None:
    canonical = np.ones((1, 7, 1, 7), dtype=np.complex64)
    canonical[0, 3, 0, 3] = 10
    data = np.transpose(canonical, (2, 0, 3, 1)).copy()

    indices, magnitudes, noise, snr = _native.detect_range_doppler_cfar_complex(
        data,
        (1, 3, 0, 2),
        0,
        (1, 1, 20.0, 0, False, 0, 0),
        None,
        1,
    )

    np.testing.assert_array_equal(indices, np.array([[0, 3, 3]], dtype=np.int64))
    np.testing.assert_array_equal(magnitudes, np.array([10], dtype=np.float32))
    np.testing.assert_array_equal(noise, np.array([1], dtype=np.float32))
    np.testing.assert_array_equal(snr, np.array([100], dtype=np.float32))


def test_native_two_dimensional_cfar_keeps_guard_window_and_edge_policy() -> None:
    data = np.ones((1, 5, 1, 5), dtype=np.complex64)
    data[0, 0, 0, 0] = 100
    data[0, 2, 0, 2] = 8

    indices, magnitudes, noise, snr = _native.detect_cfar_2d_complex(
        data,
        (0, 1, 2, 3),
        0,
        (1, 0, 4.0),
    )

    np.testing.assert_array_equal(indices, np.array([[0, 2, 2]], dtype=np.int64))
    np.testing.assert_array_equal(magnitudes, np.array([8], dtype=np.float32))
    np.testing.assert_array_equal(noise, np.array([1], dtype=np.float32))
    np.testing.assert_array_equal(snr, np.array([8], dtype=np.float32))


def test_native_cfar_rejects_invalid_direct_boundary_inputs() -> None:
    power = np.ones(7, dtype=np.float32)

    with pytest.raises(ValueError, match="contiguous"):
        _native.detect_cfar_1d(power[::-1], (1, 0, 1.0, 0, False, 0, 0))
    with pytest.raises(ValueError, match="Unsupported native CFAR mode"):
        _native.detect_cfar_1d(power, (1, 0, 1.0, 7, False, 0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        _native.detect_cfar_1d(
            np.array([1, -1, 1], dtype=np.float32),
            (1, 0, 1.0, 0, False, 0, 0),
        )
