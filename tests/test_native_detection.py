from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


@pytest.mark.parametrize(
    ("aggregation", "expected_magnitude"),
    [
        (0, np.array([[[5.0, 10.0], [4.0, 7.0]]], dtype=np.float32)),
        (1, np.array([[[6.0, 12.0], [7.0, 11.0]]], dtype=np.float32)),
        (2, np.array([[[3.0, 6.0], [3.5, 5.5]]], dtype=np.float32)),
    ],
)
def test_native_range_doppler_magnitude_and_threshold_preserve_reordered_axes(
    aggregation: int,
    expected_magnitude: np.ndarray,
) -> None:
    canonical = np.array(
        [
            [
                [[3 + 4j, 2], [1, 6 + 8j]],
                [[4, 7], [3, 4]],
            ]
        ],
        dtype=np.complex64,
    )
    data = np.transpose(canonical, (2, 0, 3, 1)).copy()

    actual_magnitude = _native.range_doppler_magnitude_complex(
        data,
        (1, 3, 0, 2),
        aggregation,
    )
    indices, magnitudes = _native.threshold_range_doppler_complex(
        data,
        (1, 3, 0, 2),
        aggregation,
        5.0,
    )
    expected_indices = np.argwhere(expected_magnitude >= 5.0)
    expected_values = expected_magnitude[tuple(expected_indices.T)]

    np.testing.assert_allclose(actual_magnitude, expected_magnitude, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(magnitudes, expected_values, rtol=1e-6, atol=1e-6)


def test_native_range_doppler_magnitude_matches_numpy_for_random_reordered_axes() -> None:
    random = np.random.default_rng(29)
    canonical = (
        random.normal(size=(2, 3, 4, 5)).astype(np.float32)
        + 1j * random.normal(size=(2, 3, 4, 5)).astype(np.float32)
    ).astype(np.complex64)
    data = np.transpose(canonical, (2, 0, 3, 1)).copy()

    actual = _native.range_doppler_magnitude_complex(data, (1, 3, 0, 2), 1)
    expected = np.abs(canonical).sum(axis=2)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_native_azimuth_threshold_matches_explicit_local_peak_policy() -> None:
    data = np.zeros((1, 1, 5, 1), dtype=np.complex64)
    data[0, 0, :, 0] = [2, 6, 10, 7, 3]

    peaks, peak_magnitudes = _native.threshold_range_doppler_azimuth_complex(
        data,
        (0, 1, 2, 3),
        5.0,
        1,
        True,
    )
    threshold_hits, threshold_magnitudes = _native.threshold_range_doppler_azimuth_complex(
        data,
        (0, 1, 2, 3),
        5.0,
        0,
        True,
    )

    np.testing.assert_array_equal(peaks, np.array([[0, 0, 2, 0]], dtype=np.int64))
    np.testing.assert_array_equal(peak_magnitudes, np.array([10], dtype=np.float32))
    np.testing.assert_array_equal(
        threshold_hits,
        np.array([[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(threshold_magnitudes, np.array([6, 10, 7], dtype=np.float32))


def test_native_threshold_detection_rejects_invalid_direct_boundary_inputs() -> None:
    data = np.ones((2, 2, 2, 2), dtype=np.complex64)

    with pytest.raises(ValueError, match="contiguous"):
        _native.range_doppler_magnitude_complex(data.transpose(0, 2, 1, 3), (0, 2, 1, 3), 0)
    with pytest.raises(ValueError, match="Unsupported native receiver aggregation"):
        _native.range_doppler_magnitude_complex(data, (0, 1, 2, 3), 3)
    with pytest.raises(ValueError, match="must be distinct"):
        _native.threshold_range_doppler_complex(data, (0, 1, 1, 3), 0, 1.0)
