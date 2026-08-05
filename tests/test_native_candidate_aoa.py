from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_candidate_azimuth_recovers_peak_with_reordered_cube_axes() -> None:
    canonical = np.ones((1, 1, 4, 2), dtype=np.complex64)
    cube = np.transpose(canonical, (2, 0, 3, 1)).copy()
    positions = np.array(
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
        dtype=np.float32,
    )
    candidates = np.array(((0.0, 1.0, 0.0),), dtype=np.float32)

    peak_bins, angles, magnitudes = _native.candidate_azimuth_peaks(
        cube,
        (1, 3, 0, 2),
        candidates,
        (0, 1, 2),
        positions,
        (4, 0, True, 0),
    )

    np.testing.assert_array_equal(peak_bins, [2])
    np.testing.assert_allclose(angles, [0.0], atol=1e-6)
    np.testing.assert_allclose(magnitudes, [4.0], atol=1e-6)


def test_native_candidate_elevation_recovers_paired_row_direction() -> None:
    lateral_direction = np.float32(0.25)
    vertical_direction = np.float32(0.25)
    azimuth_positions = np.array(
        ((0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (1.0, 0.0, 0.5), (1.5, 0.0, 0.5)),
        dtype=np.float64,
    )
    elevation_positions = np.array(
        (
            (1.123456789, 0.0, 0.0),
            (1.623456789, 0.0, 0.0),
            (2.123456789, 0.0, 0.0),
            (2.623456789, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    positions = np.concatenate((azimuth_positions, elevation_positions))
    phase = (
        2.0 * np.pi * (positions[:, 0] * lateral_direction + positions[:, 2] * vertical_direction)
    )
    cube = np.exp(1j * phase).astype(np.complex64).reshape(1, 1, 8, 1)
    candidates = np.array(
        ((0.0, 0.0, 0.0, 5.0, np.arcsin(lateral_direction)),),
        dtype=np.float32,
    )

    valid_indices, angles, magnitudes, offsets = _native.candidate_elevations(
        cube,
        (0, 1, 2, 3),
        candidates,
        (0, 1, 2, 3, 4),
        (
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            azimuth_positions,
            elevation_positions,
        ),
        (8, 0, True),
    )

    np.testing.assert_array_equal(valid_indices, [0])
    np.testing.assert_allclose(angles, [np.arcsin(vertical_direction)], atol=1e-5)
    assert magnitudes[0] > 0.0
    assert offsets == pytest.approx((1.123456789, -0.5), rel=0.0, abs=1e-12)


def test_native_candidate_aoa_rejects_invalid_boundary_inputs() -> None:
    cube = np.ones((1, 1, 2, 1), dtype=np.complex64)
    positions = np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float32)
    candidates = np.array(((0.0, 0.0, 0.0),), dtype=np.float32)

    with pytest.raises(ValueError, match="contiguous"):
        _native.candidate_azimuth_peaks(
            cube,
            (0, 1, 2, 3),
            candidates[:, ::-1],
            (0, 1, 2),
            positions,
            (2, 0, True, 0),
        )
    with pytest.raises(ValueError, match="outside the angle-estimation cube"):
        _native.candidate_azimuth_peaks(
            cube,
            (0, 1, 2, 3),
            np.array(((1.0, 0.0, 0.0),), dtype=np.float32),
            (0, 1, 2),
            positions,
            (2, 0, True, 0),
        )
    with pytest.raises(ValueError, match="must have rank 4"):
        _native.candidate_azimuth_peaks(
            np.ones((1, 1, 2, 1, 1), dtype=np.complex64),
            (0, 1, 2, 3),
            candidates,
            (0, 1, 2),
            positions,
            (2, 0, True, 0),
        )
