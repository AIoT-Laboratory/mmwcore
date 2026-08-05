from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_angle_calibration_matches_shifted_uniform_linear_bins() -> None:
    positions = np.array(
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
        dtype=np.float32,
    )

    angles = _native.calibrate_angle_bins(positions, 4, 0, True)

    np.testing.assert_allclose(
        angles,
        np.array([-np.pi / 2, -np.pi / 6, 0.0, np.pi / 6], dtype=np.float32),
        atol=1e-6,
    )


def test_native_angle_calibration_supports_elevation_and_unshifted_bins() -> None:
    positions = np.array(((1.0, 0.0, 0.0), (1.0, 0.0, 0.5)), dtype=np.float32)

    angles = _native.calibrate_angle_bins(positions, 2, 1, False)

    np.testing.assert_allclose(angles, [0.0, -np.pi / 2], atol=1e-6)


def test_native_angle_calibration_preserves_odd_fft_bin_order() -> None:
    positions = np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float32)

    shifted = _native.calibrate_angle_bins(positions, 5, 0, True)
    unshifted = _native.calibrate_angle_bins(positions, 5, 0, False)

    expected_shifted = np.arcsin(np.array([-0.8, -0.4, 0.0, 0.4, 0.8], dtype=np.float32))
    expected_unshifted = np.arcsin(np.array([0.0, 0.4, 0.8, -0.8, -0.4], dtype=np.float32))
    np.testing.assert_allclose(shifted, expected_shifted, atol=1e-6)
    np.testing.assert_allclose(unshifted, expected_unshifted, atol=1e-6)


def test_native_angle_calibration_rejects_invalid_boundary_inputs() -> None:
    positions = np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float32)

    with pytest.raises(ValueError, match="contiguous"):
        _native.calibrate_angle_bins(positions[:, ::-1], 2, 0, True)
    with pytest.raises(ValueError, match="num_bins"):
        _native.calibrate_angle_bins(positions, 0, 0, True)
    with pytest.raises(ValueError, match="num_bins must be positive; got -1"):
        _native.calibrate_angle_bins(positions, -1, 0, True)
    with pytest.raises(ValueError, match="Unsupported native angle axis"):
        _native.calibrate_angle_bins(positions, 2, 2, True)
