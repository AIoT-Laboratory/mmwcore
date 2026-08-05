from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_vital_phase_unwrap_and_displacement_preserve_float32_contract() -> None:
    angles = np.array([2.8, -2.9, -2.7], dtype=np.float32)
    samples = np.exp(1j * angles).astype(np.complex64)

    phase = _native.unwrap_vital_phase(samples, False)
    displacement = _native.vital_phase_to_displacement(phase, 0.004)

    expected_phase = np.unwrap(angles).astype(np.float32)
    np.testing.assert_allclose(phase, expected_phase, atol=1e-6)
    np.testing.assert_allclose(
        displacement,
        expected_phase * np.float32(0.004 / (4.0 * np.pi)),
        atol=1e-9,
    )
    assert phase.dtype == np.float32
    assert displacement.dtype == np.float32


def test_native_vital_phase_rejects_invalid_boundary_inputs() -> None:
    samples = np.ones(4, dtype=np.complex64)

    with pytest.raises(ValueError, match="C-contiguous"):
        _native.unwrap_vital_phase(samples[::-1], False)
    with pytest.raises(ValueError, match="at least one"):
        _native.unwrap_vital_phase(np.empty(0, dtype=np.complex64), False)
    with pytest.raises(ValueError, match="NaN or Inf"):
        _native.unwrap_vital_phase(np.array([np.nan + 0j], dtype=np.complex64), False)
    with pytest.raises(ValueError, match="finite and positive"):
        _native.vital_phase_to_displacement(np.zeros(1, dtype=np.float32), 0.0)
