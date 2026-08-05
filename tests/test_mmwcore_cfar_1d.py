from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import CFAR1DSpec, CFARMode
from mmwcore.dsp import detect_cfar_1d


@pytest.mark.parametrize(
    ("mode", "expected_noise"),
    [
        (CFARMode.CA, 1.5),
        (CFARMode.GO, 2.5),
        (CFARMode.SO, 0.5),
        (CFARMode.CACC, 6.0),
    ],
)
def test_cfar_1d_matches_ti_window_reduction_modes(
    mode: CFARMode,
    expected_noise: float,
) -> None:
    power = np.array([1, 1, 0, 20, 0, 5, 5], dtype=np.float32)
    spec = CFAR1DSpec(
        training_cells=2,
        guard_cells=0,
        threshold_scale=1.1,
        mode=mode,
    )

    result = detect_cfar_1d(power, spec)

    np.testing.assert_array_equal(result.indices, np.array([3]))
    np.testing.assert_allclose(result.noise, np.array([expected_noise]))


def test_cfar_1d_cyclic_mode_wraps_doppler_window() -> None:
    power = np.array([10, 1, 1, 1, 1], dtype=np.float32)
    spec = CFAR1DSpec(
        training_cells=1,
        guard_cells=0,
        threshold_scale=2.0,
        cyclic=True,
    )

    result = detect_cfar_1d(power, spec)

    np.testing.assert_array_equal(result.indices, np.array([0]))
    np.testing.assert_allclose(result.noise, np.array([1.0]))


def test_cfar_1d_noncyclic_mode_skips_incomplete_edges_and_explicit_regions() -> None:
    power = np.array([100, 1, 1, 10, 1, 1, 100], dtype=np.float32)
    spec = CFAR1DSpec(
        training_cells=1,
        guard_cells=0,
        threshold_scale=2.0,
        left_skip=1,
        right_skip=1,
    )

    result = detect_cfar_1d(power, spec)

    np.testing.assert_array_equal(result.indices, np.array([3]))


def test_cfar_1d_returns_empty_when_the_window_exactly_spans_the_input() -> None:
    result = detect_cfar_1d(
        np.array([1, 20, 1, 1], dtype=np.float32),
        CFAR1DSpec(training_cells=2, guard_cells=0, threshold_scale=1.0),
    )

    assert result.indices.shape == (0,)
    assert result.noise.shape == (0,)


def test_cfar_1d_rejects_non_power_input() -> None:
    spec = CFAR1DSpec(training_cells=1, guard_cells=0, threshold_scale=1.0)

    with pytest.raises(ValueError, match="non-negative"):
        detect_cfar_1d(np.array([1.0, -1.0, 2.0]), spec)
