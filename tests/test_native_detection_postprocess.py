from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_peak_grouping_preserves_reordered_axes_and_cyclic_doppler() -> None:
    canonical = np.ones((1, 5, 1, 3), dtype=np.complex64)
    canonical[0, 0, 0, 1] = 8.0
    canonical[0, 4, 0, 1] = 10.0
    data = np.transpose(canonical, (2, 0, 3, 1)).copy()
    candidates = np.array(
        ((0.0, 1.0, 0.0, 8.0), (0.0, 1.0, 4.0, 10.0)),
        dtype=np.float32,
    )

    retained = _native.group_range_doppler_candidates(
        data,
        (1, 3, 0, 2),
        1,
        candidates,
        (0, 1, 2),
        (0, 1, True, True),
    )

    np.testing.assert_array_equal(retained, [1])


def test_native_peak_grouping_honors_non_strict_plateaus() -> None:
    data = np.ones((1, 1, 1, 3), dtype=np.complex64)
    data[0, 0, 0, 1:] = 5.0
    candidates = np.array(((0.0, 1.0, 0.0), (0.0, 2.0, 0.0)), dtype=np.float32)

    strict = _native.group_range_doppler_candidates(
        data,
        (0, 1, 2, 3),
        0,
        candidates,
        (0, 1, 2),
        (1, 0, False, True),
    )
    non_strict = _native.group_range_doppler_candidates(
        data,
        (0, 1, 2, 3),
        0,
        candidates,
        (0, 1, 2),
        (1, 0, False, False),
    )

    np.testing.assert_array_equal(strict, [])
    np.testing.assert_array_equal(non_strict, [0, 1])


def test_native_quality_filter_keeps_inclusive_snr_threshold() -> None:
    candidates = np.array(((5.0, 1.0), (10.0, 2.0), (15.0, 3.0)), dtype=np.float32)

    retained = _native.filter_detection_quality_rows(candidates, 0, 10.0)

    np.testing.assert_array_equal(retained, [1, 2])


def test_native_detection_postprocess_rejects_invalid_boundary_inputs() -> None:
    data = np.ones((1, 1, 1, 1), dtype=np.complex64)
    candidates = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="contiguous"):
        _native.group_range_doppler_candidates(
            data,
            (0, 1, 2, 3),
            0,
            candidates[:, ::-1],
            (0, 1, 2),
            (1, 0, False, True),
        )
    with pytest.raises(ValueError, match="outside shape"):
        _native.group_range_doppler_candidates(
            data,
            (0, 1, 2, 3),
            0,
            np.array(((0.0, 1.0, 0.0),), dtype=np.float32),
            (0, 1, 2),
            (1, 0, False, True),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        _native.filter_detection_quality_rows(candidates, 0, 0.0)
