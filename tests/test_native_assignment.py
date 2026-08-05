from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_assignment_matches_minimum_cost_square_and_rectangular_cases() -> None:
    rows, columns = _native.linear_sum_assignment(
        np.array(
            [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]],
            dtype=np.float64,
        )
    )
    np.testing.assert_array_equal(rows, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(columns, np.array([1, 0, 2], dtype=np.int64))

    rows, columns = _native.linear_sum_assignment(
        np.array([[10.0, 1.0], [1.0, 10.0], [2.0, 2.0]], dtype=np.float64)
    )
    np.testing.assert_array_equal(rows, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(columns, np.array([1, 0], dtype=np.int64))


def test_native_assignment_is_stable_for_empty_and_tied_costs() -> None:
    rows, columns = _native.linear_sum_assignment(np.ones((2, 3), dtype=np.float64))
    np.testing.assert_array_equal(rows, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(columns, np.array([0, 1], dtype=np.int64))

    rows, columns = _native.linear_sum_assignment(np.empty((0, 3), dtype=np.float64))
    assert rows.shape == (0,)
    assert columns.shape == (0,)


def test_native_assignment_rejects_noncontiguous_and_nonfinite_costs() -> None:
    costs = np.ones((3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="contiguous"):
        _native.linear_sum_assignment(costs[:, ::-1])
    with pytest.raises(ValueError, match="finite"):
        _native.linear_sum_assignment(np.array([[0.0, np.inf]], dtype=np.float64))
