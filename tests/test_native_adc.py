from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


@pytest.mark.parametrize(
    ("layout", "raw", "num_chirps", "num_rx", "num_samples", "expected"),
    [
        (
            0,
            [1, 10, 2, 20, 3, 30, 4, 40],
            1,
            2,
            2,
            [[[[1 + 10j, 2 + 20j], [3 + 30j, 4 + 40j]]]],
        ),
        (
            1,
            [1, 3, 10, 30, 2, 4, 20, 40],
            1,
            2,
            2,
            [[[[1 + 10j, 2 + 20j], [3 + 30j, 4 + 40j]]]],
        ),
        (
            2,
            [1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60, 7, 8, 70, 80],
            1,
            2,
            4,
            [[[[1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j], [5 + 50j, 6 + 60j, 7 + 70j, 8 + 80j]]]],
        ),
        (
            3,
            [1, 5, 2, 6, 10, 50, 20, 60, 3, 7, 4, 8, 30, 70, 40, 80],
            1,
            2,
            4,
            [[[[1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j], [5 + 50j, 6 + 60j, 7 + 70j, 8 + 80j]]]],
        ),
        (
            3,
            [1, 2, 3, 4, 10, 20, 30, 40],
            2,
            1,
            2,
            [[[[1 + 10j, 2 + 20j]], [[3 + 30j, 4 + 40j]]]],
        ),
    ],
)
def test_native_adc_decoder_preserves_layout_contract(
    layout: int,
    raw: list[int],
    num_chirps: int,
    num_rx: int,
    num_samples: int,
    expected: list[object],
) -> None:
    actual = _native.decode_adc_i16(
        np.array(raw, dtype=np.int16),
        num_chirps,
        num_rx,
        num_samples,
        layout,
        False,
    )

    assert actual.dtype == np.complex64
    np.testing.assert_array_equal(actual, np.array(expected, dtype=np.complex64))


def test_native_adc_decoder_drops_only_an_incomplete_tail() -> None:
    actual = _native.decode_adc_i16(
        np.array([1, 10, 2, 20, 999], dtype=np.int16),
        1,
        1,
        2,
        0,
        True,
    )

    np.testing.assert_array_equal(actual, np.array([[[[1 + 10j, 2 + 20j]]]], dtype=np.complex64))


def test_native_adc_decoder_preserves_frame_and_chirp_order() -> None:
    raw = np.array(
        [
            1,
            3,
            10,
            30,
            2,
            4,
            20,
            40,
            101,
            103,
            110,
            130,
            102,
            104,
            120,
            140,
            1001,
            1003,
            1010,
            1030,
            1002,
            1004,
            1020,
            1040,
            1101,
            1103,
            1110,
            1130,
            1102,
            1104,
            1120,
            1140,
        ],
        dtype=np.int16,
    )

    actual = _native.decode_adc_i16(raw, 2, 2, 2, 1, False)

    expected = np.array(
        [
            [
                [[1 + 10j, 2 + 20j], [3 + 30j, 4 + 40j]],
                [[101 + 110j, 102 + 120j], [103 + 130j, 104 + 140j]],
            ],
            [
                [[1001 + 1010j, 1002 + 1020j], [1003 + 1030j, 1004 + 1040j]],
                [[1101 + 1110j, 1102 + 1120j], [1103 + 1130j, 1104 + 1140j]],
            ],
        ],
        dtype=np.complex64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_native_adc_decoder_rejects_invalid_native_inputs() -> None:
    with pytest.raises(ValueError, match="whole number of frames"):
        _native.decode_adc_i16(np.array([1, 2, 3, 4, 5], dtype=np.int16), 1, 1, 2, 0, False)
    with pytest.raises(ValueError, match="even num_samples"):
        _native.decode_adc_i16(np.arange(6, dtype=np.int16), 1, 1, 3, 2, False)
    with pytest.raises(
        ValueError,
        match="num_chirps \\* num_rx \\* num_samples to be divisible by 4",
    ):
        _native.decode_adc_i16(np.arange(4, dtype=np.int16), 1, 2, 1, 3, False)
    with pytest.raises(ValueError, match="whole number of frames"):
        _native.decode_adc_i16(np.arange(9, dtype=np.int16), 1, 2, 2, 3, False)
    with pytest.raises(ValueError, match="contiguous"):
        _native.decode_adc_i16(np.arange(10, dtype=np.int16)[::2], 1, 1, 2, 0, False)
