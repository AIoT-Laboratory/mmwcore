from __future__ import annotations

import struct

import numpy as np
import pytest

from mmwcore import _native


def test_native_dca1000_parser_preserves_header_and_payload() -> None:
    packet_number, byte_count, payload = _native.parse_dca1000_packet(
        _packet_bytes(7, 1456, np.array([1, -2, 3], dtype=np.int16))
    )

    assert packet_number == 7
    assert byte_count == 1456
    np.testing.assert_array_equal(payload, np.array([1, -2, 3], dtype=np.int16))


def test_native_dca1000_reorder_preserves_packet_loss_contract() -> None:
    result = _native.reorder_dca1000_packets(
        np.array([5, 7, 7, 9], dtype=np.uint32),
        (
            np.array([50], dtype=np.int16),
            np.array([70], dtype=np.int16),
            np.array([71], dtype=np.int16),
            np.array([90], dtype=np.int16),
        ),
        5,
        3,
        1,
        -1,
    )
    samples, expected, received, missing, duplicates, out_of_frame = result

    np.testing.assert_array_equal(samples, np.array([50, -1, 70], dtype=np.int16))
    assert expected == 3
    assert received == 4
    assert missing == [6]
    assert duplicates == [7]
    assert out_of_frame == [9]


def test_native_dca1000_batch_assembly_parses_before_reordering() -> None:
    result = _native.assemble_dca1000_frame_bytes(
        (
            _packet_bytes(2, 4, np.array([3, 4], dtype=np.int16)),
            _packet_bytes(1, 0, np.array([1, 2], dtype=np.int16)),
        ),
        4,
        2,
        0,
    )
    samples, expected, received, missing, duplicates, out_of_frame = result

    np.testing.assert_array_equal(samples, np.array([1, 2, 3, 4], dtype=np.int16))
    assert expected == 2
    assert received == 2
    assert missing == []
    assert duplicates == []
    assert out_of_frame == []


def test_native_dca1000_rejects_invalid_boundary_inputs() -> None:
    with pytest.raises(ValueError, match="10-byte header"):
        _native.parse_dca1000_packet(b"\x01\x00")
    with pytest.raises(ValueError, match="whole int16"):
        _native.parse_dca1000_packet(b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01")
    with pytest.raises(ValueError, match="same length"):
        _native.reorder_dca1000_packets(
            np.array([1], dtype=np.uint32),
            (),
            1,
            1,
            1,
            0,
        )


def _packet_bytes(packet_number: int, byte_count: int, payload: np.ndarray) -> bytes:
    return (
        struct.pack("<I", packet_number)
        + byte_count.to_bytes(6, byteorder="little", signed=False)
        + np.asarray(payload, dtype=np.int16).tobytes()
    )
