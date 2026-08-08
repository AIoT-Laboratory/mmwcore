from __future__ import annotations

import struct
from collections.abc import Iterator

import numpy as np
import pytest

from mmwcore.core import ADCFrameSpec
from mmwcore.io import (
    DCA1000Packet,
    assemble_dca1000_frame,
    assemble_dca1000_frame_bytes,
    parse_dca1000_packet,
    read_dca1000_frame_from_packets,
    reorder_dca1000_packets,
)


def test_parse_dca1000_packet_reads_header_and_payload() -> None:
    raw = _packet_bytes(packet_number=7, byte_count=1456, payload=np.array([1, -2, 3]))

    packet = parse_dca1000_packet(raw)

    assert packet.packet_number == 7
    assert packet.byte_count == 1456
    np.testing.assert_array_equal(packet.payload, np.array([1, -2, 3], dtype=np.int16))


def test_parse_dca1000_packet_preserves_high_u32_sequence_values() -> None:
    packet = parse_dca1000_packet(
        _packet_bytes(packet_number=(1 << 32) - 1, byte_count=0, payload=np.array([1]))
    )

    assert packet.packet_number == (1 << 32) - 1


def test_parse_dca1000_packet_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="10-byte header"):
        parse_dca1000_packet(b"\x01\x00")


def test_dca1000_packet_preserves_safe_native_integer_values() -> None:
    packet = DCA1000Packet(
        packet_number=np.uint32(np.iinfo(np.uint32).max),  # type: ignore[arg-type]
        byte_count=np.uint64((1 << 48) - 1),  # type: ignore[arg-type]
        payload=np.array([-32768, 32767], dtype=np.int32),
    )

    assert packet.packet_number == np.iinfo(np.uint32).max
    assert packet.byte_count == (1 << 48) - 1
    assert type(packet.packet_number) is int
    assert type(packet.byte_count) is int
    assert packet.payload.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(packet.payload, [-32768, 32767])


def test_dca1000_packet_reuses_native_int16_payload() -> None:
    payload = np.array([1, -2, 3], dtype=np.int16)

    packet = DCA1000Packet(1, 0, payload)

    assert packet.payload is payload


def test_dca1000_packet_normalizes_object_integer_payload() -> None:
    packet = DCA1000Packet(1, 0, np.array([-2, 3], dtype=object))

    assert packet.payload.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(packet.payload, [-2, 3])


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("packet_number", 1.0),
        ("packet_number", True),
        ("byte_count", 0.0),
        ("byte_count", False),
    ],
)
def test_dca1000_packet_rejects_non_integer_header_fields(
    field_name: str,
    value: object,
) -> None:
    packet_number = value if field_name == "packet_number" else 1
    byte_count = value if field_name == "byte_count" else 0

    with pytest.raises(TypeError, match=rf"DCA1000Packet\.{field_name} must be an integer"):
        DCA1000Packet(
            packet_number,  # type: ignore[arg-type]
            byte_count,  # type: ignore[arg-type]
            np.array([0], dtype=np.int16),
        )


@pytest.mark.parametrize("packet_number", [-1, 1 << 32])
def test_dca1000_packet_rejects_packet_numbers_outside_wire_uint32(
    packet_number: int,
) -> None:
    with pytest.raises(ValueError, match="packet_number must fit the unsigned 32-bit"):
        DCA1000Packet(packet_number, 0, np.array([0], dtype=np.int16))


@pytest.mark.parametrize("byte_count", [-1, 1 << 48])
def test_dca1000_packet_rejects_byte_count_outside_wire_uint48(byte_count: int) -> None:
    with pytest.raises(
        ValueError, match="byte_count must (?:be non-negative|fit the unsigned 48-bit)"
    ):
        DCA1000Packet(1, byte_count, np.array([0], dtype=np.int16))


@pytest.mark.parametrize("payload", [np.array([1.0]), np.array([True])])
def test_dca1000_packet_rejects_non_integer_payload(payload: np.ndarray) -> None:
    with pytest.raises(TypeError, match="payload must contain integer values"):
        DCA1000Packet(1, 0, payload)


@pytest.mark.parametrize(
    "payload",
    [
        np.array([-32769], dtype=np.int32),
        np.array([32768], dtype=np.uint16),
        np.array([1 << 63], dtype=np.uint64),
    ],
)
def test_dca1000_packet_rejects_payload_outside_int16(payload: np.ndarray) -> None:
    with pytest.raises(ValueError, match="payload contains values outside the int16 range"):
        DCA1000Packet(1, 0, payload)


def test_reorder_dca1000_packets_places_out_of_order_payloads() -> None:
    packets = [
        DCA1000Packet(2, 4, np.array([20, 21], dtype=np.int16)),
        DCA1000Packet(1, 0, np.array([10, 11], dtype=np.int16)),
        DCA1000Packet(3, 8, np.array([30, 31], dtype=np.int16)),
    ]

    frame, stats = reorder_dca1000_packets(
        packets,
        frame_start_packet_number=1,
        packets_per_frame=3,
    )

    np.testing.assert_array_equal(frame, np.array([10, 11, 20, 21, 30, 31]))
    assert stats.expected_packets == 3
    assert stats.received_packets == 3
    assert stats.missing_count == 0
    assert stats.duplicate_count == 0


def test_reorder_dca1000_packets_reports_missing_duplicate_and_out_of_frame() -> None:
    packets = [
        DCA1000Packet(5, 0, np.array([50], dtype=np.int16)),
        DCA1000Packet(7, 0, np.array([70], dtype=np.int16)),
        DCA1000Packet(7, 0, np.array([71], dtype=np.int16)),
        DCA1000Packet(9, 0, np.array([90], dtype=np.int16)),
    ]

    frame, stats = reorder_dca1000_packets(
        packets,
        frame_start_packet_number=5,
        packets_per_frame=3,
        payload_values_per_packet=1,
        fill_value=-1,
    )

    np.testing.assert_array_equal(frame, np.array([50, -1, 70], dtype=np.int16))
    assert stats.missing_packet_numbers == (6,)
    assert stats.duplicate_packet_numbers == (7,)
    assert stats.out_of_frame_packet_numbers == (9,)


def test_reorder_dca1000_packets_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        reorder_dca1000_packets([], frame_start_packet_number=0, packets_per_frame=1)


def test_reorder_dca1000_packets_uses_explicit_u32_wrapping_origin() -> None:
    packets = [
        DCA1000Packet(0, 0, np.array([2], dtype=np.int16)),
        DCA1000Packet((1 << 32) - 1, 0, np.array([1], dtype=np.int16)),
        DCA1000Packet(2, 0, np.array([9], dtype=np.int16)),
    ]

    frame, stats = reorder_dca1000_packets(
        packets,
        frame_start_packet_number=(1 << 32) - 1,
        packets_per_frame=2,
        payload_values_per_packet=1,
        fill_value=-1,
    )

    np.testing.assert_array_equal(frame, np.array([1, 2], dtype=np.int16))
    assert stats.missing_packet_numbers == ()
    assert stats.out_of_frame_packet_numbers == (2,)


def test_assemble_dca1000_frame_sizes_raw_adc_frame_from_spec() -> None:
    packets = [
        DCA1000Packet(2, 8, np.array([5, 6, 7, 8], dtype=np.int16)),
        DCA1000Packet(1, 0, np.array([1, 2, 3, 4], dtype=np.int16)),
    ]
    spec = ADCFrameSpec(num_chirps=1, num_rx=2, num_samples=2)

    raw, stats = assemble_dca1000_frame(
        packets,
        spec,
        frame_start_byte_count=0,
        frame_id="frame-0",
        source="udp",
        payload_values_per_packet=4,
    )

    assert raw.frame_id == "frame-0"
    assert raw.source == "udp"
    np.testing.assert_array_equal(raw.samples, np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    assert stats.missing_count == 0
    assert raw.metadata["packets_per_frame"] == 2


def test_assemble_dca1000_frame_rejects_incomplete_batches() -> None:
    packets = [DCA1000Packet(3, 0, np.array([9, 10], dtype=np.int16))]
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match=r"requires exactly 2 packet\(s\); got 1"):
        assemble_dca1000_frame(
            packets,
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=2,
        )


@pytest.mark.parametrize(
    ("packets", "message"),
    [
        (
            [
                DCA1000Packet(1, 0, np.array([1, 2], dtype=np.int16)),
                DCA1000Packet(2, 4, np.array([3, 4], dtype=np.int16)),
                DCA1000Packet(3, 8, np.array([5, 6], dtype=np.int16)),
            ],
            r"requires exactly 2 packet\(s\); got 3",
        ),
        (
            [
                DCA1000Packet(1, 0, np.array([1], dtype=np.int16)),
                DCA1000Packet(2, 4, np.array([3, 4], dtype=np.int16)),
            ],
            "payload contains 1 int16 value",
        ),
        (
            [
                DCA1000Packet(2, 4, np.array([3, 4], dtype=np.int16)),
                DCA1000Packet(3, 8, np.array([5, 6], dtype=np.int16)),
            ],
            "byte_count 4; expected 0",
        ),
        (
            [
                DCA1000Packet(1, 0, np.array([1, 2], dtype=np.int16)),
                DCA1000Packet(2, 8, np.array([5, 6], dtype=np.int16)),
            ],
            "byte_count 8; expected 4",
        ),
        (
            [
                DCA1000Packet(1, 0, np.array([1, 2], dtype=np.int16)),
                DCA1000Packet(1, 4, np.array([3, 4], dtype=np.int16)),
            ],
            "packet_number 1; expected 2",
        ),
    ],
)
def test_assemble_dca1000_frame_rejects_unproven_frame_batches(
    packets: list[DCA1000Packet],
    message: str,
) -> None:
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match=message):
        assemble_dca1000_frame(
            packets,
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=2,
        )


def test_assemble_dca1000_frame_rejects_non_integral_packet_counts() -> None:
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match="must be divisible"):
        assemble_dca1000_frame(
            [],
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=3,
        )


@pytest.mark.parametrize("payload_values_per_packet", [True, 2.0])
def test_assemble_dca1000_frame_rejects_non_integer_payload_counts(
    payload_values_per_packet: object,
) -> None:
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(TypeError, match="payload_values_per_packet must be an integer"):
        assemble_dca1000_frame(
            [],
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=payload_values_per_packet,  # type: ignore[arg-type]
        )


def test_assemble_dca1000_frame_accepts_u32_and_u48_counter_wrap() -> None:
    frame_start_byte_count = (1 << 48) - 4
    packets = [
        DCA1000Packet(0, 0, np.array([3, 4], dtype=np.int16)),
        DCA1000Packet(
            (1 << 32) - 1,
            frame_start_byte_count,
            np.array([1, 2], dtype=np.int16),
        ),
    ]
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    raw, _stats = assemble_dca1000_frame(
        packets,
        spec,
        frame_start_byte_count=frame_start_byte_count,
        payload_values_per_packet=2,
    )

    np.testing.assert_array_equal(raw.samples, np.array([1, 2, 3, 4], dtype=np.int16))


def test_assemble_dca1000_frame_bytes_parses_packets_before_assembly() -> None:
    packets = [
        _packet_bytes(packet_number=2, byte_count=4, payload=np.array([3, 4])),
        _packet_bytes(packet_number=1, byte_count=0, payload=np.array([1, 2])),
    ]
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    raw, stats = assemble_dca1000_frame_bytes(
        packets,
        spec,
        frame_start_byte_count=0,
        source="udp-bytes",
        payload_values_per_packet=2,
    )

    assert raw.source == "udp-bytes"
    np.testing.assert_array_equal(raw.samples, np.array([1, 2, 3, 4], dtype=np.int16))
    assert stats.received_packets == 2


def test_assemble_dca1000_frame_bytes_rejects_inexact_payloads() -> None:
    packets = [
        _packet_bytes(packet_number=1, byte_count=0, payload=np.array([1])),
        _packet_bytes(packet_number=2, byte_count=4, payload=np.array([3, 4])),
    ]
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match="contains 12 bytes; expected exactly 14"):
        assemble_dca1000_frame_bytes(
            packets,
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=2,
        )


def test_assemble_dca1000_frame_bytes_rejects_extra_packets_with_bounded_read() -> None:
    consumed = 0

    def packets() -> Iterator[bytes]:
        nonlocal consumed
        packet_number = 1
        while True:
            consumed += 1
            yield _packet_bytes(
                packet_number=packet_number,
                byte_count=(packet_number - 1) * 4,
                payload=np.array([packet_number, packet_number]),
            )
            packet_number += 1

    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match=r"exactly 2 packet\(s\); got at least 3"):
        assemble_dca1000_frame_bytes(
            packets(),
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=2,
        )

    assert consumed == 3


def test_assemble_dca1000_frame_bytes_preflights_u32_count_before_consuming() -> None:
    def packets() -> Iterator[bytes]:
        raise AssertionError("packet source was consumed")
        yield b""

    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=(1 << 31) + 1)

    with pytest.raises(ValueError, match="packets_per_frame exceeds the unsigned 32-bit"):
        assemble_dca1000_frame_bytes(
            packets(),
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=1,
        )


def test_read_dca1000_frame_from_packets_consumes_one_expected_frame() -> None:
    packet_iter = iter(
        [
            _packet_bytes(packet_number=1, byte_count=0, payload=np.array([1, 2])),
            _packet_bytes(packet_number=2, byte_count=4, payload=np.array([3, 4])),
            _packet_bytes(packet_number=3, byte_count=8, payload=np.array([5, 6])),
        ]
    )
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    raw, stats = read_dca1000_frame_from_packets(
        packet_iter,
        spec,
        frame_start_byte_count=0,
        frame_id="stream-0",
        payload_values_per_packet=2,
    )

    assert raw.frame_id == "stream-0"
    np.testing.assert_array_equal(raw.samples, np.array([1, 2, 3, 4], dtype=np.int16))
    assert stats.received_packets == 2
    remaining = parse_dca1000_packet(next(packet_iter))
    assert remaining.packet_number == 3


def test_read_dca1000_frame_from_packets_rejects_short_sources() -> None:
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match="expected 2 packets, got 1"):
        read_dca1000_frame_from_packets(
            [_packet_bytes(packet_number=1, byte_count=0, payload=np.array([1, 2]))],
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=2,
        )


def test_read_dca1000_frame_rejects_non_integral_counts_before_consuming_source() -> None:
    def packets() -> Iterator[bytes]:
        raise AssertionError("packet source was consumed")
        yield b""

    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match="must be divisible"):
        read_dca1000_frame_from_packets(
            packets(),
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=3,
        )


def test_read_dca1000_frame_rejects_invalid_origin_before_consuming_source() -> None:
    def packets() -> Iterator[bytes]:
        raise AssertionError("packet source was consumed")
        yield b""

    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match="frame_start_byte_count must fit the unsigned 48-bit"):
        read_dca1000_frame_from_packets(
            packets(),
            spec,
            frame_start_byte_count=1 << 48,
            payload_values_per_packet=2,
        )


@pytest.mark.parametrize(
    ("spec", "payload_values_per_packet", "message"),
    [
        (
            ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=(1 << 31) + 1),
            1,
            "packets_per_frame exceeds the unsigned 32-bit",
        ),
        (
            ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=(1 << 47) + 1),
            (1 << 48) + 2,
            "frame byte span exceeds the unsigned 48-bit",
        ),
    ],
)
def test_read_dca1000_frame_preflights_counter_limits_before_consuming_source(
    spec: ADCFrameSpec,
    payload_values_per_packet: int,
    message: str,
) -> None:
    def packets() -> Iterator[bytes]:
        raise AssertionError("packet source was consumed")
        yield b""

    with pytest.raises(ValueError, match=message):
        read_dca1000_frame_from_packets(
            packets(),
            spec,
            frame_start_byte_count=0,
            payload_values_per_packet=payload_values_per_packet,
        )


def _packet_bytes(packet_number: int, byte_count: int, payload: np.ndarray) -> bytes:
    packet_header = struct.pack("<I", packet_number)
    byte_count_header = byte_count.to_bytes(6, byteorder="little", signed=False)
    return packet_header + byte_count_header + np.asarray(payload, dtype=np.int16).tobytes()
