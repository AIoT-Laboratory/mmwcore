"""Python contracts and stream boundaries for native DCA1000 assembly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from operator import index as integer_index

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import ADCFrameSpec, RawADCFrame

DCA1000_PACKET_HEADER_BYTES = 10
DCA1000_PACKET_PAYLOAD_BYTES = 1456
DCA1000_PACKET_PAYLOAD_INT16_VALUES = DCA1000_PACKET_PAYLOAD_BYTES // np.dtype(np.int16).itemsize
_INT16_MIN = -(1 << 15)
_INT16_MAX = (1 << 15) - 1
_UINT32_MAX = (1 << 32) - 1
DCA1000_BYTE_COUNT_MODULUS = 1 << 48
DCA1000_BYTE_COUNT_MASK = DCA1000_BYTE_COUNT_MODULUS - 1


@dataclass(frozen=True)
class DCA1000Packet:
    """One DCA1000 data packet with parsed header and int16 payload."""

    packet_number: int
    byte_count: int
    payload: np.ndarray

    def __post_init__(self) -> None:
        packet_number = _integer(self.packet_number, name="DCA1000Packet.packet_number")
        byte_count = _integer(self.byte_count, name="DCA1000Packet.byte_count")
        payload_values = np.asarray(self.payload)
        if payload_values.ndim != 1:
            raise ValueError(f"DCA1000Packet.payload must be 1-D; got {payload_values.shape}.")
        payload = _as_int16_array(payload_values, name="DCA1000Packet.payload")
        if packet_number < 0 or packet_number > _UINT32_MAX:
            raise ValueError(
                f"DCA1000Packet.packet_number must fit the unsigned 32-bit wire counter; "
                f"got {packet_number}."
            )
        if byte_count < 0:
            raise ValueError(f"DCA1000Packet.byte_count must be non-negative; got {byte_count}.")
        if byte_count > DCA1000_BYTE_COUNT_MASK:
            raise ValueError(
                f"DCA1000Packet.byte_count must fit the unsigned 48-bit wire counter; "
                f"got {byte_count}."
            )
        object.__setattr__(self, "packet_number", packet_number)
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "payload", payload)


def _integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    try:
        return integer_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc


def _as_int16_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == np.dtype(np.int16):
        return array

    if np.issubdtype(array.dtype, np.integer):
        if array.size and (int(array.min()) < _INT16_MIN or int(array.max()) > _INT16_MAX):
            raise ValueError(f"{name} contains values outside the int16 range.")
        return array.astype(np.int16, copy=False)

    if array.dtype == np.dtype(object):
        normalized = np.empty(array.shape, dtype=np.int16)
        for offset, value in enumerate(array.flat):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must contain integer values.")
            try:
                item = integer_index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must contain integer values.") from exc
            if item < _INT16_MIN or item > _INT16_MAX:
                raise ValueError(f"{name} contains values outside the int16 range.")
            normalized.flat[offset] = item
        return normalized

    raise TypeError(f"{name} must contain integer values; got dtype {array.dtype}.")


@dataclass(frozen=True)
class PacketLossStats:
    """Packet ordering summary for one expected frame."""

    expected_packets: int
    received_packets: int
    missing_packet_numbers: tuple[int, ...]
    duplicate_packet_numbers: tuple[int, ...]
    out_of_frame_packet_numbers: tuple[int, ...] = ()

    @property
    def missing_count(self) -> int:
        return len(self.missing_packet_numbers)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_packet_numbers)

    @property
    def out_of_frame_count(self) -> int:
        return len(self.out_of_frame_packet_numbers)


def parse_dca1000_packet(data: bytes | bytearray | memoryview) -> DCA1000Packet:
    """Parse one raw UDP packet through the native DCA1000 core."""

    packet_number, byte_count, payload = _native.parse_dca1000_packet(bytes(data))
    return DCA1000Packet(
        packet_number=packet_number,
        byte_count=byte_count,
        payload=payload,
    )


def reorder_dca1000_packets(
    packets: list[DCA1000Packet] | tuple[DCA1000Packet, ...],
    *,
    frame_start_packet_number: int,
    packets_per_frame: int,
    payload_values_per_packet: int | None = None,
    fill_value: int = 0,
) -> tuple[np.ndarray, PacketLossStats]:
    """Reorder from a caller-proven u32 packet-sequence origin."""

    packet_numbers, payloads = _packet_arrays(packets)
    frame_start_packet_number = _wire_packet_number(
        frame_start_packet_number,
        name="frame_start_packet_number",
    )
    if payload_values_per_packet is not None:
        payload_values_per_packet = _payload_value_count(payload_values_per_packet)
    result = _native.reorder_dca1000_packets(
        packet_numbers,
        payloads,
        frame_start_packet_number,
        packets_per_frame,
        payload_values_per_packet,
        fill_value,
    )
    return _native_assembly(result)


def assemble_dca1000_frame(
    packets: list[DCA1000Packet] | tuple[DCA1000Packet, ...],
    spec: ADCFrameSpec,
    *,
    frame_start_byte_count: int,
    frame_id: str | int | None = None,
    source: str | None = None,
    payload_values_per_packet: int = DCA1000_PACKET_PAYLOAD_INT16_VALUES,
) -> tuple[RawADCFrame, PacketLossStats]:
    """Assemble one frame from an externally proven radar-frame byte origin.

    The origin is never inferred from packet modulo or the first packet.
    """

    payload_values_per_packet = _payload_value_count(payload_values_per_packet)
    ordered_packets = _validate_exact_frame_packets(
        packets,
        raw_values_per_frame=spec.raw_values_per_frame,
        payload_values_per_packet=payload_values_per_packet,
        frame_start_byte_count=_wire_byte_count(
            frame_start_byte_count,
            name="frame_start_byte_count",
        ),
    )
    samples = np.concatenate([packet.payload for packet in ordered_packets])
    stats = PacketLossStats(
        expected_packets=len(ordered_packets),
        received_packets=len(ordered_packets),
        missing_packet_numbers=(),
        duplicate_packet_numbers=(),
    )
    return _raw_adc_frame(
        samples,
        stats,
        frame_id=frame_id,
        source=source,
    )


def assemble_dca1000_frame_bytes(
    packets: Iterable[bytes],
    spec: ADCFrameSpec,
    *,
    frame_start_byte_count: int,
    frame_id: str | int | None = None,
    source: str | None = None,
    payload_values_per_packet: int = DCA1000_PACKET_PAYLOAD_INT16_VALUES,
) -> tuple[RawADCFrame, PacketLossStats]:
    """Parse one frame using an externally proven radar-frame byte origin.

    The origin is never inferred from packet modulo or the first packet.
    """

    payload_values_per_packet = _payload_value_count(payload_values_per_packet)
    packets_per_frame = _packets_per_frame(
        spec.raw_values_per_frame,
        payload_values_per_packet,
    )
    frame_start_byte_count = _wire_byte_count(
        frame_start_byte_count,
        name="frame_start_byte_count",
    )
    packet_iter = iter(packets)
    packet_batch: list[bytes] = []
    for _ in range(packets_per_frame):
        try:
            packet_batch.append(bytes(next(packet_iter)))
        except StopIteration as error:
            raise ValueError(
                "Stateless DCA1000 frame assembly requires exactly "
                f"{packets_per_frame} packet(s); got {len(packet_batch)}."
            ) from error
    try:
        next(packet_iter)
    except StopIteration:
        pass
    else:
        raise ValueError(
            "Stateless DCA1000 frame assembly requires exactly "
            f"{packets_per_frame} packet(s); got at least {packets_per_frame + 1}."
        )
    result = _native.assemble_dca1000_frame_bytes(
        packet_batch,
        spec.raw_values_per_frame,
        payload_values_per_packet,
        frame_start_byte_count,
    )
    samples, stats = _native_assembly(result)
    return _raw_adc_frame(samples, stats, frame_id=frame_id, source=source)


def read_dca1000_frame_from_packets(
    packet_source: Iterable[bytes],
    spec: ADCFrameSpec,
    *,
    frame_start_byte_count: int,
    frame_id: str | int | None = None,
    source: str | None = None,
    payload_values_per_packet: int = DCA1000_PACKET_PAYLOAD_INT16_VALUES,
) -> tuple[RawADCFrame, PacketLossStats]:
    """Read one frame using an externally proven radar-frame byte origin.

    The origin is never inferred from packet modulo or the first packet.
    """

    payload_values_per_packet = _payload_value_count(payload_values_per_packet)
    packets_per_frame = _packets_per_frame(spec.raw_values_per_frame, payload_values_per_packet)
    frame_start_byte_count = _wire_byte_count(
        frame_start_byte_count,
        name="frame_start_byte_count",
    )
    packet_iter = iter(packet_source)
    packets: list[bytes] = []
    for _ in range(packets_per_frame):
        try:
            packets.append(next(packet_iter))
        except StopIteration as error:
            raise ValueError(
                "packet_source ended before one complete frame could be read; "
                f"expected {packets_per_frame} packets, got {len(packets)}."
            ) from error

    return assemble_dca1000_frame_bytes(
        packets,
        spec,
        frame_start_byte_count=frame_start_byte_count,
        frame_id=frame_id,
        source=source,
        payload_values_per_packet=payload_values_per_packet,
    )


def _packet_arrays(
    packets: list[DCA1000Packet] | tuple[DCA1000Packet, ...],
) -> tuple[NDArray[np.uint32], tuple[NDArray[np.int16], ...]]:
    packet_numbers = np.asarray([packet.packet_number for packet in packets], dtype=np.uint32)
    payloads = tuple(np.ascontiguousarray(packet.payload, dtype=np.int16) for packet in packets)
    return packet_numbers, payloads


def _native_assembly(
    result: _native.DCA1000AssemblyResult,
) -> tuple[NDArray[np.int16], PacketLossStats]:
    (
        samples,
        expected_packets,
        received_packets,
        missing_packet_numbers,
        duplicate_packet_numbers,
        out_of_frame_packet_numbers,
    ) = result
    stats = PacketLossStats(
        expected_packets=expected_packets,
        received_packets=received_packets,
        missing_packet_numbers=tuple(missing_packet_numbers),
        duplicate_packet_numbers=tuple(duplicate_packet_numbers),
        out_of_frame_packet_numbers=tuple(out_of_frame_packet_numbers),
    )
    return samples, stats


def _raw_adc_frame(
    samples: NDArray[np.int16],
    stats: PacketLossStats,
    *,
    frame_id: str | int | None,
    source: str | None,
) -> tuple[RawADCFrame, PacketLossStats]:
    raw = RawADCFrame(
        samples=samples,
        frame_id=frame_id,
        source=source,
        metadata={
            "packet_loss": stats.missing_count,
            "packet_duplicates": stats.duplicate_count,
            "packet_out_of_frame": stats.out_of_frame_count,
            "packets_per_frame": stats.expected_packets,
        },
    )
    return raw, stats


def _packets_per_frame(raw_values_per_frame: int, payload_values_per_packet: int) -> int:
    payload_values_per_packet = _payload_value_count(payload_values_per_packet)
    packets_per_frame, remainder = divmod(raw_values_per_frame, payload_values_per_packet)
    if remainder:
        raise ValueError(
            "raw_values_per_frame must be divisible by payload_values_per_packet "
            "for stateless DCA1000 frame assembly."
        )
    if packets_per_frame > 1 << 32:
        raise ValueError("DCA1000 packets_per_frame exceeds the unsigned 32-bit sequence space.")
    frame_bytes = raw_values_per_frame * np.dtype(np.int16).itemsize
    if frame_bytes > DCA1000_BYTE_COUNT_MODULUS:
        raise ValueError("DCA1000 frame byte span exceeds the unsigned 48-bit wire counter.")
    return packets_per_frame


def _validate_exact_frame_packets(
    packets: list[DCA1000Packet] | tuple[DCA1000Packet, ...],
    *,
    raw_values_per_frame: int,
    payload_values_per_packet: int,
    frame_start_byte_count: int,
) -> tuple[DCA1000Packet, ...]:
    packets_per_frame = _packets_per_frame(
        raw_values_per_frame,
        payload_values_per_packet,
    )
    if len(packets) != packets_per_frame:
        raise ValueError(
            "Stateless DCA1000 frame assembly requires exactly "
            f"{packets_per_frame} packet(s); got {len(packets)}."
        )
    for packet in packets:
        if packet.payload.size != payload_values_per_packet:
            raise ValueError(
                f"DCA1000 packet {packet.packet_number} payload contains "
                f"{packet.payload.size} int16 value(s); expected exactly "
                f"{payload_values_per_packet}."
            )

    payload_bytes = payload_values_per_packet * np.dtype(np.int16).itemsize
    packets_with_offsets = tuple(
        sorted(
            (
                ((packet.byte_count - frame_start_byte_count) & DCA1000_BYTE_COUNT_MASK, packet)
                for packet in packets
            ),
            key=lambda item: item[0],
        )
    )
    for offset, (relative_byte_count, packet) in enumerate(packets_with_offsets):
        expected_relative_byte_count = offset * payload_bytes
        expected_byte_count = (
            frame_start_byte_count + expected_relative_byte_count
        ) & DCA1000_BYTE_COUNT_MASK
        if relative_byte_count != expected_relative_byte_count:
            raise ValueError(
                f"DCA1000 packet {packet.packet_number} has byte_count "
                f"{packet.byte_count}; expected {expected_byte_count}."
            )
    first_packet_number = packets_with_offsets[0][1].packet_number
    for offset, (_relative_byte_count, packet) in enumerate(packets_with_offsets):
        expected_packet_number = (first_packet_number + offset) & _UINT32_MAX
        if packet.packet_number != expected_packet_number:
            raise ValueError(
                f"DCA1000 packet sequence has packet_number {packet.packet_number}; "
                f"expected {expected_packet_number} in byte_count order."
            )
    return tuple(packet for _relative_byte_count, packet in packets_with_offsets)


def _wire_packet_number(value: int, *, name: str) -> int:
    number = _integer(value, name=name)
    if number < 0 or number > _UINT32_MAX:
        raise ValueError(f"{name} must fit the unsigned 32-bit wire counter; got {number}.")
    return number


def _payload_value_count(value: int) -> int:
    count = _integer(value, name="payload_values_per_packet")
    if count <= 0:
        raise ValueError("payload_values_per_packet must be positive.")
    return count


def _wire_byte_count(value: int, *, name: str) -> int:
    byte_count = _integer(value, name=name)
    if byte_count < 0 or byte_count > DCA1000_BYTE_COUNT_MASK:
        raise ValueError(f"{name} must fit the unsigned 48-bit wire counter; got {byte_count}.")
    return byte_count
