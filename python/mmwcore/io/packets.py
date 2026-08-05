"""Python contracts and stream boundaries for native DCA1000 assembly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import ADCFrameSpec, RawADCFrame

DCA1000_PACKET_HEADER_BYTES = 10
DCA1000_PACKET_PAYLOAD_BYTES = 1456
DCA1000_PACKET_PAYLOAD_INT16_VALUES = DCA1000_PACKET_PAYLOAD_BYTES // np.dtype(np.int16).itemsize


@dataclass(frozen=True)
class DCA1000Packet:
    """One DCA1000 data packet with parsed header and int16 payload."""

    packet_number: int
    byte_count: int
    payload: np.ndarray

    def __post_init__(self) -> None:
        payload = np.asarray(self.payload, dtype=np.int16)
        if payload.ndim != 1:
            raise ValueError(f"DCA1000Packet.payload must be 1-D; got {payload.shape}.")
        if self.packet_number <= 0:
            raise ValueError(
                f"DCA1000Packet.packet_number must be positive; got {self.packet_number}."
            )
        if self.byte_count < 0:
            raise ValueError(
                f"DCA1000Packet.byte_count must be non-negative; got {self.byte_count}."
            )
        object.__setattr__(self, "payload", payload)


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
    packets_per_frame: int,
    payload_values_per_packet: int | None = None,
    fill_value: int = 0,
) -> tuple[np.ndarray, PacketLossStats]:
    """Reorder parsed packets through the native DCA1000 core."""

    packet_numbers, payloads = _packet_arrays(packets)
    result = _native.reorder_dca1000_packets(
        packet_numbers,
        payloads,
        packets_per_frame,
        payload_values_per_packet,
        fill_value,
    )
    return _native_assembly(result)


def assemble_dca1000_frame(
    packets: list[DCA1000Packet] | tuple[DCA1000Packet, ...],
    spec: ADCFrameSpec,
    *,
    frame_id: str | int | None = None,
    source: str | None = None,
    payload_values_per_packet: int = DCA1000_PACKET_PAYLOAD_INT16_VALUES,
    fill_value: int = 0,
) -> tuple[RawADCFrame, PacketLossStats]:
    """Assemble parsed packets into one typed ADC frame."""

    packets_per_frame = _packets_per_frame(spec.raw_values_per_frame, payload_values_per_packet)
    samples, stats = reorder_dca1000_packets(
        packets,
        packets_per_frame=packets_per_frame,
        payload_values_per_packet=payload_values_per_packet,
        fill_value=fill_value,
    )
    return _raw_adc_frame(
        samples[: spec.raw_values_per_frame],
        stats,
        frame_id=frame_id,
        source=source,
    )


def assemble_dca1000_frame_bytes(
    packets: Iterable[bytes],
    spec: ADCFrameSpec,
    *,
    frame_id: str | int | None = None,
    source: str | None = None,
    payload_values_per_packet: int = DCA1000_PACKET_PAYLOAD_INT16_VALUES,
    fill_value: int = 0,
) -> tuple[RawADCFrame, PacketLossStats]:
    """Parse and assemble one packet batch through the native DCA1000 core."""

    _packets_per_frame(spec.raw_values_per_frame, payload_values_per_packet)
    result = _native.assemble_dca1000_frame_bytes(
        tuple(bytes(packet) for packet in packets),
        spec.raw_values_per_frame,
        payload_values_per_packet,
        fill_value,
    )
    samples, stats = _native_assembly(result)
    return _raw_adc_frame(samples, stats, frame_id=frame_id, source=source)


def read_dca1000_frame_from_packets(
    packet_source: Iterable[bytes],
    spec: ADCFrameSpec,
    *,
    frame_id: str | int | None = None,
    source: str | None = None,
    payload_values_per_packet: int = DCA1000_PACKET_PAYLOAD_INT16_VALUES,
    fill_value: int = 0,
) -> tuple[RawADCFrame, PacketLossStats]:
    """Read one expected packet batch from a Python stream boundary."""

    packets_per_frame = _packets_per_frame(spec.raw_values_per_frame, payload_values_per_packet)
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
        frame_id=frame_id,
        source=source,
        payload_values_per_packet=payload_values_per_packet,
        fill_value=fill_value,
    )


def _packet_arrays(
    packets: list[DCA1000Packet] | tuple[DCA1000Packet, ...],
) -> tuple[NDArray[np.int64], tuple[NDArray[np.int16], ...]]:
    packet_numbers = np.asarray([packet.packet_number for packet in packets], dtype=np.int64)
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
    if payload_values_per_packet <= 0:
        raise ValueError("payload_values_per_packet must be positive.")
    return -(-raw_values_per_frame // payload_values_per_packet)
