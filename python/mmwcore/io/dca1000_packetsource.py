"""DCA1000 UDP packet source and frame reader."""

from __future__ import annotations

import socket
from collections.abc import Callable
from types import TracebackType
from typing import cast

from mmwcore._compat import Self
from mmwcore.core import ADCFrameSpec, RawADCFrame
from mmwcore.io.dca1000_types import DatagramSocket, DCA1000NetworkConfig
from mmwcore.io.packets import (
    DCA1000_PACKET_PAYLOAD_INT16_VALUES,
    PacketLossStats,
    read_dca1000_frame_from_packets,
)

DCA1000_MAX_PACKET_BYTES = 4096


class DCA1000PacketSource:
    """Iterable raw UDP packet source for DCA1000 ADC data."""

    def __init__(
        self,
        config: DCA1000NetworkConfig | None = None,
        *,
        max_packet_bytes: int = DCA1000_MAX_PACKET_BYTES,
        socket_factory: Callable[[], DatagramSocket] | None = None,
    ) -> None:
        self.config = config or DCA1000NetworkConfig()
        if max_packet_bytes <= 0:
            raise ValueError(f"max_packet_bytes must be positive; got {max_packet_bytes}.")
        self.max_packet_bytes = max_packet_bytes
        self._socket_factory = socket_factory or _udp_socket
        self._socket: DatagramSocket | None = None

    def open(self) -> Self:
        if self._socket is None:
            udp_socket = self._socket_factory()
            udp_socket.bind(self.config.data_bind_address)
            udp_socket.settimeout(self.config.timeout_s)
            self._socket = udp_socket
        return self

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def read_packet(self) -> bytes:
        if self._socket is None:
            raise RuntimeError("DCA1000PacketSource must be opened before reading.")
        data, _address = self._socket.recvfrom(self.max_packet_bytes)
        return data

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> bytes:
        return self.read_packet()

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _udp_socket() -> DatagramSocket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    return cast(DatagramSocket, sock)


class DCA1000FrameReader:
    """Read RawADCFrame objects from a DCA1000 packet source."""

    def __init__(
        self,
        adc_spec: ADCFrameSpec,
        packet_source: DCA1000PacketSource | None = None,
        *,
        frame_source: str | None = None,
        payload_values_per_packet: int = DCA1000_PACKET_PAYLOAD_INT16_VALUES,
        fill_value: int = 0,
    ) -> None:
        if payload_values_per_packet <= 0:
            raise ValueError("payload_values_per_packet must be positive.")
        self.adc_spec = adc_spec
        self.packet_source = packet_source or DCA1000PacketSource()
        self.frame_source = frame_source
        self.payload_values_per_packet = payload_values_per_packet
        self.fill_value = fill_value

    def open(self) -> Self:
        self.packet_source.open()
        return self

    def close(self) -> None:
        self.packet_source.close()

    def read_frame(
        self,
        *,
        frame_id: str | int | None = None,
    ) -> tuple[RawADCFrame, PacketLossStats]:
        return read_dca1000_frame_from_packets(
            self.packet_source,
            self.adc_spec,
            frame_id=frame_id,
            source=self.frame_source or _default_frame_source(self.packet_source.config),
            payload_values_per_packet=self.payload_values_per_packet,
            fill_value=self.fill_value,
        )

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _default_frame_source(config: DCA1000NetworkConfig) -> str:
    host, port = config.data_bind_address
    return f"dca1000://{host}:{port}"
