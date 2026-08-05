"""Offline and hardware IO entry points for mmwcore."""

from __future__ import annotations

from .adc_file import ADCFileFrameReader, load_adc_cube, load_adc_file
from .dca1000_cli import DCA1000CLIController
from .dca1000_packetsource import (
    DCA1000_MAX_PACKET_BYTES,
    DCA1000FrameReader,
    DCA1000PacketSource,
)
from .dca1000_types import (
    DCA1000CLIError,
    DCA1000CLIResult,
    DCA1000CLITimeoutError,
    DCA1000NetworkConfig,
)
from .packets import (
    DCA1000_PACKET_HEADER_BYTES,
    DCA1000_PACKET_PAYLOAD_BYTES,
    DCA1000_PACKET_PAYLOAD_INT16_VALUES,
    DCA1000Packet,
    PacketLossStats,
    assemble_dca1000_frame,
    assemble_dca1000_frame_bytes,
    parse_dca1000_packet,
    read_dca1000_frame_from_packets,
    reorder_dca1000_packets,
)
from .serial import RadarSerialConfig, RadarSerialController

__all__ = [
    "DCA1000_PACKET_HEADER_BYTES",
    "ADCFileFrameReader",
    "DCA1000_PACKET_PAYLOAD_BYTES",
    "DCA1000_PACKET_PAYLOAD_INT16_VALUES",
    "DCA1000CLIController",
    "DCA1000CLIError",
    "DCA1000CLIResult",
    "DCA1000CLITimeoutError",
    "DCA1000_MAX_PACKET_BYTES",
    "DCA1000FrameReader",
    "DCA1000NetworkConfig",
    "DCA1000PacketSource",
    "DCA1000Packet",
    "PacketLossStats",
    "RadarSerialConfig",
    "RadarSerialController",
    "assemble_dca1000_frame",
    "assemble_dca1000_frame_bytes",
    "load_adc_cube",
    "load_adc_file",
    "parse_dca1000_packet",
    "read_dca1000_frame_from_packets",
    "reorder_dca1000_packets",
]
