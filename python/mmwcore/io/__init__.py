"""Offline capture IO entry points for mmwcore."""

from __future__ import annotations

from ._mmwcli_contract import MmwcliRawCaptureContract
from .adc_file import ADCFileFrameReader, load_adc_cube, load_adc_file
from .capture_stream import (
    MMWCLI_CAPTURE_STREAM_SCHEMA_V1,
    MMWCLI_CAPTURE_STREAM_TERMINAL_SCHEMA_V1,
    CaptureStream,
    CaptureStreamAbort,
    CaptureStreamAborted,
    CaptureStreamCommit,
    CaptureStreamContract,
    CaptureStreamError,
    CaptureStreamReader,
    CaptureStreamStateError,
    ProvisionalADCFrame,
    ProvisionalRangeDopplerFrame,
    open_capture_stream,
)
from .mmwcli_capture import (
    MMWCLI_CAPTURE_SESSION_SCHEMA_V1,
    ADCFileCapture,
    RangeDopplerPreset,
    open_capture,
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

__all__ = [
    "DCA1000_PACKET_HEADER_BYTES",
    "ADCFileFrameReader",
    "ADCFileCapture",
    "DCA1000_PACKET_PAYLOAD_BYTES",
    "DCA1000_PACKET_PAYLOAD_INT16_VALUES",
    "DCA1000Packet",
    "PacketLossStats",
    "MMWCLI_CAPTURE_SESSION_SCHEMA_V1",
    "MMWCLI_CAPTURE_STREAM_SCHEMA_V1",
    "MMWCLI_CAPTURE_STREAM_TERMINAL_SCHEMA_V1",
    "MmwcliRawCaptureContract",
    "CaptureStreamAbort",
    "CaptureStreamAborted",
    "CaptureStream",
    "CaptureStreamCommit",
    "CaptureStreamContract",
    "CaptureStreamError",
    "CaptureStreamReader",
    "CaptureStreamStateError",
    "ProvisionalADCFrame",
    "ProvisionalRangeDopplerFrame",
    "RangeDopplerPreset",
    "assemble_dca1000_frame",
    "assemble_dca1000_frame_bytes",
    "load_adc_cube",
    "load_adc_file",
    "open_capture",
    "open_capture_stream",
    "parse_dca1000_packet",
    "read_dca1000_frame_from_packets",
    "reorder_dca1000_packets",
]
