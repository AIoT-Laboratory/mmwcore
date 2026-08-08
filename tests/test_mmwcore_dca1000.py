from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass
from inspect import Parameter, signature

import numpy as np
import pytest

from mmwcore.core import ADCFrameSpec
from mmwcore.io import (
    DCA1000CLIController,
    DCA1000CLIError,
    DCA1000CLITimeoutError,
    DCA1000FrameReader,
    DCA1000NetworkConfig,
    DCA1000PacketSource,
)


def test_dca1000_network_config_exposes_addresses() -> None:
    config = DCA1000NetworkConfig(
        system_ip="192.168.1.10",
        dca_ip="192.168.1.20",
        data_port=5000,
        config_port=5002,
    )

    assert config.data_bind_address == ("192.168.1.10", 5000)
    assert config.config_bind_address == ("192.168.1.10", 5002)
    assert config.config_destination == ("192.168.1.20", 5002)


def test_dca1000_packet_source_binds_only_when_opened() -> None:
    fake_socket = FakeSocket([b"packet-0"])
    source = DCA1000PacketSource(socket_factory=lambda: fake_socket)

    assert fake_socket.bound_address is None

    source.open()

    assert fake_socket.bound_address == ("192.168.33.30", 4098)
    assert fake_socket.timeout_s == 1.0
    assert source.read_packet() == b"packet-0"


def test_dca1000_packet_source_is_iterable_and_closes_context() -> None:
    fake_socket = FakeSocket([b"packet-0", b"packet-1"])

    with DCA1000PacketSource(socket_factory=lambda: fake_socket) as source:
        assert next(source) == b"packet-0"
        assert next(source) == b"packet-1"

    assert fake_socket.closed is True


def test_dca1000_packet_source_rejects_read_before_open() -> None:
    source = DCA1000PacketSource(socket_factory=lambda: FakeSocket([]))

    with pytest.raises(RuntimeError, match="opened"):
        source.read_packet()


def test_dca1000_network_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="data_port"):
        DCA1000NetworkConfig(data_port=0)

    with pytest.raises(ValueError, match="timeout_s"):
        DCA1000NetworkConfig(timeout_s=0)


def test_dca1000_frame_reader_reads_raw_adc_frame_from_packet_source() -> None:
    fake_socket = FakeSocket(
        [
            _packet_bytes(packet_number=1, byte_count=0, payload=np.array([1, 2])),
            _packet_bytes(packet_number=2, byte_count=4, payload=np.array([3, 4])),
        ]
    )
    packet_source = DCA1000PacketSource(socket_factory=lambda: fake_socket)
    reader = DCA1000FrameReader(
        ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2),
        packet_source,
        initial_frame_byte_count=0,
        payload_values_per_packet=2,
    )

    with reader:
        raw, stats = reader.read_frame(frame_id="frame-0")

    assert raw.frame_id == "frame-0"
    assert raw.source == "dca1000://192.168.33.30:4098"
    np.testing.assert_array_equal(raw.samples, np.array([1, 2, 3, 4], dtype=np.int16))
    assert raw.metadata["packet_loss"] == 0
    assert stats.received_packets == 2
    assert fake_socket.closed is True


def test_dca1000_frame_reader_has_no_ignored_fill_value_parameter() -> None:
    parameters = signature(DCA1000FrameReader).parameters

    assert "fill_value" not in parameters
    assert parameters["initial_frame_byte_count"].default is Parameter.empty


def test_dca1000_frame_reader_supports_custom_frame_source() -> None:
    fake_socket = FakeSocket(
        [
            _packet_bytes(packet_number=1, byte_count=0, payload=np.array([1, 2])),
            _packet_bytes(packet_number=2, byte_count=4, payload=np.array([3, 4])),
        ]
    )
    packet_source = DCA1000PacketSource(socket_factory=lambda: fake_socket)
    reader = DCA1000FrameReader(
        ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2),
        packet_source,
        initial_frame_byte_count=0,
        frame_source="fixture",
        payload_values_per_packet=2,
    )

    reader.open()
    raw, _stats = reader.read_frame()
    reader.close()

    assert raw.source == "fixture"


def test_dca1000_frame_reader_advances_u48_origin_and_cannot_reopen() -> None:
    byte_mask = (1 << 48) - 1
    initial_byte_count = byte_mask - 3
    fake_socket = FakeSocket(
        [
            _packet_bytes((1 << 32) - 1, initial_byte_count, np.array([1, 2])),
            _packet_bytes(0, 0, np.array([3, 4])),
            _packet_bytes(1, 4, np.array([5, 6])),
            _packet_bytes(2, 8, np.array([7, 8])),
        ]
    )
    packet_source = DCA1000PacketSource(socket_factory=lambda: fake_socket)
    reader = DCA1000FrameReader(
        ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2),
        packet_source,
        initial_frame_byte_count=initial_byte_count,
        payload_values_per_packet=2,
    )

    with reader:
        first, _stats = reader.read_frame()
        second, _stats = reader.read_frame()
    with pytest.raises(RuntimeError, match="cannot be reopened"):
        reader.open()

    np.testing.assert_array_equal(first.samples, [1, 2, 3, 4])
    np.testing.assert_array_equal(second.samples, [5, 6, 7, 8])


def test_dca1000_frame_reader_rejects_double_open() -> None:
    packet_source = DCA1000PacketSource(socket_factory=lambda: FakeSocket([]))
    reader = DCA1000FrameReader(
        ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2),
        packet_source,
        initial_frame_byte_count=0,
        payload_values_per_packet=2,
    )

    reader.open()
    with pytest.raises(RuntimeError, match="already open"):
        reader.open()
    reader.close()


def test_dca1000_cli_controller_runs_command_with_config_path() -> None:
    calls = []

    def runner(args, **kwargs):
        calls.append((tuple(args), kwargs))
        return Completed(returncode=0, stdout=b"ok", stderr=b"")

    controller = DCA1000CLIController("DCA1000.exe", "config.json", runner=runner)

    result = controller.configure_fpga(timeout_s=7)

    assert result.command == ("DCA1000.exe", "fpga", "config.json")
    assert result.stdout == "ok"
    assert calls == [
        (
            ("DCA1000.exe", "fpga", "config.json"),
            {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "timeout": 7,
            },
        )
    ]


def test_dca1000_cli_controller_exposes_record_commands() -> None:
    commands = []

    def runner(args, **kwargs):
        commands.append((args[1], kwargs["timeout"]))
        return Completed(returncode=0)

    controller = DCA1000CLIController("cli", "config.json", runner=runner)

    controller.start_record()
    controller.stop_record(timeout_s=9)

    assert commands == [("start_record", 30.0), ("stop_record", 9)]


def test_dca1000_cli_controller_raises_on_nonzero_exit() -> None:
    controller = DCA1000CLIController(
        "cli",
        "config.json",
        runner=lambda *_args, **_kwargs: Completed(returncode=2, stderr=b"bad"),
    )

    with pytest.raises(DCA1000CLIError, match="failed with code 2"):
        controller.run_command("fpga")


def test_dca1000_cli_controller_raises_on_timeout() -> None:
    def runner(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    controller = DCA1000CLIController("cli", "config.json", runner=runner)

    with pytest.raises(DCA1000CLITimeoutError, match="timed out"):
        controller.run_command("fpga", timeout_s=1)


@dataclass
class Completed:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class FakeSocket:
    def __init__(self, packets: list[bytes]) -> None:
        self.packets = list(packets)
        self.bound_address: tuple[str, int] | None = None
        self.timeout_s: float | None = None
        self.closed = False

    def bind(self, address: tuple[str, int]) -> None:
        self.bound_address = address

    def settimeout(self, timeout: float | None) -> None:
        self.timeout_s = timeout

    def recvfrom(self, bufsize: int) -> tuple[bytes, object]:
        packet = self.packets.pop(0)
        return packet[:bufsize], ("192.168.33.180", 4098)

    def close(self) -> None:
        self.closed = True


def _packet_bytes(packet_number: int, byte_count: int, payload: np.ndarray) -> bytes:
    packet_header = struct.pack("<I", packet_number)
    byte_count_header = byte_count.to_bytes(6, byteorder="little", signed=False)
    return packet_header + byte_count_header + np.asarray(payload, dtype=np.int16).tobytes()
