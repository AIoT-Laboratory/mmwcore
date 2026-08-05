from __future__ import annotations

import pytest

from mmwcore.io import RadarSerialConfig, RadarSerialController


def test_radar_serial_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="port"):
        RadarSerialConfig(port="")

    with pytest.raises(ValueError, match="baudrate"):
        RadarSerialConfig(port="COM1", baudrate=0)


def test_radar_serial_controller_opens_lazily_and_sends_commands() -> None:
    factory = FakeSerialFactory()
    sleeps = []
    controller = RadarSerialController(
        RadarSerialConfig(port="COM9", command_delay_s=0.1),
        serial_factory=factory,
        sleep=sleeps.append,
    )

    assert factory.created == []
    controller.open()
    sent = controller.send_command(" sensorStop ")

    assert sent == "sensorStop"
    assert factory.created == [("COM9", 115200, 1.0)]
    assert factory.port.writes == [b"sensorStop\n"]
    assert sleeps == [0.1]


def test_radar_serial_controller_loads_config_lines() -> None:
    factory = FakeSerialFactory()
    controller = RadarSerialController(
        RadarSerialConfig(port="COM9", command_delay_s=0.0),
        serial_factory=factory,
    )

    with controller:
        commands = controller.load_config_lines(
            [
                "",
                "% comment",
                "sensorStop",
                "profileCfg 0 60",
                "sensorStart",
            ]
        )

    assert commands == ("sensorStop", "profileCfg 0 60")
    assert factory.port.writes == [b"sensorStop\n", b"profileCfg 0 60\n"]
    assert factory.port.closed is True


def test_radar_serial_controller_can_include_sensor_start_and_use_helpers() -> None:
    factory = FakeSerialFactory()
    controller = RadarSerialController(
        RadarSerialConfig(port="COM9", command_delay_s=0.0),
        serial_factory=factory,
    )

    controller.open()
    commands = controller.load_config_lines(["sensorStart"], include_sensor_start=True)
    controller.sensor_stop()
    controller.sensor_start()

    assert commands == ("sensorStart",)
    assert factory.port.writes == [
        b"sensorStart\n",
        b"sensorStop\n",
        b"sensorStart\n",
    ]


def test_radar_serial_controller_rejects_send_before_open() -> None:
    controller = RadarSerialController(
        RadarSerialConfig(port="COM9"),
        serial_factory=FakeSerialFactory(),
    )

    with pytest.raises(RuntimeError, match="opened"):
        controller.send_command("sensorStop")


class FakeSerialFactory:
    def __init__(self) -> None:
        self.created: list[tuple[str, int, float]] = []
        self.port = FakeSerialPort()

    def __call__(self, port: str, baudrate: int, *, timeout: float) -> FakeSerialPort:
        self.created.append((port, baudrate, timeout))
        return self.port


class FakeSerialPort:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def close(self) -> None:
        self.closed = True
