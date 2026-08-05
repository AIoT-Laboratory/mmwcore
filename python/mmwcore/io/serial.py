"""Narrow serial controller for TI mmWave radar CLI ports."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self


class SerialPort(Protocol):
    """Minimal serial-port protocol used by RadarSerialController."""

    def write(self, data: bytes) -> object: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RadarSerialConfig:
    """Serial settings for a radar CLI port."""

    port: str
    baudrate: int = 115200
    timeout_s: float = 1.0
    command_delay_s: float = 0.2

    def __post_init__(self) -> None:
        if not self.port:
            raise ValueError("RadarSerialConfig.port must not be empty.")
        if self.baudrate <= 0:
            raise ValueError(f"RadarSerialConfig.baudrate must be positive; got {self.baudrate}.")
        if self.timeout_s < 0:
            raise ValueError(
                f"RadarSerialConfig.timeout_s must be non-negative; got {self.timeout_s}."
            )
        if self.command_delay_s < 0:
            raise ValueError(
                "RadarSerialConfig.command_delay_s must be non-negative; "
                f"got {self.command_delay_s}."
            )


class RadarSerialController:
    """Upload radar CLI commands through an injected serial-port factory."""

    def __init__(
        self,
        config: RadarSerialConfig,
        *,
        serial_factory: Callable[..., SerialPort],
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._serial_factory = serial_factory
        self._sleep = sleep
        self._port: SerialPort | None = None

    def open(self) -> Self:
        if self._port is None:
            self._port = self._serial_factory(
                self.config.port,
                self.config.baudrate,
                timeout=self.config.timeout_s,
            )
        return self

    def close(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None

    def send_command(self, command: str, *, delay_s: float | None = None) -> str:
        if self._port is None:
            raise RuntimeError("RadarSerialController must be opened before sending commands.")
        normalized = command.strip()
        if not normalized:
            raise ValueError("command must not be empty.")

        self._port.write(f"{normalized}\n".encode("ascii"))
        wait_s = self.config.command_delay_s if delay_s is None else delay_s
        if wait_s > 0:
            self._sleep(wait_s)
        return normalized

    def load_config_lines(
        self,
        lines: Iterable[str],
        *,
        include_sensor_start: bool = False,
    ) -> tuple[str, ...]:
        commands = tuple(_iter_config_commands(lines, include_sensor_start=include_sensor_start))
        for command in commands:
            self.send_command(command)
        return commands

    def load_config_file(
        self,
        path: str | Path,
        *,
        include_sensor_start: bool = False,
    ) -> tuple[str, ...]:
        config_path = Path(path)
        return self.load_config_lines(
            config_path.read_text(encoding="utf-8").splitlines(),
            include_sensor_start=include_sensor_start,
        )

    def sensor_start(self) -> str:
        return self.send_command("sensorStart")

    def sensor_stop(self) -> str:
        return self.send_command("sensorStop")

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _iter_config_commands(
    lines: Iterable[str],
    *,
    include_sensor_start: bool,
) -> Iterable[str]:
    for line in lines:
        command = line.strip()
        if not command or command.startswith("%"):
            continue
        if command == "sensorStart" and not include_sensor_start:
            continue
        yield command
