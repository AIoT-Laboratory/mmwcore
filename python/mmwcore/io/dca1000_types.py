"""Shared DCA1000 types and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class DatagramSocket(Protocol):
    """Minimal socket protocol used by DCA1000PacketSource."""

    def bind(self, address: tuple[str, int], /) -> None: ...

    def settimeout(self, timeout: float | None, /) -> None: ...

    def recvfrom(self, bufsize: int, /) -> tuple[bytes, object]: ...

    def close(self) -> None: ...


class CompletedProcessLike(Protocol):
    """Minimal subprocess result protocol used by DCA1000CLIController."""

    returncode: int
    stdout: bytes | str
    stderr: bytes | str


@dataclass(frozen=True)
class DCA1000CLIResult:
    """Result from one DCA1000 CLI command."""

    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class DCA1000CLIError(RuntimeError):
    """DCA1000 CLI command failed."""


class DCA1000CLITimeoutError(TimeoutError):
    """DCA1000 CLI command timed out."""


@dataclass(frozen=True)
class DCA1000NetworkConfig:
    """Network addresses for receiving DCA1000 ADC data packets."""

    system_ip: str = "192.168.33.30"
    dca_ip: str = "192.168.33.180"
    data_port: int = 4098
    config_port: int = 4096
    timeout_s: float | None = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("data_port", self.data_port),
            ("config_port", self.config_port),
        ):
            if value <= 0:
                raise ValueError(f"DCA1000NetworkConfig.{name} must be positive; got {value}.")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(
                f"DCA1000NetworkConfig.timeout_s must be positive or None; got {self.timeout_s}."
            )

    @property
    def data_bind_address(self) -> tuple[str, int]:
        return (self.system_ip, self.data_port)

    @property
    def config_bind_address(self) -> tuple[str, int]:
        return (self.system_ip, self.config_port)

    @property
    def config_destination(self) -> tuple[str, int]:
        return (self.dca_ip, self.config_port)
