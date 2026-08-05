"""DCA1000EVM CLI controller."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeGuard

from mmwcore.io.dca1000_types import (
    CompletedProcessLike,
    DCA1000CLIError,
    DCA1000CLIResult,
    DCA1000CLITimeoutError,
)


class DCA1000CLIController:
    """Narrow wrapper for the DCA1000EVM CLI executable."""

    def __init__(
        self,
        executable: str | Path,
        config_path: str | Path,
        *,
        runner: Callable[..., object] | None = None,
    ) -> None:
        self.executable = str(executable)
        self.config_path = str(config_path)
        self._runner = runner or subprocess.run

    def run_command(self, command: str, *, timeout_s: float = 15.0) -> DCA1000CLIResult:
        args = (self.executable, command, self.config_path)
        try:
            completed = self._runner(
                list(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
            if not _is_completed_process(completed):
                raise TypeError(f"Runner returned invalid type: {type(completed)}")
        except subprocess.TimeoutExpired as error:
            raise DCA1000CLITimeoutError(
                f"DCA1000 CLI command {command!r} timed out after {timeout_s} seconds."
            ) from error

        result = DCA1000CLIResult(
            command=args,
            returncode=int(completed.returncode),
            stdout=_decode_output(completed.stdout),
            stderr=_decode_output(completed.stderr),
        )
        if result.returncode != 0:
            raise DCA1000CLIError(
                f"DCA1000 CLI command {command!r} failed with code {result.returncode}: "
                f"{result.stderr or result.stdout}"
            )
        return result

    def configure_fpga(self, *, timeout_s: float = 15.0) -> DCA1000CLIResult:
        return self.run_command("fpga", timeout_s=timeout_s)

    def start_record(self, *, timeout_s: float = 30.0) -> DCA1000CLIResult:
        return self.run_command("start_record", timeout_s=timeout_s)

    def stop_record(self, *, timeout_s: float = 30.0) -> DCA1000CLIResult:
        return self.run_command("stop_record", timeout_s=timeout_s)


def _decode_output(output: object) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="ignore")
    return str(output or "")


def _is_completed_process(value: object) -> TypeGuard[CompletedProcessLike]:
    return (
        isinstance(getattr(value, "returncode", None), int)
        and isinstance(getattr(value, "stdout", None), bytes | str)
        and isinstance(getattr(value, "stderr", None), bytes | str)
    )
