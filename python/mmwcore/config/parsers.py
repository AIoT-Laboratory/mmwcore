"""Parsers for existing radar configuration files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mmwcore.core import ADCComplexLayout, ADCFrameSpec


@dataclass(frozen=True)
class TiCliConfigSummary:
    """Minimal capture-shape summary parsed from a TI mmWave SDK CLI config."""

    num_rx: int
    num_tx: int
    num_adc_samples: int
    num_loops: int
    num_chirps_per_loop: int
    frame_chirp_start_index: int
    frame_chirp_end_index: int
    frame_periodicity_s: float | None = None

    @property
    def chirps_per_frame(self) -> int:
        return self.num_chirps_per_loop * self.num_loops

    @property
    def num_chirps_per_tx(self) -> int:
        return self.chirps_per_frame // self.num_tx

    def to_adc_frame_spec(
        self,
        *,
        layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED,
    ) -> ADCFrameSpec:
        return ADCFrameSpec(
            num_chirps=self.chirps_per_frame,
            num_rx=self.num_rx,
            num_samples=self.num_adc_samples,
            layout=layout,
        )


@dataclass
class _TiCliParseState:
    rx_mask: int | None = None
    tx_mask: int | None = None
    num_adc_samples: int | None = None
    chirp_tx_masks: dict[int, int] = field(default_factory=dict)
    frame_chirp_start: int | None = None
    frame_chirp_end: int | None = None
    num_loops: int | None = None
    frame_periodicity_s: float | None = None


def parse_ti_cli_config_file(path: str | Path) -> TiCliConfigSummary:
    """Parse a TI mmWave SDK CLI config file from disk."""

    return parse_ti_cli_config(Path(path).read_text(encoding="utf-8"))


def parse_ti_cli_config(text: str) -> TiCliConfigSummary:
    """Parse the capture shape from a TI mmWave SDK CLI config string."""

    state = _TiCliParseState()
    for line in _config_lines(text):
        _parse_config_command(line.split(), state)
    return _summarize_state(state)


def _parse_config_command(fields: list[str], state: _TiCliParseState) -> None:
    command = fields[0]
    if command == "channelCfg":
        _require_field_count(fields, 3, command)
        state.rx_mask = int(fields[1])
        state.tx_mask = int(fields[2])
    elif command == "profileCfg":
        _require_field_count(fields, 11, command)
        state.num_adc_samples = int(fields[10])
    elif command == "chirpCfg":
        _require_field_count(fields, 9, command)
        start = int(fields[1])
        end = int(fields[2])
        chirp_tx_mask = int(fields[8])
        for chirp_index in range(start, end + 1):
            state.chirp_tx_masks[chirp_index] = chirp_tx_mask
    elif command == "frameCfg":
        _require_field_count(fields, 6, command)
        state.frame_chirp_start = int(fields[1])
        state.frame_chirp_end = int(fields[2])
        state.num_loops = int(fields[3])
        state.frame_periodicity_s = float(fields[5]) / 1e3


def _summarize_state(state: _TiCliParseState) -> TiCliConfigSummary:
    missing = [
        name
        for name, value in (
            ("channelCfg", state.rx_mask),
            ("profileCfg", state.num_adc_samples),
            ("frameCfg", state.frame_chirp_start),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"TI CLI config missing required command(s): {', '.join(missing)}")

    rx_mask = cast(int, state.rx_mask)
    tx_mask = cast(int, state.tx_mask)
    num_adc_samples = cast(int, state.num_adc_samples)
    frame_chirp_start = cast(int, state.frame_chirp_start)
    frame_chirp_end = cast(int, state.frame_chirp_end)
    num_loops = cast(int, state.num_loops)

    num_rx = rx_mask.bit_count()
    active_tx_masks = tuple(
        mask
        for chirp_index in range(frame_chirp_start, frame_chirp_end + 1)
        if (mask := state.chirp_tx_masks.get(chirp_index, 0)) != 0
    )
    num_tx = len(set(active_tx_masks)) if active_tx_masks else tx_mask.bit_count()
    if num_rx <= 0:
        raise ValueError("TI CLI config channelCfg enables no RX antennas.")
    if num_tx <= 0:
        raise ValueError("TI CLI config enables no TX antennas.")
    if frame_chirp_end < frame_chirp_start:
        raise ValueError("TI CLI config frameCfg chirp end index is before start index.")
    if num_adc_samples <= 0:
        raise ValueError("TI CLI config profileCfg num ADC samples must be positive.")
    if num_loops <= 0:
        raise ValueError("TI CLI config frameCfg num loops must be positive.")

    num_chirps_per_loop = frame_chirp_end - frame_chirp_start + 1

    return TiCliConfigSummary(
        num_rx=num_rx,
        num_tx=num_tx,
        num_adc_samples=num_adc_samples,
        num_loops=num_loops,
        num_chirps_per_loop=num_chirps_per_loop,
        frame_chirp_start_index=frame_chirp_start,
        frame_chirp_end_index=frame_chirp_end,
        frame_periodicity_s=state.frame_periodicity_s,
    )


def _config_lines(text: str) -> tuple[str, ...]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("%", maxsplit=1)[0].split("#", maxsplit=1)[0].strip()
        if line:
            lines.append(line)
    return tuple(lines)


def _require_field_count(fields: list[str], minimum: int, command: str) -> None:
    if len(fields) < minimum:
        raise ValueError(
            f"TI CLI config command {command!r} expected at least {minimum - 1} "
            f"argument(s); got {len(fields) - 1}."
        )
