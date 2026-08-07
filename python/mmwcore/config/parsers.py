"""Parsers for existing radar configuration files."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import cast

from mmwcore.core import ADCComplexLayout, ADCFrameSpec

from .capture import RadarCaptureSpec
from .profiles import RadarProfile


@dataclass(frozen=True)
class TiCliConfigSummary:
    """Minimal capture-shape summary parsed from a TI legacy CLI config."""

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
class _TiCliShapeState:
    rx_mask: int | None = None
    tx_mask: int | None = None
    num_adc_samples: int | None = None
    chirp_tx_masks: dict[int, int] = field(default_factory=dict)
    frame_chirp_start: int | None = None
    frame_chirp_end: int | None = None
    num_loops: int | None = None
    frame_periodicity_s: float | None = None


@dataclass(frozen=True)
class _TiProfile:
    profile_id: int
    start_frequency_hz: float
    frequency_slope_hz_per_s: float
    adc_sample_rate_hz: float
    adc_start_time_s: float
    ramp_end_time_s: float
    idle_time_s: float
    num_adc_samples: int


@dataclass
class _TiCliCaptureState:
    flush_seen: bool = False
    capture_command_seen: bool = False
    data_output_mode: int | None = None
    rx_mask: int | None = None
    tx_mask: int | None = None
    adc_bits: int | None = None
    adc_output_format: int | None = None
    adcbuf_configured: bool = False
    lvds_stream_configured: bool = False
    profiles: dict[int, _TiProfile] = field(default_factory=dict)
    chirp_tx_masks: dict[int, int] = field(default_factory=dict)
    chirp_profile_ids: dict[int, int] = field(default_factory=dict)
    frame_chirp_start: int | None = None
    frame_chirp_end: int | None = None
    num_loops: int | None = None
    num_frames: int | None = None
    frame_periodicity_s: float | None = None


def parse_ti_cli_config_file(path: str | Path) -> TiCliConfigSummary:
    """Parse a TI legacy CLI config file into a capture-shape summary."""

    return parse_ti_cli_config(Path(path).read_text(encoding="utf-8"))


def parse_ti_cli_config(text: str) -> TiCliConfigSummary:
    """Parse a capture-shape summary from a TI legacy CLI config string."""

    state = _TiCliShapeState()
    for line in _config_lines(text):
        _parse_shape_command(line.split(), state)
    return _summarize_shape_state(state)


def parse_ti_cli_capture_spec(
    text: str,
    *,
    layout: ADCComplexLayout,
) -> RadarCaptureSpec:
    """Parse the supported xWR68xx legacy raw-capture subset into a contract."""

    state = _TiCliCaptureState()
    for line in _config_lines(text):
        _parse_capture_command(line.split(), state)
    return _capture_spec(state, layout=layout)


def parse_ti_cli_capture_spec_file(
    path: str | Path,
    *,
    layout: ADCComplexLayout,
) -> RadarCaptureSpec:
    """Parse an xWR68xx legacy raw-capture config file into a contract."""

    return parse_ti_cli_capture_spec(
        Path(path).read_text(encoding="utf-8"),
        layout=layout,
    )


def _parse_capture_command(fields: list[str], state: _TiCliCaptureState) -> None:
    command = fields[0]
    if command == "flushCfg":
        _parse_flush_cfg(fields, state)
        return
    if command in {"advFrameCfg", "subFrameCfg"}:
        raise ValueError("TI CLI capture supports legacy frameCfg only.")
    parser = {
        "dfeDataOutputMode": _parse_data_output_mode,
        "channelCfg": _parse_channel_cfg,
        "adcCfg": _parse_adc_cfg,
        "adcbufCfg": _parse_adcbuf_cfg,
        "profileCfg": _parse_profile_cfg,
        "chirpCfg": _parse_chirp_cfg,
        "frameCfg": _parse_frame_cfg,
        "lvdsStreamCfg": _parse_lvds_stream_cfg,
    }.get(command)
    if parser is not None:
        state.capture_command_seen = True
        parser(fields, state)


def _parse_flush_cfg(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 1, fields[0])
    if state.flush_seen:
        raise ValueError("TI CLI config contains multiple flushCfg commands.")
    if state.capture_command_seen:
        raise ValueError("TI CLI config flushCfg must precede capture configuration.")
    state.flush_seen = True


def _parse_data_output_mode(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 2, fields[0])
    if state.data_output_mode is not None:
        raise ValueError("TI CLI config contains multiple dfeDataOutputMode commands.")
    state.data_output_mode = int(fields[1])
    if state.data_output_mode != 1:
        raise ValueError("TI CLI capture supports dfeDataOutputMode 1 only.")


def _parse_channel_cfg(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 4, fields[0])
    if state.rx_mask is not None:
        raise ValueError("TI CLI config contains multiple channelCfg commands.")
    state.rx_mask = int(fields[1])
    state.tx_mask = int(fields[2])
    if int(fields[3]) != 0:
        raise ValueError("TI CLI capture does not support cascaded channelCfg.")


def _parse_adc_cfg(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 3, fields[0])
    if state.adc_bits is not None:
        raise ValueError("TI CLI config contains multiple adcCfg commands.")
    state.adc_bits = int(fields[1])
    state.adc_output_format = int(fields[2])
    if (state.adc_bits, state.adc_output_format) != (2, 1):
        raise ValueError("TI CLI capture requires adcCfg 2 1 (16-bit complex ADC).")


def _parse_adcbuf_cfg(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 6, fields[0])
    if state.adcbuf_configured:
        raise ValueError("TI CLI config contains multiple adcbufCfg commands.")
    if tuple(int(value) for value in fields[1:]) != (-1, 0, 1, 1, 1):
        raise ValueError("TI CLI capture requires exact adcbufCfg -1 0 1 1 1.")
    state.adcbuf_configured = True


def _parse_profile_cfg(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 15, fields[0])
    profile = _TiProfile(
        profile_id=int(fields[1]),
        start_frequency_hz=float(fields[2]) * 1e9,
        idle_time_s=float(fields[3]) * 1e-6,
        adc_start_time_s=float(fields[4]) * 1e-6,
        ramp_end_time_s=float(fields[5]) * 1e-6,
        frequency_slope_hz_per_s=float(fields[8]) * 1e12,
        num_adc_samples=int(fields[10]),
        adc_sample_rate_hz=float(fields[11]) * 1e3,
    )
    if profile.profile_id in state.profiles:
        raise ValueError(f"TI CLI config contains duplicate profileCfg ID {profile.profile_id}.")
    if not 0 <= profile.profile_id <= 3:
        raise ValueError("xWR68xx profileCfg ID must be in range 0..3.")
    _require_finite_profile(profile)
    state.profiles[profile.profile_id] = profile


def _parse_chirp_cfg(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 9, fields[0])
    start = int(fields[1])
    end = int(fields[2])
    if start < 0 or end < start or end > 511:
        raise ValueError("xWR68xx chirpCfg indices must be in range 0..511.")
    profile_id = int(fields[3])
    variations = tuple(float(value) for value in fields[4:8])
    if any(not isfinite(value) or value != 0 for value in variations):
        raise ValueError("TI CLI capture does not support nonzero chirpCfg variations.")
    chirp_tx_mask = int(fields[8])
    for chirp_index in range(start, end + 1):
        if chirp_index in state.chirp_tx_masks:
            raise ValueError(f"TI CLI config defines chirp index {chirp_index} more than once.")
        state.chirp_tx_masks[chirp_index] = chirp_tx_mask
        state.chirp_profile_ids[chirp_index] = profile_id


def _parse_frame_cfg(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 8, fields[0])
    if state.frame_chirp_start is not None:
        raise ValueError("TI CLI config contains multiple frameCfg commands.")
    state.frame_chirp_start = int(fields[1])
    state.frame_chirp_end = int(fields[2])
    state.num_loops = int(fields[3])
    state.num_frames = int(fields[4])
    state.frame_periodicity_s = float(fields[5]) / 1e3
    if int(fields[6]) != 1 or float(fields[7]) != 0:
        raise ValueError("TI CLI capture requires software-triggered frameCfg with zero delay.")


def _parse_lvds_stream_cfg(fields: list[str], state: _TiCliCaptureState) -> None:
    _require_field_count(fields, 5, fields[0])
    if state.lvds_stream_configured:
        raise ValueError("TI CLI config contains multiple lvdsStreamCfg commands.")
    if tuple(int(value) for value in fields[1:]) != (-1, 0, 1, 0):
        raise ValueError(
            "TI CLI capture requires exact lvdsStreamCfg -1 0 1 0 "
            "(no header, hardware ADC, software off)."
        )
    state.lvds_stream_configured = True


def _require_finite_profile(profile: _TiProfile) -> None:
    values = (
        profile.start_frequency_hz,
        profile.frequency_slope_hz_per_s,
        profile.adc_sample_rate_hz,
        profile.adc_start_time_s,
        profile.ramp_end_time_s,
        profile.idle_time_s,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("TI CLI config profileCfg physical values must be finite.")


def _parse_shape_command(fields: list[str], state: _TiCliShapeState) -> None:
    command = fields[0]
    if command == "channelCfg":
        _require_minimum_field_count(fields, 3, command)
        state.rx_mask = int(fields[1])
        state.tx_mask = int(fields[2])
    elif command == "profileCfg":
        _require_minimum_field_count(fields, 11, command)
        state.num_adc_samples = int(fields[10])
    elif command == "chirpCfg":
        _require_minimum_field_count(fields, 9, command)
        start = int(fields[1])
        end = int(fields[2])
        chirp_tx_mask = int(fields[8])
        for chirp_index in range(start, end + 1):
            state.chirp_tx_masks[chirp_index] = chirp_tx_mask
    elif command == "frameCfg":
        _require_minimum_field_count(fields, 6, command)
        state.frame_chirp_start = int(fields[1])
        state.frame_chirp_end = int(fields[2])
        state.num_loops = int(fields[3])
        state.frame_periodicity_s = float(fields[5]) / 1e3


def _summarize_shape_state(state: _TiCliShapeState) -> TiCliConfigSummary:
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

    return TiCliConfigSummary(
        num_rx=num_rx,
        num_tx=num_tx,
        num_adc_samples=num_adc_samples,
        num_loops=num_loops,
        num_chirps_per_loop=frame_chirp_end - frame_chirp_start + 1,
        frame_chirp_start_index=frame_chirp_start,
        frame_chirp_end_index=frame_chirp_end,
        frame_periodicity_s=state.frame_periodicity_s,
    )


def _capture_spec(
    state: _TiCliCaptureState,
    *,
    layout: ADCComplexLayout,
) -> RadarCaptureSpec:
    _require_complete_capture_state(state)
    rx_mask = cast(int, state.rx_mask)
    tx_mask = cast(int, state.tx_mask)
    frame_fields = _validated_capture_frame_fields(state)
    frame_chirp_start, frame_chirp_end, num_loops, num_frames_raw, frame_periodicity_s = (
        frame_fields
    )
    chirp_indices = _frame_chirp_indices(state, frame_chirp_start, frame_chirp_end)
    parsed_profile = _frame_profile(state, chirp_indices)
    tx_order = _strict_tdm_tx_order(
        tuple(state.chirp_tx_masks[index] for index in chirp_indices),
        channel_tx_mask=tx_mask,
    )
    profile = _radar_profile(
        parsed_profile,
        num_loops=num_loops,
        num_tx=len(tx_order),
        num_rx=rx_mask.bit_count(),
    )
    adc = profile.to_adc_frame_spec(layout=layout)
    if adc.layout is ADCComplexLayout.GROUP2_I_THEN_Q and profile.num_adc_samples % 2:
        raise ValueError("GROUP2_I_THEN_Q capture requires an even numAdcSamples value.")

    return RadarCaptureSpec(
        profile=profile,
        adc=adc,
        tx_order=tx_order,
        frame_periodicity_s=frame_periodicity_s,
        num_frames=None if num_frames_raw == 0 else num_frames_raw,
    )


def _require_complete_capture_state(state: _TiCliCaptureState) -> None:
    missing: list[str] = []
    if state.data_output_mode is None:
        missing.append("dfeDataOutputMode")
    if state.rx_mask is None:
        missing.append("channelCfg")
    if state.adc_bits is None:
        missing.append("adcCfg")
    if not state.adcbuf_configured:
        missing.append("adcbufCfg")
    if not state.profiles:
        missing.append("profileCfg")
    if not state.chirp_tx_masks:
        missing.append("chirpCfg")
    if state.frame_chirp_start is None:
        missing.append("frameCfg")
    if not state.lvds_stream_configured:
        missing.append("lvdsStreamCfg")
    if missing:
        raise ValueError(f"TI CLI config missing required command(s): {', '.join(missing)}")


def _validated_capture_frame_fields(
    state: _TiCliCaptureState,
) -> tuple[int, int, int, int, float]:
    rx_mask = cast(int, state.rx_mask)
    tx_mask = cast(int, state.tx_mask)
    frame_chirp_start = cast(int, state.frame_chirp_start)
    frame_chirp_end = cast(int, state.frame_chirp_end)
    num_loops = cast(int, state.num_loops)
    num_frames_raw = cast(int, state.num_frames)
    frame_periodicity_s = cast(float, state.frame_periodicity_s)

    if not 1 <= rx_mask <= 0x0F:
        raise ValueError("xWR68xx channelCfg RX mask must be in range 1..15.")
    if rx_mask & (rx_mask + 1):
        raise ValueError("Sparse xWR68xx RX masks are not representable by RadarCaptureSpec.")
    if not 1 <= tx_mask <= 0x07:
        raise ValueError("xWR68xx channelCfg TX mask must be in range 1..7.")
    if frame_chirp_start < 0 or frame_chirp_end < frame_chirp_start or frame_chirp_end > 511:
        raise ValueError("xWR68xx frameCfg chirp indices must be in range 0..511.")
    if not 1 <= num_loops <= 255:
        raise ValueError("xWR68xx frameCfg num loops must be in range 1..255.")
    if not 0 <= num_frames_raw <= 65535:
        raise ValueError("xWR68xx frameCfg num frames must be in range 0..65535.")
    if not isfinite(frame_periodicity_s) or frame_periodicity_s <= 0:
        raise ValueError("TI CLI config frameCfg periodicity must be positive.")
    return (
        frame_chirp_start,
        frame_chirp_end,
        num_loops,
        num_frames_raw,
        frame_periodicity_s,
    )


def _frame_chirp_indices(
    state: _TiCliCaptureState,
    start: int,
    end: int,
) -> tuple[int, ...]:
    chirp_indices = tuple(range(start, end + 1))
    missing_chirps = tuple(
        chirp_index for chirp_index in chirp_indices if chirp_index not in state.chirp_tx_masks
    )
    if missing_chirps:
        raise ValueError(
            "TI CLI config frameCfg references undefined chirp index(es): "
            + ", ".join(str(index) for index in missing_chirps)
        )
    return chirp_indices


def _frame_profile(
    state: _TiCliCaptureState,
    chirp_indices: tuple[int, ...],
) -> _TiProfile:
    profile_ids = {state.chirp_profile_ids[index] for index in chirp_indices}
    if len(profile_ids) != 1:
        raise ValueError("TI CLI capture must use exactly one profileCfg ID per frame loop.")
    profile_id = next(iter(profile_ids))
    try:
        parsed_profile = state.profiles[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"TI CLI chirpCfg references undefined profileCfg ID {profile_id}."
        ) from exc
    return parsed_profile


def _radar_profile(
    parsed_profile: _TiProfile,
    *,
    num_loops: int,
    num_tx: int,
    num_rx: int,
) -> RadarProfile:
    return RadarProfile(
        start_frequency_hz=parsed_profile.start_frequency_hz,
        frequency_slope_hz_per_s=parsed_profile.frequency_slope_hz_per_s,
        adc_sample_rate_hz=parsed_profile.adc_sample_rate_hz,
        adc_start_time_s=parsed_profile.adc_start_time_s,
        ramp_end_time_s=parsed_profile.ramp_end_time_s,
        idle_time_s=parsed_profile.idle_time_s,
        num_adc_samples=parsed_profile.num_adc_samples,
        num_chirps_per_tx=num_loops,
        num_tx=num_tx,
        num_rx=num_rx,
    )


def _strict_tdm_tx_order(
    tx_masks: tuple[int, ...],
    *,
    channel_tx_mask: int,
) -> tuple[int, ...]:
    tx_order: list[int] = []
    for tx_mask in tx_masks:
        if tx_mask <= 0 or tx_mask & (tx_mask - 1):
            raise ValueError(
                "TI CLI capture requires each chirpCfg to enable exactly one TX antenna."
            )
        if tx_mask & ~channel_tx_mask:
            raise ValueError("TI CLI chirpCfg enables a TX disabled by channelCfg.")
        tx_identifier = tx_mask.bit_length() - 1
        if tx_identifier in tx_order:
            raise ValueError("TI CLI capture requires each active TX exactly once per frame loop.")
        tx_order.append(tx_identifier)
    return tuple(tx_order)


def _config_lines(text: str) -> tuple[str, ...]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("%", maxsplit=1)[0].split("#", maxsplit=1)[0].strip()
        if line:
            lines.append(line)
    return tuple(lines)


def _require_field_count(fields: list[str], expected: int, command: str) -> None:
    if len(fields) != expected:
        raise ValueError(
            f"TI CLI config command {command!r} expected exactly {expected - 1} "
            f"argument(s); got {len(fields) - 1}."
        )


def _require_minimum_field_count(fields: list[str], minimum: int, command: str) -> None:
    if len(fields) < minimum:
        raise ValueError(
            f"TI CLI config command {command!r} expected at least {minimum - 1} "
            f"argument(s); got {len(fields) - 1}."
        )
