"""Deterministic config exporters for radar tooling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .profiles import RadarProfile


@dataclass(frozen=True)
class TiCliConfigSpec:
    """Export options for TI mmWave SDK CLI profile files."""

    profile_id: int = 0
    frame_periodicity_s: float = 0.1
    adc_bits: int = 16
    adc_output_complex: bool = True
    tx_enable_order: tuple[int, ...] = (0, 2, 1)
    include_sensor_start: bool = False

    def __post_init__(self) -> None:
        if self.profile_id < 0:
            raise ValueError(
                f"TiCliConfigSpec.profile_id must be non-negative; got {self.profile_id}."
            )
        if self.frame_periodicity_s <= 0:
            raise ValueError(
                "TiCliConfigSpec.frame_periodicity_s must be positive; "
                f"got {self.frame_periodicity_s}."
            )
        if self.adc_bits not in {12, 14, 16}:
            raise ValueError(
                f"TiCliConfigSpec.adc_bits must be 12, 14, or 16; got {self.adc_bits}."
            )
        if not self.tx_enable_order:
            raise ValueError("TiCliConfigSpec.tx_enable_order must not be empty.")
        if len(set(self.tx_enable_order)) != len(self.tx_enable_order):
            raise ValueError("TiCliConfigSpec.tx_enable_order must not contain duplicates.")
        for tx_index in self.tx_enable_order:
            if tx_index < 0:
                raise ValueError("TiCliConfigSpec.tx_enable_order values must be non-negative.")


@dataclass(frozen=True)
class DCA1000ConfigSpec:
    """Export options for DCA1000EVM CLI JSON config files."""

    dca_mac: str
    capture_path: str = "dataset/radar"
    file_prefix: str = "adc_data"
    system_ip: str = "192.168.33.30"
    dca_ip: str = "192.168.33.180"
    config_port: int = 4096
    data_port: int = 4098
    lvds_lanes: int = 2
    adc_bits: int = 16
    packet_delay_us: int = 25
    max_record_file_size_mb: int = 1024
    sequence_number_enable: bool = True
    capture_stop_mode: str = "infinite"
    bytes_to_capture: int = 1025
    duration_to_capture_ms: int = 1000
    frames_to_capture: int = 10
    reorder_enable: bool = True
    data_type: str = "complex"

    def __post_init__(self) -> None:
        for name, value in (
            ("config_port", self.config_port),
            ("data_port", self.data_port),
            ("lvds_lanes", self.lvds_lanes),
            ("packet_delay_us", self.packet_delay_us),
            ("max_record_file_size_mb", self.max_record_file_size_mb),
            ("bytes_to_capture", self.bytes_to_capture),
            ("duration_to_capture_ms", self.duration_to_capture_ms),
            ("frames_to_capture", self.frames_to_capture),
        ):
            if value <= 0:
                raise ValueError(f"DCA1000ConfigSpec.{name} must be positive; got {value}.")
        if self.adc_bits not in {12, 14, 16}:
            raise ValueError(
                f"DCA1000ConfigSpec.adc_bits must be 12, 14, or 16; got {self.adc_bits}."
            )
        if self.data_type not in {"real", "complex"}:
            raise ValueError("DCA1000ConfigSpec.data_type must be 'real' or 'complex'.")
        if not self.dca_mac.strip():
            raise ValueError("DCA1000ConfigSpec.dca_mac must not be empty.")
        if not self.file_prefix:
            raise ValueError("DCA1000ConfigSpec.file_prefix must not be empty.")


def render_ti_cli_config(
    profile: RadarProfile,
    spec: TiCliConfigSpec | None = None,
) -> str:
    """Render a TI mmWave SDK CLI config from a pure radar profile."""

    export = spec or TiCliConfigSpec()
    if len(export.tx_enable_order) < profile.num_tx:
        raise ValueError(
            "TiCliConfigSpec.tx_enable_order must provide at least one entry per TX antenna."
        )

    rx_mask = (1 << profile.num_rx) - 1
    tx_mask = sum(1 << tx_index for tx_index in export.tx_enable_order[: profile.num_tx])
    adc_output_fmt = 1 if export.adc_output_complex else 0
    adc_config_bits = {12: 0, 14: 1, 16: 2}[export.adc_bits]

    lines = [
        "sensorStop",
        "flushCfg",
        "dfeDataOutputMode 1",
        f"channelCfg {rx_mask} {tx_mask} 0",
        f"adcCfg {adc_config_bits} {adc_output_fmt}",
        "adcbufCfg -1 0 1 1 1",
        _profile_cfg_line(profile, export),
    ]

    for chirp_index, tx_index in enumerate(export.tx_enable_order[: profile.num_tx]):
        lines.append(
            f"chirpCfg {chirp_index} {chirp_index} {export.profile_id} 0 0 0 0 {1 << tx_index}"
        )

    lines.extend(
        [
            (
                f"frameCfg 0 {profile.num_tx - 1} {profile.num_chirps_per_tx} 0 "
                f"{export.frame_periodicity_s * 1e3:.1f} 1 0"
            ),
            "lowPower 0 0",
            "guiMonitor -1 1 0 0 0 0 0",
            "cfarCfg -1 0 2 8 4 3 0 15 1",
            "cfarCfg -1 1 0 4 2 3 1 15 1",
            "multiObjBeamForming -1 1 0.5",
            "calibDcRangeSig -1 0 -5 8 256",
            "clutterRemoval -1 0",
            "aoaFovCfg -1 -90 90 -90 90",
            "cfarFovCfg -1 0 0 12",
            "cfarFovCfg -1 1 -4 4",
            _range_bias_line(profile),
            "measureRangeBiasAndRxChanPhase 0 1 0.2",
            "extendedMaxVelocity -1 0",
            "bpmCfg -1 0 0 0",
            "CQRxSatMonitor 0 3 5 121 0",
            "CQSigImgMonitor 0 127 4",
            "analogMonitor 0 0",
            "calibData 0 0 0",
            "lvdsStreamCfg -1 0 1 0",
        ]
    )
    if export.include_sensor_start:
        lines.append("sensorStart")
    return "\n".join(lines) + "\n"


def write_ti_cli_config(
    path: str | Path,
    profile: RadarProfile,
    spec: TiCliConfigSpec | None = None,
) -> Path:
    """Write a TI mmWave SDK CLI config file and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_ti_cli_config(profile, spec), encoding="utf-8")
    return output


def render_dca1000_config(
    profile: RadarProfile,
    spec: DCA1000ConfigSpec,
) -> dict[str, Any]:
    """Render a DCA1000EVM CLI JSON payload from radar capture settings."""

    export = spec
    data_format_mode = {12: 1, 14: 2, 16: 3}[export.adc_bits]
    data_port_config = [
        {"portIdx": port_idx, "dataType": export.data_type}
        for port_idx in range(max(export.lvds_lanes, profile.num_rx))
    ]

    return {
        "DCA1000Config": {
            "dataLoggingMode": "raw",
            "dataTransferMode": "LVDSCapture",
            "dataCaptureMode": "ethernetStream",
            "lvdsMode": export.lvds_lanes,
            "dataFormatMode": data_format_mode,
            "packetDelay_us": export.packet_delay_us,
            "ethernetConfig": {
                "DCA1000IPAddress": export.dca_ip,
                "DCA1000ConfigPort": export.config_port,
                "DCA1000DataPort": export.data_port,
            },
            "ethernetConfigUpdate": {
                "systemIPAddress": export.system_ip,
                "DCA1000IPAddress": export.dca_ip,
                "DCA1000MACAddress": export.dca_mac,
                "DCA1000ConfigPort": export.config_port,
                "DCA1000DataPort": export.data_port,
            },
            "captureConfig": {
                "fileBasePath": export.capture_path.replace("\\", "/"),
                "filePrefix": export.file_prefix,
                "maxRecFileSize_MB": export.max_record_file_size_mb,
                "sequenceNumberEnable": int(export.sequence_number_enable),
                "captureStopMode": export.capture_stop_mode,
                "bytesToCapture": export.bytes_to_capture,
                "durationToCapture_ms": export.duration_to_capture_ms,
                "framesToCapture": export.frames_to_capture,
            },
            "dataFormatConfig": {
                "MSBToggle": 0,
                "reorderEnable": int(export.reorder_enable),
                "laneFmtMap": 0,
                "dataPortConfig": data_port_config,
            },
        }
    }


def write_dca1000_config(
    path: str | Path,
    profile: RadarProfile,
    spec: DCA1000ConfigSpec,
) -> Path:
    """Write a DCA1000EVM CLI JSON config file and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = render_dca1000_config(profile, spec)
    output.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return output


def _profile_cfg_line(profile: RadarProfile, spec: TiCliConfigSpec) -> str:
    return (
        f"profileCfg {spec.profile_id} {profile.start_frequency_hz / 1e9:.3f} "
        f"{profile.idle_time_s * 1e6:.2f} {profile.adc_start_time_s * 1e6:.2f} "
        f"{profile.ramp_end_time_s * 1e6:.2f} 0 0 "
        f"{profile.frequency_slope_hz_per_s / 1e12:.3f} "
        f"{profile.adc_start_time_s * 1e6:.2f} {profile.num_adc_samples} "
        f"{profile.adc_sample_rate_hz / 1e3:.0f} 0 0 30"
    )


def _range_bias_line(profile: RadarProfile) -> str:
    compensation = " ".join("1 0" for _ in range(profile.virtual_antennas))
    return f"compRangeBiasAndRxChanPhase 0.0 {compensation}"
