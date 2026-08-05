"""CLI for exporting deterministic radar configuration files."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmwcore.cli._args import positive_float, positive_int
from mmwcore.config import (
    DCA1000ConfigSpec,
    RadarProfile,
    TiCliConfigSpec,
    iwr6843_profile,
    write_dca1000_config,
    write_ti_cli_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export mmwcore radar config files.")
    parser.add_argument("--ti-cfg", type=Path, help="Optional TI mmWave SDK .cfg output path.")
    parser.add_argument("--dca-json", type=Path, help="Optional DCA1000EVM JSON output path.")

    parser.add_argument("--preset", choices=["iwr6843"], help="Optional radar profile preset.")
    parser.add_argument("--start-frequency-hz", type=positive_float)
    parser.add_argument(
        "--frequency-slope-hz-per-s",
        type=positive_float,
    )
    parser.add_argument("--adc-sample-rate-hz", type=positive_float)
    parser.add_argument("--adc-start-time-s", type=positive_float)
    parser.add_argument("--ramp-end-time-s", type=positive_float)
    parser.add_argument("--idle-time-s", type=positive_float)
    parser.add_argument("--num-adc-samples", type=positive_int)
    parser.add_argument("--num-chirps-per-tx", type=positive_int)
    parser.add_argument("--num-tx", type=positive_int)
    parser.add_argument("--num-rx", type=positive_int)

    parser.add_argument("--frame-periodicity-s", type=positive_float, default=0.1)
    parser.add_argument("--include-sensor-start", action="store_true")
    parser.add_argument("--capture-path", default=DCA1000ConfigSpec.capture_path)
    parser.add_argument("--file-prefix", default=DCA1000ConfigSpec.file_prefix)
    parser.add_argument("--system-ip", default=DCA1000ConfigSpec.system_ip)
    parser.add_argument("--dca-ip", default=DCA1000ConfigSpec.dca_ip)
    parser.add_argument("--dca-mac")
    parser.add_argument(
        "--config-port",
        type=positive_int,
        default=DCA1000ConfigSpec.config_port,
    )
    parser.add_argument(
        "--data-port",
        type=positive_int,
        default=DCA1000ConfigSpec.data_port,
    )
    parser.add_argument(
        "--packet-delay-us",
        type=positive_int,
        default=DCA1000ConfigSpec.packet_delay_us,
    )
    parser.add_argument(
        "--frames-to-capture",
        type=positive_int,
        default=DCA1000ConfigSpec.frames_to_capture,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ti_cfg is None and args.dca_json is None:
        raise SystemExit("at least one of --ti-cfg or --dca-json is required")

    try:
        profile = _build_profile(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.ti_cfg is not None:
        try:
            ti_spec = TiCliConfigSpec(
                frame_periodicity_s=args.frame_periodicity_s,
                include_sensor_start=args.include_sensor_start,
            )
            ti_path = write_ti_cli_config(args.ti_cfg, profile, ti_spec)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"saved_ti_cfg={ti_path}")

    if args.dca_json is not None:
        if args.dca_mac is None:
            raise SystemExit("--dca-mac is required with --dca-json")
        try:
            dca_spec = DCA1000ConfigSpec(
                dca_mac=args.dca_mac,
                capture_path=args.capture_path,
                file_prefix=args.file_prefix,
                system_ip=args.system_ip,
                dca_ip=args.dca_ip,
                config_port=args.config_port,
                data_port=args.data_port,
                packet_delay_us=args.packet_delay_us,
                frames_to_capture=args.frames_to_capture,
            )
            dca_path = write_dca1000_config(args.dca_json, profile, dca_spec)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"saved_dca_json={dca_path}")

    return 0


def _build_profile(args: argparse.Namespace) -> RadarProfile:
    overrides = {
        key: value
        for key, value in {
            "start_frequency_hz": args.start_frequency_hz,
            "frequency_slope_hz_per_s": args.frequency_slope_hz_per_s,
            "adc_sample_rate_hz": args.adc_sample_rate_hz,
            "adc_start_time_s": args.adc_start_time_s,
            "ramp_end_time_s": args.ramp_end_time_s,
            "idle_time_s": args.idle_time_s,
            "num_adc_samples": args.num_adc_samples,
            "num_chirps_per_tx": args.num_chirps_per_tx,
            "num_tx": args.num_tx,
            "num_rx": args.num_rx,
        }.items()
        if value is not None
    }
    if args.preset == "iwr6843":
        return iwr6843_profile(**overrides)
    return RadarProfile(**overrides)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
