"""CLI helpers for inspecting mmwcore offline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmwcore.cli._args import positive_int
from mmwcore.cli._inspect_adc import (
    ADCFileInspection,
    ADCShapeCandidate,
    ADCSpecRecord,
    adc_spec_from_args,
    infer_adc_shapes,
    inspect_adc_file,
)

__all__ = [
    "ADCFileInspection",
    "ADCShapeCandidate",
    "ADCSpecRecord",
    "build_parser",
    "infer_adc_shapes",
    "inspect_adc_file",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect mmwcore offline artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    adc = subparsers.add_parser("adc", help="Inspect an int16 ADC binary file.")
    adc.add_argument("path", type=Path)
    adc.add_argument("--num-chirps", type=positive_int)
    adc.add_argument("--num-rx", type=positive_int)
    adc.add_argument("--num-samples", type=positive_int)
    adc.add_argument(
        "--ti-cfg",
        type=Path,
        help="Use a TI mmWave SDK CLI config to derive the ADC frame shape.",
    )
    adc.add_argument(
        "--infer-shapes",
        action="store_true",
        help="List candidate ADC frame shapes from file-size arithmetic.",
    )
    adc.add_argument(
        "--candidate-num-chirps",
        type=positive_int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 8, 16, 32, 64, 96, 128, 192],
        help="Candidate chirp counts used with --infer-shapes.",
    )
    adc.add_argument(
        "--candidate-num-rx",
        type=positive_int,
        nargs="+",
        default=[1, 2, 4],
        help="Candidate RX counts used with --infer-shapes.",
    )
    adc.add_argument(
        "--candidate-num-samples",
        type=positive_int,
        nargs="+",
        default=[64, 128, 256],
        help="Candidate ADC sample counts used with --infer-shapes.",
    )
    adc.add_argument(
        "--max-candidates",
        type=positive_int,
        default=12,
        help="Maximum inferred shape candidates to print.",
    )
    adc.add_argument("--json", action="store_true", help="Print JSON instead of key=value lines.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "adc":
        inspection = inspect_adc_file(
            args.path,
            spec=adc_spec_from_args(args),
            infer_shapes=args.infer_shapes,
            candidate_num_chirps=args.candidate_num_chirps,
            candidate_num_rx=args.candidate_num_rx,
            candidate_num_samples=args.candidate_num_samples,
            max_candidates=args.max_candidates,
        )
        _print_inspection(inspection, json_output=args.json)
        return 0
    raise SystemExit(f"unsupported command: {args.command}")


def _print_inspection(
    inspection: ADCFileInspection,
    *,
    json_output: bool,
) -> None:
    record = inspection.to_record()
    if json_output:
        print(json.dumps(record, indent=2))
        return
    for key, value in record.items():
        if value is not None and value != []:
            if key in {"adc_spec", "shape_candidates"}:
                value = json.dumps(value)
            print(f"{key}={value}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
