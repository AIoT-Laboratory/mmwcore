"""Command-line wrapper for the offline ADC storage benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmarks.adc_storage_benchmark import DEFAULT_CASES, DEFAULT_FILENAME, run_benchmark
from benchmarks.adc_storage_codecs import DEFAULT_ZLIB_LEVEL, SUPPORTED_CODECS
from benchmarks.adc_storage_inputs import StorageCase


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _zlib_level(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 9:
        raise argparse.ArgumentTypeError("value must be in [0, 9]")
    return parsed


def _case(value: str) -> StorageCase:
    codec, separator, frames_text = value.rpartition(":")
    if not separator or codec not in SUPPORTED_CODECS:
        choices = ", ".join(SUPPORTED_CODECS)
        raise argparse.ArgumentTypeError(f"case must be CODEC:FRAMES; codecs: {choices}")
    try:
        frames = _positive_integer(frames_text)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("case frame count must be positive") from error
    return StorageCase(codec=codec, chunk_frames=frames)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark strictly reversible chunk storage for offline radar ADC data."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="ADC files or directories to scan.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--frame-bytes", required=True, type=_positive_integer)
    parser.add_argument(
        "--case",
        action="append",
        type=_case,
        dest="cases",
        metavar="CODEC:FRAMES",
        help="Benchmark one codec and chunk size; repeat for additional cases.",
    )
    parser.add_argument("--start-frame", type=_non_negative_integer, default=0)
    parser.add_argument("--max-frames", type=_positive_integer)
    parser.add_argument("--random-windows", type=_non_negative_integer, default=32)
    parser.add_argument("--window-frames", type=_positive_integer, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zlib-level", type=_zlib_level, default=DEFAULT_ZLIB_LEVEL)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _write_json_atomic(path: Path, serialized: str) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {path.parent}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmark(
        args.inputs,
        frame_bytes=args.frame_bytes,
        filename=args.filename,
        cases=args.cases or DEFAULT_CASES,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        random_windows=args.random_windows,
        window_frames=args.window_frames,
        seed=args.seed,
        zlib_level=args.zlib_level,
        scratch_dir=args.scratch_dir,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        _write_json_atomic(args.output, serialized)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
