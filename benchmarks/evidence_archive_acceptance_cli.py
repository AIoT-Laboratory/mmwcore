"""CLI for the implemented evidence archive acceptance run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmarks.evidence_archive_acceptance import (
    DEFAULT_FILENAME,
    run_archive_acceptance,
)


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the implemented offline evidence archive on complete ADC files."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--frame-bytes", required=True, type=_positive_integer)
    parser.add_argument("--random-windows", type=_non_negative_integer, default=128)
    parser.add_argument("--window-frames", type=_positive_integer, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_archive_acceptance(
        args.inputs,
        frame_bytes=args.frame_bytes,
        filename=args.filename,
        random_windows=args.random_windows,
        window_frames=args.window_frames,
        seed=args.seed,
        scratch_dir=args.scratch_dir,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {args.output.parent}")
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
