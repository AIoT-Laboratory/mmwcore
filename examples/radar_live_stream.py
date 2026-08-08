"""Consume a finite mmwcli radar stream from binary stdin or a file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import BinaryIO

from mmwcore import open_capture_stream


def consume(source: BinaryIO) -> dict[str, object]:
    stream = open_capture_stream(source)
    seen = 0
    for item in stream.frames():
        seen += 1
        print(json.dumps({"provisional": True, "frame_index": item.frame_index}), flush=True)
    commit = stream.require_commit()
    return {
        "committed": True,
        "family": stream.contract.raw_capture.family,
        "frames_seen": seen,
        "frames_committed": commit.frames,
        "adc_sha256": commit.adc_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="omit to read binary stdin")
    args = parser.parse_args()
    if args.input is None:
        result = consume(sys.stdin.buffer)
    else:
        with args.input.open("rb") as source:
            result = consume(source)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
