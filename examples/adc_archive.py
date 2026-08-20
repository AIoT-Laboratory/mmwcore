"""Archive and verify a completed fixed-frame ADC file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmwcore.config import RadarCaptureSpec
from mmwcore.io import write_adc_archive


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--capture-spec", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    capture = RadarCaptureSpec.from_record(
        json.loads(args.capture_spec.read_text(encoding="utf-8"))
    )
    archive = write_adc_archive(
        args.source,
        args.destination,
        capture,
    )
    archive.verify_all()
    print(f"frames={archive.frame_count}")
    print(f"archive_bytes={archive.archive_size}")
    print(f"adc_sha256={archive.adc_sha256}")


if __name__ == "__main__":
    main()
