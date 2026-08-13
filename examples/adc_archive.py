"""Archive and verify a completed fixed-frame ADC file."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmwcore.io import write_adc_archive


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--frame-bytes", type=int, required=True)
    parser.add_argument("--capture-contract-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    archive = write_adc_archive(
        args.source,
        args.destination,
        frame_bytes=args.frame_bytes,
        capture_contract_sha256=args.capture_contract_sha256,
    )
    archive.verify_all()
    print(f"frames={archive.frame_count}")
    print(f"archive_bytes={archive.archive_size}")
    print(f"adc_sha256={archive.adc_sha256}")


if __name__ == "__main__":
    main()
