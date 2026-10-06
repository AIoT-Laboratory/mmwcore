"""Restore raw ADC bytes and the decoding contract from a compressed .mmwa file."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmwcore.io import decompress_adc_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    capture = decompress_adc_file(args.source, args.destination)
    print(f"frames={capture.num_frames}")
    print(f"raw_bytes={args.destination.stat().st_size}")


if __name__ == "__main__":
    main()
