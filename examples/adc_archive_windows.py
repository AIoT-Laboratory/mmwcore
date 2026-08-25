"""Read ordered fixed-length windows from a completed ADC archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mmwcore.io import open_adc_archive


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("starts", nargs="+", type=int)
    parser.add_argument("--window-frames", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    archive = open_adc_archive(args.archive)
    payload = archive.read_windows(args.starts, args.window_frames)
    windows = np.frombuffer(payload, dtype="<i2").reshape(
        len(args.starts),
        args.window_frames,
        archive.frame_bytes // 2,
    )
    print(f"shape={windows.shape}")
    print(f"starts={args.starts}")


if __name__ == "__main__":
    main()
