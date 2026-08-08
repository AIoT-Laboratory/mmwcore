"""Read a strict mmwcli capture directory or a headerless int16 ADC file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmwcore import open_capture
from mmwcore.core import ADCComplexLayout, ADCFrameSpec
from mmwcore.io import ADCFileFrameReader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture_parser = modes.add_parser("capture")
    capture_parser.add_argument("path", type=Path)
    capture_parser.add_argument("--frame", type=int, default=0)
    raw_parser = modes.add_parser("raw")
    raw_parser.add_argument("path", type=Path)
    raw_parser.add_argument("--chirps", type=int, required=True)
    raw_parser.add_argument("--rx", type=int, required=True)
    raw_parser.add_argument("--samples", type=int, required=True)
    raw_parser.add_argument(
        "--layout", choices=[layout.value for layout in ADCComplexLayout], required=True
    )
    raw_parser.add_argument("--frame-period-s", type=float)
    raw_parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "capture":
        capture = open_capture(args.path)
        frame = capture.frame(args.frame)
        summary = {
            "source": "capture",
            "family": capture.raw_capture.family,
            "frames": capture.num_frames,
            "tx_order": list(capture.radar_capture.tx_order),
            "raw_int16_values": int(frame.samples.size),
        }
    else:
        spec = ADCFrameSpec(
            num_chirps=args.chirps,
            num_rx=args.rx,
            num_samples=args.samples,
            layout=ADCComplexLayout(args.layout),
        )
        reader = ADCFileFrameReader(args.path, spec, frame_periodicity_s=args.frame_period_s)
        frame = reader.read_frame(args.frame)
        summary = {
            "source": "raw",
            "layout": spec.layout.value,
            "frames": reader.num_frames,
            "raw_int16_values": int(frame.samples.size),
            "timestamp_s": frame.timestamp,
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
