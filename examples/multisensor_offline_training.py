"""Read causal radar/camera pairs from a published training session."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from mmwcore import open_multisensor_capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--radar-source", default="radar-0")
    parser.add_argument("--camera-source", default="camera-0")
    parser.add_argument("--lag-min-ns", type=int, default=0)
    parser.add_argument("--lag-max-ns", type=int, default=50_000_000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    session = open_multisensor_capture(args.session)
    radar = session.source(args.radar_source)
    camera = session.source(args.camera_source)
    nested = radar.open_radar_capture()
    pairs = session.causal_pairs(
        radar.source_id,
        camera.source_id,
        lag_min_ns=args.lag_min_ns,
        lag_max_ns=args.lag_max_ns,
    )
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be non-negative")
        pairs = islice(pairs, args.limit)
    count = 0
    for radar_item, camera_item in pairs:
        raw = nested.frame(radar_item.item_index)
        print(
            json.dumps(
                {
                    "radar_key": list(radar_item.training_key),
                    "camera_key": list(camera_item.training_key),
                    "radar_values": int(raw.samples.size),
                    "camera_bytes": len(camera_item.payload),
                },
                sort_keys=True,
            )
        )
        count += 1
    print(json.dumps({"session_id": session.session_id, "pairs": count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
