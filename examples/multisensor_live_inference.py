"""Compute a provisional live feature and retain it only after aggregate COMMIT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import BinaryIO

from mmwcore import ProvisionalMultisensorItem, open_multisensor_stream


def infer(item: ProvisionalMultisensorItem) -> float:
    """Replace this deterministic feature with an application model call."""

    return sum(item.payload) / len(item.payload) if item.payload else 0.0


def consume(source: BinaryIO) -> tuple[int, int]:
    stream = open_multisensor_stream(source)
    provisional: list[tuple[ProvisionalMultisensorItem, float]] = []
    for item in stream.items():
        score = infer(item)
        provisional.append((item, score))
        bounds = (
            None
            if item.mapped_time is None
            else [item.mapped_time.start_ns, item.mapped_time.end_ns]
        )
        print(
            json.dumps(
                {
                    "provisional": True,
                    "source_id": item.source_id,
                    "item_index": item.item_index,
                    "mapped_time_ns": bounds,
                    "score": score,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    commit = stream.require_commit()
    accepted = [(item, score) for item, score in provisional if commit.accepts(item)]
    for item, score in accepted:
        print(
            json.dumps(
                {
                    "provisional": False,
                    "source_id": item.source_id,
                    "item_index": item.item_index,
                    "score": score,
                },
                sort_keys=True,
            )
        )
    return len(provisional), len(accepted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="omit to read binary stdin")
    args = parser.parse_args()
    if args.input is None:
        seen, accepted = consume(sys.stdin.buffer)
    else:
        with args.input.open("rb") as source:
            seen, accepted = consume(source)
    print(json.dumps({"items_seen": seen, "items_accepted": accepted}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
