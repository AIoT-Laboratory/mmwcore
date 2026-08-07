"""Shared argparse value parsers for mmwcore CLIs."""

from __future__ import annotations

import argparse


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive; got {parsed}")
    return parsed
