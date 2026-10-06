"""Shared structural contract for finite random-access ADC frame readers."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Protocol

from mmwcore.core import ADCFrame, ADCFrameSpec


class ADCReader(Protocol):
    """Read fixed-shape raw ADC frames by zero-based index."""

    @property
    def path(self) -> str | Path: ...

    @property
    def spec(self) -> ADCFrameSpec: ...

    @property
    def frame_periodicity_s(self) -> float | None: ...

    @property
    def num_frames(self) -> int: ...

    def read_frame(self, index: int) -> ADCFrame: ...


__all__ = ["ADCReader"]


def _frame_index(index: int, num_frames: int) -> int:
    """Normalize integer-like indices consistently for raw files and archives."""
    if isinstance(index, bool):
        raise TypeError("ADC frame index must be an integer, not bool.")
    try:
        index = operator.index(index)
    except TypeError as exc:
        raise TypeError("ADC frame index must be an integer.") from exc
    if not 0 <= index < num_frames:
        raise IndexError(f"ADC frame index {index} is outside [0, {num_frames}).")
    return index
