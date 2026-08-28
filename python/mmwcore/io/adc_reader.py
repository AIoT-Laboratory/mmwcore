"""Shared structural contract for finite random-access ADC frame readers."""

from __future__ import annotations

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
