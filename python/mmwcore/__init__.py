"""Typed, physically explicit mmWave radar processing."""

from __future__ import annotations

from importlib.metadata import version

from .io import CaptureStream, ProvisionalRangeDopplerFrame, open_capture, open_capture_stream

__all__ = [
    "__version__",
    "CaptureStream",
    "ProvisionalRangeDopplerFrame",
    "open_capture",
    "open_capture_stream",
]

__version__ = version("mmwcore")
