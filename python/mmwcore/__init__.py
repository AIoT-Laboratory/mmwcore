"""Typed, physically explicit mmWave radar processing."""

from __future__ import annotations

from importlib.metadata import version

from .io import (
    CaptureStream,
    MappedTimeInterval,
    MultisensorCapture,
    MultisensorItem,
    MultisensorSource,
    MultisensorSyncEvent,
    ProvisionalRangeDopplerFrame,
    TrainingKey,
    causal_match,
    open_capture,
    open_capture_stream,
    open_multisensor_capture,
)

__all__ = [
    "__version__",
    "CaptureStream",
    "MappedTimeInterval",
    "MultisensorCapture",
    "MultisensorItem",
    "MultisensorSource",
    "MultisensorSyncEvent",
    "ProvisionalRangeDopplerFrame",
    "TrainingKey",
    "causal_match",
    "open_capture",
    "open_capture_stream",
    "open_multisensor_capture",
]

__version__ = version("mmwcore")
