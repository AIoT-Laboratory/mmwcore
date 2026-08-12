"""Typed, physically explicit mmWave radar processing."""

from __future__ import annotations

from importlib.metadata import version

from .io import (
    CaptureStream,
    EvidenceArchive,
    MappedTimeInterval,
    MultisensorCapture,
    MultisensorItem,
    MultisensorSource,
    MultisensorStream,
    MultisensorStreamCommit,
    MultisensorSyncEvent,
    ProvisionalMultisensorItem,
    ProvisionalRangeDopplerFrame,
    TrainingKey,
    causal_match,
    open_capture,
    open_capture_stream,
    open_evidence_archive,
    open_multisensor_capture,
    open_multisensor_stream,
    write_evidence_archive,
)

__all__ = [
    "__version__",
    "CaptureStream",
    "EvidenceArchive",
    "MappedTimeInterval",
    "MultisensorCapture",
    "MultisensorItem",
    "MultisensorSource",
    "MultisensorStream",
    "MultisensorStreamCommit",
    "MultisensorSyncEvent",
    "ProvisionalMultisensorItem",
    "ProvisionalRangeDopplerFrame",
    "TrainingKey",
    "causal_match",
    "open_capture",
    "open_capture_stream",
    "open_evidence_archive",
    "open_multisensor_capture",
    "open_multisensor_stream",
    "write_evidence_archive",
]

__version__ = version("mmwcore")
