"""Integrity and timing inspection for synchronized radar-camera capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._capture_paths import manifest_path, resolve_relative_reference
from .radar_camera_alignment import build_causal_radar_camera_matches
from .sync_protocol import (
    CaptureSyncEvent,
    CaptureSyncEventKind,
    load_capture_sync_events,
)
from .synchronized_capture import (
    SynchronizationMode,
    load_synchronized_capture_manifest,
    validate_capture_session,
)


@dataclass(frozen=True)
class SynchronizedCaptureInspection:
    """Validated temporal coverage around one radar trigger."""

    capture_id: str
    synchronization_mode: SynchronizationMode
    camera_frame_count: int
    frames_before_radar: int
    frames_during_radar: int
    radar_duration_s: float
    expected_radar_duration_s: float
    radar_duration_error_ms: float
    causal_match_count: int
    causal_unmatched_frame_count: int
    causal_frame_lag_ms_median: float | None
    causal_frame_lag_ms_p95: float | None
    causal_frame_lag_ms_max: float | None
    frame_interval_ms_median: float | None
    frame_interval_ms_p95: float | None
    radar_adc_exists: bool
    radar_adc_size_bytes: int
    expected_radar_adc_size_bytes: int
    radar_adc_complete: bool
    radar_capture: dict[str, Any]
    session: dict[str, Any]
    synchronization_claim: str

    def to_record(self) -> dict[str, Any]:
        return {
            "ok": True,
            "capture_id": self.capture_id,
            "synchronization_mode": self.synchronization_mode.value,
            "camera_frame_count": self.camera_frame_count,
            "frames_before_radar": self.frames_before_radar,
            "frames_during_radar": self.frames_during_radar,
            "radar_duration_s": self.radar_duration_s,
            "expected_radar_duration_s": self.expected_radar_duration_s,
            "radar_duration_error_ms": self.radar_duration_error_ms,
            "causal_match_count": self.causal_match_count,
            "causal_unmatched_frame_count": self.causal_unmatched_frame_count,
            "causal_frame_lag_ms_median": self.causal_frame_lag_ms_median,
            "causal_frame_lag_ms_p95": self.causal_frame_lag_ms_p95,
            "causal_frame_lag_ms_max": self.causal_frame_lag_ms_max,
            "frame_interval_ms_median": self.frame_interval_ms_median,
            "frame_interval_ms_p95": self.frame_interval_ms_p95,
            "radar_adc_exists": self.radar_adc_exists,
            "radar_adc_size_bytes": self.radar_adc_size_bytes,
            "expected_radar_adc_size_bytes": self.expected_radar_adc_size_bytes,
            "radar_adc_complete": self.radar_adc_complete,
            "radar_capture": self.radar_capture,
            "session": self.session,
            "synchronization_claim": self.synchronization_claim,
        }


def inspect_synchronized_capture(
    path: str | Path,
) -> SynchronizedCaptureInspection:
    """Validate references, event ordering, and camera coverage of radar time."""

    manifest = manifest_path(path)
    root = manifest.parent
    artifact = load_synchronized_capture_manifest(manifest)
    events = load_capture_sync_events(
        resolve_relative_reference(root, artifact.event_log, "event_log")
    )
    if any(event.capture_id != artifact.capture_id for event in events):
        raise ValueError("Synchronized capture manifest and event log disagree.")

    monotonic_times = np.asarray(
        [event.monotonic_ns for event in events],
        dtype=np.int64,
    )
    if np.any(np.diff(monotonic_times) < 0):
        raise ValueError("Synchronized capture event timestamps must not decrease.")

    start_ns, stop_ns = _validated_radar_window(events)

    frame_events = _events_of_kind(events, CaptureSyncEventKind.CAMERA_FRAME)
    if len(frame_events) != artifact.camera_frame_count or not frame_events:
        raise ValueError("Synchronized capture camera frame count is inconsistent.")
    frames_root = resolve_relative_reference(
        root,
        artifact.camera_frames,
        "camera_frames",
    )
    if not frames_root.is_dir():
        raise ValueError("Synchronized capture camera frame directory does not exist.")
    _validate_camera_frames(frames_root, frame_events)
    frame_times = np.asarray(
        [event.monotonic_ns for event in frame_events],
        dtype=np.int64,
    )
    if np.any(np.diff(frame_times) <= 0):
        raise ValueError("Synchronized camera frame timestamps must increase.")
    during = (frame_times >= start_ns) & (frame_times <= stop_ns)
    if not bool(during.any()):
        raise ValueError("Synchronized capture has no camera frames during radar capture.")

    intervals_ms = np.diff(frame_times).astype(np.float64) / 1.0e6
    matches = build_causal_radar_camera_matches(
        events,
        artifact.radar_timing,
    )
    causal_lags_ms = np.asarray(
        [match.lag_ms for match in matches if match.accepted],
        dtype=np.float64,
    )
    radar_adc_size, expected_adc_size = _validated_radar_adc_size(
        root,
        reference=artifact.radar_adc,
        expected_size=artifact.radar_capture.expected_size_bytes,
    )
    radar_duration_s = (stop_ns - start_ns) / 1.0e9
    expected_duration_s = artifact.radar_timing.expected_duration_s
    return SynchronizedCaptureInspection(
        capture_id=artifact.capture_id,
        synchronization_mode=artifact.synchronization_mode,
        camera_frame_count=len(frame_events),
        frames_before_radar=int(np.count_nonzero(frame_times < start_ns)),
        frames_during_radar=int(np.count_nonzero(during)),
        radar_duration_s=radar_duration_s,
        expected_radar_duration_s=expected_duration_s,
        radar_duration_error_ms=(radar_duration_s - expected_duration_s) * 1_000.0,
        causal_match_count=int(causal_lags_ms.size),
        causal_unmatched_frame_count=len(matches) - int(causal_lags_ms.size),
        causal_frame_lag_ms_median=(
            float(np.median(causal_lags_ms)) if causal_lags_ms.size else None
        ),
        causal_frame_lag_ms_p95=(
            float(np.quantile(causal_lags_ms, 0.95)) if causal_lags_ms.size else None
        ),
        causal_frame_lag_ms_max=(float(np.max(causal_lags_ms)) if causal_lags_ms.size else None),
        frame_interval_ms_median=(float(np.median(intervals_ms)) if intervals_ms.size else None),
        frame_interval_ms_p95=(
            float(np.quantile(intervals_ms, 0.95)) if intervals_ms.size else None
        ),
        radar_adc_exists=True,
        radar_adc_size_bytes=radar_adc_size,
        expected_radar_adc_size_bytes=expected_adc_size,
        radar_adc_complete=True,
        radar_capture=artifact.radar_capture.to_record(),
        session=validate_capture_session(artifact.session),
        synchronization_claim=_synchronization_claim(artifact.synchronization_mode),
    )


def _events_of_kind(
    events: tuple[CaptureSyncEvent, ...],
    kind: CaptureSyncEventKind,
) -> tuple[CaptureSyncEvent, ...]:
    return tuple(event for event in events if event.kind is kind)


def _validated_radar_window(events: tuple[CaptureSyncEvent, ...]) -> tuple[int, int]:
    kinds = (
        CaptureSyncEventKind.CAMERA_ARMED,
        CaptureSyncEventKind.RADAR_START,
        CaptureSyncEventKind.RADAR_STOP,
        CaptureSyncEventKind.CAPTURE_CLOSED,
    )
    groups = tuple(_events_of_kind(events, kind) for kind in kinds)
    if any(len(group) != 1 for group in groups):
        raise ValueError(
            "Synchronized capture requires exactly one arm, radar start, stop, and close event."
        )
    armed, started, stopped, closed = (group[0] for group in groups)
    start_ns = started.monotonic_ns
    stop_ns = stopped.monotonic_ns
    if (
        not armed.event_index < started.event_index < stopped.event_index < closed.event_index
        or stop_ns <= start_ns
        or closed.monotonic_ns < stop_ns
    ):
        raise ValueError("Synchronized capture event ordering is invalid.")
    _validate_control_sequence_order((armed, started, stopped, closed))
    if closed.metadata.get("semantics") != "adc_file_flushed_and_size_verified":
        raise ValueError("Synchronized capture close event does not prove ADC flush validation.")
    return start_ns, stop_ns


def _validated_radar_adc_size(
    root: Path,
    *,
    reference: str | None,
    expected_size: int | None,
) -> tuple[int, int]:
    if reference is None:
        raise ValueError("Synchronized capture does not reference a radar ADC file.")
    radar_adc = resolve_relative_reference(root, reference, "radar_adc")
    if not radar_adc.is_file():
        raise ValueError("Synchronized capture radar ADC reference does not exist.")
    if expected_size is None:
        raise ValueError("Synchronized capture does not declare a finite ADC size.")
    actual_size = radar_adc.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            "Synchronized capture radar ADC size no longer matches its contract: "
            f"{actual_size} != {expected_size} bytes."
        )
    return actual_size, expected_size


def _validate_camera_frames(
    frames_root: Path,
    events: tuple[CaptureSyncEvent, ...],
) -> None:
    indices: list[int] = []
    for event in events:
        index = _required_int(event.metadata, "frame_index")
        relative = str(event.metadata.get("path", ""))
        frame = resolve_relative_reference(frames_root, relative, "camera frame")
        if not frame.is_file():
            raise ValueError(f"Synchronized camera frame does not exist: {relative}.")
        indices.append(index)
    if indices != list(range(len(events))):
        raise ValueError("Synchronized camera frame indices must be contiguous from zero.")


def _required_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Synchronized capture {key} must be a non-negative integer.")
    return value


def _validate_control_sequence_order(
    events: tuple[CaptureSyncEvent, ...],
) -> None:
    sequences = tuple(event.control_sequence for event in events)
    if any(sequence is None for sequence in sequences):
        raise ValueError("Synchronized capture control events require sequence numbers.")
    ordered = tuple(int(sequence) for sequence in sequences if sequence is not None)
    if any(current >= following for current, following in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("Synchronized capture control sequences must increase.")


def _synchronization_claim(mode: SynchronizationMode) -> str:
    if mode is SynchronizationMode.HARDWARE_TRIGGERED:
        return "shared_hardware_trigger_declared"
    return "same_host_software_timestamped_not_hardware_synchronized"


__all__ = [
    "SynchronizedCaptureInspection",
    "inspect_synchronized_capture",
]
