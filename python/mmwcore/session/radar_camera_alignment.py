"""Causal alignment between radar marker-grid frames and camera frames."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from .sync_protocol import CaptureSyncEvent, CaptureSyncEventKind
from .synchronized_capture import RadarCaptureTiming

RADAR_CAMERA_ALIGNMENT_POLICY = "causal_strictly_before"
RADAR_TIMESTAMP_SEMANTICS = "radar_start_marker_lower_bound_periodic_grid"


class RadarCameraMatchStatus(StrEnum):
    """Outcome of matching one radar frame to a causal camera frame."""

    MATCHED = "matched"
    NO_CAUSAL_FRAME = "no_causal_camera_frame"
    LAG_EXCEEDS_LIMIT = "lag_exceeds_limit"


@dataclass(frozen=True)
class RadarCameraFrameMatch:
    """One radar frame and its latest strictly earlier camera frame."""

    radar_frame_index: int
    radar_monotonic_ns: int
    status: RadarCameraMatchStatus
    camera_frame_index: int | None = None
    camera_event_index: int | None = None
    camera_monotonic_ns: int | None = None
    camera_path: str | None = None
    lag_ns: int | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.radar_frame_index, "radar_frame_index")
        _require_non_negative_int(self.radar_monotonic_ns, "radar_monotonic_ns")
        if not isinstance(self.status, RadarCameraMatchStatus):
            raise TypeError("status must be a RadarCameraMatchStatus.")

        camera_values = (
            self.camera_frame_index,
            self.camera_event_index,
            self.camera_monotonic_ns,
            self.camera_path,
            self.lag_ns,
        )
        if self.status is RadarCameraMatchStatus.NO_CAUSAL_FRAME:
            if any(value is not None for value in camera_values):
                raise ValueError("An unmatched radar frame cannot reference a camera frame.")
            return
        if any(value is None for value in camera_values):
            raise ValueError("A camera match requires complete camera-frame metadata.")

        assert self.camera_frame_index is not None
        assert self.camera_event_index is not None
        assert self.camera_monotonic_ns is not None
        assert self.camera_path is not None
        assert self.lag_ns is not None
        _require_non_negative_int(self.camera_frame_index, "camera_frame_index")
        _require_non_negative_int(self.camera_event_index, "camera_event_index")
        _require_non_negative_int(self.camera_monotonic_ns, "camera_monotonic_ns")
        _require_non_negative_int(self.lag_ns, "lag_ns")
        if self.camera_monotonic_ns >= self.radar_monotonic_ns:
            raise ValueError("A causal camera frame must precede its radar timestamp.")
        if self.lag_ns != self.radar_monotonic_ns - self.camera_monotonic_ns:
            raise ValueError("Camera match lag does not agree with its timestamps.")
        _validate_camera_path(self.camera_path)

    @property
    def accepted(self) -> bool:
        return self.status is RadarCameraMatchStatus.MATCHED

    @property
    def lag_ms(self) -> float | None:
        return self.lag_ns / 1.0e6 if self.lag_ns is not None else None

    def to_record(self) -> dict[str, Any]:
        return {
            "radar_frame_index": self.radar_frame_index,
            "radar_monotonic_ns": self.radar_monotonic_ns,
            "radar_timestamp_semantics": RADAR_TIMESTAMP_SEMANTICS,
            "alignment_policy": RADAR_CAMERA_ALIGNMENT_POLICY,
            "status": self.status.value,
            "camera_frame_index": self.camera_frame_index,
            "camera_event_index": self.camera_event_index,
            "camera_monotonic_ns": self.camera_monotonic_ns,
            "camera_path": self.camera_path,
            "lag_ns": self.lag_ns,
            "lag_ms": self.lag_ms,
        }

    @classmethod
    def from_record(cls, record: object) -> Self:
        """Restore one validated alignment record."""

        if not isinstance(record, dict):
            raise ValueError("Radar-camera alignment record must be a JSON object.")
        if record.get("alignment_policy") != RADAR_CAMERA_ALIGNMENT_POLICY:
            raise ValueError("Radar-camera alignment record uses an unsupported policy.")
        if record.get("radar_timestamp_semantics") != RADAR_TIMESTAMP_SEMANTICS:
            raise ValueError("Radar-camera alignment record uses unsupported radar timestamps.")
        try:
            status = RadarCameraMatchStatus(record["status"])
            radar_frame_index = _record_int(record, "radar_frame_index")
            radar_monotonic_ns = _record_int(record, "radar_monotonic_ns")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Radar-camera alignment record is incomplete or invalid.") from exc

        if status is RadarCameraMatchStatus.NO_CAUSAL_FRAME:
            camera_fields = (
                "camera_frame_index",
                "camera_event_index",
                "camera_monotonic_ns",
                "camera_path",
                "lag_ns",
                "lag_ms",
            )
            if any(record.get(field) is not None for field in camera_fields):
                raise ValueError("An unmatched alignment record cannot reference a camera frame.")
            return cls(
                radar_frame_index=radar_frame_index,
                radar_monotonic_ns=radar_monotonic_ns,
                status=status,
            )
        try:
            camera_path = record["camera_path"]
            if not isinstance(camera_path, str):
                raise TypeError("camera_path must be a string.")
            match = cls(
                radar_frame_index=radar_frame_index,
                radar_monotonic_ns=radar_monotonic_ns,
                status=status,
                camera_frame_index=_record_int(record, "camera_frame_index"),
                camera_event_index=_record_int(record, "camera_event_index"),
                camera_monotonic_ns=_record_int(record, "camera_monotonic_ns"),
                camera_path=camera_path,
                lag_ns=_record_int(record, "lag_ns"),
            )
            lag_ms = record.get("lag_ms")
            if (
                not isinstance(lag_ms, int | float)
                or isinstance(lag_ms, bool)
                or not math.isfinite(lag_ms)
                or not math.isclose(float(lag_ms), match.lag_ms or 0.0, abs_tol=1.0e-9)
            ):
                raise ValueError("Radar-camera alignment lag_ms disagrees with lag_ns.")
            return match
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Matched radar-camera record is incomplete or invalid.") from exc


def build_causal_radar_camera_matches(
    events: tuple[CaptureSyncEvent, ...],
    timing: RadarCaptureTiming,
    *,
    max_lag_ms: float | None = None,
) -> tuple[RadarCameraFrameMatch, ...]:
    """Match each radar marker-grid timestamp to the latest earlier camera frame.

    Equality is rejected. A non-earlier image is never selected, even when it
    is temporally closer than the preceding image.
    """

    if not isinstance(timing, RadarCaptureTiming):
        raise TypeError("timing must be a RadarCaptureTiming.")
    max_lag_ns = _max_lag_ns(max_lag_ms)
    starts = tuple(event for event in events if event.kind is CaptureSyncEventKind.RADAR_START)
    if len(starts) != 1:
        raise ValueError("Radar-camera alignment requires exactly one radar-start event.")
    frames = tuple(event for event in events if event.kind is CaptureSyncEventKind.CAMERA_FRAME)
    if not frames:
        raise ValueError("Radar-camera alignment requires camera-frame events.")

    camera_times = tuple(frame.monotonic_ns for frame in frames)
    if any(
        current >= following
        for current, following in zip(camera_times, camera_times[1:], strict=False)
    ):
        raise ValueError("Camera frame timestamps must increase strictly.")
    frame_references = tuple(_camera_reference(frame) for frame in frames)

    start_ns = starts[0].monotonic_ns
    period_ns = timing.frame_periodicity_ms * 1.0e6
    matches: list[RadarCameraFrameMatch] = []
    for radar_index in range(timing.num_frames):
        radar_ns = start_ns + round(radar_index * period_ns)
        camera_position = bisect_left(camera_times, radar_ns) - 1
        if camera_position < 0:
            matches.append(
                RadarCameraFrameMatch(
                    radar_frame_index=radar_index,
                    radar_monotonic_ns=radar_ns,
                    status=RadarCameraMatchStatus.NO_CAUSAL_FRAME,
                )
            )
            continue

        camera_frame_index, camera_path = frame_references[camera_position]
        camera_event = frames[camera_position]
        lag_ns = radar_ns - camera_event.monotonic_ns
        status = (
            RadarCameraMatchStatus.LAG_EXCEEDS_LIMIT
            if max_lag_ns is not None and lag_ns > max_lag_ns
            else RadarCameraMatchStatus.MATCHED
        )
        matches.append(
            RadarCameraFrameMatch(
                radar_frame_index=radar_index,
                radar_monotonic_ns=radar_ns,
                status=status,
                camera_frame_index=camera_frame_index,
                camera_event_index=camera_event.event_index,
                camera_monotonic_ns=camera_event.monotonic_ns,
                camera_path=camera_path,
                lag_ns=lag_ns,
            )
        )
    return tuple(matches)


def _camera_reference(event: CaptureSyncEvent) -> tuple[int, str]:
    frame_index = event.metadata.get("frame_index")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError("Camera frame metadata requires a non-negative frame_index.")
    path = event.metadata.get("path")
    if not isinstance(path, str):
        raise ValueError("Camera frame metadata requires a path.")
    _validate_camera_path(path)
    return frame_index, path


def _max_lag_ns(value: float | None) -> int | None:
    if value is None:
        return None
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("max_lag_ms must be finite and non-negative.")
    return round(float(value) * 1.0e6)


def _validate_camera_path(value: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("Camera frame path must be a safe relative path.")


def _require_non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _record_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Radar-camera alignment {key} must be a non-negative integer.")
    return value


__all__ = [
    "RADAR_CAMERA_ALIGNMENT_POLICY",
    "RADAR_TIMESTAMP_SEMANTICS",
    "RadarCameraFrameMatch",
    "RadarCameraMatchStatus",
    "build_causal_radar_camera_matches",
]
