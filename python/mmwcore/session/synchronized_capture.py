"""Artifact contract for synchronized radar-camera capture."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmwcore._compat import Self, StrEnum
from mmwcore.config import RadarCaptureSpec

from ._capture_paths import (
    manifest_path,
    validate_relative_reference,
)
from .sync_protocol import validate_capture_id

SYNCHRONIZED_CAPTURE_SCHEMA = "openmmw.synchronized_camera_capture.v2"
CAMERA_AGENT_CLOCK_DOMAIN = "camera_agent_host_monotonic"
CAPTURE_COORDINATE_FRAME = "mount_compensated_forward_lateral_up"


class SynchronizationMode(StrEnum):
    """Declared physical strength of a synchronized capture."""

    SOFTWARE_TIMESTAMPED = "software_timestamped"
    HARDWARE_TRIGGERED = "hardware_triggered"


class RadarFrameTriggerMode(StrEnum):
    """Physical frame-trigger mode configured on the radar."""

    SOFTWARE = "software"
    HARDWARE = "hardware"


@dataclass(frozen=True)
class RadarCaptureTiming:
    """Finite radar frame schedule declared by the acquisition script."""

    num_frames: int
    frame_periodicity_ms: float
    frame_trigger_mode: RadarFrameTriggerMode

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_frames, int)
            or isinstance(self.num_frames, bool)
            or self.num_frames <= 0
        ):
            raise ValueError("Radar capture num_frames must be a positive integer.")
        if (
            not isinstance(self.frame_periodicity_ms, int | float)
            or isinstance(self.frame_periodicity_ms, bool)
            or not math.isfinite(self.frame_periodicity_ms)
            or self.frame_periodicity_ms <= 0
        ):
            raise ValueError("Radar capture frame_periodicity_ms must be finite and positive.")
        if not isinstance(self.frame_trigger_mode, RadarFrameTriggerMode):
            raise TypeError("frame_trigger_mode must be a RadarFrameTriggerMode.")

    @property
    def expected_duration_s(self) -> float:
        return self.num_frames * self.frame_periodicity_ms / 1_000.0

    def to_record(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "frame_periodicity_ms": self.frame_periodicity_ms,
            "frame_trigger_mode": self.frame_trigger_mode.value,
        }

    @classmethod
    def from_record(cls, record: object) -> Self:
        if not isinstance(record, dict):
            raise ValueError("Radar capture timing must be a JSON object.")
        try:
            num_frames = record["num_frames"]
            frame_periodicity_ms = record["frame_periodicity_ms"]
            frame_trigger_mode = RadarFrameTriggerMode(record["frame_trigger_mode"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Radar capture timing is incomplete or invalid.") from exc
        if not isinstance(num_frames, int) or isinstance(num_frames, bool):
            raise ValueError("Radar capture num_frames must be an integer.")
        if not isinstance(frame_periodicity_ms, int | float) or isinstance(
            frame_periodicity_ms, bool
        ):
            raise ValueError("Radar frame periodicity must be numeric.")
        return cls(
            num_frames=num_frames,
            frame_periodicity_ms=float(frame_periodicity_ms),
            frame_trigger_mode=frame_trigger_mode,
        )


@dataclass(frozen=True)
class SynchronizedCaptureArtifact:
    """Relative references and acquisition metadata for one capture."""

    capture_id: str
    synchronization_mode: SynchronizationMode
    event_log: str
    camera_frames: str
    camera_frame_count: int
    camera: dict[str, Any]
    radar: dict[str, Any]
    radar_capture: RadarCaptureSpec
    radar_timing: RadarCaptureTiming
    session: dict[str, Any]
    radar_adc: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = SYNCHRONIZED_CAPTURE_SCHEMA

    def __post_init__(self) -> None:
        validate_capture_id(self.capture_id)
        if not isinstance(self.synchronization_mode, SynchronizationMode):
            raise TypeError("synchronization_mode must be a SynchronizationMode.")
        if self.schema != SYNCHRONIZED_CAPTURE_SCHEMA:
            raise ValueError("Synchronized capture uses an unsupported schema.")
        if (
            not isinstance(self.camera_frame_count, int)
            or isinstance(self.camera_frame_count, bool)
            or self.camera_frame_count < 0
        ):
            raise ValueError("camera_frame_count must be a non-negative integer.")
        if not isinstance(self.radar_timing, RadarCaptureTiming):
            raise TypeError("radar_timing must be a RadarCaptureTiming.")
        if not isinstance(self.radar_capture, RadarCaptureSpec):
            raise TypeError("radar_capture must be a RadarCaptureSpec.")
        if (
            self.radar_capture.num_frames != self.radar_timing.num_frames
            or self.radar_capture.frame_periodicity_s is None
            or not math.isclose(
                self.radar_capture.frame_periodicity_s * 1_000.0,
                self.radar_timing.frame_periodicity_ms,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("Radar capture spec and synchronized frame timing do not match.")
        validate_relative_reference(self.event_log, "event_log")
        validate_relative_reference(self.camera_frames, "camera_frames")
        if self.radar_adc is not None:
            validate_relative_reference(self.radar_adc, "radar_adc")
        json.dumps(self.camera)
        json.dumps(self.radar)
        json.dumps(self.session)
        json.dumps(self.metadata)
        object.__setattr__(self, "session", validate_capture_session(self.session))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capture_id": self.capture_id,
            "synchronization_mode": self.synchronization_mode.value,
            "clock_domain": CAMERA_AGENT_CLOCK_DOMAIN,
            "event_log": self.event_log,
            "camera_frames": self.camera_frames,
            "camera_frame_count": self.camera_frame_count,
            "camera": dict(self.camera),
            "radar": dict(self.radar),
            "radar_capture": self.radar_capture.to_record(),
            "radar_timing": self.radar_timing.to_record(),
            "session": dict(self.session),
            "radar_adc": self.radar_adc,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: object) -> Self:
        if not isinstance(record, dict):
            raise ValueError("Synchronized capture manifest must be a JSON object.")
        try:
            if record.get("clock_domain") != CAMERA_AGENT_CLOCK_DOMAIN:
                raise ValueError("Synchronized capture uses an unsupported clock domain.")
            camera = record["camera"]
            radar = record["radar"]
            metadata = record.get("metadata", {})
            if (
                not isinstance(camera, dict)
                or not isinstance(radar, dict)
                or not isinstance(metadata, dict)
            ):
                raise ValueError(
                    "Synchronized capture camera, radar, and metadata must be objects."
                )
            return cls(
                schema=str(record.get("schema", "")),
                capture_id=str(record["capture_id"]),
                synchronization_mode=SynchronizationMode(record["synchronization_mode"]),
                event_log=str(record["event_log"]),
                camera_frames=str(record["camera_frames"]),
                camera_frame_count=_required_int(record, "camera_frame_count"),
                camera=camera,
                radar=radar,
                radar_capture=RadarCaptureSpec.from_record(record["radar_capture"]),
                radar_timing=RadarCaptureTiming.from_record(record["radar_timing"]),
                session=validate_capture_session(record["session"]),
                radar_adc=(
                    str(record["radar_adc"]) if record.get("radar_adc") is not None else None
                ),
                metadata=metadata,
            )
        except KeyError as exc:
            raise ValueError(f"Synchronized capture manifest is missing {exc.args[0]!r}.") from exc


def write_synchronized_capture_manifest(
    path: str | Path,
    artifact: SynchronizedCaptureArtifact,
) -> Path:
    """Atomically publish one synchronized-capture manifest."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Synchronized capture manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    payload = json.dumps(artifact.to_record(), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_synchronized_capture_manifest(
    path: str | Path,
) -> SynchronizedCaptureArtifact:
    """Load one synchronized-capture manifest without touching payload files."""

    manifest = manifest_path(path)
    return SynchronizedCaptureArtifact.from_record(json.loads(manifest.read_text(encoding="utf-8")))


def _required_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Synchronized capture {key} must be a non-negative integer.")
    return value


def validate_capture_session(record: object) -> dict[str, Any]:
    """Validate identity, mount, and measured-region metadata for one take."""

    if not isinstance(record, dict):
        raise ValueError("Synchronized capture requires session metadata.")
    session = dict(record)
    _validate_session_identity(session)
    _bounded_integer(session, "take_index", minimum=0)
    expected_people = _bounded_integer(session, "expected_people", minimum=0, maximum=5)
    if (session["action"] == "null") != (expected_people == 0):
        raise ValueError("Capture session requires expected_people 0 exactly for action 'null'.")
    _validate_radar_mount(_required_object(session, "radar_mount"))
    _validate_detection_region(_required_object(session, "detection_region"))
    return session


def _validate_session_identity(session: dict[str, Any]) -> None:
    for name in ("subject_id", "scene_id", "action"):
        value = session.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Capture session {name} must be a non-empty string.")


def _validate_radar_mount(mount: dict[str, Any]) -> None:
    if _finite_number(mount, "height_m", owner="radar_mount") < 0.0:
        raise ValueError("Capture radar_mount height_m must be non-negative.")
    for name in ("yaw_deg", "pitch_deg", "roll_deg"):
        _finite_number(mount, name, owner="radar_mount")


def _validate_detection_region(region: dict[str, Any]) -> None:
    if region.get("coordinate_frame") != CAPTURE_COORDINATE_FRAME:
        raise ValueError("Capture detection_region uses an unsupported coordinate frame.")
    center = region.get("center_xyz_m")
    if not isinstance(center, list | tuple) or len(center) != 3:
        raise ValueError("Capture detection_region center_xyz_m must contain three values.")
    if any(
        not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value)
        for value in center
    ):
        raise ValueError("Capture detection_region center_xyz_m must be finite and numeric.")
    for name in ("length_m", "width_m", "height_m"):
        if _finite_number(region, name, owner="detection_region") <= 0.0:
            raise ValueError(f"Capture detection_region {name} must be positive.")


def _required_object(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"Capture session requires a {field} object.")
    return value


def _finite_number(record: dict[str, Any], field: str, *, owner: str) -> float:
    value = record.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Capture {owner} {field} must be finite and numeric.")
    return float(value)


def _bounded_integer(
    record: dict[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = record.get(field)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        limit = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"Capture session {field} must be an integer in {limit}.")
    return value


__all__ = [
    "CAMERA_AGENT_CLOCK_DOMAIN",
    "CAPTURE_COORDINATE_FRAME",
    "RadarCaptureTiming",
    "RadarFrameTriggerMode",
    "SYNCHRONIZED_CAPTURE_SCHEMA",
    "SynchronizationMode",
    "SynchronizedCaptureArtifact",
    "load_synchronized_capture_manifest",
    "validate_capture_session",
    "write_synchronized_capture_manifest",
]
