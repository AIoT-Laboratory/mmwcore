"""Immutable radar-camera frame-alignment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from mmwcore.config import RadarCaptureSpec

from ._capture_paths import (
    manifest_path,
    resolve_relative_reference,
    validate_relative_reference,
)
from .radar_camera_alignment import (
    RADAR_CAMERA_ALIGNMENT_POLICY,
    RADAR_TIMESTAMP_SEMANTICS,
    RadarCameraFrameMatch,
    build_causal_radar_camera_matches,
)
from .sync_protocol import load_capture_sync_events
from .synchronized_capture import (
    CAMERA_AGENT_CLOCK_DOMAIN,
    load_synchronized_capture_manifest,
    validate_capture_session,
)
from .synchronized_capture_inspection import inspect_synchronized_capture

RADAR_CAMERA_ALIGNMENT_SCHEMA = "openmmw.radar_camera_alignment.v3"
RADAR_CAMERA_ALIGNMENT_INDEX = "frames.jsonl"


@dataclass(frozen=True)
class RadarCameraAlignmentExport:
    """Exported alignment index and compact quality summary."""

    output_root: Path
    manifest_path: Path
    index_path: Path
    capture_id: str
    radar_frame_count: int
    matched_frame_count: int
    rejected_frame_count: int
    status_counts: dict[str, int]
    max_lag_ms: float

    def to_record(self) -> dict[str, Any]:
        return {
            "ok": True,
            "output_root": str(self.output_root),
            "manifest_path": str(self.manifest_path),
            "index_path": str(self.index_path),
            "capture_id": self.capture_id,
            "radar_frame_count": self.radar_frame_count,
            "matched_frame_count": self.matched_frame_count,
            "rejected_frame_count": self.rejected_frame_count,
            "status_counts": dict(self.status_counts),
            "max_lag_ms": self.max_lag_ms,
            "alignment_policy": RADAR_CAMERA_ALIGNMENT_POLICY,
        }


@dataclass(frozen=True)
class RadarCameraAlignmentArtifact:
    """Validated immutable alignment manifest and frame records."""

    manifest_path: Path
    capture_id: str
    source_capture_manifest_sha256: str
    source_event_log_sha256: str
    index_sha256: str
    camera_path_root: str
    radar_capture: RadarCaptureSpec
    session: dict[str, Any]
    max_lag_ms: float
    matches: tuple[RadarCameraFrameMatch, ...]

    @classmethod
    def from_record(
        cls,
        record: object,
        *,
        manifest: Path,
        matches: tuple[RadarCameraFrameMatch, ...],
        actual_index_sha256: str,
    ) -> Self:
        payload = _alignment_record(record)
        _validate_alignment_header(payload)
        camera_path_root = _alignment_camera_root(payload)
        _validate_alignment_summary(payload, matches)
        max_lag_ms = _validated_max_lag_ms(payload, matches)
        capture_id, source_manifest_digest, source_event_digest, index_digest = _alignment_identity(
            payload,
            actual_index_sha256=actual_index_sha256,
        )
        radar_capture = RadarCaptureSpec.from_record(payload.get("radar_capture"))
        session = validate_capture_session(payload.get("session"))

        return cls(
            manifest_path=manifest,
            capture_id=capture_id,
            source_capture_manifest_sha256=source_manifest_digest,
            source_event_log_sha256=source_event_digest,
            index_sha256=index_digest,
            camera_path_root=camera_path_root,
            radar_capture=radar_capture,
            session=session,
            max_lag_ms=max_lag_ms,
            matches=matches,
        )


def _alignment_record(record: object) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Radar-camera alignment manifest must be a JSON object.")
    return record


def _validate_alignment_header(record: dict[str, Any]) -> None:
    expected = {
        "schema": RADAR_CAMERA_ALIGNMENT_SCHEMA,
        "alignment_policy": RADAR_CAMERA_ALIGNMENT_POLICY,
        "radar_timestamp_semantics": RADAR_TIMESTAMP_SEMANTICS,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"Radar-camera alignment uses unsupported {field}.")


def _alignment_camera_root(record: dict[str, Any]) -> str:
    index_reference = record.get("index")
    camera_path_root = record.get("camera_path_root")
    if not isinstance(index_reference, str) or not isinstance(camera_path_root, str):
        raise ValueError("Radar-camera alignment path references are invalid.")
    validate_relative_reference(index_reference, "alignment_index")
    validate_relative_reference(camera_path_root, "camera_path_root")
    if index_reference != RADAR_CAMERA_ALIGNMENT_INDEX:
        raise ValueError("Radar-camera alignment index name is unsupported.")
    return camera_path_root


def _validate_alignment_summary(
    record: dict[str, Any],
    matches: tuple[RadarCameraFrameMatch, ...],
) -> None:
    expected_counts = {
        "radar_frame_count": len(matches),
        "matched_frame_count": sum(match.accepted for match in matches),
        "rejected_frame_count": sum(not match.accepted for match in matches),
    }
    for field, expected in expected_counts.items():
        if _manifest_int(record, field) != expected:
            raise ValueError(f"Radar-camera alignment {field} does not match its index.")
    indices = tuple(match.radar_frame_index for match in matches)
    if indices != tuple(range(len(matches))):
        raise ValueError("Radar-camera alignment frame indices must be contiguous.")
    expected_statuses = dict(sorted(Counter(match.status.value for match in matches).items()))
    if record.get("status_counts") != expected_statuses:
        raise ValueError("Radar-camera alignment status counts do not match its index.")


def _validated_max_lag_ms(
    record: dict[str, Any],
    matches: tuple[RadarCameraFrameMatch, ...],
) -> float:
    value = record.get("max_lag_ms")
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Radar-camera alignment max_lag_ms must be finite and non-negative.")
    max_lag_ms = float(value)
    max_lag_ns = round(max_lag_ms * 1.0e6)
    for match in matches:
        if match.lag_ns is not None and (match.lag_ns <= max_lag_ns) != match.accepted:
            raise ValueError("Radar-camera alignment status disagrees with max_lag_ms.")
    return max_lag_ms


def _alignment_identity(
    record: dict[str, Any],
    *,
    actual_index_sha256: str,
) -> tuple[str, str, str, str]:
    capture_id = record.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id:
        raise ValueError("Radar-camera alignment source identity is incomplete.")
    source_manifest_digest = _sha256_digest(
        record.get("source_capture_manifest_sha256"),
        "source_capture_manifest_sha256",
    )
    source_event_digest = _sha256_digest(
        record.get("source_event_log_sha256"),
        "source_event_log_sha256",
    )
    index_digest = _sha256_digest(record.get("index_sha256"), "index_sha256")
    if index_digest != actual_index_sha256:
        raise ValueError("Radar-camera alignment index digest does not match its manifest.")
    return capture_id, source_manifest_digest, source_event_digest, index_digest


def export_radar_camera_alignment(
    capture: str | Path,
    output_root: str | Path,
    *,
    max_lag_ms: float,
) -> RadarCameraAlignmentExport:
    """Publish a causal frame index for an immutable synchronized capture."""

    source_manifest = manifest_path(capture)
    inspect_synchronized_capture(source_manifest)
    capture_root = source_manifest.parent
    artifact = load_synchronized_capture_manifest(source_manifest)
    event_log = resolve_relative_reference(capture_root, artifact.event_log, "event_log")
    events = load_capture_sync_events(event_log)
    matches = build_causal_radar_camera_matches(
        events,
        artifact.radar_timing,
        max_lag_ms=max_lag_ms,
    )
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"Radar-camera alignment output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        index_path = temporary / RADAR_CAMERA_ALIGNMENT_INDEX
        _write_index(index_path, matches)
        status_counts = Counter(match.status.value for match in matches)
        manifest_record = {
            "schema": RADAR_CAMERA_ALIGNMENT_SCHEMA,
            "capture_id": artifact.capture_id,
            "source_capture_manifest_sha256": _sha256(source_manifest),
            "source_event_log_sha256": _sha256(event_log),
            "source_synchronization_mode": artifact.synchronization_mode.value,
            "clock_domain": CAMERA_AGENT_CLOCK_DOMAIN,
            "alignment_policy": RADAR_CAMERA_ALIGNMENT_POLICY,
            "radar_timestamp_semantics": RADAR_TIMESTAMP_SEMANTICS,
            "camera_timestamp_semantics": artifact.camera.get("timestamp_semantics"),
            "radar_capture": artifact.radar_capture.to_record(),
            "session": validate_capture_session(artifact.session),
            "camera_path_root": artifact.camera_frames,
            "camera_path_semantics": "relative_to_source_camera_frames",
            "radar_timing": artifact.radar_timing.to_record(),
            "max_lag_ms": float(max_lag_ms),
            "index": RADAR_CAMERA_ALIGNMENT_INDEX,
            "index_sha256": _sha256(index_path),
            "radar_frame_count": len(matches),
            "matched_frame_count": sum(match.accepted for match in matches),
            "rejected_frame_count": sum(not match.accepted for match in matches),
            "status_counts": dict(sorted(status_counts.items())),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    except BaseException:
        _remove_incomplete_export(temporary)
        raise

    manifest_output = destination / "manifest.json"
    index_output = destination / RADAR_CAMERA_ALIGNMENT_INDEX
    return RadarCameraAlignmentExport(
        output_root=destination,
        manifest_path=manifest_output,
        index_path=index_output,
        capture_id=artifact.capture_id,
        radar_frame_count=len(matches),
        matched_frame_count=sum(match.accepted for match in matches),
        rejected_frame_count=sum(not match.accepted for match in matches),
        status_counts=dict(sorted(status_counts.items())),
        max_lag_ms=float(max_lag_ms),
    )


def load_radar_camera_alignment(
    path: str | Path,
) -> RadarCameraAlignmentArtifact:
    """Load and validate one immutable radar-camera alignment artifact."""

    source_manifest = manifest_path(path)
    record = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Radar-camera alignment manifest must be a JSON object.")
    index_reference = record.get("index")
    if not isinstance(index_reference, str):
        raise ValueError("Radar-camera alignment manifest is missing its index.")
    index_path = resolve_relative_reference(
        source_manifest.parent,
        index_reference,
        "alignment_index",
    )
    matches = tuple(
        RadarCameraFrameMatch.from_record(json.loads(line))
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return RadarCameraAlignmentArtifact.from_record(
        record,
        manifest=source_manifest,
        matches=matches,
        actual_index_sha256=_sha256(index_path),
    )


def _write_index(
    path: Path,
    matches: tuple[RadarCameraFrameMatch, ...],
) -> None:
    with path.open("x", encoding="utf-8") as file:
        for match in matches:
            file.write(json.dumps(match.to_record(), sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_incomplete_export(root: Path) -> None:
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_file():
            child.unlink()
    root.rmdir()


def _manifest_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Radar-camera alignment {key} must be a non-negative integer.")
    return value


def _sha256_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Radar-camera alignment {name} must be a lowercase SHA-256 digest.")
    return value


__all__ = [
    "RADAR_CAMERA_ALIGNMENT_INDEX",
    "RADAR_CAMERA_ALIGNMENT_SCHEMA",
    "RadarCameraAlignmentArtifact",
    "RadarCameraAlignmentExport",
    "export_radar_camera_alignment",
    "load_radar_camera_alignment",
]
