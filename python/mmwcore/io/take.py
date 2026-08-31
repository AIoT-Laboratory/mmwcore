"""Write and read the one fixed OpenMMW take layout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .adc_archive import ADCArchive, open_adc_archive, write_adc_archive
from .capture import (
    CameraRecord,
    Capture,
    FileRecord,
    HostTimeRange,
    SceneROI,
    SetupSnapshot,
    _read_setup_snapshot,
)

TAKE_SCHEMA = "openmmw.take.v3"
type SampleKey = tuple[str, int]

_CAMERA_MAGIC = b"MMWCAM01"
_CAMERA_HEADER = struct.Struct("<8sIIQ")
_CAMERA_ENTRY = struct.Struct("<QQQ")


@dataclass(frozen=True)
class CameraFrame:
    index: int
    offset: int
    bytes: int
    received_ns: int


@dataclass(frozen=True)
class Camera:
    payload_path: Path
    index_path: Path
    frames: tuple[CameraFrame, ...]

    def read(self, index: int) -> bytes:
        if type(index) is not int or not 0 <= index < len(self.frames):
            raise IndexError(index)
        frame = self.frames[index]
        with self.payload_path.open("rb") as payload:
            payload.seek(frame.offset)
            encoded = payload.read(frame.bytes)
        if (
            len(encoded) != frame.bytes
            or not encoded.startswith(b"\xff\xd8")
            or not encoded.endswith(b"\xff\xd9")
        ):
            raise ValueError(f"camera frame {index} is not a complete JPEG")
        return encoded


@dataclass(frozen=True)
class Take:
    root: Path
    session_id: str
    frame_count: int
    frame_period_ns: int
    radar_start: HostTimeRange
    setup: SetupSnapshot
    archive: ADCArchive = field(repr=False)
    config_path: Path
    camera: Camera | None

    @property
    def setup_path(self) -> Path:
        return self.setup.path

    @property
    def height_m(self) -> float:
        return self.setup.height_m

    @property
    def pitch_deg(self) -> float:
        return self.setup.pitch_deg

    @property
    def roi(self) -> SceneROI | None:
        return self.setup.roi

    def radar_time(self, index: int) -> HostTimeRange:
        if type(index) is not int or not 0 <= index < self.frame_count:
            raise IndexError(index)
        offset = index * self.frame_period_ns
        return HostTimeRange(
            self.radar_start.lower_ns + offset,
            self.radar_start.upper_ns + offset,
        )

    def sample_key(self, index: int) -> SampleKey:
        self.radar_time(index)
        return (self.session_id, index)


def write_take(capture: Capture, destination: str | Path) -> Take:
    if not isinstance(capture, Capture):
        raise TypeError("capture must be a Capture")
    target = Path(destination).resolve(strict=False)
    stage = Path(f"{target}.part")
    if target.exists() or stage.exists():
        raise ValueError(f"take destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    try:
        _copy_verified_file(capture.setup_path, capture.setup.record, stage / "setup.json")
        config_path = stage / "radar.cfg"
        shutil.copyfile(capture.config_path, config_path)
        archive = write_adc_archive(
            capture.adc_path,
            stage / "radar.mmwa",
            capture.radar,
            expected_adc_sha256=capture.adc.sha256,
        )
        archive.verify_all()
        camera_record = _copy_camera(capture, stage)
        session = _session_record(capture, archive, camera_record)
        _write_json(stage / "session.json", session)
        open_take(stage)
        stage.rename(target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return open_take(target)


def open_take(path: str | Path) -> Take:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"take is not a directory: {root}")
    session = json.loads((root / "session.json").read_text(encoding="utf-8"))
    if not isinstance(session, dict) or session.get("schema") != TAKE_SCHEMA:
        raise ValueError("take schema is unsupported")
    required = {
        "schema",
        "session_id",
        "frame_count",
        "frame_period_ns",
        "radar_start",
        "setup",
        "radar",
    }
    if not required <= set(session) or set(session) - required - {"camera"}:
        raise ValueError("take fields are invalid")
    session_id = _text(session["session_id"], "session_id")
    frame_count = _positive_int(session["frame_count"], "frame_count")
    frame_period_ns = _positive_int(session["frame_period_ns"], "frame_period_ns")
    setup = _read_setup_snapshot(root, session["setup"], "take")
    start = _object(session["radar_start"], "radar_start")
    if set(start) != {"lower_ns", "upper_ns"}:
        raise ValueError("radar_start fields are invalid")
    radar_start = HostTimeRange(
        _uint(start["lower_ns"], "radar_start.lower_ns"),
        _uint(start["upper_ns"], "radar_start.upper_ns"),
    )
    archive = _open_radar(root, session["radar"], frame_count, frame_period_ns)
    camera_value = session.get("camera")
    camera = None if camera_value is None else _open_camera(root, camera_value)
    if setup.has_camera != (camera is not None):
        raise ValueError("take camera disagrees with setup.json")
    _validate_names(root, camera is not None)
    return Take(
        root=root,
        session_id=session_id,
        frame_count=frame_count,
        frame_period_ns=frame_period_ns,
        radar_start=radar_start,
        setup=setup,
        archive=archive,
        config_path=root / "radar.cfg",
        camera=camera,
    )


def _open_radar(root: Path, value: object, frame_count: int, frame_period_ns: int) -> ADCArchive:
    record = _object(value, "radar")
    if set(record) != {"archive", "config"}:
        raise ValueError("radar fields are invalid")
    archive_record = _object(record["archive"], "radar.archive")
    if set(archive_record) != {"path", "bytes", "capture_sha256", "adc_sha256"}:
        raise ValueError("radar archive fields are invalid")
    if archive_record["path"] != "radar.mmwa":
        raise ValueError("radar archive path must be radar.mmwa")
    archive = open_adc_archive(root / "radar.mmwa")
    if (
        archive.archive_size != _positive_int(archive_record["bytes"], "radar.archive.bytes")
        or archive.capture_sha256 != _sha256(archive_record["capture_sha256"], "capture_sha256")
        or archive.adc_sha256 != _sha256(archive_record["adc_sha256"], "adc_sha256")
        or archive.frame_count != frame_count
    ):
        raise ValueError("radar archive disagrees with session.json")
    period = archive.capture.frame_periodicity_s
    if period is None or round(period * 1_000_000_000) != frame_period_ns:
        raise ValueError("radar archive period disagrees with session.json")
    config_record = _file_record(record["config"], "radar.cfg", "radar.config")
    _verify_file(root, config_record)
    return archive


def _copy_camera(capture: Capture, stage: Path) -> CameraRecord | None:
    if capture.camera is None:
        return None
    source = capture.camera
    shutil.copyfile(capture.root / source.payload.path, stage / source.payload.path)
    shutil.copyfile(capture.root / source.index.path, stage / source.index.path)
    return source


def _session_record(
    capture: Capture,
    archive: ADCArchive,
    camera: CameraRecord | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": TAKE_SCHEMA,
        "session_id": capture.session_id,
        "frame_count": capture.frame_count,
        "frame_period_ns": capture.frame_period_ns,
        "radar_start": {
            "lower_ns": capture.radar_start.lower_ns,
            "upper_ns": capture.radar_start.upper_ns,
        },
        "setup": _record(capture.setup.record),
        "radar": {
            "archive": {
                "path": "radar.mmwa",
                "bytes": archive.archive_size,
                "capture_sha256": archive.capture_sha256,
                "adc_sha256": archive.adc_sha256,
            },
            "config": _record(capture.config),
        },
    }
    if camera is not None:
        record["camera"] = {
            "frames": camera.frames,
            "time_semantics": "delivery_observed",
            "tick_hz": 1_000_000_000,
            "payload": _record(camera.payload),
            "index": _record(camera.index),
        }
    return record


def _record(value: FileRecord) -> dict[str, object]:
    return {"path": value.path, "bytes": value.bytes, "sha256": value.sha256}


def _copy_verified_file(source: Path, record: FileRecord, destination: Path) -> None:
    payload = source.read_bytes()
    if len(payload) != record.bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ValueError(f"capture artifact changed after validation: {record.path}")
    destination.write_bytes(payload)


def _open_camera(root: Path, value: object) -> Camera:
    record = _object(value, "camera")
    if set(record) != {"frames", "time_semantics", "tick_hz", "payload", "index"}:
        raise ValueError("camera fields are invalid")
    if record["time_semantics"] != "delivery_observed" or record["tick_hz"] != 1_000_000_000:
        raise ValueError("camera time contract is invalid")
    frame_count = _positive_int(record["frames"], "camera.frames")
    payload_record = _file_record(record["payload"], "camera.mjpeg", "camera.payload")
    index_record = _file_record(record["index"], "camera.index.bin", "camera.index")
    _verify_file(root, payload_record)
    index = _verify_file(root, index_record)
    frames = _decode_camera_index(index, frame_count, payload_record.bytes)
    return Camera(
        payload_path=root / payload_record.path,
        index_path=root / index_record.path,
        frames=frames,
    )


def _decode_camera_index(
    payload: bytes, frame_count: int, payload_bytes: int
) -> tuple[CameraFrame, ...]:
    if len(payload) < _CAMERA_HEADER.size:
        raise ValueError("camera index is truncated")
    magic, version, header_bytes, count = _CAMERA_HEADER.unpack_from(payload)
    if magic != _CAMERA_MAGIC or version != 1 or header_bytes != _CAMERA_HEADER.size:
        raise ValueError("camera index header is invalid")
    if count != frame_count or len(payload) != _CAMERA_HEADER.size + count * _CAMERA_ENTRY.size:
        raise ValueError("camera index frame count is invalid")
    frames = []
    expected_offset = 0
    previous_time = 0
    for index in range(frame_count):
        offset = _CAMERA_HEADER.size + index * _CAMERA_ENTRY.size
        payload_offset, size, received_ns = _CAMERA_ENTRY.unpack_from(payload, offset)
        if (
            payload_offset != expected_offset
            or size == 0
            or (index != 0 and received_ns < previous_time)
        ):
            raise ValueError("camera index is not contiguous and time ordered")
        frames.append(CameraFrame(index, payload_offset, size, received_ns))
        expected_offset += size
        previous_time = received_ns
    if expected_offset != payload_bytes:
        raise ValueError("camera index does not cover camera.mjpeg")
    return tuple(frames)


def _write_json(path: Path, value: dict[str, object]) -> None:
    encoded = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="ascii", newline="\n") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def _validate_names(root: Path, has_camera: bool) -> None:
    expected = {"session.json", "setup.json", "radar.cfg", "radar.mmwa"}
    if has_camera:
        expected.update({"camera.mjpeg", "camera.index.bin"})
    if {item.name for item in root.iterdir()} != expected:
        raise ValueError("take directory contains unexpected files")


def _file_record(value: object, expected_path: str, label: str) -> FileRecord:
    record = _object(value, label)
    if set(record) != {"path", "bytes", "sha256"} or record.get("path") != expected_path:
        raise ValueError(f"{label} fields are invalid")
    return FileRecord(
        path=expected_path,
        bytes=_positive_int(record["bytes"], f"{label}.bytes"),
        sha256=_sha256(record["sha256"], f"{label}.sha256"),
    )


def _verify_file(root: Path, record: FileRecord) -> bytes:
    payload = (root / record.path).read_bytes()
    if len(payload) != record.bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ValueError(f"take artifact does not match session.json: {record.path}")
    return payload


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _uint(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


__all__ = [
    "Camera",
    "CameraFrame",
    "HostTimeRange",
    "SampleKey",
    "TAKE_SCHEMA",
    "Take",
    "open_take",
    "write_take",
]
