"""Read the finite raw capture transaction emitted by mmwcli."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from mmwcore.config import RadarCaptureSpec, parse_ti_cli_capture_spec
from mmwcore.core import ADCComplexLayout

SCHEMA = "mmwcli.take.v1"


@dataclass(frozen=True)
class HostTimeRange:
    lower_ns: int
    upper_ns: int

    def __post_init__(self) -> None:
        if type(self.lower_ns) is not int or type(self.upper_ns) is not int:
            raise TypeError("host time bounds must be integers")
        if self.lower_ns < 0 or self.lower_ns > self.upper_ns:
            raise ValueError("host time bounds are invalid")


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CameraRecord:
    frames: int
    payload: FileRecord
    index: FileRecord


@dataclass(frozen=True)
class Capture:
    root: Path
    session_id: str
    frame_count: int
    frame_period_ns: int
    radar_start: HostTimeRange
    radar_height_m: float
    radar: RadarCaptureSpec
    adc: FileRecord
    config: FileRecord
    camera: CameraRecord | None

    @property
    def adc_path(self) -> Path:
        return self.root / self.adc.path

    @property
    def config_path(self) -> Path:
        return self.root / self.config.path


def read_capture(path: str | Path) -> Capture:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"capture is not a directory: {root}")
    session = _object(json.loads((root / "session.json").read_text(encoding="utf-8")), "capture")
    _keys(
        session,
        {
            "schema",
            "session_id",
            "frame_count",
            "frame_period_ns",
            "radar_start",
            "radar_height_m",
            "radar",
        },
        {"camera"},
        "capture",
    )
    if session["schema"] != SCHEMA:
        raise ValueError("capture schema is unsupported")
    session_id = _text(session["session_id"], "session_id")
    frame_count = _positive_int(session["frame_count"], "frame_count")
    frame_period_ns = _positive_int(session["frame_period_ns"], "frame_period_ns")
    radar_height_m = _positive_float(session["radar_height_m"], "radar_height_m")
    start_record = _object(session["radar_start"], "radar_start")
    _keys(start_record, {"lower_ns", "upper_ns"}, set(), "radar_start")
    radar_start = HostTimeRange(
        _uint(start_record["lower_ns"], "radar_start.lower_ns"),
        _uint(start_record["upper_ns"], "radar_start.upper_ns"),
    )

    radar_record = _object(session["radar"], "radar")
    _keys(radar_record, {"model", "revision", "adc", "config"}, set(), "radar")
    if radar_record["model"] != "iwr6843" or radar_record["revision"] != "es2":
        raise ValueError("capture must be the gated IWR6843 ES2 contract")
    adc = _file_record(radar_record["adc"], "adc.bin", "radar.adc")
    config = _file_record(radar_record["config"], "radar.cfg", "radar.config")
    config_bytes = _verify_file(root, config)
    radar = _radar_capture(config_bytes)
    _verify_file(root, adc)
    if radar.num_frames != frame_count or radar.expected_size_bytes != adc.bytes:
        raise ValueError("capture radar geometry disagrees with session.json")
    expected_period = radar.frame_periodicity_s
    if expected_period is None or round(expected_period * 1_000_000_000) != frame_period_ns:
        raise ValueError("capture radar period disagrees with session.json")

    camera_value = session.get("camera")
    camera = None if camera_value is None else _camera_record(root, camera_value)
    expected_names = {"session.json", "adc.bin", "radar.cfg"}
    if camera is not None:
        expected_names.update({camera.payload.path, camera.index.path})
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("capture directory contains unexpected files")
    return Capture(
        root=root,
        session_id=session_id,
        frame_count=frame_count,
        frame_period_ns=frame_period_ns,
        radar_start=radar_start,
        radar_height_m=radar_height_m,
        radar=radar,
        adc=adc,
        config=config,
        camera=camera,
    )


def _radar_capture(
    config_bytes: bytes,
) -> RadarCaptureSpec:
    try:
        text = config_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("radar.cfg must be UTF-8") from error
    return parse_ti_cli_capture_spec(
        text,
        layout=ADCComplexLayout.GROUP2_I_THEN_Q,
    )


def _camera_record(root: Path, value: object) -> CameraRecord:
    record = _object(value, "camera")
    _keys(
        record,
        {"frames", "time_semantics", "tick_hz", "payload", "index"},
        set(),
        "camera",
    )
    if record["time_semantics"] != "delivery_observed" or record["tick_hz"] != 1_000_000_000:
        raise ValueError("camera must use the delivery_observed 1GHz host clock")
    frames = _positive_int(record["frames"], "camera.frames")
    payload = _file_record(record["payload"], "camera.mjpeg", "camera.payload")
    index = _file_record(record["index"], "camera.index.bin", "camera.index")
    _verify_file(root, payload)
    _verify_file(root, index)
    return CameraRecord(frames=frames, payload=payload, index=index)


def _file_record(value: object, expected_path: str, label: str) -> FileRecord:
    record = _object(value, label)
    _keys(record, {"path", "bytes", "sha256"}, set(), label)
    path = _text(record["path"], f"{label}.path")
    if path != expected_path:
        raise ValueError(f"{label}.path must be {expected_path!r}")
    size = _positive_int(record["bytes"], f"{label}.bytes")
    digest = _text(record["sha256"], f"{label}.sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label}.sha256 is invalid")
    return FileRecord(path=path, bytes=size, sha256=digest)


def _verify_file(root: Path, record: FileRecord) -> bytes:
    path = root / record.path
    payload = path.read_bytes()
    if len(payload) != record.bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ValueError(f"capture artifact does not match session.json: {record.path}")
    return payload


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return value


def _keys(
    value: dict[str, object],
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    keys = set(value)
    if not required <= keys or keys - required - optional:
        raise ValueError(f"{label} fields are invalid")


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


def _positive_float(value: object, label: str) -> float:
    if type(value) is not int and type(value) is not float:
        raise ValueError(f"{label} must be positive")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


__all__ = [
    "CameraRecord",
    "Capture",
    "FileRecord",
    "HostTimeRange",
    "SCHEMA",
    "read_capture",
]
