"""Read the finite raw capture transaction emitted by mmwcli."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
from dataclasses import dataclass
from pathlib import Path

from mmwcore.config import RadarCaptureSpec, parse_ti_cli_capture_spec
from mmwcore.core import ADCComplexLayout

SCHEMA = "mmwcli.take.v3"
SETUP_SCHEMA = "mmwcli.snapshot.v1"


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
class SceneROI:
    """Axis-aligned scene volume frozen with a capture setup."""

    frame: str
    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]


@dataclass(frozen=True)
class SetupSnapshot:
    path: Path
    record: FileRecord
    height_m: float
    pitch_deg: float
    roi: SceneROI | None
    has_camera: bool


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
    setup: SetupSnapshot
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
            "setup",
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
    setup = _read_setup_snapshot(root, session["setup"], "capture")
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
    if setup.has_camera != (camera is not None):
        raise ValueError("capture camera disagrees with setup.json")
    expected_names = {"session.json", "setup.json", "adc.bin", "radar.cfg"}
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
        setup=setup,
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


def _read_setup_snapshot(root: Path, value: object, owner: str) -> SetupSnapshot:
    file_record = _file_record(value, "setup.json", "setup")
    payload = _verify_file(root, file_record, owner)
    try:
        snapshot = _object(json.loads(payload.decode("utf-8")), "setup")
    except UnicodeDecodeError as error:
        raise ValueError("setup.json must be UTF-8") from error
    _keys(snapshot, {"schema", "radar", "dca", "mount", "camera"}, {"roi"}, "setup")
    if snapshot["schema"] != SETUP_SCHEMA:
        raise ValueError("setup schema is unsupported")

    radar = _object(snapshot["radar"], "setup.radar")
    _keys(
        radar,
        {"model", "revision", "port", "bss", "mss", "d2xx"},
        set(),
        "setup.radar",
    )
    if radar["model"] != "iwr6843" or radar["revision"] != "es2":
        raise ValueError("setup must describe IWR6843 ES2")
    _exact_text(radar["port"], "setup.radar.port")
    _firmware_record(radar["bss"], "setup.radar.bss")
    _firmware_record(radar["mss"], "setup.radar.mss")
    _exact_text(radar["d2xx"], "setup.radar.d2xx")

    dca = _object(snapshot["dca"], "setup.dca")
    _keys(dca, {"host", "device", "delay_us"}, set(), "setup.dca")
    _canonical_ipv4(dca["host"], "setup.dca.host")
    _canonical_ipv4(dca["device"], "setup.dca.device")
    delay_us = _uint(dca["delay_us"], "setup.dca.delay_us")
    if not 5 <= delay_us <= 500:
        raise ValueError("setup.dca.delay_us must be in 5..500")

    mount = _object(snapshot["mount"], "setup.mount")
    _keys(mount, {"height_m", "pitch_deg"}, set(), "setup.mount")
    height_m = _positive_float(mount["height_m"], "setup.mount.height_m")
    if height_m > 10:
        raise ValueError("setup.mount.height_m must be in (0, 10]")
    pitch_deg = _mount_pitch(mount["pitch_deg"])

    roi_value = snapshot.get("roi")
    roi = None if roi_value is None else _scene_roi(roi_value)

    camera_value = snapshot["camera"]
    if camera_value is not None:
        camera = _object(camera_value, "setup.camera")
        _keys(
            camera,
            {"device", "width", "height", "fps", "max_bytes"},
            set(),
            "setup.camera",
        )
        _exact_text(camera["device"], "setup.camera.device")
        width = _positive_int(camera["width"], "setup.camera.width")
        height = _positive_int(camera["height"], "setup.camera.height")
        fps = _positive_int(camera["fps"], "setup.camera.fps")
        max_bytes = _positive_int(camera["max_bytes"], "setup.camera.max_bytes")
        if width > 16_384 or height > 16_384:
            raise ValueError("setup camera width and height must be in 1..16384")
        if fps > 240:
            raise ValueError("setup.camera.fps must be in 1..240")
        if not 4 <= max_bytes <= 64 << 20:
            raise ValueError("setup.camera.max_bytes must be in [4, 67108864]")
    return SetupSnapshot(
        path=root / file_record.path,
        record=file_record,
        height_m=height_m,
        pitch_deg=pitch_deg,
        roi=roi,
        has_camera=camera_value is not None,
    )


def _scene_roi(value: object) -> SceneROI:
    record = _object(value, "setup.roi")
    _keys(record, {"frame", "min_m", "max_m"}, set(), "setup.roi")
    if record["frame"] != "level_forward_lateral_up":
        raise ValueError("setup.roi.frame must be level_forward_lateral_up")
    lower = _finite_triplet(record["min_m"], "setup.roi.min_m")
    upper = _finite_triplet(record["max_m"], "setup.roi.max_m")
    if any(minimum >= maximum for minimum, maximum in zip(lower, upper, strict=True)):
        raise ValueError("setup.roi min_m must be below max_m on every axis")
    if lower[0] < 0.0 or lower[2] < 0.0:
        raise ValueError("setup.roi forward and up minima must be non-negative")
    return SceneROI(
        frame="level_forward_lateral_up",
        min_m=lower,
        max_m=upper,
    )


def _finite_triplet(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three finite numbers")
    normalized: list[float] = []
    for item in value:
        if type(item) is not int and type(item) is not float:
            raise ValueError(f"{label} must contain three finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain three finite numbers")
        normalized.append(number)
    return normalized[0], normalized[1], normalized[2]


def _firmware_record(value: object, label: str) -> None:
    record = _object(value, label)
    _keys(record, {"name", "bytes", "sha256"}, set(), label)
    name = _text(record["name"], f"{label}.name")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"{label}.name must be a filename")
    _positive_int(record["bytes"], f"{label}.bytes")
    _sha256(record["sha256"], f"{label}.sha256")


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


def _verify_file(root: Path, record: FileRecord, owner: str = "capture") -> bytes:
    path = root / record.path
    payload = path.read_bytes()
    if len(payload) != record.bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ValueError(f"{owner} artifact does not match session.json: {record.path}")
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


def _exact_text(value: object, label: str) -> str:
    text = _text(value, label)
    if text != text.strip() or "\0" in text:
        raise ValueError(f"{label} must be an exact value")
    return text


def _canonical_ipv4(value: object, label: str) -> str:
    text = _exact_text(value, label)
    try:
        address = ipaddress.IPv4Address(text)
    except ipaddress.AddressValueError as error:
        raise ValueError(f"{label} must be an IPv4 address") from error
    if str(address) != text:
        raise ValueError(f"{label} must be a canonical IPv4 address")
    return text


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
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{label} must be positive")
    return float(value)


def _mount_pitch(value: object) -> float:
    if type(value) is not int and type(value) is not float:
        raise ValueError("setup.mount.pitch_deg must be 0 or 90")
    pitch = float(value)
    if pitch not in {0.0, 90.0}:
        raise ValueError("setup.mount.pitch_deg must be 0 or 90")
    return pitch


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


__all__ = [
    "CameraRecord",
    "Capture",
    "FileRecord",
    "HostTimeRange",
    "SceneROI",
    "SCHEMA",
    "SETUP_SCHEMA",
    "SetupSnapshot",
    "read_capture",
]
