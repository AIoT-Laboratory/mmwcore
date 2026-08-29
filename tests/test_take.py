from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from mmwcore.io import open_take, read_capture, write_take

_CONFIG = b"""\
flushCfg
dfeDataOutputMode 1
channelCfg 1 1 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60 7 3 24 0 0 166 1 4 12500 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
frameCfg 0 0 1 2 10 1 0
lvdsStreamCfg -1 0 1 0
"""


def test_raw_capture_becomes_one_flat_verified_take(tmp_path: Path) -> None:
    source = _raw_capture(tmp_path / "raw", camera=True)
    capture = read_capture(source)
    take = write_take(capture, tmp_path / "take")

    assert {path.name for path in take.root.iterdir()} == {
        "session.json",
        "radar.cfg",
        "radar.mmwa",
        "camera.mjpeg",
        "camera.index.bin",
    }
    assert take.frame_count == 2
    assert take.archive.frame_count == 2
    assert take.radar_height_m == 1.5
    assert take.radar_tilt_deg == 90.0
    assert take.radar_time(1).lower_ns == 10_001_000
    assert take.sample_key(1) == (capture.session_id, 1)
    assert take.camera is not None
    assert take.camera.read(1) == b"\xff\xd8second\xff\xd9"
    reopened = open_take(take.root)
    assert reopened.session_id == take.session_id
    assert reopened.archive.adc_sha256 == take.archive.adc_sha256


def test_radar_only_take_has_exactly_three_files(tmp_path: Path) -> None:
    capture = read_capture(_raw_capture(tmp_path / "raw", camera=False))
    take = write_take(capture, tmp_path / "take")

    assert take.camera is None
    assert {path.name for path in take.root.iterdir()} == {
        "session.json",
        "radar.cfg",
        "radar.mmwa",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema", "mmwcli.take.v1"), ("radar_tilt_deg", 89.0)],
)
def test_capture_rejects_old_or_non_upright_mount(
    tmp_path: Path, field: str, value: object
) -> None:
    root = _raw_capture(tmp_path / "raw", camera=False)
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        read_capture(root)


def test_invalid_tilt_is_not_published(tmp_path: Path) -> None:
    capture = read_capture(_raw_capture(tmp_path / "raw", camera=False))
    target = tmp_path / "take"

    with pytest.raises(ValueError):
        write_take(replace(capture, radar_tilt_deg=89.0), target)

    assert not target.exists()
    assert not Path(f"{target}.part").exists()


def _raw_capture(root: Path, *, camera: bool) -> Path:
    root.mkdir()
    adc = struct.pack("<16h", *range(16))
    (root / "adc.bin").write_bytes(adc)
    (root / "radar.cfg").write_bytes(_CONFIG)
    adc_record = _file("adc.bin", adc)
    config_record = _file("radar.cfg", _CONFIG)
    record: dict[str, object] = {
        "schema": "mmwcli.take.v2",
        "session_id": "123e4567-e89b-42d3-a456-426614174000",
        "frame_count": 2,
        "frame_period_ns": 10_000_000,
        "radar_start": {"lower_ns": 1_000, "upper_ns": 2_000},
        "radar_height_m": 1.5,
        "radar_tilt_deg": 90.0,
        "radar": {
            "model": "iwr6843",
            "revision": "es2",
            "adc": adc_record,
            "config": config_record,
        },
    }
    if camera:
        first = b"\xff\xd8first\xff\xd9"
        second = b"\xff\xd8second\xff\xd9"
        payload = first + second
        index = struct.pack("<8sIIQ", b"MMWCAM01", 1, 24, 2)
        index += struct.pack("<QQQ", 0, len(first), 500)
        index += struct.pack("<QQQ", len(first), len(second), 700)
        (root / "camera.mjpeg").write_bytes(payload)
        (root / "camera.index.bin").write_bytes(index)
        record["camera"] = {
            "frames": 2,
            "time_semantics": "delivery_observed",
            "tick_hz": 1_000_000_000,
            "payload": _file("camera.mjpeg", payload),
            "index": _file("camera.index.bin", index),
        }
    (root / "session.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def _file(path: str, payload: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
