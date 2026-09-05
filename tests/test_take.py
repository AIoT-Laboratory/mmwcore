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
        "setup.json",
        "radar.cfg",
        "radar.mmwa",
        "camera.mjpeg",
        "camera.index.bin",
    }
    assert take.frame_count == 2
    assert take.archive.frame_count == 2
    assert take.height_m == 1.5
    assert take.pitch_deg == 0.0
    assert take.roi is None
    assert take.setup_path.read_bytes() == (source / "setup.json").read_bytes()
    assert take.radar_time(1).lower_ns == 10_001_000
    assert take.sample_key(1) == (capture.session_id, 1)
    assert take.camera is not None
    assert take.camera.read(1) == b"\xff\xd8second\xff\xd9"
    reopened = open_take(take.root)
    assert reopened.session_id == take.session_id
    assert reopened.archive.adc_sha256 == take.archive.adc_sha256
    assert reopened.setup.path == reopened.root / "setup.json"

    session = json.loads((take.root / "session.json").read_text(encoding="utf-8"))
    assert session["schema"] == "openmmw.take.v3"
    assert session["setup"] == _file("setup.json", (take.root / "setup.json").read_bytes())
    assert "radar_height_m" not in session
    assert "radar_tilt_deg" not in session


def test_radar_only_take_has_exactly_four_files(tmp_path: Path) -> None:
    capture = read_capture(_raw_capture(tmp_path / "raw", camera=False))
    take = write_take(capture, tmp_path / "take")

    assert take.camera is None
    assert {path.name for path in take.root.iterdir()} == {
        "session.json",
        "setup.json",
        "radar.cfg",
        "radar.mmwa",
    }


def test_context_is_hashed_and_published_atomically(tmp_path: Path) -> None:
    capture = read_capture(_raw_capture(tmp_path / "raw", camera=False))
    context = b'{"schema":"openmmw.capture-context.v1"}\n'

    take = write_take(capture, tmp_path / "take", context=context)

    context_path = take.context_path
    assert context_path == take.root / "context.json"
    assert context_path is not None
    assert context_path.read_bytes() == context
    session = json.loads((take.root / "session.json").read_text(encoding="utf-8"))
    assert session["schema"] == "openmmw.take.v4"
    assert session["context"] == _file("context.json", context)
    assert open_take(take.root).context_path == take.context_path

    context_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="context.json"):
        open_take(take.root)


def test_capture_rejects_old_schema(tmp_path: Path) -> None:
    root = _raw_capture(tmp_path / "raw", camera=False)
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "mmwcli.take.v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema is unsupported"):
        read_capture(root)


def test_capture_rejects_unsupported_pitch(tmp_path: Path) -> None:
    root = _raw_capture(tmp_path / "raw", camera=False)
    _set_pitch(root, 1.0)

    with pytest.raises(ValueError, match="pitch_deg must be 0, 30, or 90"):
        read_capture(root)


def test_capture_accepts_downward_pitch(tmp_path: Path) -> None:
    root = _raw_capture(tmp_path / "raw", camera=False)
    _set_pitch(root, 90.0)

    assert read_capture(root).pitch_deg == 90.0


def test_capture_accepts_thirty_degree_pitch(tmp_path: Path) -> None:
    root = _raw_capture(tmp_path / "raw", camera=False)
    _set_pitch(root, 30.0)

    assert read_capture(root).pitch_deg == 30.0


def test_capture_preserves_scene_roi(tmp_path: Path) -> None:
    root = _raw_capture(tmp_path / "raw", camera=False)
    setup_path = root / "setup.json"
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["roi"] = {
        "frame": "level_forward_lateral_up",
        "min_m": [0.5, -1.5, 0.0],
        "max_m": [5.5, 1.5, 2.2],
    }
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    session_path = root / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["setup"] = _file("setup.json", setup_path.read_bytes())
    session_path.write_text(json.dumps(session), encoding="utf-8")

    capture = read_capture(root)
    take = write_take(capture, tmp_path / "take")

    assert take.roi is not None
    assert take.roi.frame == "level_forward_lateral_up"
    assert take.roi.min_m == (0.5, -1.5, 0.0)
    assert take.roi.max_m == (5.5, 1.5, 2.2)


@pytest.mark.parametrize(
    "roi",
    [
        {"frame": "radar", "min_m": [0, -1, 0], "max_m": [5, 1, 2]},
        {
            "frame": "level_forward_lateral_up",
            "min_m": [5, -1, 0],
            "max_m": [5, 1, 2],
        },
        {
            "frame": "level_forward_lateral_up",
            "min_m": [-1, -1, 0],
            "max_m": [5, 1, 2],
        },
    ],
)
def test_capture_rejects_invalid_scene_roi(tmp_path: Path, roi: object) -> None:
    root = _raw_capture(tmp_path / "raw", camera=False)
    setup_path = root / "setup.json"
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["roi"] = roi
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    session_path = root / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["setup"] = _file("setup.json", setup_path.read_bytes())
    session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="setup.roi"):
        read_capture(root)


def test_changed_setup_is_not_published(tmp_path: Path) -> None:
    capture = read_capture(_raw_capture(tmp_path / "raw", camera=False))
    target = tmp_path / "take"
    capture.setup_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="changed after validation"):
        write_take(capture, target)

    assert not target.exists()
    assert not Path(f"{target}.part").exists()


def test_inconsistent_capture_is_not_published(tmp_path: Path) -> None:
    capture = read_capture(_raw_capture(tmp_path / "raw", camera=True))
    target = tmp_path / "take"

    with pytest.raises(ValueError, match="camera disagrees"):
        write_take(replace(capture, camera=None), target)

    assert not target.exists()
    assert not Path(f"{target}.part").exists()


def test_open_take_rejects_old_schema(tmp_path: Path) -> None:
    capture = read_capture(_raw_capture(tmp_path / "raw", camera=False))
    take = write_take(capture, tmp_path / "take")
    session_path = take.root / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["schema"] = "openmmw.take.v2"
    session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="schema is unsupported"):
        open_take(take.root)


def test_capture_rejects_camera_that_disagrees_with_setup(tmp_path: Path) -> None:
    root = _raw_capture(tmp_path / "raw", camera=False)
    setup_path = root / "setup.json"
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["camera"] = {
        "device": "camera-0",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "max_bytes": 2_097_152,
    }
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    session_path = root / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["setup"] = _file("setup.json", setup_path.read_bytes())
    session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="camera disagrees"):
        read_capture(root)


def _set_pitch(root: Path, pitch_deg: float) -> None:
    setup_path = root / "setup.json"
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["mount"]["pitch_deg"] = pitch_deg
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    session_path = root / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["setup"] = _file("setup.json", setup_path.read_bytes())
    session_path.write_text(json.dumps(session), encoding="utf-8")


def _raw_capture(root: Path, *, camera: bool) -> Path:
    root.mkdir()
    adc = struct.pack("<16h", *range(16))
    (root / "adc.bin").write_bytes(adc)
    (root / "radar.cfg").write_bytes(_CONFIG)
    setup = {
        "schema": "mmwcli.snapshot.v1",
        "radar": {
            "model": "iwr6843",
            "revision": "es2",
            "port": "COM3",
            "bss": {"name": "bss.bin", "bytes": 4, "sha256": "a" * 64},
            "mss": {"name": "mss.bin", "bytes": 4, "sha256": "b" * 64},
            "d2xx": "AR-DevPack-EVM-012",
        },
        "dca": {"host": "192.168.33.30", "device": "192.168.33.180", "delay_us": 50},
        "mount": {"height_m": 1.5, "pitch_deg": 0.0},
        "camera": (
            {
                "device": "camera-0",
                "width": 1280,
                "height": 720,
                "fps": 30,
                "max_bytes": 2_097_152,
            }
            if camera
            else None
        ),
    }
    setup_bytes = json.dumps(setup, sort_keys=True).encode()
    (root / "setup.json").write_bytes(setup_bytes)
    adc_record = _file("adc.bin", adc)
    config_record = _file("radar.cfg", _CONFIG)
    record: dict[str, object] = {
        "schema": "mmwcli.take.v3",
        "session_id": "123e4567-e89b-42d3-a456-426614174000",
        "frame_count": 2,
        "frame_period_ns": 10_000_000,
        "radar_start": {"lower_ns": 1_000, "upper_ns": 2_000},
        "setup": _file("setup.json", setup_bytes),
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


@pytest.mark.parametrize("legacy_label", [False, True])
def test_take_reader_corrects_only_legacy_layout_without_changing_bytes(
    tmp_path: Path, legacy_label: bool
) -> None:
    import numpy as np

    from mmwcore.core import ADCComplexLayout
    from mmwcore.dsp import organize_adc_samples
    from mmwcore.io import ADCArchiveReader

    capture = read_capture(_raw_capture(tmp_path / "raw", camera=False))
    assert capture.radar.adc.layout is ADCComplexLayout.GROUP2_Q_THEN_I
    if legacy_label:
        capture = replace(
            capture,
            radar=replace(
                capture.radar,
                adc=replace(capture.radar.adc, layout=ADCComplexLayout.GROUP2_I_THEN_Q),
            ),
        )
    take = write_take(capture, tmp_path / "take")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in take.root.iterdir()}
    reader = ADCArchiveReader.from_take(take)
    assert reader.spec.layout is ADCComplexLayout.GROUP2_Q_THEN_I
    assert reader.archive.capture == capture.radar
    assert ADCArchiveReader(take.archive.path).capture == capture.radar
    raw = reader.read_frame(0)
    np.testing.assert_array_equal(raw.samples, np.arange(8, dtype=np.int16))
    cube = organize_adc_samples(raw, reader.spec)
    np.testing.assert_array_equal(cube.data.ravel(), [2 + 0j, 3 + 1j, 6 + 4j, 7 + 5j])
    assert ("adc_layout_correction" in raw.metadata) is legacy_label
    reader.verify_all()
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in take.root.iterdir()
    }


def test_take_reader_rejects_non_layout_contract_mismatch(tmp_path: Path) -> None:
    from mmwcore.io import ADCArchiveReader

    capture = read_capture(_raw_capture(tmp_path / "raw", camera=False))
    capture = replace(
        capture,
        radar=replace(
            capture.radar, profile=replace(capture.radar.profile, adc_sample_rate_hz=10e6)
        ),
    )
    take = write_take(capture, tmp_path / "take")
    with pytest.raises(ValueError, match="CFG disagrees"):
        ADCArchiveReader.from_take(take)
