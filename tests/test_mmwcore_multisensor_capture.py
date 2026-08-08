from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from mmwcore import open_multisensor_capture as public_open_multisensor_capture
from mmwcore.io import (
    MMWCLI_MULTISENSOR_SESSION_SCHEMA_V1,
    MMWCLI_SENSOR_INDEX_SCHEMA_V1,
    MappedTimeInterval,
    MultisensorCapture,
    causal_match,
    open_multisensor_capture,
)

_SESSION_ID = "12345678-1234-4abc-8def-1234567890ab"
_HEADER = struct.Struct("<8sHHHHQQ")
_ENTRY = struct.Struct("<QQQQQQQII")
_EVENT_ID = 7


def test_open_multisensor_capture_reads_training_items_and_time(tmp_path: Path) -> None:
    root, _record = _write_fixture(tmp_path)

    capture = open_multisensor_capture(root)

    assert public_open_multisensor_capture is open_multisensor_capture
    assert MMWCLI_MULTISENSOR_SESSION_SCHEMA_V1 == "mmwcli.multisensor_session.v1"
    assert MMWCLI_SENSOR_INDEX_SCHEMA_V1 == "mmwcli.sensor_index.v1"
    assert isinstance(capture, MultisensorCapture)
    assert capture.session_id == _SESSION_ID
    assert capture.synchronization_grade == "software_barrier"
    assert capture.item_count == 2
    assert capture.payload_bytes == 13
    assert [source.source_id for source in capture.sources] == ["radar-0", "camera-0"]

    radar = capture.source("radar-0").read_item(0)
    camera = list(capture.source("camera-0").items())[0]
    assert radar.payload == b"RADAR"
    assert camera.payload == b"JPEGDATA"
    assert radar.training_key == (_SESSION_ID, "radar-0", 0, _EVENT_ID)
    assert camera.training_key == (_SESSION_ID, "camera-0", 0, _EVENT_ID)
    assert radar.mapped_time == MappedTimeInterval(99_999_995, 110_000_005)
    assert camera.mapped_time == MappedTimeInterval(102_999_990, 107_000_010)
    assert causal_match(radar, camera, lag_min_ns=-1_000_000, lag_max_ns=1_000_000)
    assert not causal_match(radar, camera, lag_min_ns=20_000_000, lag_max_ns=30_000_000)
    assert [item.training_key for item in capture.items()] == [
        radar.training_key,
        camera.training_key,
    ]
    assert capture.sync_events[0].mapped_time == MappedTimeInterval(
        99_999_975,
        100_000_025,
    )
    with pytest.raises(FrozenInstanceError):
        capture.session_id = "changed"  # type: ignore[misc]


def test_opens_the_exact_go_two_source_golden(tmp_path: Path) -> None:
    root = _write_go_golden(tmp_path)

    capture = open_multisensor_capture(root)
    radar = list(capture.source("radar-0").items())
    camera = list(capture.source("camera-0").items())

    assert capture.session_id == "123e4567-e89b-42d3-a456-426614174000"
    assert capture.item_count == 4
    assert capture.payload_bytes == 28
    assert [item.payload for item in radar] == [bytes(range(8)), bytes(range(8, 16))]
    assert [item.payload for item in camera] == [b"camera", b"-frame"]
    assert radar[0].training_key == (capture.session_id, "radar-0", 0)
    assert camera[0].training_key == (capture.session_id, "camera-0", 0)
    assert radar[0].mapped_time == MappedTimeInterval(1_000_095_000, 1_000_115_000)
    assert camera[0].mapped_time == MappedTimeInterval(1_008_980_000, 1_011_020_000)


def test_rejects_non_exact_schema_hashes_and_undeclared_leaves(tmp_path: Path) -> None:
    root, record = _write_fixture(tmp_path)
    record["unknown"] = True
    _write_session(root, record)
    with pytest.raises(ValueError, match="exact key"):
        open_multisensor_capture(root)

    root, _record = _write_fixture(tmp_path, name="bad-hash")
    payload = root / "sensors" / "radar-0" / "adc.bin"
    payload.write_bytes(b"radar")
    with pytest.raises(ValueError, match="SHA-256"):
        open_multisensor_capture(root)

    root, _record = _write_fixture(tmp_path, name="extra-leaf")
    (root / "sensors" / "camera-0" / "notes.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        open_multisensor_capture(root)


def test_sensor_index_header_bounds_are_checked_before_entries(tmp_path: Path) -> None:
    root, record = _write_fixture(tmp_path)
    index_path = root / "sensors" / "radar-0" / "index.bin"
    payload = bytearray(index_path.read_bytes())
    payload[: _HEADER.size] = _HEADER.pack(b"MMWSIDX1", 1, 32, 64, 0, 2, 5)
    index_path.write_bytes(payload)
    radar = _source(record, "radar-0")
    index_artifact = _artifact(radar, "index")
    index_artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    _write_session(root, record)

    with pytest.raises(ValueError, match="header does not match"):
        open_multisensor_capture(root)


def test_read_item_validates_indices_and_does_not_own_external_resources(
    tmp_path: Path,
) -> None:
    root, _record = _write_fixture(tmp_path)
    source = open_multisensor_capture(root).source("radar-0")

    with pytest.raises(TypeError):
        source.read_item(True)
    with pytest.raises(IndexError):
        source.read_item(1)
    with pytest.raises(KeyError):
        open_multisensor_capture(root).source("missing")
    with pytest.raises(ValueError):
        causal_match(
            MappedTimeInterval(0, 1),
            MappedTimeInterval(2, 3),
            lag_min_ns=2,
            lag_max_ns=1,
        )


def _write_fixture(tmp_path: Path, *, name: str = "capture") -> tuple[Path, dict[str, Any]]:
    root = tmp_path / name
    sensors = root / "sensors"
    radar_payload = b"RADAR"
    camera_payload = b"JPEGDATA"
    radar_index = _index(radar_payload, ticks=100, duration=10)
    camera_index = _index(camera_payload, ticks=105, duration=4)
    for source_id, payload_name, payload, index in (
        ("radar-0", "adc.bin", radar_payload, radar_index),
        ("camera-0", "frames.bin", camera_payload, camera_index),
    ):
        source_root = sensors / source_id
        source_root.mkdir(parents=True)
        (source_root / payload_name).write_bytes(payload)
        (source_root / "index.bin").write_bytes(index)

    radar = _source_record(
        source_id="radar-0",
        kind="radar",
        payload_name="adc.bin",
        payload_format="ti.raw_adc.v1",
        payload=radar_payload,
        index=radar_index,
        tick=100,
        uncertainty_ns=5,
    )
    camera = _source_record(
        source_id="camera-0",
        kind="camera",
        payload_name="frames.bin",
        payload_format="image.jpeg.v1",
        payload=camera_payload,
        index=camera_index,
        tick=105,
        uncertainty_ns=10,
    )
    record: dict[str, Any] = {
        "schema": "mmwcli.multisensor_session.v1",
        "session_id": _SESSION_ID,
        "synchronization_grade": "software_barrier",
        "host_clock": {
            "clock_id": "host-0",
            "tick_hz": 1_000_000_000,
            "wrap_ticks": 0,
            "timestamp_semantics": "host_monotonic",
        },
        "sources": [radar, camera],
        "sync_events": [
            {
                "sync_event_id": _EVENT_ID,
                "clock_id": "radar-0-clock",
                "tick": 100,
                "wrap_count": 0,
                "edge": "rising",
                "evidence_kind": "hardware_observation",
                "generator": "trigger-0",
                "observer": "radar-0",
                "routing_id": "line-0",
                "observation_ids": ["radar-0-obs"],
                "uncertainty_ns": 20,
            }
        ],
        "totals": {
            "source_count": 2,
            "required_source_count": 2,
            "complete_source_count": 2,
            "item_count": 2,
            "payload_bytes": len(radar_payload) + len(camera_payload),
        },
        "application_metadata": {"org.openmmw.training": {"split": "train"}},
    }
    _write_session(root, record)
    return root, record


def _source_record(
    *,
    source_id: str,
    kind: str,
    payload_name: str,
    payload_format: str,
    payload: bytes,
    index: bytes,
    tick: int,
    uncertainty_ns: int,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "kind": kind,
        "required": True,
        "outcome": "complete",
        "producer": {"name": "fixture", "version": "1.0"},
        "limits": {
            "max_items": 1,
            "max_item_bytes": len(payload),
            "max_payload_bytes": len(payload),
        },
        "payload": {"filename": payload_name, "format": payload_format},
        "item_count": 1,
        "payload_bytes": len(payload),
        "clock": {
            "clock_id": f"{source_id}-clock",
            "tick_hz": 1000,
            "wrap_ticks": 0,
            "timestamp_semantics": "frame_start" if kind == "radar" else "exposure_midpoint",
        },
        "clock_observations": [
            {
                "observation_id": f"{source_id}-obs",
                "tick": tick,
                "wrap_count": 0,
                "host_before_ns": tick * 1_000_000 - uncertainty_ns,
                "host_after_ns": tick * 1_000_000 + uncertainty_ns,
            }
        ],
        "affine_segments": [
            {
                "start_unwrapped_tick": 100,
                "end_unwrapped_tick": 1000,
                "source_origin_tick": 100,
                "host_origin_ns": 100_000_000,
                "scale_num": 1_000_000,
                "scale_den": 1,
                "observation_ids": [f"{source_id}-obs"],
                "uncertainty_ns": uncertainty_ns,
            }
        ],
        "sync_event_cardinality": {"required": True, "min_items": 1, "max_items": 1},
        "artifacts": [
            {
                "role": "payload",
                "path": payload_name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            {
                "role": "index",
                "path": "index.bin",
                "size_bytes": len(index),
                "sha256": hashlib.sha256(index).hexdigest(),
            },
        ],
        "application_metadata": {},
    }


def _index(payload: bytes, *, ticks: int, duration: int) -> bytes:
    return _HEADER.pack(b"MMWSIDX1", 1, 32, 64, 0, 1, len(payload)) + _ENTRY.pack(
        0,
        0,
        len(payload),
        ticks,
        0,
        duration,
        _EVENT_ID,
        0,
        0,
    )


def _write_session(root: Path, record: dict[str, Any]) -> None:
    (root / "session.json").write_text(
        json.dumps(record, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_go_golden(tmp_path: Path) -> Path:
    golden_path = Path(__file__).with_name("multisensor_two_source_golden.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert isinstance(golden, dict)
    session_json = golden["session_json"]
    sources = golden["sources"]
    assert isinstance(session_json, str) and isinstance(sources, list)
    session = json.loads(session_json)
    root = tmp_path / "go-golden"
    for source_record, source_payload in zip(session["sources"], sources, strict=True):
        source_root = root / "sensors" / source_record["source_id"]
        source_root.mkdir(parents=True)
        (source_root / source_record["payload"]["filename"]).write_bytes(
            bytes.fromhex(source_payload["payload_hex"])
        )
        (source_root / "index.bin").write_bytes(bytes.fromhex(source_payload["index_hex"]))
    (root / "session.json").write_text(session_json, encoding="utf-8")
    return root


def _source(record: dict[str, Any], source_id: str) -> dict[str, Any]:
    return next(source for source in record["sources"] if source["source_id"] == source_id)


def _artifact(source: dict[str, Any], role: str) -> dict[str, Any]:
    return next(artifact for artifact in source["artifacts"] if artifact["role"] == role)
