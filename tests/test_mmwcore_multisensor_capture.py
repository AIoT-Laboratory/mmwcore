from __future__ import annotations

import hashlib
import json
import shutil
import struct
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from mmwcore import open_multisensor_capture as public_open_multisensor_capture
from mmwcore.core import ADCDecodeRecipe, RangeDopplerRecipe
from mmwcore.io import (
    MMWCLI_MULTISENSOR_SESSION_SCHEMA_V1,
    MMWCLI_SENSOR_INDEX_SCHEMA_V1,
    ADCFileCapture,
    MappedTimeInterval,
    MultisensorCapture,
    causal_match,
    open_multisensor_capture,
)

_SESSION_ID = "12345678-1234-4abc-8def-1234567890ab"
_HEADER = struct.Struct("<8sHHHHQQ")
_ENTRY = struct.Struct("<QQQQQQQII")
_EVENT_ID = 7
_NESTED_RADAR_CONFIG = b"""\
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
_NESTED_RADAR_ADC = struct.pack("<16h", *range(16))


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


def test_radar_source_opens_nested_capture_with_explicit_recipe(tmp_path: Path) -> None:
    root, _record = _write_nested_radar_fixture(tmp_path)
    source = open_multisensor_capture(root).source("radar-0")

    nested = source.open_radar_capture()

    assert isinstance(nested, ADCFileCapture)
    assert source.payload_path is not None
    assert nested.root == source.payload_path.parent
    assert nested.adc_path == source.payload_path
    assert nested.num_frames == source.item_count == 2
    assert nested.frame(1).samples.tolist() == list(range(8, 16))
    with pytest.raises(TypeError, match="explicit RangeDopplerRecipe"):
        nested.range_doppler()

    recipe = RangeDopplerRecipe(decode=ADCDecodeRecipe(nested.radar_capture.adc))
    bound = source.open_radar_capture(range_doppler=recipe)
    assert bound.range_doppler(frame_index=0).frame_id == 0


def test_nested_radar_capture_is_parsed_only_when_requested(tmp_path: Path) -> None:
    root, record = _write_nested_radar_fixture(tmp_path, name="lazy-nested")
    manifest_path = root / "sensors" / "radar-0" / "capture.json"
    manifest_path.write_bytes(b"{}")
    manifest_artifact = _artifact(_source(record, "radar-0"), "manifest")
    manifest_artifact["size_bytes"] = 2
    manifest_artifact["sha256"] = hashlib.sha256(b"{}").hexdigest()
    _write_session(root, record)

    source = open_multisensor_capture(root).source("radar-0")

    with pytest.raises(ValueError, match="unsupported schema"):
        source.open_radar_capture()


@pytest.mark.parametrize(
    ("mismatch", "error"),
    [
        ("frame count", "frame count"),
        ("frame geometry", "offset/size"),
    ],
)
def test_nested_radar_capture_rejects_source_index_mismatch(
    tmp_path: Path,
    mismatch: str,
    error: str,
) -> None:
    root, record = _write_nested_radar_fixture(tmp_path, name=mismatch.replace(" ", "-"))
    if mismatch == "frame count":
        entries = [(0, len(_NESTED_RADAR_ADC), 100)]
    else:
        entries = [(0, 8, 100), (8, len(_NESTED_RADAR_ADC) - 8, 110)]
    _replace_radar_index(root, record, entries)

    source = open_multisensor_capture(root).source("radar-0")

    with pytest.raises(ValueError, match=error):
        source.open_radar_capture()


@pytest.mark.parametrize("outcome", ["failed", "omitted"])
def test_nested_radar_capture_rejects_noncomplete_and_camera_sources(
    tmp_path: Path,
    outcome: str,
) -> None:
    root, record = _write_fixture(tmp_path, name=outcome)
    radar = _source(record, "radar-0")
    radar.update(
        required=False,
        outcome=outcome,
        item_count=0,
        payload_bytes=0,
        artifacts=[],
    )
    radar.pop("sync_event_cardinality")
    shutil.rmtree(root / "sensors" / "radar-0")
    totals = record["totals"]
    totals.update(
        required_source_count=1,
        complete_source_count=1,
        item_count=1,
        payload_bytes=8,
    )
    _write_session(root, record)
    capture = open_multisensor_capture(root)

    with pytest.raises(ValueError, match="complete radar"):
        capture.source("radar-0").open_radar_capture()
    with pytest.raises(ValueError, match="radar multisensor"):
        capture.source("camera-0").open_radar_capture()


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


def _write_nested_radar_fixture(
    tmp_path: Path,
    *,
    name: str = "nested-radar",
) -> tuple[Path, dict[str, Any]]:
    root, record = _write_fixture(tmp_path, name=name)
    radar_root = root / "sensors" / "radar-0"
    radar_index = _sensor_index(
        len(_NESTED_RADAR_ADC),
        [(0, 16, 100), (16, 16, 110)],
    )
    capture_manifest = {
        "schema": "mmwcli.capture_session.v1",
        "hardware": {
            "vendor": "ti",
            "family": "xwr68xx",
            "model": "",
            "revision": "",
            "identity_source": "route_declaration",
        },
        "adc": {
            "path": "adc.bin",
            "dtype": "int16",
            "byte_order": "little",
            "lane_count": 2,
            "layout": "group2_i_then_q",
            "size_bytes": len(_NESTED_RADAR_ADC),
            "sha256": hashlib.sha256(_NESTED_RADAR_ADC).hexdigest(),
        },
        "radar_config": {
            "path": "radar.cfg",
            "format": "ti_mmwave_legacy_cli.v1",
            "sha256": hashlib.sha256(_NESTED_RADAR_CONFIG).hexdigest(),
        },
    }
    capture_manifest_bytes = json.dumps(
        capture_manifest,
        separators=(",", ":"),
    ).encode()
    files = {
        "adc.bin": _NESTED_RADAR_ADC,
        "index.bin": radar_index,
        "radar.cfg": _NESTED_RADAR_CONFIG,
        "capture.json": capture_manifest_bytes,
    }
    for filename, payload in files.items():
        (radar_root / filename).write_bytes(payload)

    radar = _source(record, "radar-0")
    radar["limits"] = {
        "max_items": 2,
        "max_item_bytes": 16,
        "max_payload_bytes": len(_NESTED_RADAR_ADC),
    }
    radar["item_count"] = 2
    radar["payload_bytes"] = len(_NESTED_RADAR_ADC)
    radar["sync_event_cardinality"] = {
        "required": True,
        "min_items": 1,
        "max_items": 2,
    }
    radar["artifacts"] = [
        {
            "role": role,
            "path": filename,
            "size_bytes": len(files[filename]),
            "sha256": hashlib.sha256(files[filename]).hexdigest(),
        }
        for role, filename in (
            ("payload", "adc.bin"),
            ("index", "index.bin"),
            ("configuration", "radar.cfg"),
            ("manifest", "capture.json"),
        )
    ]
    record["totals"].update(
        item_count=3,
        payload_bytes=len(_NESTED_RADAR_ADC) + 8,
    )
    _write_session(root, record)
    return root, record


def _replace_radar_index(
    root: Path,
    record: dict[str, Any],
    entries: list[tuple[int, int, int]],
) -> None:
    index = _sensor_index(len(_NESTED_RADAR_ADC), entries)
    (root / "sensors" / "radar-0" / "index.bin").write_bytes(index)
    radar = _source(record, "radar-0")
    radar["item_count"] = len(entries)
    radar["limits"]["max_items"] = max(1, len(entries))
    radar["limits"]["max_item_bytes"] = max(size for _, size, _ in entries)
    radar["sync_event_cardinality"]["max_items"] = max(1, len(entries))
    index_artifact = _artifact(radar, "index")
    index_artifact["size_bytes"] = len(index)
    index_artifact["sha256"] = hashlib.sha256(index).hexdigest()
    record["totals"]["item_count"] = len(entries) + 1
    _write_session(root, record)


def _sensor_index(payload_bytes: int, entries: list[tuple[int, int, int]]) -> bytes:
    header = _HEADER.pack(b"MMWSIDX1", 1, 32, 64, 0, len(entries), payload_bytes)
    encoded = [header]
    for item_index, (offset, size, tick) in enumerate(entries):
        encoded.append(
            _ENTRY.pack(
                item_index,
                offset,
                size,
                tick,
                0,
                10,
                _EVENT_ID,
                0,
                0,
            )
        )
    return b"".join(encoded)


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
