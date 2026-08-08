from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from mmwcore import open_multisensor_stream as public_open_multisensor_stream
from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCDecodeRecipe, ADCFrameSpec, RangeDopplerRecipe
from mmwcore.io import (
    MMWCLI_MULTISENSOR_STREAM_SCHEMA_V1,
    MultisensorStreamAborted,
    MultisensorStreamError,
    MultisensorStreamStateError,
    ProvisionalMultisensorRangeDoppler,
    open_multisensor_stream,
)

_HEADER = struct.Struct("<8sHHHHQQQQ32s")
_PREFIX = struct.Struct("<8sHHHHQQQQ")
_DOMAIN = b"mmwcli.multisensor_stream.record.v1\x00"
_SESSION_ID = "123e4567-e89b-42d3-a456-426614174000"


def _golden() -> bytes:
    encoded = (Path(__file__).parent / "multisensor_two_source_stream.hex").read_text(
        encoding="ascii"
    )
    return bytes.fromhex(encoded)


def test_decodes_exact_go_golden_and_discards_failed_optional_source() -> None:
    source = io.BytesIO(_golden())
    stream = open_multisensor_stream(source)

    assert public_open_multisensor_stream is open_multisensor_stream
    assert MMWCLI_MULTISENSOR_STREAM_SCHEMA_V1 == "mmwcli.multisensor_stream.v1"
    assert stream.contract.session_id == _SESSION_ID
    assert stream.source("radar-0").kind == "radar"
    assert stream.source("radar-0").required is True
    assert stream.source("camera-0").timestamp_semantics == "exposure_midpoint"

    items = list(stream.items())

    assert [(item.source_id, item.item_index, item.payload) for item in items] == [
        ("radar-0", 0, b"R000"),
        ("camera-0", 0, b"JPEG-A"),
        ("radar-0", 1, b"R1"),
        ("camera-0", 1, b"JPEG-B"),
    ]
    assert items[0].sync_event_id == 7
    assert items[2].sync_event_id is None
    assert stream.radar_config is not None
    assert stream.radar_config.payload == b"sensorStop\n"

    commit = stream.require_commit()

    assert commit.session_id == _SESSION_ID
    assert commit.session_json_size_bytes == 42
    assert commit.source("radar-0").outcome == "complete"
    assert commit.source("camera-0").outcome == "failed"
    assert [item.payload for item in items if commit.accepts(item)] == [b"R000", b"R1"]
    assert not source.closed


def test_preserves_delivery_observed_camera_ticks_without_exposure_claim() -> None:
    stream = open_multisensor_stream(io.BytesIO(_delivery_observed_stream()))

    camera = stream.source("camera-0")
    assert camera.clock_id == "camera-0-delivery-observed"
    assert camera.tick_hz == 1_000_000_000
    assert camera.wrap_ticks == 0
    assert camera.timestamp_semantics == "delivery_observed"

    camera_items = [item for item in stream.items() if item.source_id == "camera-0"]
    assert [item.tick for item in camera_items] == [1_000_000_123, 1_000_000_456]
    assert all(item.timestamp_semantics == "delivery_observed" for item in camera_items)
    assert all(item.duration_ticks == 0 for item in camera_items)
    stream.require_commit()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clock_id", "camera-clock"),
        ("tick_hz", 1_000_000),
        ("wrap_ticks", 1 << 32),
    ],
)
def test_rejects_invalid_delivery_observed_stream_clock(field: str, value: object) -> None:
    records = _records(_delivery_observed_stream())
    camera = next(
        source for source in records[0][2]["sources"] if source["source_id"] == "camera-0"
    )
    camera["clock"][field] = value
    encoded = b"".join(_record(*record) for record in records)

    with pytest.raises(MultisensorStreamError) as failure:
        open_multisensor_stream(io.BytesIO(encoded))
    assert failure.value.__cause__ is not None
    assert "delivery_observed" in str(failure.value.__cause__)


def test_rejects_delivery_observed_stream_duration() -> None:
    records = _records(_delivery_observed_stream())
    item = next(
        metadata
        for kind, _sequence, metadata, _payload in records
        if kind == 3 and metadata.get("source_id") == "camera-0"
    )
    item["duration_ticks"] = 1
    stream = open_multisensor_stream(io.BytesIO(b"".join(_record(*record) for record in records)))

    with pytest.raises(MultisensorStreamError) as failure:
        list(stream.items())
    assert failure.value.__cause__ is not None
    assert "delivery_observed" in str(failure.value.__cause__)


def test_radar_item_supports_explicit_and_caller_bound_preset() -> None:
    stream = open_multisensor_stream(io.BytesIO(_golden()))
    iterator = stream.items()
    radar = next(iterator)
    camera = next(iterator)
    adc = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=1)
    recipe = RangeDopplerRecipe(decode=ADCDecodeRecipe(adc))

    transformed = radar.range_doppler(recipe)

    assert isinstance(transformed, ProvisionalMultisensorRangeDoppler)
    assert transformed.item is radar
    assert transformed.cube.metadata["provisional"] is True
    with pytest.raises(TypeError, match="Only radar"):
        camera.range_doppler(recipe)

    calls: list[tuple[object, object, object]] = []

    def preset(
        profile: RadarProfile, *, adc_layout: object, tx_order: object
    ) -> RangeDopplerRecipe:
        calls.append((profile, adc_layout, tx_order))
        return recipe

    with pytest.raises(TypeError, match="explicit RadarCaptureSpec"):
        radar.range_doppler(preset)

    profile = RadarProfile(
        num_adc_samples=1,
        num_chirps_per_tx=1,
        num_tx=1,
        num_rx=1,
    )
    capture = RadarCaptureSpec(profile=profile, adc=adc, tx_order=(0,))
    preset_product = radar.range_doppler(preset, radar_capture=capture)

    assert preset_product.cube.axes == transformed.cube.axes
    assert calls == [(profile, adc.layout, (0,))]


def test_requires_item_exhaustion_commit_explicit_eof_and_transport_eof() -> None:
    stream = open_multisensor_stream(io.BytesIO(_golden()))
    with pytest.raises(MultisensorStreamStateError):
        stream.require_commit()

    trailing_source = io.BytesIO(_golden() + b"x")
    trailing = open_multisensor_stream(trailing_source)
    assert len(list(trailing.items())) == 4
    with pytest.raises(MultisensorStreamError, match="trailing data"):
        trailing.require_commit()
    assert not trailing_source.closed

    corrupted = bytearray(_golden())
    corrupted[-1] ^= 1
    invalid = open_multisensor_stream(io.BytesIO(corrupted))
    assert len(list(invalid.items())) == 4
    with pytest.raises(MultisensorStreamError, match="SHA-256"):
        invalid.require_commit()


def test_rejects_end_lineage_and_required_failed_commit() -> None:
    records = _records(_golden())
    camera_end = records[6][2]
    camera_end["payload_sha256"] = "0" * 64
    broken_lineage = b"".join(_record(*record) for record in records)
    stream = open_multisensor_stream(io.BytesIO(broken_lineage))
    with pytest.raises(MultisensorStreamError):
        list(stream.items())

    records = _records(_golden())
    radar_end = records[7][2]
    radar_end["outcome"] = "failed"
    required_failed = b"".join(_record(*record) for record in records)
    stream = open_multisensor_stream(io.BytesIO(required_failed))
    assert len(list(stream.items())) == 4
    with pytest.raises(MultisensorStreamError) as failure:
        stream.require_commit()
    assert isinstance(failure.value.__cause__, ValueError)
    assert "required" in str(failure.value.__cause__)


def test_abort_is_not_a_success_and_does_not_close_source() -> None:
    session = _records(_golden())[0]
    abort = {
        "schema": "mmwcli.multisensor_stream_terminal.v1",
        "session_id": _SESSION_ID,
        "outcome": "abort",
        "reason_code": "source_failed",
    }
    eof = {"schema": "mmwcli.multisensor_stream_eof.v1", "session_id": _SESSION_ID}
    encoded = b"".join(
        (
            _record(*session),
            _record(6, 1, abort, b""),
            _record(7, 2, eof, b""),
        )
    )
    source = io.BytesIO(encoded)
    stream = open_multisensor_stream(source)

    with pytest.raises(MultisensorStreamAborted) as failure:
        list(stream.items())

    assert failure.value.abort.reason_code == "source_failed"
    with pytest.raises(MultisensorStreamStateError):
        stream.require_commit()
    assert not source.closed


def _records(stream: bytes) -> list[tuple[int, int, dict[str, Any], bytes]]:
    records: list[tuple[int, int, dict[str, Any], bytes]] = []
    offset = 0
    while offset < len(stream):
        header = _HEADER.unpack(stream[offset : offset + _HEADER.size])
        kind = header[3]
        sequence = header[5]
        metadata_size = header[6]
        payload_size = header[7]
        metadata_start = offset + _HEADER.size
        payload_start = metadata_start + metadata_size
        end = payload_start + payload_size
        metadata = json.loads(stream[metadata_start:payload_start])
        assert isinstance(metadata, dict)
        records.append((kind, sequence, metadata, stream[payload_start:end]))
        offset = end
    return records


def _record(kind: int, sequence: int, metadata: dict[str, Any], payload: bytes) -> bytes:
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    prefix = _PREFIX.pack(
        b"MMWMSTR1",
        1,
        80,
        kind,
        0,
        sequence,
        len(encoded),
        len(payload),
        0,
    )
    digest = hashlib.sha256(_DOMAIN + prefix + encoded + payload).digest()
    return prefix + digest + encoded + payload


def _delivery_observed_stream() -> bytes:
    records = _records(_golden())
    camera = next(
        source for source in records[0][2]["sources"] if source["source_id"] == "camera-0"
    )
    camera["clock"] = {
        "clock_id": "camera-0-delivery-observed",
        "tick_hz": 1_000_000_000,
        "wrap_ticks": 0,
        "timestamp_semantics": "delivery_observed",
    }
    ticks = iter((1_000_000_123, 1_000_000_456))
    for kind, _sequence, metadata, _payload in records:
        if kind == 3 and metadata.get("source_id") == "camera-0":
            metadata["tick"] = next(ticks)
            metadata["wrap_count"] = 0
            metadata["duration_ticks"] = 0
    return b"".join(_record(*record) for record in records)
