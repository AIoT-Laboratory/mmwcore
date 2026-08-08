from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mmwcore import open_capture as public_open_capture
from mmwcore.config import RadarProfile, iwr6843_isk_range_doppler_recipe
from mmwcore.core import (
    ADCComplexLayout,
    ADCDecodeRecipe,
    AntennaArrayGeometry,
    RangeDopplerRecipe,
    TDMVirtualArraySpec,
)
from mmwcore.dsp import process_adc_to_range_doppler
from mmwcore.io import (
    MMWCLI_CAPTURE_SESSION_SCHEMA_V1,
    ADCFileCapture,
    ADCFileFrameReader,
    MmwcliRawCaptureContract,
    RangeDopplerPreset,
    open_capture,
)

_CONFIG_TEXT = """\
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
_CONFIG_BYTES = _CONFIG_TEXT.encode()
_ADC_BYTES = struct.pack("<16h", *range(16))
_TDM_CONFIG_BYTES = b"""\
flushCfg
dfeDataOutputMode 1
channelCfg 15 5 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60 7 3 24 0 0 166 1 4 12500 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
frameCfg 0 1 1 2 10 1 0
lvdsStreamCfg -1 0 1 0
"""
_TDM_ADC_BYTES = struct.pack("<128h", *range(128))


def test_open_capture_validates_and_opens_mmwcli_session(tmp_path: Path) -> None:
    assert MMWCLI_CAPTURE_SESSION_SCHEMA_V1 == "mmwcli.capture_session.v1"
    record = _manifest_record()
    record["producer"] = {"name": "fixture"}
    _object(record, "hardware")["optional_note"] = "additive v1 metadata"
    _object(record, "adc")["optional_note"] = "additive v1 metadata"
    _object(record, "radar_config")["optional_note"] = "additive v1 metadata"
    root = _write_capture(tmp_path, record=record)
    (root / "notes.txt").write_text("ignored extra file", encoding="utf-8")

    opened = open_capture(root)

    assert isinstance(opened, ADCFileCapture)
    assert opened.root == root.absolute()
    assert opened.manifest_path == opened.root / "capture.json"
    assert opened.adc_path == opened.root / "adc.bin"
    assert opened.radar_config_path == opened.root / "radar.cfg"
    assert isinstance(opened.raw_capture, MmwcliRawCaptureContract)
    assert opened.raw_capture == MmwcliRawCaptureContract(
        vendor="ti",
        family="xwr68xx",
        model="",
        revision="",
        identity_source="route_declaration",
        config_format="ti_mmwave_legacy_cli.v1",
        dtype="int16",
        byte_order="little",
        lane_count=2,
        layout="group2_i_then_q",
    )
    with pytest.raises(FrozenInstanceError):
        opened.raw_capture.family = "xwr18xx"  # type: ignore[misc]
    assert opened.radar_capture.expected_size_bytes == len(_ADC_BYTES)
    assert opened.radar_capture.tx_order == (0,)
    assert opened.radar_capture.adc.layout is ADCComplexLayout.GROUP2_I_THEN_Q
    assert opened.reader.num_frames == 2
    first = opened.reader.read_frame(0)
    second = opened.reader.read_frame(1)
    np.testing.assert_array_equal(first.samples, np.arange(8, dtype=np.int16))
    np.testing.assert_array_equal(second.samples, np.arange(8, 16, dtype=np.int16))
    assert first.timestamp == pytest.approx(0.0)
    assert second.timestamp == pytest.approx(0.01)
    assert first.metadata["tx_order"] == [0]


def test_capture_facade_reads_and_iterates_frames_lazily(tmp_path: Path) -> None:
    capture = open_capture(_write_capture(tmp_path))

    assert capture.num_frames == 2
    second = capture.frame(1)
    assert isinstance(second.samples, np.memmap)
    np.testing.assert_array_equal(second.samples, np.arange(8, 16, dtype=np.int16))
    assert second.timestamp == pytest.approx(0.01)
    assert second.source == str(capture.adc_path)

    frames = capture.frames(start=1, stop=2)
    assert iter(frames) is frames
    assert [frame.frame_id for frame in frames] == [1]
    assert list(capture.frames(start=2, stop=2)) == []


@pytest.mark.parametrize(
    ("start", "stop", "error"),
    [
        (True, None, TypeError),
        ("0", None, TypeError),
        (0, False, TypeError),
        (-1, None, ValueError),
        (3, None, ValueError),
        (1, 0, ValueError),
        (0, 3, ValueError),
    ],
)
def test_capture_facade_rejects_invalid_frame_intervals(
    tmp_path: Path,
    start: object,
    stop: object,
    error: type[Exception],
) -> None:
    capture = open_capture(_write_capture(tmp_path))

    with pytest.raises(error):
        capture.frames(start=start, stop=stop)  # type: ignore[arg-type]


def test_capture_facade_rejects_non_integer_frame_index(tmp_path: Path) -> None:
    capture = open_capture(_write_capture(tmp_path))

    with pytest.raises(TypeError, match="index"):
        capture.frame(True)


def test_capture_facade_range_doppler_matches_explicit_runner(tmp_path: Path) -> None:
    capture = open_capture(_write_capture(tmp_path))
    recipe = RangeDopplerRecipe(decode=ADCDecodeRecipe(capture.radar_capture.adc))

    actual = capture.range_doppler(recipe, frame_index=1)
    expected = process_adc_to_range_doppler(capture.frame(1), recipe)

    assert actual.axes == expected.axes
    assert actual.frame_id == 1
    assert actual.timestamp == pytest.approx(0.01)
    np.testing.assert_allclose(actual.data, expected.data)


def test_open_capture_binds_explicit_recipe_as_default(tmp_path: Path) -> None:
    root = _write_capture(tmp_path)
    contract = open_capture(root).radar_capture
    recipe = RangeDopplerRecipe(decode=ADCDecodeRecipe(contract.adc))

    capture = public_open_capture(root, range_doppler=recipe)
    actual = capture.range_doppler(frame_index=1)
    expected = process_adc_to_range_doppler(capture.frame(1), recipe)

    assert public_open_capture is open_capture
    np.testing.assert_allclose(actual.data, expected.data)


def test_open_capture_calls_preset_once_with_validated_contract(tmp_path: Path) -> None:
    root = _write_capture(
        tmp_path,
        config=_TDM_CONFIG_BYTES,
        adc=_TDM_ADC_BYTES,
    )
    calls: list[tuple[RadarProfile, ADCComplexLayout, tuple[int, ...]]] = []
    recipes: list[RangeDopplerRecipe] = []

    def preset(
        profile: RadarProfile,
        *,
        adc_layout: ADCComplexLayout,
        tx_order: tuple[int, ...],
    ) -> RangeDopplerRecipe:
        calls.append((profile, adc_layout, tx_order))
        recipe = iwr6843_isk_range_doppler_recipe(
            profile,
            adc_layout=adc_layout,
            tx_order=tx_order,
        )
        recipes.append(recipe)
        return recipe

    typed_preset: RangeDopplerPreset = preset
    capture = open_capture(root, range_doppler=typed_preset)

    assert calls == [
        (
            capture.radar_capture.profile,
            capture.radar_capture.adc.layout,
            capture.radar_capture.tx_order,
        )
    ]
    actual = capture.range_doppler(frame_index=1)
    expected = process_adc_to_range_doppler(capture.frame(1), recipes[0])
    np.testing.assert_allclose(actual.data, expected.data)


def test_capture_facade_rejects_invalid_range_doppler_policy(tmp_path: Path) -> None:
    capture = open_capture(_write_capture(tmp_path))
    recipe = RangeDopplerRecipe(decode=ADCDecodeRecipe(capture.radar_capture.adc))

    with pytest.raises(TypeError, match="explicit RangeDopplerRecipe"):
        capture.range_doppler()

    with pytest.raises(TypeError, match="RangeDopplerRecipe"):
        capture.range_doppler(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="preset callable"):
        open_capture(capture.root, range_doppler=object())  # type: ignore[arg-type]

    loop_recipe = replace(
        recipe,
        doppler_fft=replace(recipe.doppler_fft, input_axis="loop"),
    )
    with pytest.raises(ValueError, match="chirp Doppler axis"):
        capture.range_doppler(loop_recipe)


def test_capture_facade_rejects_mismatched_reader_and_recipe_specs(tmp_path: Path) -> None:
    capture = open_capture(_write_capture(tmp_path))
    alternate_adc = replace(
        capture.radar_capture.adc,
        layout=ADCComplexLayout.IQ_INTERLEAVED,
    )
    alternate_reader = ADCFileFrameReader.from_capture(
        capture.adc_path,
        replace(capture.radar_capture, adc=alternate_adc),
    )
    recipe = RangeDopplerRecipe(decode=ADCDecodeRecipe(capture.radar_capture.adc))

    with pytest.raises(ValueError, match="reader ADC spec"):
        replace(capture, reader=alternate_reader)

    wrong_recipe = replace(recipe, decode=ADCDecodeRecipe(alternate_adc))
    with pytest.raises(ValueError, match="recipe ADC spec"):
        capture.range_doppler(wrong_recipe)
    with pytest.raises(ValueError, match="recipe ADC spec"):
        open_capture(capture.root, range_doppler=wrong_recipe)

    alternate_path = tmp_path / "alternate-adc.bin"
    alternate_path.write_bytes(_ADC_BYTES)
    alternate_path_reader = ADCFileFrameReader.from_capture(
        alternate_path,
        capture.radar_capture,
    )
    with pytest.raises(ValueError, match="reader path"):
        replace(capture, reader=alternate_path_reader)


def test_capture_facade_processes_explicit_two_tx_isk_recipe(tmp_path: Path) -> None:
    root = _write_capture(
        tmp_path,
        config=_TDM_CONFIG_BYTES,
        adc=_TDM_ADC_BYTES,
    )
    capture = open_capture(root)
    contract = capture.radar_capture
    recipe = iwr6843_isk_range_doppler_recipe(
        contract.profile,
        adc_layout=contract.adc.layout,
        tx_order=contract.tx_order,
    )

    actual = capture.range_doppler(recipe, frame_index=1)
    expected = process_adc_to_range_doppler(capture.frame(1), recipe)

    assert actual.axes == ("frame", "doppler_bin", "virtual_rx", "range_bin")
    assert actual.data.shape[2] == 8
    assert actual.metadata["tdm_doppler_compensation"]["tx_order"] == [0, 2]
    np.testing.assert_allclose(actual.data, expected.data)


def test_capture_facade_requires_matching_explicit_tdm_geometry(tmp_path: Path) -> None:
    root = _write_capture(
        tmp_path,
        config=_TDM_CONFIG_BYTES,
        adc=_TDM_ADC_BYTES,
    )
    capture = open_capture(root)
    contract = capture.radar_capture
    no_tdm = RangeDopplerRecipe(decode=ADCDecodeRecipe(contract.adc))

    with pytest.raises(ValueError, match="explicit TDM"):
        capture.range_doppler(no_tdm)

    wrong_order = iwr6843_isk_range_doppler_recipe(
        contract.profile,
        adc_layout=contract.adc.layout,
        tx_order=(2, 0),
    )
    with pytest.raises(ValueError, match="Tx order"):
        capture.range_doppler(wrong_order)

    wrong_rx_geometry = AntennaArrayGeometry(
        tx_positions_wavelengths=((0.0, 0.0, 0.0),) * 3,
        rx_positions_wavelengths=((0.0, 0.0, 0.0),) * 3,
    )
    wrong_rx = replace(
        wrong_order,
        tdm_virtual_array=TDMVirtualArraySpec(wrong_rx_geometry, contract.tx_order),
    )
    with pytest.raises(ValueError, match="receiver geometry"):
        capture.range_doppler(wrong_rx)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (None, "adc"),
        (None, "hardware"),
        (None, "radar_config"),
        ("hardware", "vendor"),
        ("hardware", "family"),
        ("hardware", "model"),
        ("hardware", "revision"),
        ("hardware", "identity_source"),
        ("adc", "path"),
        ("adc", "lane_count"),
        ("adc", "size_bytes"),
        ("adc", "sha256"),
        ("radar_config", "path"),
        ("radar_config", "sha256"),
    ],
)
def test_open_capture_requires_manifest_fields(
    tmp_path: Path,
    section: str | None,
    field: str,
) -> None:
    record = _manifest_record()
    target = record if section is None else _object(record, section)
    target.pop(field)
    root = _write_capture(tmp_path, record=record)

    with pytest.raises(ValueError):
        open_capture(root)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "schema", "mmwcli.capture_session.v2"),
        ("hardware", "vendor", "TI"),
        ("hardware", "vendor", 1),
        ("hardware", "family", "xwr18xx"),
        ("hardware", "model", "iwr6843"),
        ("hardware", "revision", "es2"),
        ("hardware", "identity_source", "observed_device"),
        ("adc", "path", "../adc.bin"),
        ("adc", "path", "ADC.BIN"),
        ("adc", "dtype", "uint16"),
        ("adc", "byte_order", "big"),
        ("adc", "lane_count", True),
        ("adc", "lane_count", 2.0),
        ("adc", "lane_count", 4),
        ("adc", "layout", "iq_interleaved"),
        ("adc", "size_bytes", True),
        ("adc", "size_bytes", 32.0),
        ("adc", "size_bytes", 0),
        ("adc", "size_bytes", 3),
        ("adc", "size_bytes", 1 << 63),
        ("adc", "sha256", "A" * 64),
        ("adc", "sha256", "g" * 64),
        ("radar_config", "path", "/radar.cfg"),
        ("radar_config", "format", "ti_xwr68xx_legacy_cli"),
        ("radar_config", "format", "unknown"),
        ("radar_config", "sha256", "0" * 63),
    ],
)
def test_open_capture_rejects_invalid_required_manifest_values(
    tmp_path: Path,
    section: str | None,
    field: str,
    value: object,
) -> None:
    record = _manifest_record()
    target = record if section is None else _object(record, section)
    target[field] = value
    root = _write_capture(tmp_path, record=record)

    with pytest.raises(ValueError):
        open_capture(root)


@pytest.mark.parametrize(
    "payload",
    [
        b"[]",
        b"{",
        b'[{"invalid": "top-level"}]',
        b'{"schema": NaN}',
        (
            b'{"schema":"mmwcli.capture_session.v1",'
            b'"schema":"mmwcli.capture_session.v1","adc":{},"radar_config":{}}'
        ),
        b"\xff",
    ],
)
def test_open_capture_rejects_malformed_manifest(tmp_path: Path, payload: bytes) -> None:
    root = _write_capture(tmp_path)
    (root / "capture.json").write_bytes(payload)

    with pytest.raises(ValueError):
        open_capture(root)


def test_open_capture_rejects_oversize_manifest(tmp_path: Path) -> None:
    root = _write_capture(tmp_path)
    (root / "capture.json").write_bytes(b" " * ((64 << 10) + 1))

    with pytest.raises(ValueError, match="exceeds"):
        open_capture(root)


@pytest.mark.parametrize("leaf", ["capture.json", "adc.bin", "radar.cfg"])
def test_open_capture_requires_fixed_regular_files(tmp_path: Path, leaf: str) -> None:
    root = _write_capture(tmp_path)
    path = root / leaf
    path.unlink()
    path.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        open_capture(root)


def test_open_capture_rejects_leaf_symlink(tmp_path: Path) -> None:
    root = _write_capture(tmp_path)
    target = root / "payload"
    (root / "adc.bin").rename(target)
    try:
        (root / "adc.bin").symlink_to(target.name)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="regular file"):
        open_capture(root)


@pytest.mark.parametrize(
    "failure",
    [
        "manifest size",
        "ADC size",
        "ADC digest",
        "config digest",
        "config UTF-8",
        "config semantics",
        "infinite frames",
        "CFG-derived size",
        "oversize config",
    ],
)
def test_open_capture_rejects_integrity_and_contract_failures(
    tmp_path: Path,
    failure: str,
) -> None:
    config = _CONFIG_BYTES
    adc = _ADC_BYTES
    record = _manifest_record(config=config, adc=adc)
    if failure == "manifest size":
        _object(record, "adc")["size_bytes"] = len(adc) + 2
    elif failure == "ADC size":
        adc += b"\x00\x00"
    elif failure == "ADC digest":
        adc = bytes([adc[0] ^ 1]) + adc[1:]
    elif failure == "config digest":
        config += b"% changed\n"
    elif failure == "config UTF-8":
        config += b"% invalid \xff\n"
        _object(record, "radar_config")["sha256"] = _sha256(config)
    elif failure == "config semantics":
        config = b"sensorStart\n"
        _object(record, "radar_config")["sha256"] = _sha256(config)
    elif failure == "infinite frames":
        config = _CONFIG_BYTES.replace(b"frameCfg 0 0 1 2", b"frameCfg 0 0 1 0")
        _object(record, "radar_config")["sha256"] = _sha256(config)
    elif failure == "CFG-derived size":
        config = _CONFIG_BYTES.replace(b"frameCfg 0 0 1 2", b"frameCfg 0 0 1 3")
        _object(record, "radar_config")["sha256"] = _sha256(config)
    elif failure == "oversize config":
        config = b"%" + b"x" * (4 << 20)
        _object(record, "radar_config")["sha256"] = _sha256(config)
    root = _write_capture(tmp_path, config=config, adc=adc, record=record)

    with pytest.raises(ValueError):
        open_capture(root)


def test_open_capture_does_not_fall_back_to_raw_files(tmp_path: Path) -> None:
    raw = tmp_path / "capture.bin"
    raw.write_bytes(_ADC_BYTES)

    with pytest.raises(ValueError, match="directory"):
        open_capture(raw)


def _write_capture(
    tmp_path: Path,
    *,
    config: bytes = _CONFIG_BYTES,
    adc: bytes = _ADC_BYTES,
    record: dict[str, Any] | None = None,
) -> Path:
    root = tmp_path / "capture-session"
    root.mkdir()
    (root / "radar.cfg").write_bytes(config)
    (root / "adc.bin").write_bytes(adc)
    payload = record if record is not None else _manifest_record(config=config, adc=adc)
    (root / "capture.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _manifest_record(
    *,
    config: bytes = _CONFIG_BYTES,
    adc: bytes = _ADC_BYTES,
) -> dict[str, Any]:
    return {
        "schema": MMWCLI_CAPTURE_SESSION_SCHEMA_V1,
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
            "size_bytes": len(adc),
            "sha256": _sha256(adc),
        },
        "radar_config": {
            "path": "radar.cfg",
            "format": "ti_mmwave_legacy_cli.v1",
            "sha256": _sha256(config),
        },
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record[field]
    if not isinstance(value, dict):
        raise AssertionError(f"fixture field {field!r} is not an object")
    return value
