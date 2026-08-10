from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from mmwcore import open_capture_stream as public_open_capture_stream
from mmwcore.config import RadarProfile
from mmwcore.core import ADCComplexLayout, ADCDecodeRecipe, RangeDopplerRecipe
from mmwcore.dsp import process_adc_to_range_doppler
from mmwcore.io import (
    MMWCLI_CAPTURE_STREAM_SCHEMA_V1,
    CaptureStream,
    CaptureStreamAbort,
    CaptureStreamAborted,
    CaptureStreamError,
    CaptureStreamReader,
    CaptureStreamStateError,
    MmwcliRawCaptureContract,
    ProvisionalRangeDopplerFrame,
    RangeDopplerPreset,
    open_capture_stream,
)

_HEADER = struct.Struct("<8sHHHHQQQQ32s")
_PREFIX = struct.Struct("<8sHHHHQQQQ")
_DOMAIN = b"mmwcli.capture_stream.record.v1\x00"
_STREAM_ID = "000102030405060708090a0b0c0d0e0f"
_CONFIG = b"""\
flushCfg
dfeDataOutputMode 1
channelCfg 1 1 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60 7 3 24 0 0 166 1 16 12500 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
frameCfg 0 0 1 1 10 1 0
lowPower 0 0
lvdsStreamCfg -1 0 1 0
sensorStart
"""
_FRAME = bytes(range(64))

# Copied from mmwcli internal/capturestream goldenV1Hex. This is a byte-for-byte
# cross-language fixture, not a Python-generated equivalent.
_GO_GOLDEN_V1_HEX = (
    "4d4d575354524d31010050000100000000000000000000000000000000000000bd020000000000000000000000000000"
    "0c472b63a5169b7cd8cd592237c781464bdbd38d2d13126aae57e8a2358dac3b7b22736368656d61223a226d6d77636c"
    "692e636170747572655f73747265616d2e7631222c2273747265616d5f6964223a223030303130323033303430353036"
    "303730383039306130623063306430653066222c2270726f6475636572223a7b226e616d65223a226d6d77636c69222c"
    "2276657273696f6e223a2274657374227d2c226d6f6465223a2273747564696f2d636c69222c22686172647761726522"
    "3a7b2276656e646f72223a227469222c2266616d696c79223a2278777236387878222c226d6f64656c223a22222c2272"
    "65766973696f6e223a22222c226964656e746974795f736f75726365223a22726f7574655f6465636c61726174696f6e"
    "227d2c2263617074757265223a7b226672616d655f636f756e74223a312c226672616d655f6279746573223a36342c22"
    "65787065637465645f6279746573223a36342c227265636f72645f73657175656e63655f6f726967696e223a302c2266"
    "72616d655f696e6465785f6f726967696e223a302c226164635f627974655f6f66667365745f6f726967696e223a307d"
    "2c22616463223a7b226474797065223a22696e743136222c22627974655f6f72646572223a226c6974746c65222c226c"
    "616e655f636f756e74223a322c226c61796f7574223a2267726f7570325f695f7468656e5f71227d2c2272616461725f"
    "636f6e666967223a7b22666f726d6174223a2274695f6d6d776176655f6c65676163795f636c692e7631222c2273697a"
    "655f6279746573223a3232352c22736861323536223a2262396565626638613466656534303239353936333863373232"
    "633964613539353734386665383938373537393664613034343466333262316339656535646562227d2c226172746966"
    "616374223a7b227265717569726564223a747275652c22736368656d61223a226d6d77636c692e636170747572655f73"
    "657373696f6e2e7631227d7d0a4d4d575354524d31010050000200000001000000000000000000000000000000e10000"
    "000000000000000000000000002ed2891e73145ea55d3fe68f550123e3a1801c02875291d3581df8196e9eebeb666c75"
    "73684366670a646665446174614f75747075744d6f646520310a6368616e6e656c4366672031203120300a6164634366"
    "67203220310a616463627566436667202d3120302031203120310a70726f66696c654366672030203630203720332032"
    "342030203020313636203120313620313235303020302030203135380a63686972704366672030203020302030203020"
    "30203020310a6672616d654366672030203020312031203130203120300a6c6f77506f776572203020300a6c76647353"
    "747265616d436667202d312030203120300a73656e736f7253746172740a4d4d575354524d3101005000030000000200"
    "00000000000000000000000000004000000000000000000000000000000014cae892709325bd9f8e84a1d832feaf9589"
    "b669fdb32a1375d368524b606d00000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f2021"
    "22232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f4d4d575354524d3101005000040000000300"
    "0000000000000100000000000000db0000000000000000000000000000001157156605f2c9d58b5c7e23256e67533376"
    "76f79ccc801728c06a1dc5486e1d7b22736368656d61223a226d6d77636c692e636170747572655f73747265616d5f74"
    "65726d696e616c2e7631222c2273747265616d5f6964223a223030303130323033303430353036303730383039306130"
    "623063306430653066222c226f7574636f6d65223a22636f6d6d6974222c226672616d6573223a312c226164635f6279"
    "746573223a36342c226164635f736861323536223a226664656162396163663337313033363262643236353863646339"
    "6132396538663963373537666366393831313630336138633434376364316439313531313038227d0a"
)
_GO_GOLDEN_V1 = bytes.fromhex(_GO_GOLDEN_V1_HEX)


def test_decodes_exact_go_golden_as_provisional_until_commit() -> None:
    source = io.BytesIO(_GO_GOLDEN_V1)
    reader = CaptureStreamReader(source)

    contract = reader.read_contract()

    assert MMWCLI_CAPTURE_STREAM_SCHEMA_V1 == "mmwcli.capture_stream.v1"
    assert contract.stream_id == _STREAM_ID
    assert contract.producer_name == "mmwcli"
    assert contract.producer_version == "test"
    assert contract.mode == "studio-cli"
    assert contract.frame_count == 1
    assert contract.frame_bytes == 64
    assert contract.expected_bytes == 64
    assert isinstance(contract.raw_capture, MmwcliRawCaptureContract)
    assert contract.raw_capture == MmwcliRawCaptureContract(
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
        contract.raw_capture.family = "xwr18xx"  # type: ignore[misc]
    assert contract.radar_config == _CONFIG
    assert contract.radar_capture.num_frames == 1
    assert contract.radar_capture.expected_size_bytes == 64
    assert contract.radar_capture.tx_order == (0,)

    provisional = list(reader.provisional_frames())

    assert len(provisional) == 1
    assert provisional[0].sequence == 2
    assert provisional[0].frame_index == 0
    assert provisional[0].adc_byte_offset == 0
    np.testing.assert_array_equal(
        provisional[0].frame.samples,
        np.frombuffer(_FRAME, dtype="<i2").astype(np.int16, copy=False),
    )
    assert provisional[0].frame.timestamp == pytest.approx(0.0)
    assert provisional[0].frame.metadata["provisional"] is True

    commit = reader.require_commit()

    assert commit.frames == 1
    assert commit.adc_bytes == 64
    assert commit.adc_sha256 == hashlib.sha256(_FRAME).hexdigest()
    assert not source.closed


@pytest.mark.parametrize(
    ("family", "expected_tx_order"),
    [
        ("xwr16xx", (1, 0)),
        ("xwr18xx", (2, 0, 1)),
    ],
)
def test_capture_stream_supports_versioned_77_ghz_family_descriptors(
    family: str,
    expected_tx_order: tuple[int, ...],
) -> None:
    source = io.BytesIO(_family_capture_stream(family, family))
    reader = CaptureStreamReader(source)

    contract = reader.read_contract()
    frames = list(reader.provisional_frames())
    commit = reader.require_commit()

    assert contract.raw_capture.family == family
    assert contract.raw_capture.model == ""
    assert contract.raw_capture.revision == ""
    assert contract.raw_capture.identity_source == "route_declaration"
    assert contract.radar_capture.profile.start_frequency_hz == pytest.approx(77e9)
    assert contract.radar_capture.tx_order == expected_tx_order
    assert len(frames) == 1
    assert commit.frames == 1
    assert commit.adc_bytes == len(frames[0].frame.samples) * 2
    assert not source.closed


@pytest.mark.parametrize(
    ("declared_family", "config_family"),
    [
        ("xwr16xx", "xwr18xx"),
        ("xwr16xx", "xwr68xx"),
        ("xwr18xx", "xwr68xx"),
        ("xwr68xx", "xwr18xx"),
    ],
)
def test_capture_stream_rejects_descriptor_family_config_mismatch(
    declared_family: str,
    config_family: str,
) -> None:
    source = io.BytesIO(_family_capture_stream(declared_family, config_family))

    with pytest.raises(CaptureStreamError):
        CaptureStreamReader(source).read_contract()


def test_capture_stream_rejects_infinite_cfg_even_with_finite_session_shape() -> None:
    config = _CONFIG.replace(b"frameCfg 0 0 1 1", b"frameCfg 0 0 1 0")
    session = _golden_session()
    radar_config = _object(session, "radar_config")
    radar_config["size_bytes"] = len(config)
    radar_config["sha256"] = hashlib.sha256(config).hexdigest()
    records = _golden_records()
    session_kind, session_sequence, session_item_index, _ = records[0]
    config_kind, config_sequence, config_item_index, _ = records[1]
    records[0] = (session_kind, session_sequence, session_item_index, _json_line(session))
    records[1] = (config_kind, config_sequence, config_item_index, config)
    stream = b"".join(_record(*record) for record in records)

    with pytest.raises(CaptureStreamError) as caught:
        CaptureStreamReader(io.BytesIO(stream)).read_contract()
    assert isinstance(caught.value.__cause__, ValueError)
    assert "requires a finite radar frame count" in str(caught.value.__cause__)


class _ChunkedSource(io.BytesIO):
    def read(self, size: int = -1, /) -> bytes:
        return super().read(min(size, 7) if size >= 0 else 7)


def test_accepts_partial_binary_reads_without_owning_the_source() -> None:
    source = _ChunkedSource(_GO_GOLDEN_V1)
    reader = CaptureStreamReader(source)

    reader.read_contract()
    frames = list(reader.provisional_frames())
    commit = reader.require_commit()

    assert len(frames) == 1
    assert commit.frames == 1
    assert not source.closed


def test_open_capture_stream_reads_contract_and_calls_preset_once() -> None:
    source = io.BytesIO(_GO_GOLDEN_V1)
    calls: list[tuple[RadarProfile, ADCComplexLayout, tuple[int, ...]]] = []
    recipes: list[RangeDopplerRecipe] = []

    def preset(
        profile: RadarProfile,
        *,
        adc_layout: ADCComplexLayout,
        tx_order: tuple[int, ...],
    ) -> RangeDopplerRecipe:
        calls.append((profile, adc_layout, tx_order))
        recipe = RangeDopplerRecipe(
            decode=ADCDecodeRecipe(profile.to_adc_frame_spec(layout=adc_layout))
        )
        recipes.append(recipe)
        return recipe

    typed_preset: RangeDopplerPreset = preset
    stream = public_open_capture_stream(source, range_doppler=typed_preset)

    leading_records = _golden_records()[:2]
    leading_size = sum(_HEADER.size + len(payload) for _, _, _, payload in leading_records)
    assert public_open_capture_stream is open_capture_stream
    assert isinstance(stream, CaptureStream)
    assert source.tell() == leading_size
    assert source.tell() < len(_GO_GOLDEN_V1)
    assert calls == [
        (
            stream.contract.radar_capture.profile,
            stream.contract.radar_capture.adc.layout,
            stream.contract.radar_capture.tx_order,
        )
    ]

    range_doppler = stream.range_doppler()
    with pytest.raises(CaptureStreamStateError):
        stream.frames()
    items = list(range_doppler)

    assert len(items) == 1
    assert isinstance(items[0], ProvisionalRangeDopplerFrame)
    assert items[0].adc_frame.frame_index == 0
    expected = process_adc_to_range_doppler(items[0].adc_frame.frame, recipes[0])
    assert items[0].cube.axes == expected.axes
    np.testing.assert_allclose(items[0].cube.data, expected.data)
    assert stream.require_commit().frames == 1
    assert not source.closed


def test_capture_stream_facade_has_one_frame_state_and_requires_terminal_eof() -> None:
    contract = CaptureStreamReader(io.BytesIO(_GO_GOLDEN_V1)).read_contract()
    recipe = RangeDopplerRecipe(decode=ADCDecodeRecipe(contract.radar_capture.adc))
    stream = open_capture_stream(io.BytesIO(_GO_GOLDEN_V1), range_doppler=recipe)

    frames = stream.frames()
    with pytest.raises(CaptureStreamStateError):
        stream.range_doppler()
    assert len(list(frames)) == 1
    assert stream.require_commit().frames == 1

    trailing = open_capture_stream(io.BytesIO(_GO_GOLDEN_V1 + b"x"))
    assert len(list(trailing.frames())) == 1
    with pytest.raises(CaptureStreamError, match="trailing data"):
        trailing.require_commit()


def test_capture_stream_facade_requires_recipe_without_claiming_frames() -> None:
    stream = open_capture_stream(io.BytesIO(_GO_GOLDEN_V1))

    with pytest.raises(TypeError, match="bound recipe"):
        stream.range_doppler()

    assert len(list(stream.frames())) == 1
    assert stream.require_commit().frames == 1


def test_capture_stream_facade_never_commits_abort_or_missing_terminal() -> None:
    session = _golden_records()[0][3]
    abort = _json_line(
        {
            "schema": "mmwcli.capture_stream_terminal.v1",
            "stream_id": _STREAM_ID,
            "outcome": "abort",
            "frames": 0,
            "adc_bytes": 0,
            "adc_sha256": hashlib.sha256(b"").hexdigest(),
            "reason_code": "capture_failed",
        }
    )
    aborted_source = io.BytesIO(
        b"".join(
            (
                _record(1, 0, 0, session),
                _record(2, 1, 0, _CONFIG),
                _record(5, 2, 0, abort),
            )
        )
    )
    aborted = open_capture_stream(aborted_source)

    with pytest.raises(CaptureStreamAborted):
        list(aborted.frames())
    with pytest.raises(CaptureStreamStateError):
        aborted.require_commit()
    assert not aborted_source.closed

    records = _golden_records()[:3]
    missing_terminal_source = io.BytesIO(b"".join(_record(*record) for record in records))
    missing_terminal = open_capture_stream(missing_terminal_source)

    provisional = list(missing_terminal.frames())
    assert len(provisional) == 1
    with pytest.raises(CaptureStreamError, match="ended during record header"):
        missing_terminal.require_commit()
    assert not missing_terminal_source.closed


def test_open_capture_stream_rejects_recipe_mismatch_before_frame_read() -> None:
    source = io.BytesIO(_GO_GOLDEN_V1)
    contract = CaptureStreamReader(io.BytesIO(_GO_GOLDEN_V1)).read_contract()
    wrong_adc = replace(contract.radar_capture.adc, num_samples=8)
    wrong_recipe = RangeDopplerRecipe(decode=ADCDecodeRecipe(wrong_adc))

    with pytest.raises(ValueError, match="recipe ADC spec"):
        open_capture_stream(source, range_doppler=wrong_recipe)

    leading_records = _golden_records()[:2]
    leading_size = sum(_HEADER.size + len(payload) for _, _, _, payload in leading_records)
    assert source.tell() == leading_size
    assert not source.closed


def test_requires_natural_frame_iterator_exhaustion_before_commit() -> None:
    reader = CaptureStreamReader(io.BytesIO(_GO_GOLDEN_V1))
    reader.read_contract()
    frames = reader.provisional_frames()

    with pytest.raises(CaptureStreamStateError):
        reader.require_commit()
    next(frames)
    with pytest.raises(CaptureStreamStateError):
        reader.require_commit()
    with pytest.raises(StopIteration):
        next(frames)

    assert reader.require_commit().frames == 1


def test_accepts_valid_early_abort_and_preserves_provisional_outcome() -> None:
    session = _golden_records()[0][3]
    terminal = _json_line(
        {
            "schema": "mmwcli.capture_stream_terminal.v1",
            "stream_id": _STREAM_ID,
            "outcome": "abort",
            "frames": 0,
            "adc_bytes": 0,
            "adc_sha256": hashlib.sha256(b"").hexdigest(),
            "reason_code": "cancelled",
        }
    )
    stream = b"".join(
        (
            _record(1, 0, 0, session),
            _record(2, 1, 0, _CONFIG),
            _record(5, 2, 0, terminal),
        )
    )
    reader = CaptureStreamReader(io.BytesIO(stream))
    reader.read_contract()

    with pytest.raises(CaptureStreamAborted) as caught:
        next(reader.provisional_frames())

    assert isinstance(caught.value.abort, CaptureStreamAbort)
    assert caught.value.abort.frames == 0
    assert caught.value.abort.adc_bytes == 0
    assert caught.value.abort.reason_code == "cancelled"
    with pytest.raises(CaptureStreamStateError):
        reader.require_commit()


def test_session_json_is_closed_typed_and_bound_to_cfg() -> None:
    variants: list[dict[str, Any]] = []

    unknown = _golden_session()
    unknown["extension"] = True
    variants.append(unknown)

    missing = _golden_session()
    missing.pop("artifact")
    variants.append(missing)

    missing_hardware = _golden_session()
    missing_hardware.pop("hardware")
    variants.append(missing_hardware)

    missing_lane_count = _golden_session()
    _object(missing_lane_count, "adc").pop("lane_count")
    variants.append(missing_lane_count)

    unknown_hardware_key = _golden_session()
    _object(unknown_hardware_key, "hardware")["board"] = "IWR6843ISK"
    variants.append(unknown_hardware_key)

    boolean_lane_count = _golden_session()
    _object(boolean_lane_count, "adc")["lane_count"] = True
    variants.append(boolean_lane_count)

    numeric_lane_count = _golden_session()
    _object(numeric_lane_count, "adc")["lane_count"] = 2.0
    variants.append(numeric_lane_count)

    unknown_family = _golden_session()
    _object(unknown_family, "hardware")["family"] = "xwr14xx"
    variants.append(unknown_family)

    invented_model = _golden_session()
    _object(invented_model, "hardware")["model"] = "iwr6843"
    variants.append(invented_model)

    wrong_identity_source = _golden_session()
    _object(wrong_identity_source, "hardware")["identity_source"] = "observed_device"
    variants.append(wrong_identity_source)

    old_config_format = _golden_session()
    _object(old_config_format, "radar_config")["format"] = "ti_xwr68xx_legacy_cli"
    variants.append(old_config_format)

    boolean = _golden_session()
    _object(boolean, "capture")["frame_count"] = True
    variants.append(boolean)

    wrong_config_digest = _golden_session()
    _object(wrong_config_digest, "radar_config")["sha256"] = "0" * 64
    variants.append(wrong_config_digest)

    wrong_cfg_binding = _golden_session()
    capture = _object(wrong_cfg_binding, "capture")
    capture["frame_bytes"] = 62
    capture["expected_bytes"] = 62
    variants.append(wrong_cfg_binding)

    for variant in variants:
        stream = _replace_golden_record(0, payload=_json_line(variant))
        with pytest.raises(CaptureStreamError):
            CaptureStreamReader(io.BytesIO(stream)).read_contract()


def test_capture_stream_accepts_debug_cli_and_rejects_retired_mode() -> None:
    current = _golden_session()
    current["mode"] = "debug-cli"
    stream = _replace_golden_record(0, payload=_json_line(current))
    assert CaptureStreamReader(io.BytesIO(stream)).read_contract().mode == "debug-cli"

    retired = _golden_session()
    retired["mode"] = "debug-capture"
    stream = _replace_golden_record(0, payload=_json_line(retired))
    with pytest.raises(CaptureStreamError):
        CaptureStreamReader(io.BytesIO(stream)).read_contract()


class _HeaderOnlySource:
    def __init__(self, header: bytes) -> None:
        self.header = header
        self.calls = 0

    def read(self, size: int = -1, /) -> bytes:
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("decoder read an over-limit payload")
        assert size == 80
        return self.header


def test_rejects_payload_bound_from_header_before_payload_read() -> None:
    prefix = _PREFIX.pack(b"MMWSTRM1", 1, 80, 1, 0, 0, 0, (64 << 10) + 1, 0)
    source = _HeaderOnlySource(prefix + bytes(32))

    with pytest.raises(CaptureStreamError, match="contract"):
        CaptureStreamReader(source).read_contract()  # type: ignore[arg-type]

    assert source.calls == 1


def test_rejects_record_digest_truncation_order_and_sequence() -> None:
    bad_digest = bytearray(_GO_GOLDEN_V1)
    bad_digest[48] ^= 1
    with pytest.raises(CaptureStreamError, match="SHA-256"):
        CaptureStreamReader(io.BytesIO(bad_digest)).read_contract()

    out_of_order = _replace_golden_record(0, kind=2)
    with pytest.raises(CaptureStreamError, match="out of order"):
        CaptureStreamReader(io.BytesIO(out_of_order)).read_contract()

    bad_sequence = _replace_golden_record(2, sequence=3)
    reader = CaptureStreamReader(io.BytesIO(bad_sequence))
    reader.read_contract()
    with pytest.raises(CaptureStreamError, match="sequence"):
        list(reader.provisional_frames())

    truncated = _GO_GOLDEN_V1[:-1]
    reader = CaptureStreamReader(io.BytesIO(truncated))
    reader.read_contract()
    list(reader.provisional_frames())
    with pytest.raises(CaptureStreamError, match="ended during COMMIT payload"):
        reader.require_commit()


@pytest.mark.parametrize("failure", ["unknown key", "ADC digest", "trailing data"])
def test_terminal_requires_exact_fields_digest_and_eof(failure: str) -> None:
    if failure == "trailing data":
        stream = _GO_GOLDEN_V1 + b"x"
    else:
        terminal = json.loads(_golden_records()[3][3])
        assert isinstance(terminal, dict)
        if failure == "unknown key":
            terminal["reason_code"] = "cancelled"
        else:
            terminal["adc_sha256"] = "0" * 64
        stream = _replace_golden_record(3, payload=_json_line(terminal))

    reader = CaptureStreamReader(io.BytesIO(stream))
    reader.read_contract()
    list(reader.provisional_frames())

    with pytest.raises(CaptureStreamError):
        reader.require_commit()


class _DeadlineAtEOF(io.BytesIO):
    def read(self, size: int = -1, /) -> bytes:
        if self.tell() == len(self.getbuffer()):
            raise TimeoutError("caller deadline expired")
        return super().read(size)


def test_final_eof_wait_uses_caller_source_deadline_semantics() -> None:
    reader = CaptureStreamReader(_DeadlineAtEOF(_GO_GOLDEN_V1))
    reader.read_contract()
    list(reader.provisional_frames())

    with pytest.raises(CaptureStreamError, match="final EOF") as caught:
        reader.require_commit()

    assert isinstance(caught.value.__cause__, TimeoutError)


def test_wire_failure_poisons_reader_with_stable_error() -> None:
    malformed = bytearray(_GO_GOLDEN_V1)
    malformed[0] ^= 1
    reader = CaptureStreamReader(io.BytesIO(malformed))

    with pytest.raises(CaptureStreamError) as first:
        reader.read_contract()
    with pytest.raises(CaptureStreamError) as second:
        reader.read_contract()

    assert second.value is first.value


def _golden_records() -> list[tuple[int, int, int, bytes]]:
    records: list[tuple[int, int, int, bytes]] = []
    offset = 0
    while offset < len(_GO_GOLDEN_V1):
        header = _GO_GOLDEN_V1[offset : offset + _HEADER.size]
        unpacked = _HEADER.unpack(header)
        payload_size = int(unpacked[7])
        payload_start = offset + _HEADER.size
        payload_end = payload_start + payload_size
        records.append(
            (
                int(unpacked[3]),
                int(unpacked[5]),
                int(unpacked[6]),
                _GO_GOLDEN_V1[payload_start:payload_end],
            )
        )
        offset = payload_end
    assert offset == len(_GO_GOLDEN_V1)
    return records


def _replace_golden_record(
    index: int,
    *,
    kind: int | None = None,
    sequence: int | None = None,
    item_index: int | None = None,
    payload: bytes | None = None,
) -> bytes:
    records = _golden_records()
    current_kind, current_sequence, current_item, current_payload = records[index]
    records[index] = (
        current_kind if kind is None else kind,
        current_sequence if sequence is None else sequence,
        current_item if item_index is None else item_index,
        current_payload if payload is None else payload,
    )
    return b"".join(_record(*record) for record in records)


def _record(kind: int, sequence: int, item_index: int, payload: bytes) -> bytes:
    prefix = _PREFIX.pack(
        b"MMWSTRM1",
        1,
        80,
        kind,
        0,
        sequence,
        item_index,
        len(payload),
        0,
    )
    digest = hashlib.sha256(_DOMAIN + prefix + payload).digest()
    return prefix + digest + payload


def _golden_session() -> dict[str, Any]:
    value = json.loads(_golden_records()[0][3])
    assert isinstance(value, dict)
    return value


def _json_line(record: object) -> bytes:
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def _family_capture_stream(declared_family: str, config_family: str) -> bytes:
    config, frame = _family_stream_fixture(config_family)
    session = _golden_session()
    _object(session, "hardware")["family"] = declared_family
    capture = _object(session, "capture")
    capture["frame_count"] = 1
    capture["frame_bytes"] = len(frame)
    capture["expected_bytes"] = len(frame)
    radar_config = _object(session, "radar_config")
    radar_config["size_bytes"] = len(config)
    radar_config["sha256"] = hashlib.sha256(config).hexdigest()
    terminal = _json_line(
        {
            "schema": "mmwcli.capture_stream_terminal.v1",
            "stream_id": _STREAM_ID,
            "outcome": "commit",
            "frames": 1,
            "adc_bytes": len(frame),
            "adc_sha256": hashlib.sha256(frame).hexdigest(),
        }
    )
    return b"".join(
        (
            _record(1, 0, 0, _json_line(session)),
            _record(2, 1, 0, config),
            _record(3, 2, 0, frame),
            _record(4, 3, 1, terminal),
        )
    )


def _family_stream_fixture(family: str) -> tuple[bytes, bytes]:
    if family == "xwr16xx":
        start_frequency_ghz, channel_tx_mask, chirp_tx_masks = 77, 3, (2, 1)
    elif family == "xwr18xx":
        start_frequency_ghz, channel_tx_mask, chirp_tx_masks = 77, 7, (4, 1, 2)
    elif family == "xwr68xx":
        start_frequency_ghz, channel_tx_mask, chirp_tx_masks = 60, 7, (1, 4, 2)
    else:
        raise AssertionError(f"unsupported test family {family!r}")
    chirps = [
        f"chirpCfg {index} {index} 0 0 0 0 0 {tx_mask}"
        for index, tx_mask in enumerate(chirp_tx_masks)
    ]
    config = (
        "\n".join(
            [
                "flushCfg",
                "dfeDataOutputMode 1",
                f"channelCfg 1 {channel_tx_mask} 0",
                "adcCfg 2 1",
                "adcbufCfg -1 0 1 1 1",
                f"profileCfg 0 {start_frequency_ghz} 7 3 24 0 0 166 1 4 12500 0 0 158",
                *chirps,
                f"frameCfg 0 {len(chirps) - 1} 1 1 10 1 0",
                "lvdsStreamCfg -1 0 1 0",
            ]
        )
        + "\n"
    ).encode()
    return config, bytes(range(16 * len(chirp_tx_masks)))


def _object(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record[field]
    assert isinstance(value, dict)
    return value
