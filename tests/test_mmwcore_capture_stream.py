from __future__ import annotations

import hashlib
import io
import json
import struct
from typing import Any

import numpy as np
import pytest

from mmwcore.io import (
    MMWCLI_CAPTURE_STREAM_SCHEMA_V1,
    CaptureStreamAbort,
    CaptureStreamAborted,
    CaptureStreamError,
    CaptureStreamReader,
    CaptureStreamStateError,
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

# Produced by mmwcli internal/capturestream at Go commit cd9b948. This is a
# byte-for-byte cross-language fixture, not a Python-generated equivalent.
_GO_GOLDEN_V1_HEX = (
    "4d4d575354524d310100500001000000000000000000000000000000000000003f02000000000000"
    "0000000000000000291a3813531ad4da324a0d53d92b0944de232de93a07ce1ad2089921ae878d7c"
    "7b22736368656d61223a226d6d77636c692e636170747572655f73747265616d2e7631222c227374"
    "7265616d5f6964223a22303030313032303330343035303630373038303930613062306330643065"
    "3066222c2270726f6475636572223a7b226e616d65223a226d6d77636c69222c2276657273696f6e"
    "223a2274657374227d2c226d6f6465223a2273747564696f2d636c69222c2263617074757265223a"
    "7b226672616d655f636f756e74223a312c226672616d655f6279746573223a36342c226578706563"
    "7465645f6279746573223a36342c227265636f72645f73657175656e63655f6f726967696e223a30"
    "2c226672616d655f696e6465785f6f726967696e223a302c226164635f627974655f6f6666736574"
    "5f6f726967696e223a307d2c22616463223a7b226474797065223a22696e743136222c2262797465"
    "5f6f72646572223a226c6974746c65222c226c61796f7574223a2267726f7570325f695f7468656e"
    "5f71227d2c2272616461725f636f6e666967223a7b22666f726d6174223a2274695f787772363878"
    "785f6c65676163795f636c69222c2273697a655f6279746573223a3232352c22736861323536223a"
    "2262396565626638613466656534303239353936333863373232633964613539353734386665383938"
    "373537393664613034343466333262316339656535646562227d2c226172746966616374223a7b22"
    "7265717569726564223a747275652c22736368656d61223a226d6d77636c692e636170747572655f"
    "73657373696f6e2e7631227d7d0a4d4d575354524d31010050000200000001000000000000000000"
    "000000000000e10000000000000000000000000000002ed2891e73145ea55d3fe68f550123e3a180"
    "1c02875291d3581df8196e9eebeb666c7573684366670a646665446174614f75747075744d6f6465"
    "20310a6368616e6e656c4366672031203120300a616463436667203220310a61646362756643666720"
    "2d3120302031203120310a70726f66696c654366672030203630203720332032342030203020313636"
    "203120313620313235303020302030203135380a636869727043666720302030203020302030203020"
    "3020310a6672616d654366672030203020312031203130203120300a6c6f77506f776572203020300a"
    "6c76647353747265616d436667202d312030203120300a73656e736f7253746172740a4d4d57535452"
    "4d310100500003000000020000000000000000000000000000004000000000000000000000000000"
    "000014cae892709325bd9f8e84a1d832feaf9589b669fdb32a1375d368524b606d0000010203040506"
    "0708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e"
    "2f303132333435363738393a3b3c3d3e3f4d4d575354524d3101005000040000000300000000000000"
    "0100000000000000db0000000000000000000000000000001157156605f2c9d58b5c7e23256e6753"
    "337676f79ccc801728c06a1dc5486e1d7b22736368656d61223a226d6d77636c692e636170747572"
    "655f73747265616d5f7465726d696e616c2e7631222c2273747265616d5f6964223a223030303130"
    "323033303430353036303730383039306130623063306430653066222c226f7574636f6d65223a22"
    "636f6d6d6974222c226672616d6573223a312c226164635f6279746573223a36342c226164635f73"
    "6861323536223a22666465616239616366333731303336326264323635386364633961323965386639"
    "63373537666366393831313630336138633434376364316439313531313038227d0a"
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


def _object(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record[field]
    assert isinstance(value, dict)
    return value
