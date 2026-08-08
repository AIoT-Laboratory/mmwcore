"""Decode finite mmwcli capture-stream v1 records from a caller-owned BinaryIO."""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
import unicodedata
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import BinaryIO, NoReturn

import numpy as np

from mmwcore.config import RadarCaptureSpec
from mmwcore.core import RawADCFrame

from ._mmwcli_contract import (
    _MAX_INT64,
    _MAX_RADAR_CONFIG_BYTES,
    MMWCLI_CAPTURE_SESSION_SCHEMA_V1,
    MmwcliRawCaptureContract,
    _parse_mmwcli_radar_config,
    _parse_mmwcli_raw_capture_contract,
    _valid_lower_sha256,
)

MMWCLI_CAPTURE_STREAM_SCHEMA_V1 = "mmwcli.capture_stream.v1"
MMWCLI_CAPTURE_STREAM_TERMINAL_SCHEMA_V1 = "mmwcli.capture_stream_terminal.v1"

_MAGIC = b"MMWSTRM1"
_PROTOCOL_MAJOR = 1
_HEADER_SIZE = 80
_HEADER_PREFIX_SIZE = 48
_HEADER = struct.Struct("<8sHHHHQQQQ32s")
_RECORD_DIGEST_DOMAIN = b"mmwcli.capture_stream.record.v1\x00"
_MAX_SESSION_PAYLOAD_BYTES = 64 << 10
_MAX_FRAME_PAYLOAD_BYTES = 64 << 20
_MAX_TERMINAL_PAYLOAD_BYTES = 4 << 10
_MAX_PRODUCER_VERSION_BYTES = 128
_STREAM_ID_ZERO = "0" * 32
_CAPTURE_MODES = frozenset({"studio-cli", "debug-capture"})
_ABORT_REASONS = frozenset(
    {
        "cancelled",
        "backpressure",
        "capture_failed",
        "integrity_failed",
        "cleanup_failed",
        "publish_failed",
    }
)


class CaptureStreamError(ValueError):
    """A capture stream is malformed, incomplete, corrupt, or unreadable."""


class CaptureStreamStateError(RuntimeError):
    """A capture-stream reader operation is invalid in its current state."""


class CaptureStreamAborted(CaptureStreamError):
    """A valid ABORT terminal record ended the provisional stream."""

    def __init__(self, abort: CaptureStreamAbort) -> None:
        self.abort = abort
        super().__init__(f"mmwcli capture stream aborted: {abort.reason_code}.")


@dataclass(frozen=True)
class CaptureStreamContract:
    """Validated immutable contract carried by SESSION and RADAR_CONFIG."""

    stream_id: str
    producer_name: str
    producer_version: str
    mode: str
    frame_count: int
    frame_bytes: int
    expected_bytes: int
    raw_capture: MmwcliRawCaptureContract
    radar_config: bytes
    radar_config_sha256: str
    radar_capture: RadarCaptureSpec


@dataclass(frozen=True)
class ProvisionalADCFrame:
    """One integrity-checked ADC frame that remains provisional until COMMIT."""

    frame: RawADCFrame
    sequence: int
    frame_index: int
    adc_byte_offset: int


@dataclass(frozen=True)
class CaptureStreamCommit:
    """Validated COMMIT evidence for all frames in a finite stream."""

    stream_id: str
    frames: int
    adc_bytes: int
    adc_sha256: str


@dataclass(frozen=True)
class CaptureStreamAbort:
    """Validated ABORT evidence for the frames emitted before failure."""

    stream_id: str
    frames: int
    adc_bytes: int
    adc_sha256: str
    reason_code: str


class _RecordType(IntEnum):
    SESSION = 1
    RADAR_CONFIG = 2
    FRAME = 3
    COMMIT = 4
    ABORT = 5


_PAYLOAD_LIMITS = {
    _RecordType.SESSION: _MAX_SESSION_PAYLOAD_BYTES,
    _RecordType.RADAR_CONFIG: _MAX_RADAR_CONFIG_BYTES,
    _RecordType.FRAME: _MAX_FRAME_PAYLOAD_BYTES,
    _RecordType.COMMIT: _MAX_TERMINAL_PAYLOAD_BYTES,
    _RecordType.ABORT: _MAX_TERMINAL_PAYLOAD_BYTES,
}


@dataclass(frozen=True)
class _Record:
    kind: _RecordType
    sequence: int
    item_index: int
    payload: bytes


@dataclass(frozen=True)
class _RecordHeader:
    kind: _RecordType
    sequence: int
    item_index: int
    payload_size: int
    digest: bytes
    prefix: bytes


@dataclass(frozen=True)
class _Session:
    stream_id: str
    producer_version: str
    mode: str
    frame_count: int
    frame_bytes: int
    expected_bytes: int
    config_size_bytes: int
    config_sha256: str
    raw_capture: MmwcliRawCaptureContract


class CaptureStreamReader:
    """Strict synchronous reader for one finite mmwcli capture stream.

    The caller owns ``source`` and its cancellation/deadline behavior. In
    particular, :meth:`require_commit` reads one byte after the terminal
    record to require EOF and can block for as long as ``source.read`` does.
    The reader never closes the source.
    """

    def __init__(self, source: BinaryIO) -> None:
        if not callable(getattr(source, "read", None)):
            raise TypeError("CaptureStreamReader source must provide read(size) -> bytes.")
        self._source = source
        self._state = "new"
        self._failure: CaptureStreamError | None = None
        self._contract: CaptureStreamContract | None = None
        self._frames_emitted = 0
        self._adc_bytes = 0
        self._adc_hash = hashlib.sha256()

    def read_contract(self) -> CaptureStreamContract:
        """Read and validate the leading SESSION and RADAR_CONFIG records."""

        self._require_state("new", operation="read_contract")
        try:
            session_record = self._read_record(
                allowed=frozenset({_RecordType.SESSION}),
                sequence=0,
                item_index=0,
            )
            session = _parse_session(session_record.payload)
            config_record = self._read_record(
                allowed=frozenset({_RecordType.RADAR_CONFIG}),
                sequence=1,
                item_index=0,
                exact_payload_bytes=session.config_size_bytes,
            )
            radar_capture = _parse_mmwcli_radar_config(
                config_record.payload,
                raw_capture=session.raw_capture,
                expected_sha256=session.config_sha256,
                context="mmwcli capture stream",
            )
            _validate_config_binding(session, radar_capture)
        except CaptureStreamError:
            raise
        except (TypeError, ValueError, RecursionError) as exc:
            self._poison("mmwcli capture-stream contract is invalid", exc)

        contract = CaptureStreamContract(
            stream_id=session.stream_id,
            producer_name="mmwcli",
            producer_version=session.producer_version,
            mode=session.mode,
            frame_count=session.frame_count,
            frame_bytes=session.frame_bytes,
            expected_bytes=session.expected_bytes,
            raw_capture=session.raw_capture,
            radar_config=config_record.payload,
            radar_config_sha256=session.config_sha256,
            radar_capture=radar_capture,
        )
        self._contract = contract
        self._state = "contract"
        return contract

    def provisional_frames(self) -> Iterator[ProvisionalADCFrame]:
        """Iterate all declared frames once, without implying stream commit."""

        self._require_state("contract", operation="provisional_frames")
        self._state = "frames"
        return self._iterate_frames()

    def _iterate_frames(self) -> Iterator[ProvisionalADCFrame]:
        contract = self._required_contract()
        while self._frames_emitted < contract.frame_count:
            index = self._frames_emitted
            try:
                record = self._read_record(
                    allowed=frozenset({_RecordType.FRAME, _RecordType.ABORT}),
                    sequence=2 + index,
                    item_index=index,
                    exact_frame_bytes=contract.frame_bytes,
                )
                if record.kind is _RecordType.ABORT:
                    abort = self._accept_terminal(record, expected_outcome="abort")
                    assert isinstance(abort, CaptureStreamAbort)
                    raise CaptureStreamAborted(abort)
                payload = record.payload
                self._adc_hash.update(payload)
                self._adc_bytes += len(payload)
                self._frames_emitted += 1
                values = np.frombuffer(payload, dtype="<i2").astype(np.int16, copy=False)
                frame = RawADCFrame(
                    samples=values,
                    frame_id=index,
                    timestamp=(
                        index * contract.radar_capture.frame_periodicity_s
                        if contract.radar_capture.frame_periodicity_s is not None
                        else None
                    ),
                    source="mmwcli.capture_stream",
                    profile=asdict(contract.radar_capture.profile),
                    metadata={
                        "stream_id": contract.stream_id,
                        "producer": {
                            "name": contract.producer_name,
                            "version": contract.producer_version,
                        },
                        "capture_mode": contract.mode,
                        "frame_index": index,
                        "num_frames": contract.frame_count,
                        "tx_order": list(contract.radar_capture.tx_order),
                        "provisional": True,
                    },
                )
            except (CaptureStreamError, CaptureStreamAborted):
                raise
            except (TypeError, ValueError, RecursionError) as exc:
                self._poison("mmwcli capture-stream frame is invalid", exc)
            yield ProvisionalADCFrame(
                frame=frame,
                sequence=record.sequence,
                frame_index=index,
                adc_byte_offset=index * contract.frame_bytes,
            )
        self._state = "frames_complete"

    def require_commit(self) -> CaptureStreamCommit:
        """Require a valid COMMIT followed by EOF and return its evidence.

        The final EOF read uses the caller-provided source semantics. Callers
        that need a bounded wait must supply a source whose reads honor their
        deadline or cancellation policy.
        """

        self._require_state("frames_complete", operation="require_commit")
        contract = self._required_contract()
        try:
            terminal = self._read_record(
                allowed=frozenset({_RecordType.COMMIT, _RecordType.ABORT}),
                sequence=2 + self._frames_emitted,
                item_index=self._frames_emitted,
            )
            expected = "commit" if terminal.kind is _RecordType.COMMIT else "abort"
            result = self._accept_terminal(terminal, expected_outcome=expected)
        except (CaptureStreamError, CaptureStreamAborted):
            raise
        except (TypeError, ValueError, RecursionError) as exc:
            self._poison("mmwcli capture-stream terminal is invalid", exc)
        if isinstance(result, CaptureStreamAbort):
            raise CaptureStreamAborted(result)
        if result.frames != contract.frame_count or result.adc_bytes != contract.expected_bytes:
            self._poison("mmwcli capture-stream COMMIT does not cover the declared capture")
        return result

    def _accept_terminal(
        self,
        record: _Record,
        *,
        expected_outcome: str,
    ) -> CaptureStreamCommit | CaptureStreamAbort:
        contract = self._required_contract()
        result = _parse_terminal(record.payload, expected_outcome=expected_outcome)
        if result.stream_id != contract.stream_id:
            self._poison("mmwcli capture-stream terminal stream_id does not match SESSION")
        if result.frames != self._frames_emitted:
            self._poison("mmwcli capture-stream terminal frame count is inconsistent")
        if result.adc_bytes != self._adc_bytes:
            self._poison("mmwcli capture-stream terminal ADC byte count is inconsistent")
        digest = self._adc_hash.hexdigest()
        if not hmac.compare_digest(result.adc_sha256, digest):
            self._poison("mmwcli capture-stream terminal ADC SHA-256 is inconsistent")
        if isinstance(result, CaptureStreamCommit) and self._frames_emitted != contract.frame_count:
            self._poison("mmwcli capture-stream COMMIT arrived before all declared frames")
        self._require_eof()
        self._state = expected_outcome
        return result

    def _read_record(
        self,
        *,
        allowed: frozenset[_RecordType],
        sequence: int,
        item_index: int,
        exact_payload_bytes: int | None = None,
        exact_frame_bytes: int | None = None,
    ) -> _Record:
        header = self._read_exact(_HEADER_SIZE, label="record header")
        decoded = _decode_header(header)
        if decoded.kind not in allowed:
            self._poison(f"mmwcli capture-stream record type {decoded.kind.name} is out of order")
        if decoded.sequence != sequence or decoded.item_index != item_index:
            self._poison(
                "mmwcli capture-stream sequence or item index is not the next zero-based value"
            )
        if exact_payload_bytes is not None and decoded.payload_size != exact_payload_bytes:
            self._poison("mmwcli capture-stream payload size does not match SESSION")
        if (
            decoded.kind is _RecordType.FRAME
            and exact_frame_bytes is not None
            and decoded.payload_size != exact_frame_bytes
        ):
            self._poison("mmwcli capture-stream FRAME size does not match SESSION")
        payload = self._read_exact(decoded.payload_size, label=f"{decoded.kind.name} payload")
        expected_digest = hashlib.sha256(_RECORD_DIGEST_DOMAIN + decoded.prefix + payload).digest()
        if not hmac.compare_digest(decoded.digest, expected_digest):
            self._poison("mmwcli capture-stream record SHA-256 is invalid")
        return _Record(
            kind=decoded.kind,
            sequence=decoded.sequence,
            item_index=decoded.item_index,
            payload=payload,
        )

    def _read_exact(self, size: int, *, label: str) -> bytes:
        payload = bytearray()
        while len(payload) < size:
            requested = size - len(payload)
            try:
                chunk = self._source.read(requested)
            except Exception as exc:
                self._poison(f"mmwcli capture-stream {label} read failed", exc)
            if type(chunk) is not bytes:
                self._poison("mmwcli capture-stream source read() must return bytes")
            if len(chunk) > requested:
                self._poison("mmwcli capture-stream source returned more bytes than requested")
            if not chunk:
                self._poison(f"mmwcli capture-stream ended during {label}")
            payload.extend(chunk)
        return bytes(payload)

    def _require_eof(self) -> None:
        try:
            trailing = self._source.read(1)
        except Exception as exc:
            self._poison("mmwcli capture-stream final EOF read failed", exc)
        if type(trailing) is not bytes:
            self._poison("mmwcli capture-stream source read() must return bytes")
        if len(trailing) > 1:
            self._poison("mmwcli capture-stream source returned more bytes than requested")
        if trailing:
            self._poison("mmwcli capture-stream has trailing data after its terminal record")

    def _required_contract(self) -> CaptureStreamContract:
        if self._contract is None:
            raise CaptureStreamStateError("CaptureStreamReader contract has not been read.")
        return self._contract

    def _require_state(self, expected: str, *, operation: str) -> None:
        if self._state == "poisoned":
            assert self._failure is not None
            raise self._failure
        if self._state != expected:
            raise CaptureStreamStateError(
                f"CaptureStreamReader.{operation} is invalid in state {self._state!r}."
            )

    def _poison(self, message: str, cause: BaseException | None = None) -> NoReturn:
        error = CaptureStreamError(f"{message}.")
        self._state = "poisoned"
        self._failure = error
        if cause is None:
            raise error
        raise error from cause


def _decode_header(header: bytes) -> _RecordHeader:
    try:
        (
            magic,
            major,
            header_size,
            kind_value,
            flags,
            sequence,
            item_index,
            payload_size,
            reserved,
            digest,
        ) = _HEADER.unpack(header)
    except struct.error as exc:
        raise ValueError("record header cannot be decoded") from exc
    if magic != _MAGIC:
        raise ValueError("record magic is invalid")
    if major != _PROTOCOL_MAJOR or header_size != _HEADER_SIZE:
        raise ValueError("protocol version or header size is unsupported")
    if flags != 0 or reserved != 0:
        raise ValueError("v1 flags and reserved fields must be zero")
    try:
        kind = _RecordType(kind_value)
    except ValueError as exc:
        raise ValueError(f"record type {kind_value} is unsupported") from exc
    maximum = _PAYLOAD_LIMITS[kind]
    if payload_size > maximum:
        raise ValueError(f"{kind.name} payload exceeds the {maximum}-byte limit")
    return _RecordHeader(
        kind=kind,
        sequence=int(sequence),
        item_index=int(item_index),
        payload_size=int(payload_size),
        digest=digest,
        prefix=header[:_HEADER_PREFIX_SIZE],
    )


def _parse_session(payload: bytes) -> _Session:
    record = _strict_json_object(payload, context="SESSION")
    _exact_keys(
        record,
        {
            "schema",
            "stream_id",
            "producer",
            "mode",
            "hardware",
            "capture",
            "adc",
            "radar_config",
            "artifact",
        },
        context="SESSION",
    )
    _literal(record, "schema", MMWCLI_CAPTURE_STREAM_SCHEMA_V1, context="SESSION")
    stream_id = _stream_id(record.get("stream_id"), context="SESSION.stream_id")

    producer = _closed_object(
        record,
        "producer",
        {"name", "version"},
        context="SESSION.producer",
    )
    _literal(producer, "name", "mmwcli", context="SESSION.producer")
    producer_version = _producer_version(producer.get("version"))

    mode = record.get("mode")
    if not isinstance(mode, str) or mode not in _CAPTURE_MODES:
        raise ValueError("SESSION.mode must be 'studio-cli' or 'debug-capture'.")

    hardware = _closed_object(
        record,
        "hardware",
        {"vendor", "family", "model", "revision", "identity_source"},
        context="SESSION.hardware",
    )

    capture = _closed_object(
        record,
        "capture",
        {
            "frame_count",
            "frame_bytes",
            "expected_bytes",
            "record_sequence_origin",
            "frame_index_origin",
            "adc_byte_offset_origin",
        },
        context="SESSION.capture",
    )
    frame_count, frame_bytes, expected_bytes = _capture_shape(capture)

    adc = _closed_object(
        record,
        "adc",
        {"dtype", "byte_order", "lane_count", "layout"},
        context="SESSION.adc",
    )

    radar_config = _closed_object(
        record,
        "radar_config",
        {"format", "size_bytes", "sha256"},
        context="SESSION.radar_config",
    )
    raw_capture = _parse_mmwcli_raw_capture_contract(
        hardware=hardware,
        adc=adc,
        radar_config=radar_config,
        context="SESSION",
    )
    config_size_bytes = _integer(
        radar_config,
        "size_bytes",
        1,
        _MAX_RADAR_CONFIG_BYTES,
        context="SESSION.radar_config",
    )
    config_sha256 = radar_config.get("sha256")
    if not _valid_lower_sha256(config_sha256):
        raise ValueError("SESSION.radar_config.sha256 must be lowercase SHA-256.")
    assert isinstance(config_sha256, str)

    artifact = _closed_object(
        record,
        "artifact",
        {"required", "schema"},
        context="SESSION.artifact",
    )
    if artifact.get("required") is not True:
        raise ValueError("SESSION.artifact.required must be true.")
    _literal(
        artifact,
        "schema",
        MMWCLI_CAPTURE_SESSION_SCHEMA_V1,
        context="SESSION.artifact",
    )
    return _Session(
        stream_id=stream_id,
        producer_version=producer_version,
        mode=mode,
        frame_count=frame_count,
        frame_bytes=frame_bytes,
        expected_bytes=expected_bytes,
        config_size_bytes=config_size_bytes,
        config_sha256=config_sha256,
        raw_capture=raw_capture,
    )


def _producer_version(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("SESSION.producer.version must be a string.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("SESSION.producer.version must be valid UTF-8.") from exc
    if (
        not encoded
        or len(encoded) > _MAX_PRODUCER_VERSION_BYTES
        or value.strip() != value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError(
            "SESSION.producer.version must contain 1..128 UTF-8 bytes without "
            "surrounding whitespace or control characters."
        )
    return value


def _capture_shape(record: dict[str, object]) -> tuple[int, int, int]:
    frame_count = _integer(record, "frame_count", 1, _MAX_INT64, context="SESSION.capture")
    frame_bytes = _integer(
        record,
        "frame_bytes",
        1,
        _MAX_FRAME_PAYLOAD_BYTES,
        context="SESSION.capture",
    )
    if frame_bytes % 2:
        raise ValueError("SESSION.capture.frame_bytes must be aligned to int16.")
    expected_bytes = _integer(
        record,
        "expected_bytes",
        1,
        _MAX_INT64,
        context="SESSION.capture",
    )
    if frame_count > _MAX_INT64 // frame_bytes or expected_bytes != frame_count * frame_bytes:
        raise ValueError("SESSION.capture.expected_bytes must be the checked frame product.")
    for origin in (
        "record_sequence_origin",
        "frame_index_origin",
        "adc_byte_offset_origin",
    ):
        _integer(record, origin, 0, 0, context="SESSION.capture")
    return frame_count, frame_bytes, expected_bytes


def _parse_terminal(
    payload: bytes,
    *,
    expected_outcome: str,
) -> CaptureStreamCommit | CaptureStreamAbort:
    record = _strict_json_object(payload, context=expected_outcome.upper())
    outcome = record.get("outcome")
    if outcome != expected_outcome:
        raise ValueError(f"terminal outcome must be {expected_outcome!r}.")
    fields = {"schema", "stream_id", "outcome", "frames", "adc_bytes", "adc_sha256"}
    if expected_outcome == "abort":
        fields.add("reason_code")
    _exact_keys(record, fields, context=expected_outcome.upper())
    _literal(
        record,
        "schema",
        MMWCLI_CAPTURE_STREAM_TERMINAL_SCHEMA_V1,
        context=expected_outcome.upper(),
    )
    stream_id = _stream_id(record.get("stream_id"), context="terminal.stream_id")
    frames = _integer(record, "frames", 0, _MAX_INT64, context="terminal")
    adc_bytes = _integer(record, "adc_bytes", 0, _MAX_INT64, context="terminal")
    digest = record.get("adc_sha256")
    if not _valid_lower_sha256(digest):
        raise ValueError("terminal.adc_sha256 must be lowercase SHA-256.")
    assert isinstance(digest, str)
    if expected_outcome == "commit":
        return CaptureStreamCommit(
            stream_id=stream_id,
            frames=frames,
            adc_bytes=adc_bytes,
            adc_sha256=digest,
        )
    reason = record.get("reason_code")
    if not isinstance(reason, str) or reason not in _ABORT_REASONS:
        raise ValueError("terminal.reason_code is not a stable capture-stream v1 reason.")
    return CaptureStreamAbort(
        stream_id=stream_id,
        frames=frames,
        adc_bytes=adc_bytes,
        adc_sha256=digest,
        reason_code=reason,
    )


def _validate_config_binding(session: _Session, capture: RadarCaptureSpec) -> None:
    if capture.num_frames != session.frame_count:
        raise ValueError("CFG frame count does not match SESSION.capture.frame_count.")
    if capture.adc.raw_values_per_frame * 2 != session.frame_bytes:
        raise ValueError("CFG-derived ADC frame size does not match SESSION.capture.frame_bytes.")
    if capture.expected_size_bytes != session.expected_bytes:
        raise ValueError("CFG-derived ADC size does not match SESSION.capture.expected_bytes.")


def _strict_json_object(payload: bytes, *, context: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"{context} must be strict UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key {key!r}")
        record[key] = value
    return record


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _exact_keys(record: dict[str, object], fields: set[str], *, context: str) -> None:
    actual = set(record)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if unknown:
            details.append(f"unknown {unknown!r}")
        raise ValueError(f"{context} has an invalid key set ({'; '.join(details)}).")


def _closed_object(
    record: dict[str, object],
    field: str,
    fields: set[str],
    *,
    context: str,
) -> dict[str, object]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    _exact_keys(value, fields, context=context)
    return value


def _literal(
    record: dict[str, object],
    field: str,
    expected: str,
    *,
    context: str,
) -> None:
    if record.get(field) != expected:
        raise ValueError(f"{context}.{field} must be {expected!r}.")


def _integer(
    record: dict[str, object],
    field: str,
    minimum: int,
    maximum: int,
    *,
    context: str,
) -> int:
    value = record.get(field)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{context}.{field} must be an integer in [{minimum}, {maximum}].")
    return value


def _stream_id(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or value == _STREAM_ID_ZERO
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a nonzero 16-byte lowercase hexadecimal ID.")
    return value


__all__ = [
    "MMWCLI_CAPTURE_STREAM_SCHEMA_V1",
    "MMWCLI_CAPTURE_STREAM_TERMINAL_SCHEMA_V1",
    "CaptureStreamAbort",
    "CaptureStreamAborted",
    "CaptureStreamCommit",
    "CaptureStreamContract",
    "CaptureStreamError",
    "CaptureStreamReader",
    "CaptureStreamStateError",
    "ProvisionalADCFrame",
]
