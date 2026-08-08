"""Decode finite mmwcli multi-sensor streams from caller-owned BinaryIO."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import struct
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import BinaryIO, NoReturn

import numpy as np

from mmwcore.config import RadarCaptureSpec
from mmwcore.core import RadarCube, RangeDopplerRecipe, RawADCFrame

from ._range_doppler import (
    RangeDopplerPreset,
    _resolve_range_doppler_recipe,
    _validate_range_doppler_recipe,
)
from .multisensor_capture import MappedTimeInterval

MMWCLI_MULTISENSOR_STREAM_SCHEMA_V1 = "mmwcli.multisensor_stream.v1"
MMWCLI_MULTISENSOR_STREAM_TERMINAL_SCHEMA_V1 = "mmwcli.multisensor_stream_terminal.v1"

_RADAR_CONFIG_SCHEMA = "mmwcli.multisensor_stream_radar_config.v1"
_RADAR_START_SCHEMA = "mmwcli.multisensor_stream_radar_start.v1"
_ITEM_SCHEMA = "mmwcli.multisensor_stream_item.v1"
_END_SCHEMA = "mmwcli.multisensor_stream_end.v1"
_EOF_SCHEMA = "mmwcli.multisensor_stream_eof.v1"
_MAGIC = b"MMWMSTR1"
_PROTOCOL_MAJOR = 1
_HEADER_SIZE = 80
_HEADER_PREFIX_SIZE = 48
_HEADER = struct.Struct("<8sHHHHQQQQ32s")
_RECORD_DIGEST_DOMAIN = b"mmwcli.multisensor_stream.record.v1\x00"
_MAX_METADATA_BYTES = 1 << 20
_MAX_RADAR_CONFIG_BYTES = 4 << 20
_MAX_ITEM_BYTES = 1 << 34
_MAX_PAYLOAD_BYTES = 1 << 50
_MAX_ITEMS = 1 << 24
_MAX_SOURCES = 32
_MAX_U64 = (1 << 64) - 1
_NO_SYNC_EVENT = _MAX_U64
_NANOSECONDS_PER_SECOND = 1_000_000_000
_SESSION_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_SOURCE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_PAYLOAD_FORMAT = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SOURCE_KINDS = frozenset({"radar", "camera"})
_SOURCE_OUTCOMES = frozenset({"complete", "failed", "omitted"})
_ABORT_REASONS = frozenset(
    {
        "cancelled",
        "backpressure",
        "source_failed",
        "integrity_failed",
        "cleanup_failed",
        "publish_failed",
    }
)


class MultisensorStreamError(ValueError):
    """A multi-sensor stream is malformed, incomplete, corrupt, or unreadable."""


class MultisensorStreamStateError(RuntimeError):
    """A multi-sensor reader operation is invalid in its current state."""


class MultisensorStreamAborted(MultisensorStreamError):
    """A valid ABORT+EOF terminated the provisional stream."""

    def __init__(self, abort: MultisensorStreamAbort) -> None:
        self.abort = abort
        super().__init__(f"mmwcli multi-sensor stream aborted: {abort.reason_code}.")


@dataclass(frozen=True)
class MultisensorStreamSource:
    """One immutable source contract declared by SESSION."""

    source_id: str
    kind: str
    required: bool
    payload_filename: str
    payload_format: str
    clock_id: str
    tick_hz: int
    wrap_ticks: int
    timestamp_semantics: str
    max_items: int
    max_item_bytes: int
    max_payload_bytes: int


@dataclass(frozen=True)
class MultisensorStreamContract:
    """Validated static aggregate contract carried by SESSION."""

    session_id: str
    synchronization_grade: str
    sources: tuple[MultisensorStreamSource, ...]

    def source(self, source_id: str) -> MultisensorStreamSource:
        if type(source_id) is not str:
            raise TypeError("MultisensorStreamContract source_id must be a string.")
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(source_id)


@dataclass(frozen=True)
class MultisensorRadarConfig:
    """Integrity-checked optional RADAR_CONFIG record."""

    source_id: str
    format: str
    payload: bytes = field(repr=False)
    sha256: str


@dataclass(frozen=True)
class MultisensorSourceEnd:
    """END lineage and the authoritative outcome for one provisional source."""

    source_id: str
    outcome: str
    item_count: int
    payload_bytes: int
    payload_sha256: str


@dataclass(frozen=True)
class ProvisionalMultisensorRangeDoppler:
    """One provisional radar item and its derived Range-Doppler cube."""

    item: ProvisionalMultisensorItem
    cube: RadarCube


@dataclass(frozen=True)
class ProvisionalMultisensorItem:
    """One cross-source item that remains provisional until COMMIT+EOF.

    A committed source outcome other than ``complete`` invalidates every item
    previously yielded for that source. Use :meth:`MultisensorStreamCommit.accepts`
    before retaining provisional data.

    A ``delivery_observed`` camera tick is the recorder's raw host-relative
    delivery time, not a camera exposure timestamp.
    """

    session_id: str
    source_id: str
    kind: str
    payload_format: str
    record_sequence: int
    item_index: int
    payload: bytes = field(repr=False)
    tick: int
    wrap_count: int
    duration_ticks: int
    sync_event_id: int | None
    tick_hz: int
    wrap_ticks: int
    timestamp_semantics: str
    mapped_time: MappedTimeInterval | None

    def range_doppler(
        self,
        binding: RangeDopplerRecipe | RangeDopplerPreset,
        *,
        radar_capture: RadarCaptureSpec | None = None,
    ) -> ProvisionalMultisensorRangeDoppler:
        """Decode one radar item with an explicit recipe or caller-bound preset."""

        if self.kind != "radar":
            raise TypeError("Only radar multi-sensor items can produce Range-Doppler data.")
        if radar_capture is not None and not isinstance(radar_capture, RadarCaptureSpec):
            raise TypeError("radar_capture must be a RadarCaptureSpec or None.")
        if isinstance(binding, RangeDopplerRecipe):
            recipe = binding
            if radar_capture is not None:
                _validate_range_doppler_recipe(
                    radar_capture,
                    recipe,
                    context="ProvisionalMultisensorItem.range_doppler",
                )
        elif callable(binding):
            if radar_capture is None:
                raise TypeError("A Range-Doppler preset requires an explicit RadarCaptureSpec.")
            resolved = _resolve_range_doppler_recipe(
                radar_capture,
                binding,
                context="ProvisionalMultisensorItem.range_doppler",
            )
            assert resolved is not None
            recipe = resolved
        else:
            raise TypeError("binding must be a RangeDopplerRecipe or preset callable.")

        expected_bytes = recipe.decode.adc.raw_values_per_frame * 2
        if len(self.payload) != expected_bytes:
            raise ValueError(
                "Radar ITEM byte count does not match the Range-Doppler recipe ADC spec."
            )
        unwrapped = _unwrap_ticks(self.tick, self.wrap_count, self.wrap_ticks)
        values = np.frombuffer(self.payload, dtype="<i2").astype(np.int16, copy=False)
        raw = RawADCFrame(
            samples=values,
            frame_id=self.item_index,
            timestamp=unwrapped / self.tick_hz,
            source="mmwcli.multisensor_stream",
            profile=asdict(radar_capture.profile) if radar_capture is not None else {},
            metadata={
                "session_id": self.session_id,
                "source_id": self.source_id,
                "record_sequence": self.record_sequence,
                "item_index": self.item_index,
                "tick": self.tick,
                "wrap_count": self.wrap_count,
                "duration_ticks": self.duration_ticks,
                "sync_event_id": self.sync_event_id,
                "timestamp_semantics": self.timestamp_semantics,
                "provisional": True,
            },
        )
        from mmwcore.dsp.runners import process_adc_to_range_doppler

        return ProvisionalMultisensorRangeDoppler(
            item=self,
            cube=process_adc_to_range_doppler(raw, recipe),
        )


@dataclass(frozen=True)
class MultisensorStreamCommit:
    """Validated COMMIT+EOF evidence and per-source retention outcomes."""

    session_id: str
    session_json_size_bytes: int
    session_json_sha256: str
    sources: tuple[MultisensorSourceEnd, ...]

    def source(self, source_id: str) -> MultisensorSourceEnd:
        if type(source_id) is not str:
            raise TypeError("MultisensorStreamCommit source_id must be a string.")
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(source_id)

    def accepts(self, item: ProvisionalMultisensorItem) -> bool:
        """Return whether COMMIT retains this provisional source item."""

        if not isinstance(item, ProvisionalMultisensorItem):
            raise TypeError("MultisensorStreamCommit.accepts requires a provisional item.")
        if item.session_id != self.session_id:
            return False
        try:
            end = self.source(item.source_id)
        except KeyError:
            return False
        return end.outcome == "complete" and item.item_index < end.item_count


@dataclass(frozen=True)
class MultisensorStreamAbort:
    """Validated ABORT+EOF evidence."""

    session_id: str
    reason_code: str


class _RecordType(IntEnum):
    SESSION = 1
    RADAR_CONFIG = 2
    RADAR_START = 3
    ITEM = 4
    END = 5
    COMMIT = 6
    ABORT = 7
    EOF = 8


@dataclass(frozen=True)
class _Record:
    kind: _RecordType
    sequence: int
    metadata: dict[str, object]
    payload: bytes


@dataclass
class _SourceProgress:
    contract: MultisensorStreamSource
    next_item: int = 0
    payload_bytes: int = 0
    payload_hash: object = field(default_factory=hashlib.sha256)
    end: MultisensorSourceEnd | None = None
    radar_start: MappedTimeInterval | None = None


class MultisensorStreamReader:
    """Strict pull reader that never closes or otherwise owns ``source``."""

    def __init__(self, source: BinaryIO) -> None:
        if not callable(getattr(source, "read", None)):
            raise TypeError("MultisensorStreamReader source must provide read(size) -> bytes.")
        self._source = source
        self._state = "new"
        self._failure: MultisensorStreamError | None = None
        self._contract: MultisensorStreamContract | None = None
        self._progress: dict[str, _SourceProgress] = {}
        self._next_record = 0
        self._records_started = False
        self._ending_started = False
        self._radar_config: MultisensorRadarConfig | None = None

    def read_contract(self) -> MultisensorStreamContract:
        """Read and validate the leading SESSION record."""

        self._require_state("new", operation="read_contract")
        try:
            record = self._read_record(frozenset({_RecordType.SESSION}))
            contract = _parse_session(record.metadata)
        except MultisensorStreamError:
            raise
        except (TypeError, ValueError, RecursionError) as exc:
            self._poison("mmwcli multi-sensor SESSION is invalid", exc)
        self._contract = contract
        self._progress = {
            source.source_id: _SourceProgress(contract=source) for source in contract.sources
        }
        self._state = "contract"
        return contract

    @property
    def radar_config(self) -> MultisensorRadarConfig | None:
        """Return RADAR_CONFIG after the item iterator has consumed it, if present."""

        return self._radar_config

    def provisional_items(self) -> Iterator[ProvisionalMultisensorItem]:
        """Yield interleaved source items once, all explicitly provisional."""

        self._require_state("contract", operation="provisional_items")
        self._state = "items"
        return self._iterate_items()

    def _iterate_items(self) -> Iterator[ProvisionalMultisensorItem]:
        try:
            while not self._all_sources_ended():
                record = self._read_record(
                    frozenset(
                        {
                            _RecordType.RADAR_CONFIG,
                            _RecordType.RADAR_START,
                            _RecordType.ITEM,
                            _RecordType.END,
                            _RecordType.ABORT,
                        }
                    )
                )
                if record.kind is _RecordType.RADAR_CONFIG:
                    self._accept_radar_config(record)
                    continue
                if record.kind is _RecordType.RADAR_START:
                    self._accept_radar_start(record)
                    continue
                if record.kind is _RecordType.ITEM:
                    yield self._accept_item(record)
                    continue
                if record.kind is _RecordType.END:
                    self._accept_end(record)
                    continue
                abort = self._accept_abort(record)
                raise MultisensorStreamAborted(abort)
        except (MultisensorStreamError, MultisensorStreamAborted):
            raise
        except (TypeError, ValueError, RecursionError) as exc:
            self._poison("mmwcli multi-sensor record is invalid", exc)
        self._state = "items_complete"

    def require_commit(self) -> MultisensorStreamCommit:
        """Require COMMIT, explicit EOF, and transport EOF before success."""

        self._require_state("items_complete", operation="require_commit")
        try:
            record = self._read_record(frozenset({_RecordType.COMMIT, _RecordType.ABORT}))
            if record.kind is _RecordType.ABORT:
                abort = self._accept_abort(record)
                raise MultisensorStreamAborted(abort)
            commit = self._accept_commit(record)
            self._require_terminal_eof()
        except (MultisensorStreamError, MultisensorStreamAborted):
            raise
        except (TypeError, ValueError, RecursionError) as exc:
            self._poison("mmwcli multi-sensor terminal is invalid", exc)
        self._state = "committed"
        return commit

    def _accept_radar_config(self, record: _Record) -> None:
        if self._radar_config is not None or self._records_started:
            raise ValueError("RADAR_CONFIG must be unique and precede ITEM/END records")
        _exact_keys(
            record.metadata,
            {"schema", "source_id", "format", "size_bytes", "sha256"},
            context="RADAR_CONFIG",
        )
        _literal(record.metadata, "schema", _RADAR_CONFIG_SCHEMA, "RADAR_CONFIG")
        source_id = _string(record.metadata, "source_id", "RADAR_CONFIG")
        progress = self._progress.get(source_id)
        if progress is None or progress.contract.kind != "radar":
            raise ValueError("RADAR_CONFIG source is not a declared radar")
        format_name = _format(record.metadata.get("format"), "RADAR_CONFIG.format")
        size = _uint(
            record.metadata,
            "size_bytes",
            1,
            _MAX_RADAR_CONFIG_BYTES,
            "RADAR_CONFIG",
        )
        digest = _digest(record.metadata.get("sha256"), "RADAR_CONFIG.sha256")
        if size != len(record.payload) or not hmac.compare_digest(
            digest, hashlib.sha256(record.payload).hexdigest()
        ):
            raise ValueError("RADAR_CONFIG size or SHA-256 does not match its payload")
        self._radar_config = MultisensorRadarConfig(
            source_id=source_id,
            format=format_name,
            payload=record.payload,
            sha256=digest,
        )

    def _accept_radar_start(self, record: _Record) -> None:
        if self._ending_started:
            raise ValueError("RADAR_START cannot follow the first source END")
        _exact_keys(
            record.metadata,
            {"schema", "source_id", "host_lower_ns", "host_upper_ns"},
            context="RADAR_START",
        )
        _literal(record.metadata, "schema", _RADAR_START_SCHEMA, "RADAR_START")
        source_id = _string(record.metadata, "source_id", "RADAR_START")
        progress = self._progress.get(source_id)
        if progress is None or progress.contract.kind != "radar":
            raise ValueError("RADAR_START source is not a declared radar")
        if progress.radar_start is not None or progress.next_item != 0 or progress.end is not None:
            raise ValueError("RADAR_START must be unique and precede its radar ITEM/END")
        lower = _uint(record.metadata, "host_lower_ns", 0, _MAX_U64, "RADAR_START")
        upper = _uint(record.metadata, "host_upper_ns", 0, _MAX_U64, "RADAR_START")
        if lower > upper:
            raise ValueError("RADAR_START host bounds are reversed")
        progress.radar_start = MappedTimeInterval(lower, upper)

    def _accept_item(self, record: _Record) -> ProvisionalMultisensorItem:
        if self._ending_started:
            raise ValueError("ITEM cannot follow the first source END")
        _exact_keys(
            record.metadata,
            {
                "schema",
                "source_id",
                "item_index",
                "provisional",
                "tick",
                "wrap_count",
                "duration_ticks",
                "sync_event_id",
            },
            context="ITEM",
        )
        _literal(record.metadata, "schema", _ITEM_SCHEMA, "ITEM")
        if record.metadata.get("provisional") is not True:
            raise ValueError("ITEM.provisional must be true")
        source_id = _string(record.metadata, "source_id", "ITEM")
        progress = self._progress.get(source_id)
        if progress is None or progress.end is not None:
            raise ValueError("ITEM source or state is invalid")
        index = _uint(record.metadata, "item_index", 0, _MAX_U64, "ITEM")
        tick = _uint(record.metadata, "tick", 0, _MAX_U64, "ITEM")
        wrap_count = _uint(record.metadata, "wrap_count", 0, _MAX_U64, "ITEM")
        duration = _uint(record.metadata, "duration_ticks", 0, _MAX_U64, "ITEM")
        sync_id = _uint(record.metadata, "sync_event_id", 0, _MAX_U64, "ITEM")
        source = progress.contract
        if index != progress.next_item:
            raise ValueError(f"source {source_id!r} item_index is not the next value")
        if (
            progress.next_item >= source.max_items
            or len(record.payload) > source.max_item_bytes
            or len(record.payload) > source.max_payload_bytes - progress.payload_bytes
        ):
            raise ValueError(f"source {source_id!r} ITEM exceeds its static limits")
        if source.timestamp_semantics == "delivery_observed" and duration != 0:
            raise ValueError("delivery_observed ITEM.duration_ticks must be zero")
        unwrapped = _unwrap_ticks(tick, wrap_count, source.wrap_ticks, duration)
        mapped_time = _map_live_item(progress, unwrapped, duration)
        progress.payload_hash.update(record.payload)  # type: ignore[attr-defined]
        progress.payload_bytes += len(record.payload)
        progress.next_item += 1
        self._records_started = True
        return ProvisionalMultisensorItem(
            session_id=self._required_contract().session_id,
            source_id=source_id,
            kind=source.kind,
            payload_format=source.payload_format,
            record_sequence=record.sequence,
            item_index=index,
            payload=record.payload,
            tick=tick,
            wrap_count=wrap_count,
            duration_ticks=duration,
            sync_event_id=None if sync_id == _NO_SYNC_EVENT else sync_id,
            tick_hz=source.tick_hz,
            wrap_ticks=source.wrap_ticks,
            timestamp_semantics=source.timestamp_semantics,
            mapped_time=mapped_time,
        )

    def _accept_end(self, record: _Record) -> None:
        _exact_keys(
            record.metadata,
            {
                "schema",
                "source_id",
                "outcome",
                "item_count",
                "payload_bytes",
                "payload_sha256",
            },
            context="END",
        )
        _literal(record.metadata, "schema", _END_SCHEMA, "END")
        source_id = _string(record.metadata, "source_id", "END")
        progress = self._progress.get(source_id)
        if progress is None or progress.end is not None:
            raise ValueError("END source or state is invalid")
        outcome = _closed_string(record.metadata, "outcome", _SOURCE_OUTCOMES, "END")
        count = _uint(record.metadata, "item_count", 0, _MAX_U64, "END")
        payload_bytes = _uint(record.metadata, "payload_bytes", 0, _MAX_U64, "END")
        digest = _digest(record.metadata.get("payload_sha256"), "END.payload_sha256")
        actual_digest = progress.payload_hash.hexdigest()  # type: ignore[attr-defined]
        if (
            count != progress.next_item
            or payload_bytes != progress.payload_bytes
            or not hmac.compare_digest(digest, actual_digest)
        ):
            raise ValueError("END does not match provisional source payload")
        progress.end = MultisensorSourceEnd(
            source_id=source_id,
            outcome=outcome,
            item_count=count,
            payload_bytes=payload_bytes,
            payload_sha256=digest,
        )
        self._records_started = True
        self._ending_started = True

    def _accept_commit(self, record: _Record) -> MultisensorStreamCommit:
        _exact_keys(
            record.metadata,
            {
                "schema",
                "session_id",
                "outcome",
                "session_json_size_bytes",
                "session_json_sha256",
            },
            context="COMMIT",
        )
        _literal(
            record.metadata,
            "schema",
            MMWCLI_MULTISENSOR_STREAM_TERMINAL_SCHEMA_V1,
            "COMMIT",
        )
        _literal(record.metadata, "outcome", "commit", "COMMIT")
        session_id = _string(record.metadata, "session_id", "COMMIT")
        contract = self._required_contract()
        if session_id != contract.session_id:
            raise ValueError("COMMIT session_id does not match SESSION")
        size = _uint(
            record.metadata,
            "session_json_size_bytes",
            1,
            _MAX_METADATA_BYTES,
            "COMMIT",
        )
        digest = _digest(
            record.metadata.get("session_json_sha256"),
            "COMMIT.session_json_sha256",
        )
        ends: list[MultisensorSourceEnd] = []
        for source in contract.sources:
            end = self._progress[source.source_id].end
            if end is None:
                raise ValueError("COMMIT requires END from every source")
            if source.required and end.outcome != "complete":
                raise ValueError("COMMIT requires every required source to be complete")
            ends.append(end)
        return MultisensorStreamCommit(
            session_id=session_id,
            session_json_size_bytes=size,
            session_json_sha256=digest,
            sources=tuple(ends),
        )

    def _accept_abort(self, record: _Record) -> MultisensorStreamAbort:
        _exact_keys(
            record.metadata,
            {"schema", "session_id", "outcome", "reason_code"},
            context="ABORT",
        )
        _literal(
            record.metadata,
            "schema",
            MMWCLI_MULTISENSOR_STREAM_TERMINAL_SCHEMA_V1,
            "ABORT",
        )
        _literal(record.metadata, "outcome", "abort", "ABORT")
        session_id = _string(record.metadata, "session_id", "ABORT")
        if session_id != self._required_contract().session_id:
            raise ValueError("ABORT session_id does not match SESSION")
        reason = _closed_string(record.metadata, "reason_code", _ABORT_REASONS, "ABORT")
        self._require_terminal_eof()
        self._state = "aborted"
        return MultisensorStreamAbort(session_id=session_id, reason_code=reason)

    def _require_terminal_eof(self) -> None:
        record = self._read_record(frozenset({_RecordType.EOF}))
        _exact_keys(record.metadata, {"schema", "session_id"}, context="EOF")
        _literal(record.metadata, "schema", _EOF_SCHEMA, "EOF")
        if _string(record.metadata, "session_id", "EOF") != self._required_contract().session_id:
            raise ValueError("EOF session_id does not match SESSION")
        try:
            trailing = self._source.read(1)
        except Exception as exc:
            self._poison("mmwcli multi-sensor final EOF read failed", exc)
        if type(trailing) is not bytes:
            self._poison("mmwcli multi-sensor source read() must return bytes")
        if len(trailing) > 1:
            self._poison("mmwcli multi-sensor source returned too many bytes")
        if trailing:
            self._poison("mmwcli multi-sensor stream has trailing data after EOF")

    def _read_record(self, allowed: frozenset[_RecordType]) -> _Record:  # noqa: C901
        header = self._read_exact(_HEADER_SIZE, label="record header")
        try:
            (
                magic,
                major,
                header_size,
                kind_value,
                flags,
                sequence,
                metadata_size,
                payload_size,
                reserved,
                digest,
            ) = _HEADER.unpack(header)
            kind = _RecordType(kind_value)
        except (struct.error, ValueError) as exc:
            self._poison("mmwcli multi-sensor record header is invalid", exc)
        if magic != _MAGIC or major != _PROTOCOL_MAJOR or header_size != _HEADER_SIZE:
            self._poison("mmwcli multi-sensor record framing is unsupported")
        if flags != 0 or reserved != 0:
            self._poison("mmwcli multi-sensor record flags or reserved bytes are nonzero")
        if kind not in allowed:
            self._poison(f"mmwcli multi-sensor record type {kind.name} is out of order")
        if sequence != self._next_record or sequence == _MAX_U64:
            self._poison("mmwcli multi-sensor record sequence is not the next zero-based value")
        if not 1 <= metadata_size <= _MAX_METADATA_BYTES:
            self._poison("mmwcli multi-sensor record metadata size is invalid")
        if kind is _RecordType.RADAR_CONFIG:
            valid_payload = 1 <= payload_size <= _MAX_RADAR_CONFIG_BYTES
        elif kind is _RecordType.ITEM:
            valid_payload = 1 <= payload_size <= _MAX_ITEM_BYTES
        else:
            valid_payload = payload_size == 0
        if not valid_payload:
            self._poison("mmwcli multi-sensor record payload size is invalid")
        metadata_bytes = self._read_exact(metadata_size, label="record metadata")
        payload = self._read_exact(payload_size, label=f"{kind.name} payload")
        expected = hashlib.sha256(
            _RECORD_DIGEST_DOMAIN + header[:_HEADER_PREFIX_SIZE] + metadata_bytes + payload
        ).digest()
        if not hmac.compare_digest(digest, expected):
            self._poison("mmwcli multi-sensor record SHA-256 is invalid")
        try:
            metadata = _strict_json_object(metadata_bytes, context=f"{kind.name} metadata")
        except (TypeError, ValueError, RecursionError) as exc:
            self._poison("mmwcli multi-sensor record metadata is invalid", exc)
        self._next_record += 1
        return _Record(kind=kind, sequence=sequence, metadata=metadata, payload=payload)

    def _read_exact(self, size: int, *, label: str) -> bytes:
        payload = bytearray()
        while len(payload) < size:
            requested = size - len(payload)
            try:
                chunk = self._source.read(requested)
            except Exception as exc:
                self._poison(f"mmwcli multi-sensor {label} read failed", exc)
            if type(chunk) is not bytes:
                self._poison("mmwcli multi-sensor source read() must return bytes")
            if len(chunk) > requested:
                self._poison("mmwcli multi-sensor source returned more bytes than requested")
            if not chunk:
                self._poison(f"mmwcli multi-sensor stream ended during {label}")
            payload.extend(chunk)
        return bytes(payload)

    def _all_sources_ended(self) -> bool:
        return bool(self._progress) and all(
            item.end is not None for item in self._progress.values()
        )

    def _required_contract(self) -> MultisensorStreamContract:
        if self._contract is None:
            raise MultisensorStreamStateError("MultisensorStreamReader contract is unavailable.")
        return self._contract

    def _require_state(self, expected: str, *, operation: str) -> None:
        if self._state == "poisoned":
            assert self._failure is not None
            raise self._failure
        if self._state != expected:
            raise MultisensorStreamStateError(
                f"MultisensorStreamReader.{operation} is invalid in state {self._state!r}."
            )

    def _poison(self, message: str, cause: BaseException | None = None) -> NoReturn:
        error = MultisensorStreamError(f"{message}.")
        self._state = "poisoned"
        self._failure = error
        if cause is None:
            raise error
        raise error from cause


@dataclass(frozen=True)
class MultisensorStream:
    """Pull facade over one finite aggregate stream without IO ownership."""

    contract: MultisensorStreamContract
    _reader: MultisensorStreamReader = field(repr=False, compare=False)

    def source(self, source_id: str) -> MultisensorStreamSource:
        return self.contract.source(source_id)

    def items(self) -> Iterator[ProvisionalMultisensorItem]:
        return self._reader.provisional_items()

    @property
    def radar_config(self) -> MultisensorRadarConfig | None:
        return self._reader.radar_config

    def require_commit(self) -> MultisensorStreamCommit:
        return self._reader.require_commit()


def open_multisensor_stream(source: BinaryIO) -> MultisensorStream:
    """Read SESSION now and return a caller-owned, pull-driven stream facade."""

    reader = MultisensorStreamReader(source)
    contract = reader.read_contract()
    return MultisensorStream(contract=contract, _reader=reader)


def _parse_session(record: dict[str, object]) -> MultisensorStreamContract:
    _exact_keys(
        record,
        {"schema", "session_id", "synchronization_grade", "sources"},
        context="SESSION",
    )
    _literal(record, "schema", MMWCLI_MULTISENSOR_STREAM_SCHEMA_V1, "SESSION")
    session_id = _string(record, "session_id", "SESSION")
    if _SESSION_ID.fullmatch(session_id) is None:
        raise ValueError("SESSION.session_id must be a lowercase UUIDv4")
    _literal(record, "synchronization_grade", "software_barrier", "SESSION")
    raw_sources = record.get("sources")
    if not isinstance(raw_sources, list) or not 1 <= len(raw_sources) <= _MAX_SOURCES:
        raise ValueError(f"SESSION.sources must contain 1..{_MAX_SOURCES} entries")
    sources = tuple(_parse_source(value, index) for index, value in enumerate(raw_sources))
    source_ids = [source.source_id for source in sources]
    clock_ids = [source.clock_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("SESSION.sources contains duplicate source_id values")
    if len(set(clock_ids)) != len(clock_ids):
        raise ValueError("SESSION.sources contains duplicate clock_id values")
    return MultisensorStreamContract(
        session_id=session_id,
        synchronization_grade="software_barrier",
        sources=sources,
    )


def _parse_source(value: object, index: int) -> MultisensorStreamSource:
    if not isinstance(value, dict):
        raise ValueError(f"SESSION source {index} must be an object")
    _exact_keys(
        value,
        {"source_id", "kind", "required", "payload", "clock", "limits"},
        context=f"source {index}",
    )
    source_id = _string(value, "source_id", f"source {index}")
    if _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError(f"source_id {source_id!r} is invalid")
    kind = _closed_string(value, "kind", _SOURCE_KINDS, f"source {source_id}")
    required = value.get("required")
    if type(required) is not bool:
        raise ValueError(f"source {source_id}.required must be a boolean")
    payload = value.get("payload")
    clock = value.get("clock")
    limits = value.get("limits")
    if not isinstance(payload, dict) or not isinstance(clock, dict) or not isinstance(limits, dict):
        raise ValueError(f"source {source_id} nested contracts must be objects")
    _exact_keys(payload, {"filename", "format"}, context=f"source {source_id}.payload")
    _exact_keys(
        clock,
        {"clock_id", "tick_hz", "wrap_ticks", "timestamp_semantics"},
        context=f"source {source_id}.clock",
    )
    _exact_keys(
        limits,
        {"max_items", "max_item_bytes", "max_payload_bytes"},
        context=f"source {source_id}.limits",
    )
    filename = _leaf(payload.get("filename"), f"source {source_id}.payload.filename")
    if filename == "index.bin":
        raise ValueError("stream payload filename cannot be index.bin")
    payload_format = _format(payload.get("format"), f"source {source_id}.payload.format")
    clock_id = _opaque(clock.get("clock_id"), f"source {source_id}.clock_id")
    tick_hz = _uint(clock, "tick_hz", 1, _MAX_U64, f"source {source_id}.clock")
    wrap_ticks = _uint(clock, "wrap_ticks", 0, _MAX_U64, f"source {source_id}.clock")
    if wrap_ticks == 1:
        raise ValueError("source clock wrap_ticks must be zero or greater than one")
    semantics = _string(clock, "timestamp_semantics", f"source {source_id}.clock")
    _validate_source_clock(source_id, kind, clock_id, tick_hz, wrap_ticks, semantics)
    max_items = _uint(limits, "max_items", 1, _MAX_ITEMS, f"source {source_id}.limits")
    max_item_bytes = _uint(
        limits,
        "max_item_bytes",
        1,
        _MAX_ITEM_BYTES,
        f"source {source_id}.limits",
    )
    max_payload_bytes = _uint(
        limits,
        "max_payload_bytes",
        1,
        _MAX_PAYLOAD_BYTES,
        f"source {source_id}.limits",
    )
    if max_item_bytes > max_payload_bytes:
        raise ValueError("source max_item_bytes exceeds max_payload_bytes")
    return MultisensorStreamSource(
        source_id=source_id,
        kind=kind,
        required=required,
        payload_filename=filename,
        payload_format=payload_format,
        clock_id=clock_id,
        tick_hz=tick_hz,
        wrap_ticks=wrap_ticks,
        timestamp_semantics=semantics,
        max_items=max_items,
        max_item_bytes=max_item_bytes,
        max_payload_bytes=max_payload_bytes,
    )


def _validate_source_clock(
    source_id: str,
    kind: str,
    clock_id: str,
    tick_hz: int,
    wrap_ticks: int,
    semantics: str,
) -> None:
    if kind == "radar" and semantics != "frame_start":
        raise ValueError(f"{kind} source clock timestamp semantics are invalid")
    if kind == "camera" and semantics not in {"exposure_midpoint", "delivery_observed"}:
        raise ValueError(f"{kind} source clock timestamp semantics are invalid")
    if semantics == "delivery_observed" and (
        clock_id != f"{source_id}-delivery-observed" or tick_hz != 1_000_000_000 or wrap_ticks != 0
    ):
        raise ValueError(
            "delivery_observed camera clock must use its dedicated clock_id, "
            "tick_hz=1000000000, and wrap_ticks=0"
        )


def _strict_json_object(encoded: bytes, *, context: str) -> dict[str, object]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{context} is not UTF-8") from exc

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{context} contains non-finite number {value}")

    value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be one JSON object")
    return value


def _exact_keys(record: dict[str, object], expected: set[str], *, context: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{context} has an invalid exact key set")


def _literal(record: dict[str, object], field_name: str, expected: str, context: str) -> None:
    if record.get(field_name) != expected:
        raise ValueError(f"{context}.{field_name} is invalid")


def _string(record: dict[str, object], field_name: str, context: str) -> str:
    value = record.get(field_name)
    if type(value) is not str:
        raise ValueError(f"{context}.{field_name} must be a string")
    return value


def _closed_string(
    record: dict[str, object],
    field_name: str,
    allowed: frozenset[str],
    context: str,
) -> str:
    value = _string(record, field_name, context)
    if value not in allowed:
        raise ValueError(f"{context}.{field_name} is unsupported")
    return value


def _uint(
    record: dict[str, object],
    field_name: str,
    minimum: int,
    maximum: int,
    context: str,
) -> int:
    value = record.get(field_name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{context}.{field_name} is outside {minimum}..{maximum}")
    return value


def _digest(value: object, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal digits")
    return value


def _leaf(value: object, context: str) -> str:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= 128:
        raise ValueError(f"{context} must contain 1..128 UTF-8 bytes")
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or value.endswith(".part")
    ):
        raise ValueError(f"{context} must be a safe published leaf")
    return value


def _format(value: object, context: str) -> str:
    if type(value) is not str or _PAYLOAD_FORMAT.fullmatch(value) is None:
        raise ValueError(f"{context} is invalid")
    return value


def _opaque(value: object, context: str) -> str:
    if type(value) is not str or _OPAQUE_ID.fullmatch(value) is None:
        raise ValueError(f"{context} is invalid")
    return value


def _unwrap_ticks(tick: int, wrap_count: int, wrap_ticks: int, duration: int = 0) -> int:
    if wrap_ticks == 0:
        if wrap_count != 0:
            raise ValueError("wrap_count must be zero for a non-wrapping clock")
        unwrapped = tick
    else:
        if tick >= wrap_ticks or wrap_count > _MAX_U64 // wrap_ticks:
            raise ValueError("source clock wrap arithmetic is invalid")
        unwrapped = wrap_count * wrap_ticks + tick
        if unwrapped > _MAX_U64:
            raise ValueError("source clock wrap arithmetic overflows uint64")
    if duration > _MAX_U64 - unwrapped:
        raise ValueError("source item duration overflows uint64")
    return unwrapped


def _map_live_item(
    progress: _SourceProgress,
    unwrapped_tick: int,
    duration_ticks: int,
) -> MappedTimeInterval | None:
    source = progress.contract
    if source.kind == "camera":
        if source.timestamp_semantics == "delivery_observed":
            return MappedTimeInterval(unwrapped_tick, unwrapped_tick)
        return None
    radar_start = progress.radar_start
    if radar_start is None:
        raise ValueError("radar ITEM requires a preceding RADAR_START")
    start_offset = _ticks_to_nanoseconds(
        unwrapped_tick,
        source.tick_hz,
        round_up=False,
    )
    end_offset = _ticks_to_nanoseconds(
        unwrapped_tick + duration_ticks,
        source.tick_hz,
        round_up=True,
    )
    return MappedTimeInterval(
        _checked_u64_add(radar_start.start_ns, start_offset, "radar mapped start"),
        _checked_u64_add(radar_start.end_ns, end_offset, "radar mapped end"),
    )


def _ticks_to_nanoseconds(tick: int, tick_hz: int, *, round_up: bool) -> int:
    if tick > _MAX_U64 // _NANOSECONDS_PER_SECOND:
        raise ValueError("radar tick-to-nanosecond multiplication overflows uint64")
    quotient, remainder = divmod(tick * _NANOSECONDS_PER_SECOND, tick_hz)
    if round_up and remainder:
        if quotient == _MAX_U64:
            raise ValueError("radar tick-to-nanosecond rounding overflows uint64")
        quotient += 1
    return quotient


def _checked_u64_add(left: int, right: int, context: str) -> int:
    if right > _MAX_U64 - left:
        raise ValueError(f"{context} addition overflows uint64")
    return left + right


__all__ = [
    "MMWCLI_MULTISENSOR_STREAM_SCHEMA_V1",
    "MMWCLI_MULTISENSOR_STREAM_TERMINAL_SCHEMA_V1",
    "MultisensorRadarConfig",
    "MultisensorSourceEnd",
    "MultisensorStream",
    "MultisensorStreamAbort",
    "MultisensorStreamAborted",
    "MultisensorStreamCommit",
    "MultisensorStreamContract",
    "MultisensorStreamError",
    "MultisensorStreamReader",
    "MultisensorStreamSource",
    "MultisensorStreamStateError",
    "ProvisionalMultisensorItem",
    "ProvisionalMultisensorRangeDoppler",
    "open_multisensor_stream",
]
