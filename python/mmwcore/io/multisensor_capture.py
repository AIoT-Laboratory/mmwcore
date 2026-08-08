"""Read strict published mmwcli multi-sensor sessions for offline training."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import stat
import struct
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, NoReturn, cast

from mmwcore.core import RangeDopplerRecipe

from .mmwcli_capture import ADCFileCapture, RangeDopplerPreset
from .mmwcli_capture import open_capture as _open_mmwcli_capture

MMWCLI_MULTISENSOR_SESSION_SCHEMA_V1 = "mmwcli.multisensor_session.v1"
MMWCLI_SENSOR_INDEX_SCHEMA_V1 = "mmwcli.sensor_index.v1"

_SESSION_NAME = "session.json"
_SENSORS_NAME = "sensors"
_INDEX_NAME = "index.bin"
_INDEX_MAGIC = b"MMWSIDX1"
_INDEX_HEADER = struct.Struct("<8sHHHHQQ")
_INDEX_ENTRY = struct.Struct("<QQQQQQQII")
_INDEX_HEADER_BYTES = 32
_INDEX_ENTRY_BYTES = 64
_NO_SYNC_EVENT = (1 << 64) - 1
_MAX_U64 = (1 << 64) - 1

_MAX_SESSION_BYTES = 1 << 20
_MAX_SOURCES = 32
_MAX_ITEMS = 1 << 24
_MAX_ITEM_BYTES = 1 << 34
_MAX_PAYLOAD_BYTES = 1 << 50
_MAX_ARTIFACTS = 16
_MAX_CLOCK_OBSERVATIONS = 4096
_MAX_AFFINE_SEGMENTS = 1024
_MAX_SYNC_EVENTS = 1 << 20
_MAX_METADATA_ENTRIES = 32
_MAX_METADATA_BYTES = 64 << 10
_MAX_METADATA_DEPTH = 16
_MAX_METADATA_KEY_BYTES = 128

_SESSION_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_SOURCE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_KINDS = frozenset({"radar", "camera"})
_OUTCOMES = frozenset({"complete", "failed", "omitted"})
_GRADES = frozenset({"software_barrier"})
_TIMESTAMP_SEMANTICS = {"radar": "frame_start", "camera": "exposure_midpoint"}
_ARTIFACT_ROLES = frozenset({"payload", "index", "configuration", "manifest", "metadata"})
_SYNC_EDGES = frozenset({"rising", "falling"})
_EVIDENCE_KINDS = frozenset({"trigger_generation", "hardware_observation"})
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_PAYLOAD_FORMAT = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_METADATA_KEY = re.compile(r"[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*){2,}\Z")

type TrainingKey = tuple[str, str, int] | tuple[str, str, int, int]


@dataclass(frozen=True)
class MappedTimeInterval:
    """Conservative host-monotonic interval in integer nanoseconds."""

    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if type(self.start_ns) is not int or type(self.end_ns) is not int:
            raise TypeError("MappedTimeInterval bounds must be integers.")
        if self.start_ns > self.end_ns:
            raise ValueError("MappedTimeInterval start_ns must not exceed end_ns.")


@dataclass(frozen=True)
class MultisensorItem:
    """One indexed payload item with immutable training and time identity."""

    session_id: str
    source_id: str
    item_index: int
    payload: bytes = field(repr=False)
    payload_offset: int
    source_ticks: int
    wrap_count: int
    duration_ticks: int
    sync_event_id: int | None
    mapped_time: MappedTimeInterval

    @property
    def training_key(self) -> TrainingKey:
        """Return the only stable offline join key defined by the contract."""

        base = (self.session_id, self.source_id, self.item_index)
        if self.sync_event_id is None:
            return base
        return (*base, self.sync_event_id)


@dataclass(frozen=True)
class MultisensorSyncEvent:
    """One physical synchronization event and its mapped host interval."""

    sync_event_id: int
    clock_id: str
    edge: str
    evidence_kind: str
    generator: str
    observer: str
    routing_id: str
    mapped_time: MappedTimeInterval


@dataclass(frozen=True)
class _Clock:
    clock_id: str
    tick_hz: int
    wrap_ticks: int
    timestamp_semantics: str


@dataclass(frozen=True)
class _AffineSegment:
    start_tick: int
    end_tick: int
    source_origin_tick: int
    host_origin_ns: int
    scale_num: int
    scale_den: int
    uncertainty_ns: int


@dataclass(frozen=True)
class _ClockObservation:
    ticks: int
    wrap_count: int
    host_before_ns: int
    host_after_ns: int


@dataclass(frozen=True)
class _Artifact:
    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class _EventCardinality:
    required: bool
    minimum: int
    maximum: int


@dataclass(frozen=True)
class _SourceContract:
    source_id: str
    kind: str
    required: bool
    outcome: str
    producer_name: str
    producer_version: str
    max_items: int
    max_item_bytes: int
    payload_filename: str
    payload_format: str
    item_count: int
    payload_bytes: int
    clock: _Clock
    observation_ids: frozenset[str]
    segments: tuple[_AffineSegment, ...]
    cardinality: _EventCardinality | None
    artifacts: tuple[_Artifact, ...]


@dataclass(frozen=True)
class _IndexEntry:
    item_index: int
    payload_offset: int
    payload_size: int
    ticks: int
    wrap_count: int
    duration_ticks: int
    sync_event_id: int | None


@dataclass(frozen=True)
class MultisensorSource:
    """One immutable source contract with lazy, integrity-bound item access."""

    session_id: str
    source_id: str
    kind: str
    required: bool
    outcome: str
    producer_name: str
    producer_version: str
    payload_format: str
    item_count: int
    payload_bytes: int
    clock_id: str
    tick_hz: int
    wrap_ticks: int
    timestamp_semantics: str
    payload_path: Path | None
    index_path: Path | None
    _max_item_bytes: int = field(repr=False, compare=False)
    _segments: tuple[_AffineSegment, ...] = field(repr=False, compare=False)
    _event_ids: frozenset[int] = field(repr=False, compare=False)

    def open_radar_capture(
        self,
        *,
        range_doppler: RangeDopplerRecipe | RangeDopplerPreset | None = None,
    ) -> ADCFileCapture:
        """Open this source's nested capture session with an explicit DSP policy."""

        if self.kind != "radar":
            raise ValueError("Only a radar multisensor source has a nested ADC capture.")
        if self.outcome != "complete":
            raise ValueError("Only a complete radar multisensor source can be opened.")
        payload_path, index_path = self._required_paths()
        capture = _open_mmwcli_capture(
            payload_path.parent,
            range_doppler=range_doppler,
        )
        _validate_nested_radar_capture(
            source=self,
            payload_path=payload_path,
            index_path=index_path,
            capture=capture,
        )
        return capture

    def items(self) -> Iterator[MultisensorItem]:
        """Yield every payload item in validated index order."""

        if self.outcome != "complete":
            return iter(())
        return self._iterate_items()

    def _iterate_items(self) -> Iterator[MultisensorItem]:
        payload_path, index_path = self._required_paths()
        _require_file_size(payload_path, self.payload_bytes, "source payload")
        _require_file_size(index_path, _checked_index_size(self.item_count), "sensor index")
        with index_path.open("rb") as index_file, payload_path.open("rb") as payload_file:
            _read_index_header(
                index_file,
                expected_items=self.item_count,
                expected_payload_bytes=self.payload_bytes,
            )
            for expected_index in range(self.item_count):
                entry = _decode_index_entry(
                    _read_exact(index_file, _INDEX_ENTRY_BYTES, "sensor index entry"),
                    expected_index=expected_index,
                    max_item_bytes=self._max_item_bytes,
                )
                payload_file.seek(entry.payload_offset)
                payload = _read_exact(payload_file, entry.payload_size, "sensor payload item")
                yield self._item(entry, payload)

    def read_item(self, index: int) -> MultisensorItem:
        """Read one payload item by its zero-based contract index."""

        if type(index) is not int:
            raise TypeError("MultisensorSource item index must be an integer.")
        if not 0 <= index < self.item_count:
            raise IndexError(
                f"MultisensorSource item index {index} is outside [0, {self.item_count})."
            )
        payload_path, index_path = self._required_paths()
        _require_file_size(payload_path, self.payload_bytes, "source payload")
        _require_file_size(index_path, _checked_index_size(self.item_count), "sensor index")
        with index_path.open("rb") as index_file:
            _read_index_header(
                index_file,
                expected_items=self.item_count,
                expected_payload_bytes=self.payload_bytes,
            )
            index_file.seek(_INDEX_HEADER_BYTES + index * _INDEX_ENTRY_BYTES)
            entry = _decode_index_entry(
                _read_exact(index_file, _INDEX_ENTRY_BYTES, "sensor index entry"),
                expected_index=index,
                max_item_bytes=self._max_item_bytes,
            )
        with payload_path.open("rb") as payload_file:
            payload_file.seek(entry.payload_offset)
            payload = _read_exact(payload_file, entry.payload_size, "sensor payload item")
        return self._item(entry, payload)

    def _required_paths(self) -> tuple[Path, Path]:
        if self.payload_path is None or self.index_path is None:
            raise RuntimeError("Only complete multisensor sources have readable items.")
        return self.payload_path, self.index_path

    def _item(self, entry: _IndexEntry, payload: bytes) -> MultisensorItem:
        if entry.sync_event_id is not None and entry.sync_event_id not in self._event_ids:
            raise ValueError("Sensor item references an unknown synchronization event.")
        clock = _Clock(self.clock_id, self.tick_hz, self.wrap_ticks, self.timestamp_semantics)
        return MultisensorItem(
            session_id=self.session_id,
            source_id=self.source_id,
            item_index=entry.item_index,
            payload=payload,
            payload_offset=entry.payload_offset,
            source_ticks=entry.ticks,
            wrap_count=entry.wrap_count,
            duration_ticks=entry.duration_ticks,
            sync_event_id=entry.sync_event_id,
            mapped_time=_map_item_interval(clock, self._segments, entry),
        )


def _validate_nested_radar_capture(
    *,
    source: MultisensorSource,
    payload_path: Path,
    index_path: Path,
    capture: ADCFileCapture,
) -> None:
    if capture.root != payload_path.parent or capture.adc_path != payload_path:
        raise ValueError(
            "Nested radar capture adc.bin does not match the multisensor payload artifact."
        )
    if capture.num_frames != source.item_count:
        raise ValueError(
            "Nested radar capture frame count does not match the multisensor source index."
        )
    frame_bytes = capture.radar_capture.adc.raw_values_per_frame * 2
    if (
        capture.radar_capture.expected_size_bytes != source.payload_bytes
        or frame_bytes * source.item_count != source.payload_bytes
    ):
        raise ValueError("Nested radar capture frame size does not match the multisensor payload.")

    _require_file_size(payload_path, source.payload_bytes, "radar source payload")
    _require_file_size(
        index_path,
        _checked_index_size(source.item_count),
        "radar source index",
    )
    with index_path.open("rb") as index_file:
        _read_index_header(
            index_file,
            expected_items=source.item_count,
            expected_payload_bytes=source.payload_bytes,
        )
        for expected_index in range(source.item_count):
            entry = _decode_index_entry(
                _read_exact(index_file, _INDEX_ENTRY_BYTES, "radar source index entry"),
                expected_index=expected_index,
                max_item_bytes=source._max_item_bytes,
            )
            if (
                entry.payload_offset != expected_index * frame_bytes
                or entry.payload_size != frame_bytes
            ):
                raise ValueError(
                    "Multisensor radar index offset/size does not match nested ADC frames."
                )
        if index_file.read(1):
            raise ValueError("Multisensor radar index contains trailing data.")


@dataclass(frozen=True)
class MultisensorCapture:
    """Validated, read-only mmwcli multi-sensor session."""

    root: Path
    session_path: Path
    session_id: str
    synchronization_grade: str
    host_clock_id: str
    sources: tuple[MultisensorSource, ...]
    sync_events: tuple[MultisensorSyncEvent, ...]
    item_count: int
    payload_bytes: int

    def source(self, source_id: str) -> MultisensorSource:
        if type(source_id) is not str:
            raise TypeError("MultisensorCapture source_id must be a string.")
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(source_id)

    def items(self) -> Iterator[MultisensorItem]:
        for source in self.sources:
            yield from source.items()

    def causal_pairs(
        self,
        earlier_source_id: str,
        later_source_id: str,
        *,
        lag_min_ns: int,
        lag_max_ns: int,
    ) -> Iterator[tuple[MultisensorItem, MultisensorItem]]:
        """Yield conservative cross-source pairs intersecting one causal lag window.

        Sources must be distinct and complete. Items are consumed lazily in mapped-time
        order; a source whose mapped intervals move backwards is rejected.
        """

        _validate_causal_lag_window(lag_min_ns, lag_max_ns)
        earlier_source = self.source(earlier_source_id)
        later_source = self.source(later_source_id)
        if earlier_source.source_id == later_source.source_id:
            raise ValueError("Causal pairs require two distinct multisensor sources.")
        for role, source in (("earlier", earlier_source), ("later", later_source)):
            if source.outcome != "complete":
                raise ValueError(
                    f"Causal pairs require a complete {role} source; "
                    f"{source.source_id!r} has outcome {source.outcome!r}."
                )
        return _iterate_causal_pairs(
            earlier_source.items(),
            later_source.items(),
            lag_min_ns=lag_min_ns,
            lag_max_ns=lag_max_ns,
        )


def causal_match(
    earlier: MultisensorItem | MappedTimeInterval,
    later: MultisensorItem | MappedTimeInterval,
    *,
    lag_min_ns: int,
    lag_max_ns: int,
) -> bool:
    """Return whether the possible physical lag intersects a causal window."""

    _validate_causal_lag_window(lag_min_ns, lag_max_ns)
    first = earlier.mapped_time if isinstance(earlier, MultisensorItem) else earlier
    second = later.mapped_time if isinstance(later, MultisensorItem) else later
    if not isinstance(first, MappedTimeInterval) or not isinstance(second, MappedTimeInterval):
        raise TypeError("causal_match requires multisensor items or mapped time intervals.")
    possible_min = second.start_ns - first.end_ns
    possible_max = second.end_ns - first.start_ns
    return possible_min <= lag_max_ns and possible_max >= lag_min_ns


def _validate_causal_lag_window(lag_min_ns: int, lag_max_ns: int) -> None:
    if type(lag_min_ns) is not int or type(lag_max_ns) is not int:
        raise TypeError("Causal lag bounds must be integers.")
    if lag_min_ns > lag_max_ns:
        raise ValueError("lag_min_ns must not exceed lag_max_ns.")


def _iterate_causal_pairs(
    earlier_items: Iterator[MultisensorItem],
    later_items: Iterator[MultisensorItem],
    *,
    lag_min_ns: int,
    lag_max_ns: int,
) -> Iterator[tuple[MultisensorItem, MultisensorItem]]:
    earlier = _monotonic_mapped_items(earlier_items, role="earlier")
    later = _monotonic_mapped_items(later_items, role="later")
    lookahead = next(earlier, None)
    candidates: deque[MultisensorItem] = deque()

    for later_item in later:
        minimum_earlier_end = later_item.mapped_time.start_ns - lag_max_ns
        maximum_earlier_start = later_item.mapped_time.end_ns - lag_min_ns

        # Retain only the temporal frontier that can intersect this or a future
        # lag window. One lookahead item prevents reading the remaining payload.
        while candidates and candidates[0].mapped_time.end_ns < minimum_earlier_end:
            candidates.popleft()
        while lookahead is not None and lookahead.mapped_time.start_ns <= maximum_earlier_start:
            if lookahead.mapped_time.end_ns >= minimum_earlier_end:
                candidates.append(lookahead)
            lookahead = next(earlier, None)

        for earlier_item in candidates:
            if causal_match(
                earlier_item,
                later_item,
                lag_min_ns=lag_min_ns,
                lag_max_ns=lag_max_ns,
            ):
                yield earlier_item, later_item


def _monotonic_mapped_items(
    items: Iterator[MultisensorItem],
    *,
    role: str,
) -> Iterator[MultisensorItem]:
    previous: MappedTimeInterval | None = None
    for item in items:
        current = item.mapped_time
        if previous is not None and (
            current.start_ns < previous.start_ns or current.end_ns < previous.end_ns
        ):
            raise ValueError(
                f"{role.capitalize()} source mapped interval order moves backwards "
                f"at item {item.item_index}."
            )
        previous = current
        yield item


def open_multisensor_capture(path: str | Path) -> MultisensorCapture:  # noqa: C901
    """Open one complete, integrity-checked multi-sensor session directory."""

    root = _session_root(path)
    session_path = _regular_leaf(root, _SESSION_NAME, "multisensor session manifest")
    session_bytes = _read_bounded_regular(
        session_path,
        maximum_bytes=_MAX_SESSION_BYTES,
        label="multisensor session manifest",
    )
    record = _strict_json_object(session_bytes, context="multisensor session manifest")
    if _json_depth(record) > 32:
        raise ValueError("multisensor session JSON nesting exceeds depth 32.")
    _exact_keys(
        record,
        {
            "schema",
            "session_id",
            "synchronization_grade",
            "host_clock",
            "sources",
            "sync_events",
            "totals",
            "application_metadata",
        },
        context="multisensor session manifest",
    )
    _literal(record, "schema", MMWCLI_MULTISENSOR_SESSION_SCHEMA_V1, "session.schema")
    session_id = _session_identifier(record.get("session_id"))
    grade = _closed_text(record, "synchronization_grade", _GRADES, "session")
    _application_metadata(record.get("application_metadata"), "session.application_metadata")

    host_clock = _parse_clock(_object(record, "host_clock", "session"), kind="host")
    if (
        host_clock.tick_hz != 1_000_000_000
        or host_clock.wrap_ticks != 0
        or host_clock.timestamp_semantics != "host_monotonic"
    ):
        raise ValueError("session.host_clock must be the 1 GHz non-wrapping host_monotonic clock.")

    source_values = _array(record, "sources", "session")
    if not 1 <= len(source_values) <= _MAX_SOURCES:
        raise ValueError(f"session.sources must contain 1..{_MAX_SOURCES} sources.")
    contracts = tuple(_parse_source(value) for value in source_values)
    source_ids = [source.source_id for source in contracts]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("session.sources contains duplicate source_id values.")
    clock_ids = [host_clock.clock_id, *(source.clock.clock_id for source in contracts)]
    if len(set(clock_ids)) != len(clock_ids):
        raise ValueError("session clock_id values must be unique.")

    observation_domains = {source.clock.clock_id: source.observation_ids for source in contracts}
    event_records = _array(record, "sync_events", "session")
    if len(event_records) > _MAX_SYNC_EVENTS:
        raise ValueError(f"session.sync_events exceeds the {_MAX_SYNC_EVENTS}-event limit.")
    raw_events = tuple(
        _parse_sync_event(
            value,
            known_clock_ids=frozenset(clock_ids),
            host_clock_id=host_clock.clock_id,
            observation_domains=observation_domains,
        )
        for value in event_records
    )
    event_ids = [cast(int, event["sync_event_id"]) for event in raw_events]
    if event_ids != sorted(event_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError("session.sync_events IDs must be unique and increasing.")
    known_event_ids = frozenset(event_ids)

    sensors_root = _directory_leaf(root, _SENSORS_NAME, "sensor directory")
    complete_ids = {source.source_id for source in contracts if source.outcome == "complete"}
    _require_directory_names(root, {_SESSION_NAME, _SENSORS_NAME}, "session root")
    _require_directory_names(sensors_root, complete_ids, "sensors directory")

    sources: list[MultisensorSource] = []
    event_counts: dict[str, Counter[int]] = {}
    for contract in contracts:
        source, counts = _open_source(
            root=sensors_root,
            session_id=session_id,
            contract=contract,
            event_ids=known_event_ids,
        )
        sources.append(source)
        event_counts[contract.source_id] = counts
    _validate_event_cardinalities(contracts, event_counts, known_event_ids, grade)

    events = tuple(
        _public_sync_event(event, host_clock=host_clock, sources=contracts) for event in raw_events
    )
    totals = _object(record, "totals", "session")
    _exact_keys(
        totals,
        {
            "source_count",
            "required_source_count",
            "complete_source_count",
            "item_count",
            "payload_bytes",
        },
        context="session.totals",
    )
    expected_totals = {
        "source_count": len(contracts),
        "required_source_count": sum(source.required for source in contracts),
        "complete_source_count": sum(source.outcome == "complete" for source in contracts),
        "item_count": sum(source.item_count for source in contracts),
        "payload_bytes": sum(source.payload_bytes for source in contracts),
    }
    for name, expected in expected_totals.items():
        if _uint(totals, name, 0, _MAX_U64, "session.totals") != expected:
            raise ValueError(f"session.totals.{name} does not match its source aggregate.")

    return MultisensorCapture(
        root=root,
        session_path=session_path,
        session_id=session_id,
        synchronization_grade=grade,
        host_clock_id=host_clock.clock_id,
        sources=tuple(sources),
        sync_events=events,
        item_count=expected_totals["item_count"],
        payload_bytes=expected_totals["payload_bytes"],
    )


def _parse_source(value: object) -> _SourceContract:
    if not isinstance(value, dict):
        raise ValueError("session.sources entries must be JSON objects.")
    required_keys = {
        "source_id",
        "kind",
        "required",
        "outcome",
        "producer",
        "limits",
        "payload",
        "item_count",
        "payload_bytes",
        "clock",
        "clock_observations",
        "affine_segments",
        "artifacts",
        "application_metadata",
    }
    _exact_keys(
        value,
        required_keys,
        optional={"sync_event_cardinality"},
        context="session source",
    )
    source_id = _source_identifier(value.get("source_id"))
    kind = _closed_text(value, "kind", _KINDS, f"source {source_id}")
    required = _boolean(value, "required", f"source {source_id}")
    outcome = _closed_text(value, "outcome", _OUTCOMES, f"source {source_id}")
    if required and outcome != "complete":
        raise ValueError(f"required source {source_id!r} must be complete.")

    producer = _object(value, "producer", f"source {source_id}")
    _exact_keys(producer, {"name", "version"}, context=f"source {source_id}.producer")
    producer_name = _opaque_id(producer.get("name"), f"source {source_id}.producer.name")
    producer_version = _opaque_id(
        producer.get("version"),
        f"source {source_id}.producer.version",
    )

    limits = _object(value, "limits", f"source {source_id}")
    _exact_keys(
        limits,
        {"max_items", "max_item_bytes", "max_payload_bytes"},
        context=f"source {source_id}.limits",
    )
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
        raise ValueError(f"source {source_id}.limits.max_item_bytes exceeds max_payload_bytes.")
    payload = _object(value, "payload", f"source {source_id}")
    _exact_keys(payload, {"filename", "format"}, context=f"source {source_id}.payload")
    filename = _leaf_name(payload.get("filename"), f"source {source_id}.payload.filename")
    if filename == _INDEX_NAME:
        raise ValueError(f"source {source_id}.payload.filename cannot be index.bin.")
    payload_format = _payload_format(payload.get("format"), f"source {source_id}.payload.format")
    item_count = _uint(value, "item_count", 0, max_items, f"source {source_id}")
    payload_bytes = _uint(
        value,
        "payload_bytes",
        0,
        max_payload_bytes,
        f"source {source_id}",
    )
    if (item_count == 0) != (payload_bytes == 0):
        raise ValueError("source item_count and payload_bytes must both be zero or both nonzero.")

    clock = _parse_clock(_object(value, "clock", f"source {source_id}"), kind=kind)
    observations = _parse_observations(value, source_id=source_id, clock=clock)
    segments = _parse_segments(
        value,
        source_id=source_id,
        clock=clock,
        observations=observations,
    )
    cardinality = (
        _parse_cardinality(value["sync_event_cardinality"], source_id, max_items)
        if "sync_event_cardinality" in value
        else None
    )
    artifacts = _parse_artifacts(value, source_id=source_id)
    _application_metadata(
        value.get("application_metadata"),
        f"source {source_id}.application_metadata",
    )

    if outcome == "complete":
        _validate_complete_artifacts(
            artifacts,
            payload_filename=filename,
            payload_bytes=payload_bytes,
            item_count=item_count,
        )
    elif item_count != 0 or payload_bytes != 0 or artifacts:
        raise ValueError(f"non-complete source {source_id!r} must not publish items or artifacts.")

    return _SourceContract(
        source_id=source_id,
        kind=kind,
        required=required,
        outcome=outcome,
        producer_name=producer_name,
        producer_version=producer_version,
        max_items=max_items,
        max_item_bytes=max_item_bytes,
        payload_filename=filename,
        payload_format=payload_format,
        item_count=item_count,
        payload_bytes=payload_bytes,
        clock=clock,
        observation_ids=frozenset(observations),
        segments=segments,
        cardinality=cardinality,
        artifacts=artifacts,
    )


def _open_source(
    *,
    root: Path,
    session_id: str,
    contract: _SourceContract,
    event_ids: frozenset[int],
) -> tuple[MultisensorSource, Counter[int]]:
    if contract.outcome != "complete":
        return _public_source(session_id, contract, None, None, event_ids), Counter()
    source_root = _directory_leaf(root, contract.source_id, f"source {contract.source_id}")
    _require_directory_names(
        source_root,
        {artifact.path for artifact in contract.artifacts},
        f"source {contract.source_id}",
    )
    artifact_paths: dict[str, Path] = {}
    for artifact in contract.artifacts:
        path = _regular_leaf(
            source_root,
            artifact.path,
            f"source {contract.source_id} artifact {artifact.role}",
        )
        _require_file_size(path, artifact.size_bytes, f"source artifact {artifact.path}")
        artifact_paths[artifact.role] = path
    payload_path = artifact_paths["payload"]
    index_path = artifact_paths["index"]
    counts = _validate_index(index_path, contract=contract, event_ids=event_ids)
    for artifact in contract.artifacts:
        path = _regular_leaf(
            source_root,
            artifact.path,
            f"source {contract.source_id} artifact {artifact.role}",
        )
        digest = _sha256_file(path, artifact.size_bytes)
        if not hmac.compare_digest(digest, artifact.sha256):
            raise ValueError(
                f"source artifact {artifact.path!r} SHA-256 does not match session.json."
            )
    return _public_source(
        session_id,
        contract,
        payload_path,
        index_path,
        event_ids,
    ), counts


def _public_source(
    session_id: str,
    contract: _SourceContract,
    payload_path: Path | None,
    index_path: Path | None,
    event_ids: frozenset[int],
) -> MultisensorSource:
    return MultisensorSource(
        session_id=session_id,
        source_id=contract.source_id,
        kind=contract.kind,
        required=contract.required,
        outcome=contract.outcome,
        producer_name=contract.producer_name,
        producer_version=contract.producer_version,
        payload_format=contract.payload_format,
        item_count=contract.item_count,
        payload_bytes=contract.payload_bytes,
        clock_id=contract.clock.clock_id,
        tick_hz=contract.clock.tick_hz,
        wrap_ticks=contract.clock.wrap_ticks,
        timestamp_semantics=contract.clock.timestamp_semantics,
        payload_path=payload_path,
        index_path=index_path,
        _max_item_bytes=contract.max_item_bytes,
        _segments=contract.segments,
        _event_ids=event_ids,
    )


def _validate_index(  # noqa: C901
    path: Path,
    *,
    contract: _SourceContract,
    event_ids: frozenset[int],
) -> Counter[int]:
    _require_file_size(path, _checked_index_size(contract.item_count), "sensor index")
    event_counts: Counter[int] = Counter()
    expected_offset = 0
    previous_tick: int | None = None
    with path.open("rb") as file:
        _read_index_header(
            file,
            expected_items=contract.item_count,
            expected_payload_bytes=contract.payload_bytes,
        )
        for expected_index in range(contract.item_count):
            entry = _decode_index_entry(
                _read_exact(file, _INDEX_ENTRY_BYTES, "sensor index entry"),
                expected_index=expected_index,
                max_item_bytes=contract.max_item_bytes,
            )
            if entry.payload_offset != expected_offset:
                raise ValueError("Sensor index payload ranges must be ordered and contiguous.")
            expected_offset += entry.payload_size
            if expected_offset > contract.payload_bytes:
                raise ValueError("Sensor index payload range exceeds the declared payload.")
            unwrapped = _unwrapped_tick(contract.clock, entry.ticks, entry.wrap_count)
            if previous_tick is not None and unwrapped < previous_tick:
                raise ValueError("Sensor index source ticks must be monotonic.")
            previous_tick = unwrapped
            if unwrapped + entry.duration_ticks > _MAX_U64:
                raise ValueError("Sensor index duration overflows the unwrapped clock.")
            _segment_for_tick(contract.segments, unwrapped)
            _map_item_interval(contract.clock, contract.segments, entry)
            if entry.sync_event_id is None:
                if contract.cardinality is not None and contract.cardinality.required:
                    raise ValueError("Sensor index item omits a required synchronization event.")
                continue
            if contract.cardinality is None:
                raise ValueError("Sensor index item declares an event without cardinality.")
            if entry.sync_event_id not in event_ids:
                raise ValueError("Sensor index references an unknown synchronization event.")
            event_counts[entry.sync_event_id] += 1
        if expected_offset != contract.payload_bytes:
            raise ValueError("Sensor index does not exactly cover the declared payload.")
        if file.read(1):
            raise ValueError("Sensor index contains trailing data.")
    return event_counts


def _read_index_header(
    file: BinaryIO,
    *,
    expected_items: int,
    expected_payload_bytes: int,
) -> None:
    payload = _read_exact(file, _INDEX_HEADER_BYTES, "sensor index header")
    magic, major, header_bytes, entry_bytes, flags, item_count, payload_bytes = (
        _INDEX_HEADER.unpack(payload)
    )
    if magic != _INDEX_MAGIC:
        raise ValueError("Sensor index magic is invalid.")
    if major != 1 or header_bytes != _INDEX_HEADER_BYTES:
        raise ValueError("Sensor index version or header size is unsupported.")
    if entry_bytes != _INDEX_ENTRY_BYTES or flags != 0:
        raise ValueError("Sensor index entry size or flags are unsupported.")
    if item_count != expected_items or payload_bytes != expected_payload_bytes:
        raise ValueError("Sensor index header does not match session.json bounds.")


def _decode_index_entry(
    payload: bytes,
    *,
    expected_index: int,
    max_item_bytes: int,
) -> _IndexEntry:
    (
        item_index,
        payload_offset,
        payload_size,
        ticks,
        wrap_count,
        duration_ticks,
        sync_event_id,
        flags,
        reserved,
    ) = _INDEX_ENTRY.unpack(payload)
    if item_index != expected_index:
        raise ValueError("Sensor index item indices must be ordered and zero-based.")
    if payload_size == 0 or payload_size > max_item_bytes:
        raise ValueError("Sensor index item size is outside the declared bound.")
    if flags != 0 or reserved != 0:
        raise ValueError("Sensor index v1 flags and reserved fields must be zero.")
    return _IndexEntry(
        item_index=int(item_index),
        payload_offset=int(payload_offset),
        payload_size=int(payload_size),
        ticks=int(ticks),
        wrap_count=int(wrap_count),
        duration_ticks=int(duration_ticks),
        sync_event_id=None if sync_event_id == _NO_SYNC_EVENT else int(sync_event_id),
    )


def _parse_clock(record: dict[str, object], *, kind: str) -> _Clock:
    _exact_keys(
        record,
        {"clock_id", "tick_hz", "wrap_ticks", "timestamp_semantics"},
        context=f"{kind} clock",
    )
    clock = _Clock(
        clock_id=_opaque_id(record.get("clock_id"), f"{kind} clock.clock_id"),
        tick_hz=_uint(record, "tick_hz", 1, _MAX_U64, f"{kind} clock"),
        wrap_ticks=_uint(record, "wrap_ticks", 0, _MAX_U64, f"{kind} clock"),
        timestamp_semantics=_closed_text(
            record,
            "timestamp_semantics",
            frozenset({"host_monotonic", "frame_start", "exposure_midpoint"}),
            f"{kind} clock",
        ),
    )
    if kind in _KINDS and clock.timestamp_semantics != _TIMESTAMP_SEMANTICS[kind]:
        raise ValueError(f"{kind} source clock timestamp semantics are invalid.")
    if clock.wrap_ticks == 1:
        raise ValueError(f"{kind} clock.wrap_ticks must be zero or greater than one.")
    return clock


def _parse_observations(
    source: dict[str, object],
    *,
    source_id: str,
    clock: _Clock,
) -> dict[str, _ClockObservation]:
    values = _array(source, "clock_observations", f"source {source_id}")
    if len(values) > _MAX_CLOCK_OBSERVATIONS:
        raise ValueError(f"source {source_id!r} has too many clock observations.")
    observations: dict[str, _ClockObservation] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Clock observations must be JSON objects.")
        _exact_keys(
            value,
            {
                "observation_id",
                "tick",
                "wrap_count",
                "host_before_ns",
                "host_after_ns",
            },
            context=f"source {source_id}.clock_observation",
        )
        identifier = _opaque_id(
            value.get("observation_id"),
            f"source {source_id}.clock_observation.observation_id",
        )
        tick = _uint(value, "tick", 0, _MAX_U64, "clock observation")
        wrap = _uint(value, "wrap_count", 0, _MAX_U64, "clock observation")
        _unwrapped_tick(clock, tick, wrap)
        before = _uint(value, "host_before_ns", 0, _MAX_U64, "clock observation")
        after = _uint(value, "host_after_ns", 0, _MAX_U64, "clock observation")
        if before > after:
            raise ValueError("Clock observation host_before_ns must not exceed host_after_ns.")
        if identifier in observations:
            raise ValueError(f"source {source_id!r} has duplicate clock observation IDs.")
        observations[identifier] = _ClockObservation(tick, wrap, before, after)
    return observations


def _parse_segments(  # noqa: C901
    source: dict[str, object],
    *,
    source_id: str,
    clock: _Clock,
    observations: dict[str, _ClockObservation],
) -> tuple[_AffineSegment, ...]:
    values = _array(source, "affine_segments", f"source {source_id}")
    if len(values) > _MAX_AFFINE_SEGMENTS:
        raise ValueError(f"source {source_id!r} has too many affine segments.")
    segments: list[_AffineSegment] = []
    previous: _AffineSegment | None = None
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Affine segments must be JSON objects.")
        _exact_keys(
            value,
            {
                "start_unwrapped_tick",
                "end_unwrapped_tick",
                "source_origin_tick",
                "host_origin_ns",
                "scale_num",
                "scale_den",
                "observation_ids",
                "uncertainty_ns",
            },
            context=f"source {source_id}.affine_segment",
        )
        observation_values = _string_array(
            value.get("observation_ids"),
            f"source {source_id}.affine_segment.observation_ids",
        )
        if not observation_values or len(set(observation_values)) != len(observation_values):
            raise ValueError("Affine segment observation IDs must be nonempty and unique.")
        if not set(observation_values) <= set(observations):
            raise ValueError("Affine segment references an unknown clock observation.")
        segment = _AffineSegment(
            start_tick=_uint(
                value,
                "start_unwrapped_tick",
                0,
                _MAX_U64,
                "affine segment",
            ),
            end_tick=_uint(value, "end_unwrapped_tick", 0, _MAX_U64, "affine segment"),
            source_origin_tick=_uint(
                value,
                "source_origin_tick",
                0,
                _MAX_U64,
                "affine segment",
            ),
            host_origin_ns=_uint(
                value,
                "host_origin_ns",
                0,
                _MAX_U64,
                "affine segment",
            ),
            scale_num=_uint(value, "scale_num", 1, _MAX_U64, "affine segment"),
            scale_den=_uint(value, "scale_den", 1, _MAX_U64, "affine segment"),
            uncertainty_ns=_uint(
                value,
                "uncertainty_ns",
                0,
                _MAX_U64,
                "affine segment",
            ),
        )
        if segment.start_tick >= segment.end_tick:
            raise ValueError("Affine segment ranges must be nonempty and half-open.")
        if not segment.start_tick <= segment.source_origin_tick < segment.end_tick:
            raise ValueError("Affine segment source_origin_tick is outside its range.")
        for observation_id in observation_values:
            observation = observations[observation_id]
            unwrapped = _unwrapped_tick(clock, observation.ticks, observation.wrap_count)
            if not segment.start_tick <= unwrapped < segment.end_tick:
                raise ValueError("Affine observation is outside its segment tick range.")
            if not _observation_interval_covered(segment, observation, unwrapped):
                raise ValueError("Affine uncertainty does not cover its observation interval.")
        if previous is not None:
            if segment.start_tick < previous.end_tick:
                raise ValueError("Affine segment ranges must not overlap.")
            previous_end = _nominal_numerator(previous, previous.end_tick)
            current_start = _nominal_numerator(segment, segment.start_tick)
            if current_start * previous.scale_den < previous_end * segment.scale_den:
                raise ValueError("Affine mapped time moves backwards at a segment boundary.")
        segments.append(segment)
        _validate_segment_domain(segment)
        previous = segment
    return tuple(segments)


def _parse_cardinality(
    value: object,
    source_id: str,
    max_source_items: int,
) -> _EventCardinality:
    if not isinstance(value, dict):
        raise ValueError(f"source {source_id}.sync_event_cardinality must be an object.")
    _exact_keys(
        value,
        {"required", "min_items", "max_items"},
        context=f"source {source_id}.sync_event_cardinality",
    )
    required = _boolean(value, "required", "sync event cardinality")
    minimum = _uint(value, "min_items", 0, _MAX_ITEMS, "sync event cardinality")
    maximum = _uint(value, "max_items", 1, max_source_items, "sync event cardinality")
    if minimum > maximum:
        raise ValueError("Sync event cardinality bounds are invalid.")
    return _EventCardinality(required, minimum, maximum)


def _parse_artifacts(source: dict[str, object], *, source_id: str) -> tuple[_Artifact, ...]:
    values = _array(source, "artifacts", f"source {source_id}")
    if len(values) > _MAX_ARTIFACTS:
        raise ValueError(f"source {source_id!r} has too many artifacts.")
    artifacts: list[_Artifact] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Source artifacts must be JSON objects.")
        _exact_keys(
            value,
            {"role", "path", "size_bytes", "sha256"},
            context="artifact",
        )
        digest = value.get("sha256")
        if not _valid_sha256(digest):
            raise ValueError("Artifact SHA-256 must be lowercase hexadecimal.")
        assert isinstance(digest, str)
        artifacts.append(
            _Artifact(
                role=_closed_text(value, "role", _ARTIFACT_ROLES, "artifact"),
                path=_leaf_name(value.get("path"), "artifact.path"),
                size_bytes=_uint(
                    value,
                    "size_bytes",
                    0,
                    _MAX_U64,
                    "artifact",
                ),
                sha256=digest,
            )
        )
    if len({artifact.path for artifact in artifacts}) != len(artifacts):
        raise ValueError("Source artifact paths must be unique.")
    return tuple(artifacts)


def _validate_complete_artifacts(
    artifacts: tuple[_Artifact, ...],
    *,
    payload_filename: str,
    payload_bytes: int,
    item_count: int,
) -> None:
    payloads = [artifact for artifact in artifacts if artifact.role == "payload"]
    indexes = [artifact for artifact in artifacts if artifact.role == "index"]
    if len(payloads) != 1 or len(indexes) != 1:
        raise ValueError("Complete sources require exactly one payload and one index artifact.")
    payload = payloads[0]
    index = indexes[0]
    if payload.path != payload_filename or payload.size_bytes != payload_bytes:
        raise ValueError("Payload artifact does not match the source payload contract.")
    if index.path != _INDEX_NAME or index.size_bytes != _checked_index_size(item_count):
        raise ValueError("Index artifact does not match the sensor-index v1 contract.")


def _parse_sync_event(
    value: object,
    *,
    known_clock_ids: frozenset[str],
    host_clock_id: str,
    observation_domains: dict[str, frozenset[str]],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Synchronization events must be JSON objects.")
    _exact_keys(
        value,
        {
            "sync_event_id",
            "clock_id",
            "tick",
            "wrap_count",
            "edge",
            "evidence_kind",
            "generator",
            "observer",
            "routing_id",
            "observation_ids",
            "uncertainty_ns",
        },
        context="sync event",
    )
    event_id = _uint(value, "sync_event_id", 0, _NO_SYNC_EVENT - 1, "sync event")
    clock_id = _opaque_id(value.get("clock_id"), "sync event.clock_id")
    if clock_id not in known_clock_ids:
        raise ValueError("Synchronization event references an unknown clock.")
    if clock_id == host_clock_id:
        raise ValueError("Synchronization event cannot use the host clock.")
    observation_ids = _string_array(
        value.get("observation_ids"),
        "sync event.observation_ids",
    )
    if not observation_ids or len(set(observation_ids)) != len(observation_ids):
        raise ValueError("Synchronization event observation IDs must be nonempty and unique.")
    if not set(observation_ids) <= observation_domains[clock_id]:
        raise ValueError("Synchronization event references an unknown observation.")
    return {
        "sync_event_id": event_id,
        "clock_id": clock_id,
        "tick": _uint(value, "tick", 0, _MAX_U64, "sync event"),
        "wrap_count": _uint(value, "wrap_count", 0, _MAX_U64, "sync event"),
        "edge": _closed_text(value, "edge", _SYNC_EDGES, "sync event"),
        "evidence_kind": _closed_text(
            value,
            "evidence_kind",
            _EVIDENCE_KINDS,
            "sync event",
        ),
        "generator": _opaque_id(value.get("generator"), "sync event.generator"),
        "observer": _opaque_id(value.get("observer"), "sync event.observer"),
        "routing_id": _opaque_id(value.get("routing_id"), "sync event.routing_id"),
        "uncertainty_ns": _uint(
            value,
            "uncertainty_ns",
            0,
            _MAX_U64,
            "sync event",
        ),
    }


def _public_sync_event(
    event: dict[str, object],
    *,
    host_clock: _Clock,
    sources: tuple[_SourceContract, ...],
) -> MultisensorSyncEvent:
    clock_id = event["clock_id"]
    tick = event["tick"]
    wrap = event["wrap_count"]
    uncertainty = event["uncertainty_ns"]
    assert isinstance(clock_id, str)
    assert isinstance(tick, int) and isinstance(wrap, int) and isinstance(uncertainty, int)
    if clock_id == host_clock.clock_id:
        raise ValueError("Synchronization event cannot use the host clock.")
    source = next(source for source in sources if source.clock.clock_id == clock_id)
    unwrapped = _unwrapped_tick(source.clock, tick, wrap)
    mapped = _map_instant(
        source.segments,
        unwrapped,
        extra_uncertainty=uncertainty,
    )
    return MultisensorSyncEvent(
        sync_event_id=cast(int, event["sync_event_id"]),
        clock_id=clock_id,
        edge=str(event["edge"]),
        evidence_kind=str(event["evidence_kind"]),
        generator=str(event["generator"]),
        observer=str(event["observer"]),
        routing_id=str(event["routing_id"]),
        mapped_time=mapped,
    )


def _validate_event_cardinalities(
    contracts: tuple[_SourceContract, ...],
    counts: dict[str, Counter[int]],
    event_ids: frozenset[int],
    grade: str,
) -> None:
    required_event_sets: list[frozenset[int]] = []
    for source in contracts:
        source_counts = counts[source.source_id]
        cardinality = source.cardinality
        if cardinality is None:
            if source_counts:
                raise ValueError("A source without event cardinality cannot reference sync events.")
            if grade == "external_trigger" and source.required:
                raise ValueError("External-trigger required sources need event cardinality.")
            continue
        relevant = event_ids if cardinality.required else frozenset(source_counts)
        for event_id in relevant:
            count = source_counts[event_id]
            if not cardinality.minimum <= count <= cardinality.maximum:
                raise ValueError("Source synchronization-event cardinality is outside its bounds.")
        if grade == "external_trigger" and source.required:
            if not cardinality.required:
                raise ValueError("External-trigger required sources must require event identity.")
            required_event_sets.append(frozenset(source_counts))
    if required_event_sets and any(
        events != required_event_sets[0] for events in required_event_sets
    ):
        raise ValueError("External-trigger required sources must share the same event set.")


def _map_item_interval(
    clock: _Clock,
    segments: tuple[_AffineSegment, ...],
    entry: _IndexEntry,
) -> MappedTimeInterval:
    tick = _unwrapped_tick(clock, entry.ticks, entry.wrap_count)
    segment = _segment_for_tick(segments, tick)
    nominal = _nominal_numerator(segment, tick)
    denominator = segment.scale_den
    uncertainty = segment.uncertainty_ns
    if entry.duration_ticks == 0:
        return MappedTimeInterval(
            nominal // denominator - uncertainty,
            _ceil_div(nominal, denominator) + uncertainty,
        )
    duration_numerator = entry.duration_ticks * segment.scale_num
    if clock.timestamp_semantics == "frame_start":
        start = nominal // denominator - uncertainty
        end = _ceil_div(nominal + duration_numerator, denominator) + uncertainty
    elif clock.timestamp_semantics == "exposure_midpoint":
        common_denominator = denominator * 2
        midpoint = nominal * 2
        start = (midpoint - duration_numerator) // common_denominator - uncertainty
        end = _ceil_div(midpoint + duration_numerator, common_denominator) + uncertainty
    else:
        raise ValueError("Source clock cannot map a physical item interval.")
    return MappedTimeInterval(start, end)


def _map_instant(
    segments: tuple[_AffineSegment, ...],
    tick: int,
    *,
    extra_uncertainty: int = 0,
) -> MappedTimeInterval:
    segment = _segment_for_tick(segments, tick)
    numerator = _nominal_numerator(segment, tick)
    uncertainty = segment.uncertainty_ns + extra_uncertainty
    return MappedTimeInterval(
        numerator // segment.scale_den - uncertainty,
        _ceil_div(numerator, segment.scale_den) + uncertainty,
    )


def _segment_for_tick(
    segments: tuple[_AffineSegment, ...],
    tick: int,
) -> _AffineSegment:
    match: _AffineSegment | None = None
    for segment in segments:
        if segment.start_tick <= tick < segment.end_tick:
            if match is not None:
                raise ValueError("Source tick has duplicate affine-segment coverage.")
            match = segment
    if match is None:
        raise ValueError("Source tick is not covered by an affine segment.")
    return match


def _nominal_numerator(segment: _AffineSegment, tick: int) -> int:
    return (
        segment.host_origin_ns * segment.scale_den
        + (tick - segment.source_origin_tick) * segment.scale_num
    )


def _observation_interval_covered(
    segment: _AffineSegment,
    observation: _ClockObservation,
    unwrapped_tick: int,
) -> bool:
    nominal = _nominal_numerator(segment, unwrapped_tick)
    uncertainty = segment.uncertainty_ns * segment.scale_den
    return (
        nominal - uncertainty <= observation.host_before_ns * segment.scale_den
        and nominal + uncertainty >= observation.host_after_ns * segment.scale_den
    )


def _validate_segment_domain(segment: _AffineSegment) -> None:
    uncertainty = segment.uncertainty_ns * segment.scale_den
    lower = _nominal_numerator(segment, segment.start_tick) - uncertainty
    upper = _nominal_numerator(segment, segment.end_tick) + uncertainty
    if lower < 0 or upper > _MAX_U64 * segment.scale_den:
        raise ValueError("Affine mapped interval plus uncertainty leaves the uint64 domain.")


def _unwrapped_tick(clock: _Clock, tick: int, wrap_count: int) -> int:
    if clock.wrap_ticks == 0:
        if wrap_count != 0:
            raise ValueError("Non-wrapping clocks require wrap_count zero.")
        return tick
    if tick >= clock.wrap_ticks:
        raise ValueError("Clock tick must be smaller than wrap_ticks.")
    unwrapped = wrap_count * clock.wrap_ticks + tick
    if unwrapped > _MAX_U64:
        raise ValueError("Unwrapped clock tick exceeds unsigned 64-bit range.")
    return unwrapped


def _checked_index_size(item_count: int) -> int:
    if not 0 <= item_count <= _MAX_ITEMS:
        raise ValueError("Sensor index item count is outside the v1 bound.")
    return _INDEX_HEADER_BYTES + item_count * _INDEX_ENTRY_BYTES


def _session_root(path: str | Path) -> Path:
    try:
        root = Path(path).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Multisensor capture directory is unavailable: {path}.") from exc
    if not root.is_dir() or root.name.endswith(".part"):
        raise ValueError(f"Multisensor capture path is not a published directory: {root}.")
    return root


def _regular_leaf(root: Path, name: str, label: str) -> Path:
    path = root / name
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}.") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}.")
    return path


def _directory_leaf(root: Path, name: str, label: str) -> Path:
    path = root / name
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}.") from exc
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{label} is not a directory: {path}.")
    return path


def _require_directory_names(root: Path, expected: set[str], label: str) -> None:
    try:
        actual = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise ValueError(f"{label} cannot be listed: {root}.") from exc
    if actual != expected:
        raise ValueError(f"{label} has undeclared or missing leaves.")


def _read_bounded_regular(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its regular-file bound.")
    payload = path.read_bytes()
    if len(payload) != status.st_size or len(payload) > maximum_bytes:
        raise ValueError(f"{label} changed while it was read.")
    return payload


def _require_file_size(path: Path, expected: int, label: str) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}.") from exc
    if not stat.S_ISREG(status.st_mode) or status.st_size != expected:
        raise ValueError(f"{label} size does not match session.json.")


def _sha256_file(path: Path, expected_size: int) -> str:
    _require_file_size(path, expected_size, "source artifact")
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    _require_file_size(path, expected_size, "source artifact")
    return digest


def _read_exact(file: BinaryIO, size: int, label: str) -> bytes:
    payload = file.read(size)
    if type(payload) is not bytes or len(payload) != size:
        raise ValueError(f"{label} is truncated.")
    return payload


def _strict_json_object(payload: bytes, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"{context} must be strict UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _exact_keys(
    record: dict[str, object],
    required: set[str],
    *,
    optional: set[str] | frozenset[str] = frozenset(),
    context: str,
) -> None:
    actual = set(record)
    if not required <= actual or not actual <= required | set(optional):
        raise ValueError(f"{context} has an invalid exact key set.")


def _object(record: dict[str, object], field: str, context: str) -> dict[str, object]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{field} must be a JSON object.")
    return value


def _array(record: dict[str, object], field: str, context: str) -> list[object]:
    value = record.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{context}.{field} must be a JSON array.")
    return value


def _string_array(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or len(item.encode("utf-8")) > 128:
            raise ValueError(f"{context} entries must be bounded nonempty strings.")
        result.append(item)
    return tuple(result)


def _literal(record: dict[str, object], field: str, expected: str, context: str) -> None:
    if record.get(field) != expected:
        raise ValueError(f"{context} must be {expected!r}.")


def _closed_text(
    record: dict[str, object],
    field: str,
    allowed: frozenset[str],
    context: str,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{context}.{field} is not a supported value.")
    return value


def _boolean(record: dict[str, object], field: str, context: str) -> bool:
    value = record.get(field)
    if type(value) is not bool:
        raise ValueError(f"{context}.{field} must be a boolean.")
    return value


def _uint(
    record: dict[str, object],
    field: str,
    minimum: int,
    maximum: int,
    context: str,
) -> int:
    value = record.get(field)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{context}.{field} is outside its unsigned integer bound.")
    return value


def _session_identifier(value: object) -> str:
    if not isinstance(value, str) or _SESSION_ID.fullmatch(value) is None:
        raise ValueError("session.session_id must be a lowercase UUIDv4.")
    return value


def _source_identifier(value: object) -> str:
    if not isinstance(value, str) or _SOURCE_ID.fullmatch(value) is None:
        raise ValueError("source_id does not match [a-z][a-z0-9-]{0,63}.")
    return value


def _leaf_name(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or value.endswith(".part")
        or len(value.encode("utf-8")) > 128
    ):
        raise ValueError(f"{context} must be one safe non-part leaf name.")
    return value


def _application_metadata(value: object, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ValueError(f"{context} exceeds the {_MAX_METADATA_BYTES}-byte limit.")
    if len(value) > _MAX_METADATA_ENTRIES:
        raise ValueError(f"{context} object has too many entries.")
    for key, item in value.items():
        if (
            len(key.encode("utf-8")) > _MAX_METADATA_KEY_BYTES
            or _METADATA_KEY.fullmatch(key) is None
        ):
            raise ValueError(f"{context} key {key!r} is not namespaced.")
        if _json_depth(item) > _MAX_METADATA_DEPTH:
            raise ValueError(f"{context}[{key!r}] exceeds the metadata depth limit.")


def _opaque_id(value: object, context: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID.fullmatch(value) is None:
        raise ValueError(f"{context} is not a valid opaque identifier.")
    return value


def _payload_format(value: object, context: str) -> str:
    if not isinstance(value, str) or _PAYLOAD_FORMAT.fullmatch(value) is None:
        raise ValueError(f"{context} is not a valid payload format.")
    return value


def _json_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 1


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


__all__ = [
    "MMWCLI_MULTISENSOR_SESSION_SCHEMA_V1",
    "MMWCLI_SENSOR_INDEX_SCHEMA_V1",
    "MappedTimeInterval",
    "MultisensorCapture",
    "MultisensorItem",
    "MultisensorSource",
    "MultisensorSyncEvent",
    "TrainingKey",
    "causal_match",
    "open_multisensor_capture",
]
