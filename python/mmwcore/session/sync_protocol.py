"""Same-host control and timestamp protocol for synchronized capture."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from mmwcore._compat import Self, StrEnum

CAPTURE_SYNC_CONTROL_VERSION = "OPENMMW_SYNC_V2"
CAPTURE_SYNC_EVENT_SCHEMA = "openmmw.capture_sync_event.v1"
_CAPTURE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class SyncControlAction(StrEnum):
    """Commands accepted by a synchronized camera agent."""

    ARM = "arm"
    RADAR_START = "radar_start"
    RADAR_STOP = "radar_stop"
    FINALIZE = "finalize"
    SHUTDOWN = "shutdown"


class CaptureSyncEventKind(StrEnum):
    """Events written in the camera-agent clock domain."""

    CAMERA_ARMED = "camera_armed"
    RADAR_START = "radar_start"
    CAMERA_FRAME = "camera_frame"
    RADAR_STOP = "radar_stop"
    CAPTURE_CLOSED = "capture_closed"


@dataclass(frozen=True)
class SyncControlMessage:
    """One versioned UDP control message sent by the radar-side script."""

    capture_id: str
    action: SyncControlAction
    sequence: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_capture_id(self.capture_id)
        if not isinstance(self.action, SyncControlAction):
            raise TypeError("Sync control action must be a SyncControlAction.")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise TypeError("Sync control sequence must be an integer.")
        if self.sequence < 0:
            raise ValueError("Sync control sequence must be non-negative.")
        json.dumps(self.metadata)

    def encode(self) -> bytes:
        return json.dumps(
            {
                "version": CAPTURE_SYNC_CONTROL_VERSION,
                "capture_id": self.capture_id,
                "action": self.action.value,
                "sequence": self.sequence,
                "metadata": self.metadata,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")

    @classmethod
    def parse(cls, payload: bytes | str) -> Self:
        try:
            text = payload.decode("ascii") if isinstance(payload, bytes) else payload
        except UnicodeDecodeError as exc:
            raise ValueError("Sync control messages must use ASCII.") from exc
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Sync control message must be valid JSON.") from exc
        if not isinstance(record, dict):
            raise ValueError("Sync control message must be a JSON object.")
        if record.get("version") != CAPTURE_SYNC_CONTROL_VERSION:
            raise ValueError("Sync control message has an unsupported format or version.")
        try:
            capture_id = record["capture_id"]
            action = SyncControlAction(record["action"])
            sequence = record["sequence"]
            metadata = record.get("metadata", {})
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Sync control message has an invalid action or sequence.") from exc
        if not isinstance(capture_id, str) or not isinstance(metadata, dict):
            raise ValueError("Sync control capture_id must be a string and metadata an object.")
        try:
            return cls(
                capture_id=capture_id,
                action=action,
                sequence=sequence,
                metadata=metadata,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Sync control message fields are invalid.") from exc


@dataclass(frozen=True)
class CaptureSyncEvent:
    """One timestamped event measured by the camera-agent process."""

    capture_id: str
    kind: CaptureSyncEventKind
    event_index: int
    monotonic_ns: int
    utc_ns: int
    control_sequence: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = CAPTURE_SYNC_EVENT_SCHEMA

    def __post_init__(self) -> None:
        validate_capture_id(self.capture_id)
        if not isinstance(self.kind, CaptureSyncEventKind):
            raise TypeError("Capture sync kind must be a CaptureSyncEventKind.")
        for name, value in (
            ("event_index", self.event_index),
            ("monotonic_ns", self.monotonic_ns),
            ("utc_ns", self.utc_ns),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"Capture sync {name} must be an integer.")
            if value < 0:
                raise ValueError(f"Capture sync {name} must be non-negative.")
        if self.control_sequence is not None and (
            not isinstance(self.control_sequence, int)
            or isinstance(self.control_sequence, bool)
            or self.control_sequence < 0
        ):
            raise ValueError("Capture sync control_sequence must be non-negative or None.")
        if self.schema != CAPTURE_SYNC_EVENT_SCHEMA:
            raise ValueError("Capture sync event uses an unsupported schema.")
        json.dumps(self.metadata)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capture_id": self.capture_id,
            "kind": self.kind.value,
            "event_index": self.event_index,
            "monotonic_ns": self.monotonic_ns,
            "utc_ns": self.utc_ns,
            "control_sequence": self.control_sequence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: object) -> Self:
        if not isinstance(record, dict):
            raise ValueError("Capture sync event record must be a JSON object.")
        try:
            kind = CaptureSyncEventKind(record["kind"])
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("Capture sync event metadata must be an object.")
            return cls(
                schema=str(record.get("schema", "")),
                capture_id=str(record["capture_id"]),
                kind=kind,
                event_index=_record_int(record, "event_index"),
                monotonic_ns=_record_int(record, "monotonic_ns"),
                utc_ns=_record_int(record, "utc_ns"),
                control_sequence=_optional_record_int(record, "control_sequence"),
                metadata=metadata,
            )
        except KeyError as exc:
            raise ValueError(f"Capture sync event is missing {exc.args[0]!r}.") from exc


class CaptureSyncEventWriter:
    """Append timestamped events with one monotonic index and clock domain."""

    def __init__(
        self,
        path: str | Path,
        *,
        capture_id: str,
        monotonic_clock: Callable[[], int] = time.monotonic_ns,
        utc_clock: Callable[[], int] = time.time_ns,
    ) -> None:
        validate_capture_id(capture_id)
        self.path = Path(path)
        self.capture_id = capture_id
        self._monotonic_clock = monotonic_clock
        self._utc_clock = utc_clock
        self._event_index = 0
        self._file: TextIO | None = None

    def open(self) -> Self:
        if self._file is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8")
        return self

    def append(
        self,
        kind: CaptureSyncEventKind,
        *,
        control_sequence: int | None = None,
        metadata: dict[str, Any] | None = None,
        monotonic_ns: int | None = None,
        utc_ns: int | None = None,
    ) -> CaptureSyncEvent:
        if self._file is None:
            raise RuntimeError("CaptureSyncEventWriter must be opened before append.")
        event = CaptureSyncEvent(
            capture_id=self.capture_id,
            kind=kind,
            event_index=self._event_index,
            monotonic_ns=self._monotonic_clock() if monotonic_ns is None else monotonic_ns,
            utc_ns=self._utc_clock() if utc_ns is None else utc_ns,
            control_sequence=control_sequence,
            metadata=dict(metadata or {}),
        )
        self._file.write(json.dumps(event.to_record(), sort_keys=True) + "\n")
        self._file.flush()
        self._event_index += 1
        return event

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def load_capture_sync_events(path: str | Path) -> tuple[CaptureSyncEvent, ...]:
    """Load a non-empty JSONL event stream with contiguous event indices."""

    events = tuple(
        CaptureSyncEvent.from_record(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not events:
        raise ValueError("Capture sync event log must not be empty.")
    expected = tuple(range(len(events)))
    actual = tuple(event.event_index for event in events)
    if actual != expected:
        raise ValueError("Capture sync event indices must be contiguous from zero.")
    capture_ids = {event.capture_id for event in events}
    if len(capture_ids) != 1:
        raise ValueError("Capture sync event log must contain one capture_id.")
    return events


def encode_sync_reply(
    message: SyncControlMessage,
    *,
    accepted: bool,
) -> bytes:
    """Encode the narrow acknowledgement consumed by the Lua script."""

    status = "ack" if accepted else "error"
    return (
        f"{CAPTURE_SYNC_CONTROL_VERSION}|{message.capture_id}|"
        f"{status}|{message.sequence}|{message.action.value}"
    ).encode("ascii")


def validate_capture_id(value: str) -> None:
    """Validate the filesystem-safe identifier shared by capture artifacts."""

    if not isinstance(value, str) or _CAPTURE_ID.fullmatch(value) is None:
        raise ValueError("capture_id must contain 1-128 ASCII letters, digits, '.', '_', or '-'.")


def _record_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Capture sync event {key} must be an integer.")
    return value


def _optional_record_int(record: dict[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Capture sync event {key} must be an integer or null.")
    return value


__all__ = [
    "CAPTURE_SYNC_CONTROL_VERSION",
    "CAPTURE_SYNC_EVENT_SCHEMA",
    "CaptureSyncEvent",
    "CaptureSyncEventKind",
    "CaptureSyncEventWriter",
    "SyncControlAction",
    "SyncControlMessage",
    "encode_sync_reply",
    "load_capture_sync_events",
    "validate_capture_id",
]
