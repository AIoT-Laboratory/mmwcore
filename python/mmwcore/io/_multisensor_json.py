"""Parse the strict JSON contract of published multi-sensor sessions."""

from __future__ import annotations

import json
import re
from typing import NoReturn

_MAX_METADATA_ENTRIES = 32
_MAX_METADATA_BYTES = 64 << 10
_MAX_METADATA_DEPTH = 16
_MAX_METADATA_KEY_BYTES = 128

_SESSION_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_SOURCE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_PAYLOAD_FORMAT = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_METADATA_KEY = re.compile(r"[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*){2,}\Z")


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
