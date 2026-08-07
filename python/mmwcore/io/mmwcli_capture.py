"""Open integrity-checked ADC capture directories published by mmwcli."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from mmwcore.config import RadarCaptureSpec, parse_ti_cli_capture_spec
from mmwcore.core import ADCComplexLayout

from .adc_file import ADCFileFrameReader

MMWCLI_CAPTURE_SESSION_SCHEMA_V1 = "mmwcli.capture_session.v1"

_MANIFEST_FILE_NAME = "capture.json"
_ADC_FILE_NAME = "adc.bin"
_RADAR_CONFIG_FILE_NAME = "radar.cfg"
_ADC_DATA_TYPE = "int16"
_ADC_BYTE_ORDER = "little"
_ADC_LAYOUT = "group2_i_then_q"
_RADAR_CONFIG_FORMAT = "ti_xwr68xx_legacy_cli"
_MAX_MANIFEST_BYTES = 64 << 10
_MAX_RADAR_CONFIG_BYTES = 4 << 20
_MAX_INT64 = (1 << 63) - 1


@dataclass(frozen=True)
class ADCFileCapture:
    """Validated mmwcli capture paths, physical contract, and frame reader.

    The directory must remain unchanged from entry into :func:`open_capture`
    through the final use of ``reader``. SHA-256 verifies bundle
    self-consistency, not authenticity against a writer that can replace the
    complete bundle.
    """

    root: Path
    manifest_path: Path
    adc_path: Path
    radar_config_path: Path
    radar_capture: RadarCaptureSpec
    reader: ADCFileFrameReader


@dataclass(frozen=True)
class _ManifestV1:
    adc_size_bytes: int
    adc_sha256: str
    radar_config_sha256: str


def open_capture(path: str | Path) -> ADCFileCapture:
    """Validate and open one published mmwcli capture-session v1 directory."""

    if sys.byteorder != "little":
        raise RuntimeError("mmwcli capture sessions require a little-endian host.")

    root = _capture_root(path)
    manifest_path = _fixed_regular_leaf(root, _MANIFEST_FILE_NAME, "manifest")
    manifest = _parse_manifest(
        _read_regular_bytes(
            manifest_path,
            label="capture manifest",
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    )

    adc_path = _fixed_regular_leaf(root, _ADC_FILE_NAME, "ADC payload")
    radar_config_path = _fixed_regular_leaf(
        root,
        _RADAR_CONFIG_FILE_NAME,
        "radar configuration",
    )
    adc_status = _regular_file_status(adc_path, "ADC payload")
    if adc_status.st_size != manifest.adc_size_bytes:
        raise ValueError(
            "mmwcli capture ADC size does not match capture.json: "
            f"{adc_status.st_size} != {manifest.adc_size_bytes}."
        )

    config_bytes = _read_regular_bytes(
        radar_config_path,
        label="radar configuration",
        maximum_bytes=_MAX_RADAR_CONFIG_BYTES,
    )
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    if not hmac.compare_digest(config_digest, manifest.radar_config_sha256):
        raise ValueError("mmwcli capture radar.cfg SHA-256 does not match capture.json.")
    try:
        config_text = config_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("mmwcli capture radar.cfg is not valid UTF-8.") from exc

    radar_capture = parse_ti_cli_capture_spec(
        config_text,
        layout=ADCComplexLayout.GROUP2_I_THEN_Q,
    )
    if radar_capture.num_frames is None or radar_capture.expected_size_bytes is None:
        raise ValueError("mmwcli capture session requires a finite radar frame count.")
    if radar_capture.expected_size_bytes != manifest.adc_size_bytes:
        raise ValueError(
            "mmwcli capture CFG-derived size does not match capture.json: "
            f"{radar_capture.expected_size_bytes} != {manifest.adc_size_bytes}."
        )

    adc_digest = _sha256_regular_file(adc_path, expected_size=manifest.adc_size_bytes)
    if not hmac.compare_digest(adc_digest, manifest.adc_sha256):
        raise ValueError("mmwcli capture adc.bin SHA-256 does not match capture.json.")

    reader = ADCFileFrameReader.from_capture(adc_path, radar_capture)
    return ADCFileCapture(
        root=root,
        manifest_path=manifest_path,
        adc_path=adc_path,
        radar_config_path=radar_config_path,
        radar_capture=radar_capture,
        reader=reader,
    )


def _capture_root(path: str | Path) -> Path:
    try:
        root = Path(path).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"mmwcli capture directory is unavailable: {path}.") from exc
    if not root.is_dir():
        raise ValueError(f"mmwcli capture path is not a regular directory: {root}.")
    return root


def _fixed_regular_leaf(root: Path, name: str, label: str) -> Path:
    path = root / name
    _regular_file_status(path, label)
    return path


def _regular_file_status(path: Path, label: str) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"mmwcli capture {label} is unavailable: {path}.") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"mmwcli capture {label} is not a regular file: {path}.")
    return status


def _read_regular_bytes(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    status = _regular_file_status(path, label)
    if status.st_size > maximum_bytes:
        raise ValueError(f"mmwcli capture {label} exceeds the {maximum_bytes}-byte limit.")
    payload = path.read_bytes()
    if len(payload) > maximum_bytes or len(payload) != status.st_size:
        raise ValueError(f"mmwcli capture {label} changed while it was read.")
    return payload


def _sha256_regular_file(path: Path, *, expected_size: int) -> str:
    status = _regular_file_status(path, "ADC payload")
    if status.st_size != expected_size:
        raise ValueError("mmwcli capture ADC size changed before hashing.")
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if path.stat().st_size != expected_size:
        raise ValueError("mmwcli capture ADC size changed while it was hashed.")
    return digest


def _parse_manifest(payload: bytes) -> _ManifestV1:
    try:
        text = payload.decode("utf-8")
        record = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError("mmwcli capture manifest is not strict UTF-8 JSON.") from exc
    if not isinstance(record, dict):
        raise ValueError("mmwcli capture manifest must be a JSON object.")
    if record.get("schema") != MMWCLI_CAPTURE_SESSION_SCHEMA_V1:
        raise ValueError("mmwcli capture manifest uses an unsupported schema.")
    # Required meanings are fixed; unknown fields remain available for
    # additive v1 provenance that does not change decoding semantics.
    adc = _required_object(record, "adc")
    radar_config = _required_object(record, "radar_config")
    _require_literal(adc, "path", _ADC_FILE_NAME, "adc.path")
    _require_literal(adc, "dtype", _ADC_DATA_TYPE, "adc.dtype")
    _require_literal(adc, "byte_order", _ADC_BYTE_ORDER, "adc.byte_order")
    _require_literal(adc, "layout", _ADC_LAYOUT, "adc.layout")
    _require_literal(
        radar_config,
        "path",
        _RADAR_CONFIG_FILE_NAME,
        "radar_config.path",
    )
    _require_literal(
        radar_config,
        "format",
        _RADAR_CONFIG_FORMAT,
        "radar_config.format",
    )
    size_bytes = adc.get("size_bytes")
    if (
        type(size_bytes) is not int
        or size_bytes <= 0
        or size_bytes > _MAX_INT64
        or size_bytes % 2 != 0
    ):
        raise ValueError("mmwcli capture adc.size_bytes must be a positive aligned int64.")
    return _ManifestV1(
        adc_size_bytes=size_bytes,
        adc_sha256=_required_sha256(adc, "sha256", "adc.sha256"),
        radar_config_sha256=_required_sha256(
            radar_config,
            "sha256",
            "radar_config.sha256",
        ),
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key {key!r}")
        record[key] = value
    return record


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _required_object(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"mmwcli capture manifest {field} must be a JSON object.")
    return value


def _require_literal(
    record: dict[str, object],
    field: str,
    expected: str,
    label: str,
) -> None:
    if record.get(field) != expected:
        raise ValueError(f"mmwcli capture manifest {label} must be {expected!r}.")


def _required_sha256(record: dict[str, object], field: str, label: str) -> str:
    value = record.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"mmwcli capture manifest {label} must be lowercase SHA-256.")
    return value


__all__ = [
    "ADCFileCapture",
    "MMWCLI_CAPTURE_SESSION_SCHEMA_V1",
    "open_capture",
]
