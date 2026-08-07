"""Open integrity-checked ADC capture directories published by mmwcli."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from mmwcore.config import RadarCaptureSpec, parse_ti_cli_capture_spec
from mmwcore.core import (
    ADCComplexLayout,
    RadarCube,
    RangeDopplerRecipe,
    RawADCFrame,
)

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
    through the final frame or reader use. SHA-256 verifies bundle
    self-consistency, not authenticity against a writer that can replace the
    complete bundle.
    """

    root: Path
    manifest_path: Path
    adc_path: Path
    radar_config_path: Path
    radar_capture: RadarCaptureSpec
    reader: ADCFileFrameReader

    def __post_init__(self) -> None:
        _validate_capture_reader(self)

    @property
    def num_frames(self) -> int:
        """Return the finite number of validated ADC frames."""

        return self.reader.num_frames

    def frame(self, index: int) -> RawADCFrame:
        """Read one validated frame by zero-based index."""

        if type(index) is not int:
            raise TypeError("ADCFileCapture frame index must be an integer.")
        return self.reader.read_frame(index)

    def frames(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[RawADCFrame]:
        """Iterate the validated half-open frame interval without loading the full file."""

        indices = _frame_interval(self.num_frames, start=start, stop=stop)
        return (self.reader.read_frame(index) for index in indices)

    def range_doppler(
        self,
        recipe: RangeDopplerRecipe,
        *,
        frame_index: int = 0,
    ) -> RadarCube:
        """Process one frame with an explicit recipe matching the capture contract."""

        _validate_range_doppler_recipe(self, recipe)
        from mmwcore.dsp.runners import process_adc_to_range_doppler

        return process_adc_to_range_doppler(self.frame(frame_index), recipe)


@dataclass(frozen=True)
class _ManifestV1:
    adc_size_bytes: int
    adc_sha256: str
    radar_config_sha256: str


def _frame_interval(
    num_frames: int,
    *,
    start: int,
    stop: int | None,
) -> range:
    if type(start) is not int or (stop is not None and type(stop) is not int):
        raise TypeError("ADCFileCapture frame bounds must be integers.")
    final = num_frames if stop is None else stop
    if start < 0 or start > num_frames:
        raise ValueError(f"ADCFileCapture start must be within [0, {num_frames}].")
    if final < start or final > num_frames:
        raise ValueError(f"ADCFileCapture stop must be within [{start}, {num_frames}].")
    return range(start, final)


def _validate_range_doppler_recipe(
    capture: ADCFileCapture,
    recipe: RangeDopplerRecipe,
) -> None:
    if not isinstance(recipe, RangeDopplerRecipe):
        raise TypeError("ADCFileCapture.range_doppler requires a RangeDopplerRecipe.")
    contract = capture.radar_capture
    if recipe.decode.adc != contract.adc:
        raise ValueError("Range-Doppler recipe ADC spec does not match the capture contract.")

    tdm = recipe.tdm_virtual_array
    if tdm is None:
        if len(contract.tx_order) > 1:
            raise ValueError("Multi-Tx capture processing requires an explicit TDM virtual array.")
        if recipe.doppler_fft.input_axis != "chirp":
            raise ValueError("Single-Tx capture processing requires the chirp Doppler axis.")
        return
    if tdm.tx_order != contract.tx_order:
        raise ValueError("Range-Doppler recipe Tx order does not match the capture contract.")
    if tdm.geometry.num_rx != contract.profile.num_rx:
        raise ValueError(
            "Range-Doppler recipe receiver geometry does not match the capture contract."
        )


def _validate_capture_reader(capture: ADCFileCapture) -> None:
    contract = capture.radar_capture
    reader = capture.reader
    expected_paths = (
        (capture.manifest_path, capture.root / _MANIFEST_FILE_NAME, "manifest"),
        (capture.adc_path, capture.root / _ADC_FILE_NAME, "ADC payload"),
        (
            capture.radar_config_path,
            capture.root / _RADAR_CONFIG_FILE_NAME,
            "radar configuration",
        ),
    )
    for actual, expected, label in expected_paths:
        if actual != expected:
            raise ValueError(f"ADCFileCapture {label} path does not match its root.")
    if Path(reader.path) != capture.adc_path:
        raise ValueError("ADCFileCapture reader path does not match its ADC payload.")
    if reader.spec != contract.adc:
        raise ValueError("ADCFileCapture reader ADC spec does not match its capture contract.")
    if reader.num_frames != contract.num_frames:
        raise ValueError("ADCFileCapture reader frame count does not match its capture contract.")
    if reader.frame_periodicity_s != contract.frame_periodicity_s:
        raise ValueError("ADCFileCapture reader period does not match its capture contract.")
    if reader.profile != asdict(contract.profile):
        raise ValueError("ADCFileCapture reader profile does not match its capture contract.")
    if reader.metadata.get("tx_order") != list(contract.tx_order):
        raise ValueError("ADCFileCapture reader Tx order does not match its capture contract.")


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
