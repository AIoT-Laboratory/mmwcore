"""Shared validation for mmwcli capture artifacts and streams."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from mmwcore.config import RadarCaptureSpec, parse_ti_cli_capture_spec
from mmwcore.core import ADCComplexLayout

MMWCLI_CAPTURE_SESSION_SCHEMA_V1 = "mmwcli.capture_session.v1"

_MANIFEST_FILE_NAME = "capture.json"
_ADC_FILE_NAME = "adc.bin"
_RADAR_CONFIG_FILE_NAME = "radar.cfg"
_MAX_RADAR_CONFIG_BYTES = 4 << 20
_MAX_INT64 = (1 << 63) - 1

type _RawCaptureKey = tuple[str, str, str, str, str, str, str, str, int, str]


def _two_lane_ti_raw_capture_key(family: str) -> _RawCaptureKey:
    return (
        "ti",
        family,
        "",
        "",
        "route_declaration",
        "ti_mmwave_legacy_cli.v1",
        "int16",
        "little",
        2,
        "group2_i_then_q",
    )


_XWR16XX_RAW_CAPTURE_KEY = _two_lane_ti_raw_capture_key("xwr16xx")
_XWR18XX_RAW_CAPTURE_KEY = _two_lane_ti_raw_capture_key("xwr18xx")
_XWR68XX_RAW_CAPTURE_KEY = _two_lane_ti_raw_capture_key("xwr68xx")
_RAW_CAPTURE_LAYOUTS: dict[_RawCaptureKey, ADCComplexLayout] = {
    _XWR16XX_RAW_CAPTURE_KEY: ADCComplexLayout.GROUP2_I_THEN_Q,
    _XWR18XX_RAW_CAPTURE_KEY: ADCComplexLayout.GROUP2_I_THEN_Q,
    _XWR68XX_RAW_CAPTURE_KEY: ADCComplexLayout.GROUP2_I_THEN_Q,
}


@dataclass(frozen=True)
class MmwcliRawCaptureContract:
    """Closed mmwcli hardware declaration and raw ADC wire contract."""

    vendor: str
    family: str
    model: str
    revision: str
    identity_source: str
    config_format: str
    dtype: str
    byte_order: str
    lane_count: int
    layout: str

    def __post_init__(self) -> None:
        strings = (
            self.vendor,
            self.family,
            self.model,
            self.revision,
            self.identity_source,
            self.config_format,
            self.dtype,
            self.byte_order,
            self.layout,
        )
        if any(type(value) is not str for value in strings):
            raise TypeError("MmwcliRawCaptureContract text fields must be strings.")
        if type(self.lane_count) is not int:
            raise TypeError("MmwcliRawCaptureContract.lane_count must be an integer.")
        if self._key() not in _RAW_CAPTURE_LAYOUTS:
            raise ValueError("MmwcliRawCaptureContract descriptor tuple is unsupported.")

    def _key(self) -> _RawCaptureKey:
        return (
            self.vendor,
            self.family,
            self.model,
            self.revision,
            self.identity_source,
            self.config_format,
            self.dtype,
            self.byte_order,
            self.lane_count,
            self.layout,
        )


def _parse_mmwcli_raw_capture_contract(
    *,
    hardware: dict[str, object],
    adc: dict[str, object],
    radar_config: dict[str, object],
    context: str,
) -> MmwcliRawCaptureContract:
    try:
        return MmwcliRawCaptureContract(
            vendor=_contract_string(hardware, "vendor", context=f"{context}.hardware"),
            family=_contract_string(hardware, "family", context=f"{context}.hardware"),
            model=_contract_string(hardware, "model", context=f"{context}.hardware"),
            revision=_contract_string(hardware, "revision", context=f"{context}.hardware"),
            identity_source=_contract_string(
                hardware,
                "identity_source",
                context=f"{context}.hardware",
            ),
            config_format=_contract_string(
                radar_config,
                "format",
                context=f"{context}.radar_config",
            ),
            dtype=_contract_string(adc, "dtype", context=f"{context}.adc"),
            byte_order=_contract_string(adc, "byte_order", context=f"{context}.adc"),
            lane_count=_contract_integer(adc, "lane_count", context=f"{context}.adc"),
            layout=_contract_string(adc, "layout", context=f"{context}.adc"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} raw capture descriptor tuple is unsupported.") from exc


def _contract_string(record: dict[str, object], field: str, *, context: str) -> str:
    value = record.get(field)
    if type(value) is not str:
        raise ValueError(f"{context}.{field} must be a string.")
    return value


def _contract_integer(record: dict[str, object], field: str, *, context: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"{context}.{field} must be an integer.")
    return value


def _valid_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_mmwcli_radar_config(
    payload: bytes,
    *,
    raw_capture: MmwcliRawCaptureContract,
    expected_sha256: str,
    context: str,
) -> RadarCaptureSpec:
    if type(raw_capture) is not MmwcliRawCaptureContract:
        raise TypeError(f"{context} raw capture contract has an invalid type.")
    layout = _RAW_CAPTURE_LAYOUTS.get(raw_capture._key())
    if layout is None:
        raise ValueError(f"{context} raw capture descriptor tuple is unsupported.")
    if not payload or not payload.strip():
        raise ValueError(f"{context} radar configuration is empty.")
    if len(payload) > _MAX_RADAR_CONFIG_BYTES:
        raise ValueError(
            f"{context} radar configuration exceeds the {_MAX_RADAR_CONFIG_BYTES}-byte limit."
        )
    if not _valid_lower_sha256(expected_sha256):
        raise ValueError(f"{context} radar configuration SHA-256 is invalid.")
    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, expected_sha256):
        raise ValueError(f"{context} radar configuration SHA-256 does not match its contract.")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{context} radar configuration is not valid UTF-8.") from exc

    capture = parse_ti_cli_capture_spec(
        text,
        layout=layout,
        family=raw_capture.family,
    )
    return capture


__all__ = ["MMWCLI_CAPTURE_SESSION_SCHEMA_V1", "MmwcliRawCaptureContract"]
