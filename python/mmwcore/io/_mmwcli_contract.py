"""Shared validation for mmwcli capture artifacts and streams."""

from __future__ import annotations

import hashlib
import hmac

from mmwcore.config import RadarCaptureSpec, parse_ti_cli_capture_spec
from mmwcore.core import ADCComplexLayout

MMWCLI_CAPTURE_SESSION_SCHEMA_V1 = "mmwcli.capture_session.v1"

_MANIFEST_FILE_NAME = "capture.json"
_ADC_FILE_NAME = "adc.bin"
_RADAR_CONFIG_FILE_NAME = "radar.cfg"
_ADC_DATA_TYPE = "int16"
_ADC_BYTE_ORDER = "little"
_ADC_LAYOUT = "group2_i_then_q"
_RADAR_CONFIG_FORMAT = "ti_xwr68xx_legacy_cli"
_MAX_RADAR_CONFIG_BYTES = 4 << 20
_MAX_INT64 = (1 << 63) - 1


def _valid_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_mmwcli_radar_config(
    payload: bytes,
    *,
    expected_sha256: str,
    context: str,
) -> RadarCaptureSpec:
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
        layout=ADCComplexLayout.GROUP2_I_THEN_Q,
    )
    if capture.num_frames is None or capture.expected_size_bytes is None:
        raise ValueError(f"{context} requires a finite radar frame count.")
    return capture


__all__ = ["MMWCLI_CAPTURE_SESSION_SCHEMA_V1"]
