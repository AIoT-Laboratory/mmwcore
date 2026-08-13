"""Explicit contracts for captured radar data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Self

from mmwcore.config.profiles import RadarProfile
from mmwcore.core import ADCComplexLayout, ADCFrameSpec

RADAR_CAPTURE_SPEC_SCHEMA = "mmwcore.radar_capture_spec.v1"
_CAPTURE_CONTRACT_DIGEST_DOMAIN = b"mmwcore.radar_capture_spec.v1\0"


@dataclass(frozen=True)
class RadarCaptureSpec:
    """User-supplied contract joining waveform, frame layout, and physical TDM Tx order."""

    profile: RadarProfile
    adc: ADCFrameSpec
    tx_order: tuple[int, ...]
    frame_periodicity_s: float | None = None
    num_frames: int | None = None

    def __post_init__(self) -> None:
        _validate_contract_types(self.profile, self.adc)
        tx_order = _validated_tx_order(self.tx_order, num_tx=self.profile.num_tx)
        object.__setattr__(self, "tx_order", tx_order)
        _validate_adc_shape(self.profile, self.adc, tx_count=len(tx_order))
        periodicity = _validated_periodicity(self.frame_periodicity_s)
        _validate_frame_periodicity(self.profile, self.adc, periodicity)
        object.__setattr__(self, "frame_periodicity_s", periodicity)
        _validate_frame_count(self.num_frames)

    @property
    def expected_size_bytes(self) -> int | None:
        """Return the complete capture size when frame count is known."""

        if self.num_frames is None:
            return None
        return self.adc.raw_values_per_frame * self.num_frames * 2

    def to_record(self) -> dict[str, Any]:
        """Serialize the complete offline decoding contract."""

        return {
            "schema": RADAR_CAPTURE_SPEC_SCHEMA,
            "profile": {
                "start_frequency_hz": self.profile.start_frequency_hz,
                "frequency_slope_hz_per_s": self.profile.frequency_slope_hz_per_s,
                "adc_sample_rate_hz": self.profile.adc_sample_rate_hz,
                "adc_start_time_s": self.profile.adc_start_time_s,
                "ramp_end_time_s": self.profile.ramp_end_time_s,
                "idle_time_s": self.profile.idle_time_s,
                "num_adc_samples": self.profile.num_adc_samples,
                "num_chirps_per_tx": self.profile.num_chirps_per_tx,
                "num_tx": self.profile.num_tx,
                "num_rx": self.profile.num_rx,
                "speed_of_light_mps": self.profile.speed_of_light_mps,
            },
            "adc": {
                "num_chirps": self.adc.num_chirps,
                "num_rx": self.adc.num_rx,
                "num_samples": self.adc.num_samples,
                "layout": self.adc.layout.value,
            },
            "tx_order": list(self.tx_order),
            "frame_periodicity_s": self.frame_periodicity_s,
            "num_frames": self.num_frames,
            "expected_size_bytes": self.expected_size_bytes,
        }

    @classmethod
    def from_record(cls, record: object) -> Self:
        """Restore a capture contract without inferring omitted values."""

        if not isinstance(record, Mapping):
            raise ValueError("Radar capture spec must be a JSON object.")
        if record.get("schema") != RADAR_CAPTURE_SPEC_SCHEMA:
            raise ValueError("Radar capture spec uses an unsupported schema.")
        profile_record = record.get("profile")
        adc_record = record.get("adc")
        tx_order = record.get("tx_order")
        if not isinstance(profile_record, Mapping) or not isinstance(adc_record, Mapping):
            raise ValueError("Radar capture profile and ADC spec must be JSON objects.")
        if not isinstance(tx_order, list | tuple):
            raise ValueError("Radar capture tx_order must be a sequence.")

        try:
            capture = cls(
                profile=RadarProfile(
                    start_frequency_hz=_number(profile_record, "start_frequency_hz"),
                    frequency_slope_hz_per_s=_number(
                        profile_record,
                        "frequency_slope_hz_per_s",
                    ),
                    adc_sample_rate_hz=_number(profile_record, "adc_sample_rate_hz"),
                    adc_start_time_s=_number(profile_record, "adc_start_time_s"),
                    ramp_end_time_s=_number(profile_record, "ramp_end_time_s"),
                    idle_time_s=_number(profile_record, "idle_time_s"),
                    num_adc_samples=_integer(profile_record, "num_adc_samples"),
                    num_chirps_per_tx=_integer(profile_record, "num_chirps_per_tx"),
                    num_tx=_integer(profile_record, "num_tx"),
                    num_rx=_integer(profile_record, "num_rx"),
                    speed_of_light_mps=_number(profile_record, "speed_of_light_mps"),
                ),
                adc=ADCFrameSpec(
                    num_chirps=_integer(adc_record, "num_chirps"),
                    num_rx=_integer(adc_record, "num_rx"),
                    num_samples=_integer(adc_record, "num_samples"),
                    layout=ADCComplexLayout(_string(adc_record, "layout")),
                ),
                tx_order=tuple(_sequence_integers(tx_order, "tx_order")),
                frame_periodicity_s=_optional_number(record, "frame_periodicity_s"),
                num_frames=_optional_integer(record, "num_frames"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Radar capture spec is incomplete or invalid.") from exc
        if record.get("expected_size_bytes") != capture.expected_size_bytes:
            raise ValueError("Radar capture expected_size_bytes does not match its frame contract.")
        return capture


def capture_contract_sha256(capture: RadarCaptureSpec) -> str:
    """Return the stable SHA-256 identity of an explicit radar capture contract."""

    if not isinstance(capture, RadarCaptureSpec):
        raise TypeError("capture must be a RadarCaptureSpec.")
    canonical = json.dumps(
        capture.to_record(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(_CAPTURE_CONTRACT_DIGEST_DOMAIN + canonical).hexdigest()


def _number(record: Mapping[str, object], field: str) -> float:
    value = record[field]
    if not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value):
        raise TypeError(f"Radar capture {field} must be finite and numeric.")
    return float(value)


def _validate_contract_types(profile: object, adc: object) -> None:
    if not isinstance(profile, RadarProfile):
        raise TypeError("RadarCaptureSpec.profile must be a RadarProfile.")
    if not isinstance(adc, ADCFrameSpec):
        raise TypeError("RadarCaptureSpec.adc must be an ADCFrameSpec.")


def _validated_tx_order(values: tuple[int, ...], *, num_tx: int) -> tuple[int, ...]:
    if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
        raise TypeError("RadarCaptureSpec.tx_order must contain integers.")
    tx_order = tuple(values)
    if not tx_order:
        raise ValueError("RadarCaptureSpec.tx_order must not be empty.")
    if len(set(tx_order)) != len(tx_order):
        raise ValueError("RadarCaptureSpec.tx_order must not contain duplicates.")
    if len(tx_order) != num_tx:
        raise ValueError(
            "RadarCaptureSpec.tx_order must contain one physical identifier per active transmitter."
        )
    if any(index < 0 for index in tx_order):
        raise ValueError("RadarCaptureSpec.tx_order identifiers must be non-negative.")
    return tx_order


def _validate_adc_shape(profile: RadarProfile, adc: ADCFrameSpec, *, tx_count: int) -> None:
    if adc.num_rx != profile.num_rx:
        raise ValueError(
            f"RadarCaptureSpec ADC/profile Rx mismatch: {adc.num_rx} != {profile.num_rx}."
        )
    if adc.num_samples != profile.num_adc_samples:
        raise ValueError(
            "RadarCaptureSpec ADC/profile sample mismatch: "
            f"{adc.num_samples} != {profile.num_adc_samples}."
        )
    expected_chirps = profile.num_chirps_per_tx * tx_count
    if adc.num_chirps != expected_chirps:
        raise ValueError(
            f"RadarCaptureSpec ADC/profile chirp mismatch: {adc.num_chirps} != {expected_chirps}."
        )


def _validated_periodicity(value: float | None) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError("RadarCaptureSpec.frame_periodicity_s must be finite and positive.")
    return float(value)


def _validate_frame_periodicity(
    profile: RadarProfile,
    adc: ADCFrameSpec,
    periodicity_s: float | None,
) -> None:
    if periodicity_s is None:
        return
    active_chirp_time_s = profile.chirp_period_s * adc.num_chirps
    if periodicity_s < active_chirp_time_s:
        raise ValueError(
            "RadarCaptureSpec.frame_periodicity_s is shorter than the active chirp time."
        )


def _validate_frame_count(value: int | None) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError("RadarCaptureSpec.num_frames must be a positive integer.")


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Radar capture {field} must be an integer.")
    return value


def _string(record: Mapping[str, object], field: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value:
        raise TypeError(f"Radar capture {field} must be a non-empty string.")
    return value


def _sequence_integers(values: list[object] | tuple[object, ...], field: str) -> tuple[int, ...]:
    integers: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Radar capture {field} must contain integers.")
        integers.append(value)
    return tuple(integers)


def _optional_number(record: Mapping[str, object], field: str) -> float | None:
    return None if record.get(field) is None else _number(record, field)


def _optional_integer(record: Mapping[str, object], field: str) -> int | None:
    return None if record.get(field) is None else _integer(record, field)


__all__ = ["RADAR_CAPTURE_SPEC_SCHEMA", "RadarCaptureSpec", "capture_contract_sha256"]
