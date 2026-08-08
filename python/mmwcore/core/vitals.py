"""Contracts for exploratory radar vital-sign waveforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from numbers import Real
from operator import index as integer_index
from sys import maxsize as _MAX_PLATFORM_INDEX
from typing import Any, SupportsFloat, SupportsIndex, cast

import numpy as np


class VitalSignQuantity(StrEnum):
    """Physical quantity represented by a vital-sign waveform."""

    PHASE_RAD = "phase_rad"
    DISPLACEMENT_M = "displacement_m"

    @property
    def units(self) -> str:
        return "rad" if self is VitalSignQuantity.PHASE_RAD else "m"


@dataclass(frozen=True)
class VitalSignWaveform:
    """Uniformly sampled candidate physiological-motion waveform.

    This contract carries radar-derived motion evidence. It does not imply a
    validated respiration, heart-rate, or clinical measurement.
    """

    values: np.ndarray
    sample_rate_hz: float
    quantity: VitalSignQuantity = VitalSignQuantity.PHASE_RAD
    start_time_s: float = 0.0
    range_bin: int | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_values = np.asarray(self.values)
        if source_values.dtype == np.dtype(np.bool_):
            raise TypeError("VitalSignWaveform.values must not have boolean dtype.")
        values = np.asarray(source_values, dtype=np.float32)
        quantity = VitalSignQuantity(self.quantity)
        if values.ndim != 1 or values.size < 2:
            raise ValueError(
                "VitalSignWaveform.values must be one-dimensional with at least two samples."
            )
        if not np.isfinite(values).all():
            raise ValueError("VitalSignWaveform.values contains NaN or Inf.")
        sample_rate_hz = _positive_real(self.sample_rate_hz, name="sample_rate_hz")
        start_time_s = _finite_real(self.start_time_s, name="start_time_s")
        range_bin = (
            None
            if self.range_bin is None
            else _non_negative_platform_integer(self.range_bin, name="range_bin")
        )

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "start_time_s", start_time_s)
        object.__setattr__(self, "range_bin", range_bin)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_samples(self) -> int:
        return int(self.values.size)

    @property
    def duration_s(self) -> float:
        return (self.num_samples - 1) / self.sample_rate_hz

    @property
    def units(self) -> str:
        return self.quantity.units

    def time_axis_s(self) -> np.ndarray:
        return self.start_time_s + np.arange(self.num_samples) / self.sample_rate_hz


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"VitalSignWaveform.{name} must be a real number.")
    normalized = float(cast(SupportsFloat, value))
    if not isfinite(normalized):
        raise ValueError(f"VitalSignWaveform.{name} must be finite.")
    return normalized


def _positive_real(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if normalized <= 0:
        raise ValueError(f"VitalSignWaveform.{name} must be finite and positive.")
    return normalized


def _non_negative_platform_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"VitalSignWaveform.{name} must be an integer.")
    try:
        normalized = int(integer_index(cast(SupportsIndex, value)))
    except TypeError as exc:
        raise TypeError(f"VitalSignWaveform.{name} must be an integer.") from exc
    if normalized > _MAX_PLATFORM_INDEX:
        raise OverflowError(f"VitalSignWaveform.{name} must fit the platform index range.")
    if normalized < 0:
        raise ValueError(f"VitalSignWaveform.{name} must be non-negative.")
    return normalized


__all__ = ["VitalSignQuantity", "VitalSignWaveform"]
