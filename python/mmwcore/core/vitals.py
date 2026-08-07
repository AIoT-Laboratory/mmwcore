"""Contracts for exploratory radar vital-sign waveforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmwcore._compat import StrEnum


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
        values = np.asarray(self.values, dtype=np.float32)
        quantity = VitalSignQuantity(self.quantity)
        if values.ndim != 1 or values.size < 2:
            raise ValueError(
                "VitalSignWaveform.values must be one-dimensional with at least two samples."
            )
        if not np.isfinite(values).all():
            raise ValueError("VitalSignWaveform.values contains NaN or Inf.")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("VitalSignWaveform.sample_rate_hz must be finite and positive.")
        if not np.isfinite(self.start_time_s):
            raise ValueError("VitalSignWaveform.start_time_s must be finite.")
        if self.range_bin is not None and (
            not isinstance(self.range_bin, int)
            or isinstance(self.range_bin, bool)
            or self.range_bin < 0
        ):
            raise ValueError("VitalSignWaveform.range_bin must be a non-negative integer.")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "start_time_s", float(self.start_time_s))
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


__all__ = ["VitalSignQuantity", "VitalSignWaveform"]
