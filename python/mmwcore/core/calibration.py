"""Radar-channel calibration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class VirtualChannelCalibration:
    """Ordered complex correction coefficients for virtual receive channels."""

    coefficients: tuple[complex, ...]
    source: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        coefficients = tuple(complex(value) for value in self.coefficients)
        if not coefficients:
            raise ValueError("VirtualChannelCalibration.coefficients must not be empty.")
        if any(
            not isfinite(coefficient.real) or not isfinite(coefficient.imag)
            for coefficient in coefficients
        ):
            raise ValueError("VirtualChannelCalibration.coefficients must be finite.")
        if any(coefficient == 0 for coefficient in coefficients):
            raise ValueError("VirtualChannelCalibration coefficients must be non-zero.")
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def num_channels(self) -> int:
        return len(self.coefficients)

    def as_metadata(self) -> dict[str, object]:
        return {
            "num_channels": self.num_channels,
            "coefficients": [
                {"real": coefficient.real, "imag": coefficient.imag}
                for coefficient in self.coefficients
            ],
            "source": self.source,
            "version": self.version,
        }


@dataclass(frozen=True)
class TimeDomainChannelCalibration:
    """Per-Tx/Rx frequency ramps and complex corrections for ADC samples."""

    frequency_rad_per_sample: tuple[tuple[float, ...], ...]
    complex_corrections: tuple[tuple[complex, ...], ...]
    source: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        frequencies = tuple(
            tuple(float(value) for value in row) for row in self.frequency_rad_per_sample
        )
        corrections = tuple(
            tuple(complex(value) for value in row) for row in self.complex_corrections
        )
        if not frequencies or not frequencies[0]:
            raise ValueError("TimeDomainChannelCalibration matrices must not be empty.")
        shape = (len(frequencies), len(frequencies[0]))
        if any(len(row) != shape[1] for row in frequencies):
            raise ValueError("Frequency calibration rows must have equal length.")
        if len(corrections) != shape[0] or any(len(row) != shape[1] for row in corrections):
            raise ValueError("Frequency and complex correction matrices must have equal shape.")
        if any(not isfinite(value) for row in frequencies for value in row):
            raise ValueError("Frequency calibration values must be finite.")
        if any(
            not isfinite(value.real) or not isfinite(value.imag)
            for row in corrections
            for value in row
        ):
            raise ValueError("Complex calibration values must be finite.")
        if any(value == 0 for row in corrections for value in row):
            raise ValueError("Complex calibration values must be non-zero.")
        object.__setattr__(self, "frequency_rad_per_sample", frequencies)
        object.__setattr__(self, "complex_corrections", corrections)

    @property
    def num_tx(self) -> int:
        return len(self.frequency_rad_per_sample)

    @property
    def num_rx(self) -> int:
        return len(self.frequency_rad_per_sample[0])

    def as_metadata(self) -> dict[str, object]:
        return {
            "num_tx": self.num_tx,
            "num_rx": self.num_rx,
            "source": self.source,
            "version": self.version,
        }
