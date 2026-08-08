"""FFT processing specs."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index
from sys import maxsize as _MAX_PLATFORM_INDEX

from .spec_adc import VirtualAntennaLayout
from .spec_enums import FFTWindow


def _positive_fft_size(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or None.")
    try:
        normalized = int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer or None.") from exc
    if not -_MAX_PLATFORM_INDEX - 1 <= normalized <= _MAX_PLATFORM_INDEX:
        raise OverflowError(f"{name} must fit the platform index range.")
    if normalized <= 0:
        raise ValueError(f"{name} must be positive; got {normalized}.")
    return normalized


def _require_bool(value: bool, *, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool.")


def _require_axis_name(value: str, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


@dataclass(frozen=True)
class RangeFFTSpec:
    """Configuration for range FFT over the ADC sample axis."""

    n_fft: int | None = None
    window: FFTWindow = FFTWindow.NONE
    one_sided: bool = False
    remove_dc: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_fft",
            _positive_fft_size(self.n_fft, name="RangeFFTSpec.n_fft"),
        )
        if not isinstance(self.window, FFTWindow):
            object.__setattr__(self, "window", FFTWindow(self.window))
        _require_bool(self.one_sided, name="RangeFFTSpec.one_sided")
        _require_bool(self.remove_dc, name="RangeFFTSpec.remove_dc")


@dataclass(frozen=True)
class DopplerFFTSpec:
    """Configuration for Doppler FFT over a named slow-time axis."""

    n_fft: int | None = None
    window: FFTWindow = FFTWindow.NONE
    fftshift: bool = True
    input_axis: str = "chirp"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_fft",
            _positive_fft_size(self.n_fft, name="DopplerFFTSpec.n_fft"),
        )
        if not isinstance(self.window, FFTWindow):
            object.__setattr__(self, "window", FFTWindow(self.window))
        _require_bool(self.fftshift, name="DopplerFFTSpec.fftshift")
        _require_axis_name(self.input_axis, name="DopplerFFTSpec.input_axis")


@dataclass(frozen=True)
class AngleFFTSpec:
    """Configuration for FFT-based angle estimation over an antenna axis."""

    n_fft: int | None = None
    window: FFTWindow = FFTWindow.NONE
    fftshift: bool = True
    input_axis: str = "rx"
    output_axis: str = "azimuth_bin"
    virtual_layout: VirtualAntennaLayout | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_fft",
            _positive_fft_size(self.n_fft, name="AngleFFTSpec.n_fft"),
        )
        if not isinstance(self.window, FFTWindow):
            object.__setattr__(self, "window", FFTWindow(self.window))
        _require_bool(self.fftshift, name="AngleFFTSpec.fftshift")
        _require_axis_name(self.input_axis, name="AngleFFTSpec.input_axis")
        _require_axis_name(self.output_axis, name="AngleFFTSpec.output_axis")


@dataclass(frozen=True)
class PlanarAngleFFTSpec:
    """Configuration for separable FFTs over a planar antenna aperture."""

    azimuth_n_fft: int | None = None
    elevation_n_fft: int | None = None
    window: FFTWindow = FFTWindow.NONE
    fftshift: bool = True
    azimuth_input_axis: str = "azimuth_aperture"
    elevation_input_axis: str = "elevation_aperture"
    azimuth_output_axis: str = "azimuth_bin"
    elevation_output_axis: str = "elevation_bin"

    def __post_init__(self) -> None:
        for name, value in (
            ("azimuth_n_fft", self.azimuth_n_fft),
            ("elevation_n_fft", self.elevation_n_fft),
        ):
            object.__setattr__(
                self,
                name,
                _positive_fft_size(value, name=f"PlanarAngleFFTSpec.{name}"),
            )
        if not isinstance(self.window, FFTWindow):
            object.__setattr__(self, "window", FFTWindow(self.window))
        _require_bool(self.fftshift, name="PlanarAngleFFTSpec.fftshift")
        for name, value in (
            ("azimuth_input_axis", self.azimuth_input_axis),
            ("elevation_input_axis", self.elevation_input_axis),
            ("azimuth_output_axis", self.azimuth_output_axis),
            ("elevation_output_axis", self.elevation_output_axis),
        ):
            _require_axis_name(value, name=f"PlanarAngleFFTSpec.{name}")
        if self.azimuth_input_axis == self.elevation_input_axis:
            raise ValueError("Planar angle input axes must be distinct.")
        if self.azimuth_output_axis == self.elevation_output_axis:
            raise ValueError("Planar angle output axes must be distinct.")
