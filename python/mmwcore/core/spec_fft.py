"""FFT processing specs."""

from __future__ import annotations

from dataclasses import dataclass

from .spec_adc import VirtualAntennaLayout
from .spec_enums import FFTWindow


@dataclass(frozen=True)
class RangeFFTSpec:
    """Configuration for range FFT over the ADC sample axis."""

    n_fft: int | None = None
    window: FFTWindow = FFTWindow.NONE
    one_sided: bool = False
    remove_dc: bool = False

    def __post_init__(self) -> None:
        if self.n_fft is not None and self.n_fft <= 0:
            raise ValueError(f"RangeFFTSpec.n_fft must be positive; got {self.n_fft}.")
        if not isinstance(self.window, FFTWindow):
            object.__setattr__(self, "window", FFTWindow(self.window))


@dataclass(frozen=True)
class DopplerFFTSpec:
    """Configuration for Doppler FFT over a named slow-time axis."""

    n_fft: int | None = None
    window: FFTWindow = FFTWindow.NONE
    fftshift: bool = True
    input_axis: str = "chirp"

    def __post_init__(self) -> None:
        if self.n_fft is not None and self.n_fft <= 0:
            raise ValueError(f"DopplerFFTSpec.n_fft must be positive; got {self.n_fft}.")
        if not isinstance(self.window, FFTWindow):
            object.__setattr__(self, "window", FFTWindow(self.window))
        if not self.input_axis:
            raise ValueError("DopplerFFTSpec.input_axis must not be empty.")


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
        if self.n_fft is not None and self.n_fft <= 0:
            raise ValueError(f"AngleFFTSpec.n_fft must be positive; got {self.n_fft}.")
        if not isinstance(self.window, FFTWindow):
            object.__setattr__(self, "window", FFTWindow(self.window))
        if not self.input_axis:
            raise ValueError("AngleFFTSpec.input_axis must not be empty.")
        if not self.output_axis:
            raise ValueError("AngleFFTSpec.output_axis must not be empty.")


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
            if value is not None and value <= 0:
                raise ValueError(f"PlanarAngleFFTSpec.{name} must be positive; got {value}.")
        if not isinstance(self.window, FFTWindow):
            object.__setattr__(self, "window", FFTWindow(self.window))
        axes = (
            self.azimuth_input_axis,
            self.elevation_input_axis,
            self.azimuth_output_axis,
            self.elevation_output_axis,
        )
        if any(not axis for axis in axes):
            raise ValueError("PlanarAngleFFTSpec axis names must not be empty.")
        if self.azimuth_input_axis == self.elevation_input_axis:
            raise ValueError("Planar angle input axes must be distinct.")
        if self.azimuth_output_axis == self.elevation_output_axis:
            raise ValueError("Planar angle output axes must be distinct.")
