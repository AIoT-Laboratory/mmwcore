"""Detection and point-cloud projection specs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
from sys import maxsize as _MAX_PLATFORM_INDEX

from .spec_enums import CFARInputScale, CFARMode

_AGGREGATE_RX_CHOICES = frozenset({"max", "sum", "mean"})


def _platform_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        normalized = int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if not -_MAX_PLATFORM_INDEX - 1 <= normalized <= _MAX_PLATFORM_INDEX:
        raise OverflowError(f"{name} must fit the platform index range.")
    return normalized


def _positive_integer(value: int, *, name: str) -> int:
    normalized = _platform_integer(value, name=name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive; got {normalized}.")
    return normalized


def _non_negative_integer(value: int, *, name: str) -> int:
    normalized = _platform_integer(value, name=name)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative; got {normalized}.")
    return normalized


def _require_finite_positive(value: float, *, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive; got {value}.")


def _require_finite_non_negative(value: float, *, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative; got {value}.")


def _require_bool(value: bool, *, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _require_aggregate_rx(value: str, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value not in _AGGREGATE_RX_CHOICES:
        raise ValueError(f"{name} must be one of max, sum, or mean; got {value!r}.")


@dataclass(frozen=True)
class PeakDetectionSpec:
    """Threshold detection with explicit angle-domain peak selection."""

    threshold: float
    aggregate_rx: str = "max"
    azimuth_peak_radius: int = 1
    azimuth_peak_strict: bool = True

    def __post_init__(self) -> None:
        _require_finite_non_negative(self.threshold, name="PeakDetectionSpec.threshold")
        _require_aggregate_rx(self.aggregate_rx, name="PeakDetectionSpec.aggregate_rx")
        object.__setattr__(
            self,
            "azimuth_peak_radius",
            _non_negative_integer(
                self.azimuth_peak_radius,
                name="PeakDetectionSpec.azimuth_peak_radius",
            ),
        )
        _require_bool(
            self.azimuth_peak_strict,
            name="PeakDetectionSpec.azimuth_peak_strict",
        )


@dataclass(frozen=True)
class CFARDetectionSpec:
    """Cell-averaging CFAR detection over range-Doppler magnitude."""

    training_cells: int
    guard_cells: int
    threshold_scale: float
    aggregate_rx: str = "max"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_cells",
            _positive_integer(
                self.training_cells,
                name="CFARDetectionSpec.training_cells",
            ),
        )
        object.__setattr__(
            self,
            "guard_cells",
            _non_negative_integer(
                self.guard_cells,
                name="CFARDetectionSpec.guard_cells",
            ),
        )
        _require_finite_non_negative(
            self.threshold_scale,
            name="CFARDetectionSpec.threshold_scale",
        )
        _require_aggregate_rx(self.aggregate_rx, name="CFARDetectionSpec.aggregate_rx")


@dataclass(frozen=True)
class CFAR1DSpec:
    """One-dimensional CFAR window and threshold policy."""

    training_cells: int
    guard_cells: int
    threshold_scale: float
    mode: CFARMode = CFARMode.CA
    cyclic: bool = False
    left_skip: int = 0
    right_skip: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_cells",
            _positive_integer(self.training_cells, name="CFAR1DSpec.training_cells"),
        )
        object.__setattr__(
            self,
            "guard_cells",
            _non_negative_integer(self.guard_cells, name="CFAR1DSpec.guard_cells"),
        )
        _require_finite_non_negative(
            self.threshold_scale,
            name="CFAR1DSpec.threshold_scale",
        )
        object.__setattr__(
            self,
            "left_skip",
            _non_negative_integer(self.left_skip, name="CFAR1DSpec.left_skip"),
        )
        object.__setattr__(
            self,
            "right_skip",
            _non_negative_integer(self.right_skip, name="CFAR1DSpec.right_skip"),
        )
        _require_bool(self.cyclic, name="CFAR1DSpec.cyclic")
        if not isinstance(self.mode, CFARMode):
            object.__setattr__(self, "mode", CFARMode(self.mode))


@dataclass(frozen=True)
class RangeDopplerCFARSpec:
    """Floating-point CFAR policy for range-Doppler research tensors."""

    range: CFAR1DSpec
    doppler: CFAR1DSpec | None = None
    input_scale: CFARInputScale = CFARInputScale.POWER
    aggregate_rx: str = "sum"

    def __post_init__(self) -> None:
        if not isinstance(self.input_scale, CFARInputScale):
            object.__setattr__(self, "input_scale", CFARInputScale(self.input_scale))
        _require_aggregate_rx(self.aggregate_rx, name="RangeDopplerCFARSpec.aggregate_rx")


@dataclass(frozen=True)
class PeakGroupingSpec:
    """Local-maximum policy for range-Doppler detection candidates."""

    range_radius: int = 1
    doppler_radius: int = 1
    cyclic_doppler: bool = True
    strict: bool = True
    aggregate_rx: str = "sum"

    def __post_init__(self) -> None:
        range_radius = _non_negative_integer(
            self.range_radius,
            name="PeakGroupingSpec.range_radius",
        )
        doppler_radius = _non_negative_integer(
            self.doppler_radius,
            name="PeakGroupingSpec.doppler_radius",
        )
        object.__setattr__(self, "range_radius", range_radius)
        object.__setattr__(self, "doppler_radius", doppler_radius)
        _require_bool(self.cyclic_doppler, name="PeakGroupingSpec.cyclic_doppler")
        _require_bool(self.strict, name="PeakGroupingSpec.strict")
        if self.range_radius == 0 and self.doppler_radius == 0:
            raise ValueError("PeakGroupingSpec requires at least one non-zero radius.")
        _require_aggregate_rx(self.aggregate_rx, name="PeakGroupingSpec.aggregate_rx")


@dataclass(frozen=True)
class DetectionQualitySpec:
    """Detector-independent minimum quality policy for linear SNR channels."""

    min_snr: float

    def __post_init__(self) -> None:
        _require_finite_positive(self.min_snr, name="DetectionQualitySpec.min_snr")


@dataclass(frozen=True)
class PointCloudProjectionSpec:
    """Projection to Cartesian points whose positive radial velocity points away."""

    range_resolution_m: float = 1.0
    doppler_resolution_mps: float = 1.0
    doppler_sign: int = 1
    center_doppler: bool = False
    doppler_bins: int | None = None
    doppler_fftshifted: bool = False

    def __post_init__(self) -> None:
        _require_finite_positive(
            self.range_resolution_m,
            name="PointCloudProjectionSpec.range_resolution_m",
        )
        _require_finite_positive(
            self.doppler_resolution_mps,
            name="PointCloudProjectionSpec.doppler_resolution_mps",
        )
        doppler_sign = _platform_integer(
            self.doppler_sign,
            name="PointCloudProjectionSpec.doppler_sign",
        )
        if doppler_sign not in {-1, 1}:
            raise ValueError("PointCloudProjectionSpec.doppler_sign must be -1 or 1")
        object.__setattr__(self, "doppler_sign", doppler_sign)
        if self.doppler_bins is not None:
            object.__setattr__(
                self,
                "doppler_bins",
                _positive_integer(
                    self.doppler_bins,
                    name="PointCloudProjectionSpec.doppler_bins",
                ),
            )
        _require_bool(self.center_doppler, name="PointCloudProjectionSpec.center_doppler")
        _require_bool(
            self.doppler_fftshifted,
            name="PointCloudProjectionSpec.doppler_fftshifted",
        )
        if self.center_doppler and self.doppler_bins is None:
            raise ValueError("PointCloudProjectionSpec.doppler_bins is required when centering.")
