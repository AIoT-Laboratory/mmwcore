"""Detection and point-cloud projection specs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .spec_enums import CFARInputScale, CFARMode


@dataclass(frozen=True)
class PeakDetectionSpec:
    """Threshold detection with explicit angle-domain peak selection."""

    threshold: float
    aggregate_rx: str = "max"
    azimuth_peak_radius: int = 1
    azimuth_peak_strict: bool = True

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError(
                f"PeakDetectionSpec.threshold must be non-negative; got {self.threshold}."
            )
        if self.aggregate_rx not in {"max", "sum", "mean"}:
            raise ValueError(f"Unsupported RX aggregation: {self.aggregate_rx}.")
        if self.azimuth_peak_radius < 0:
            raise ValueError("PeakDetectionSpec.azimuth_peak_radius must be non-negative.")


@dataclass(frozen=True)
class CFARDetectionSpec:
    """Cell-averaging CFAR detection over range-Doppler magnitude."""

    training_cells: int
    guard_cells: int
    threshold_scale: float
    aggregate_rx: str = "max"

    def __post_init__(self) -> None:
        if self.training_cells <= 0:
            raise ValueError(
                f"CFARDetectionSpec.training_cells must be positive; got {self.training_cells}."
            )
        if self.guard_cells < 0:
            raise ValueError(
                f"CFARDetectionSpec.guard_cells must be non-negative; got {self.guard_cells}."
            )
        if self.threshold_scale < 0:
            raise ValueError(
                "CFARDetectionSpec.threshold_scale must be non-negative; "
                f"got {self.threshold_scale}."
            )
        if self.aggregate_rx not in {"max", "sum", "mean"}:
            raise ValueError(f"Unsupported RX aggregation: {self.aggregate_rx}.")


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
        if self.training_cells <= 0:
            raise ValueError("CFAR1DSpec.training_cells must be positive.")
        if self.guard_cells < 0:
            raise ValueError("CFAR1DSpec.guard_cells must be non-negative.")
        if self.threshold_scale < 0:
            raise ValueError("CFAR1DSpec.threshold_scale must be non-negative.")
        if self.left_skip < 0 or self.right_skip < 0:
            raise ValueError("CFAR1DSpec skip lengths must be non-negative.")
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
        if self.aggregate_rx not in {"max", "sum", "mean"}:
            raise ValueError(f"Unsupported RX aggregation: {self.aggregate_rx}.")


@dataclass(frozen=True)
class PeakGroupingSpec:
    """Local-maximum policy for range-Doppler detection candidates."""

    range_radius: int = 1
    doppler_radius: int = 1
    cyclic_doppler: bool = True
    strict: bool = True
    aggregate_rx: str = "sum"

    def __post_init__(self) -> None:
        if self.range_radius < 0 or self.doppler_radius < 0:
            raise ValueError("PeakGroupingSpec radii must be non-negative.")
        if self.range_radius == 0 and self.doppler_radius == 0:
            raise ValueError("PeakGroupingSpec requires at least one non-zero radius.")
        if self.aggregate_rx not in {"max", "sum", "mean"}:
            raise ValueError(f"Unsupported RX aggregation: {self.aggregate_rx}.")


@dataclass(frozen=True)
class DetectionQualitySpec:
    """Detector-independent minimum quality policy for linear SNR channels."""

    min_snr: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_snr) or self.min_snr <= 0:
            raise ValueError("DetectionQualitySpec.min_snr must be finite and positive.")


@dataclass(frozen=True)
class PointCloudProjectionSpec:
    """Projection from calibrated range-Doppler-angle detections to Cartesian points."""

    range_resolution_m: float = 1.0
    doppler_resolution_mps: float = 1.0
    center_doppler: bool = False
    doppler_bins: int | None = None
    doppler_fftshifted: bool = False

    def __post_init__(self) -> None:
        if self.range_resolution_m <= 0:
            raise ValueError(
                "PointCloudProjectionSpec.range_resolution_m must be positive; "
                f"got {self.range_resolution_m}."
            )
        if self.doppler_resolution_mps <= 0:
            raise ValueError(
                "PointCloudProjectionSpec.doppler_resolution_mps must be positive; "
                f"got {self.doppler_resolution_mps}."
            )
        if self.doppler_bins is not None and self.doppler_bins <= 0:
            raise ValueError(
                f"PointCloudProjectionSpec.doppler_bins must be positive; got {self.doppler_bins}."
            )
        if self.center_doppler and self.doppler_bins is None:
            raise ValueError("PointCloudProjectionSpec.doppler_bins is required when centering.")
