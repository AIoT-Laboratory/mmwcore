"""Composable offline radar processing recipes."""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import VirtualChannelCalibration
from .spec_adc import ADCFrameSpec, TDMVirtualArraySpec, VirtualSubarraySpec
from .spec_detection import (
    CFARDetectionSpec,
    DetectionQualitySpec,
    PeakDetectionSpec,
    PeakGroupingSpec,
    PointCloudProjectionSpec,
    RangeDopplerCFARSpec,
)
from .spec_enums import DetectionMethod
from .spec_fft import AngleFFTSpec, DopplerFFTSpec, RangeFFTSpec


@dataclass(frozen=True)
class ADCDecodeRecipe:
    """Decode raw ADC words into the canonical complex radar cube."""

    adc: ADCFrameSpec
    drop_incomplete: bool = False


@dataclass(frozen=True)
class RangeDopplerRecipe:
    """Transform a decoded radar cube into range-Doppler space."""

    decode: ADCDecodeRecipe
    range_fft: RangeFFTSpec = RangeFFTSpec()
    doppler_fft: DopplerFFTSpec = DopplerFFTSpec()
    tdm_virtual_array: TDMVirtualArraySpec | None = None
    channel_calibration: VirtualChannelCalibration | None = None
    remove_static_clutter: bool = False

    def __post_init__(self) -> None:
        if self.tdm_virtual_array is not None and self.doppler_fft.input_axis != "loop":
            raise ValueError('TDM RangeDopplerRecipe requires DopplerFFTSpec(input_axis="loop").')
        if self.channel_calibration is not None and self.tdm_virtual_array is None:
            raise ValueError("Virtual-channel calibration requires a TDM virtual array.")
        if (
            self.channel_calibration is not None
            and self.tdm_virtual_array is not None
            and self.channel_calibration.num_channels != self.tdm_virtual_array.num_virtual_antennas
        ):
            raise ValueError("Calibration channel count must match the TDM virtual array.")


@dataclass(frozen=True)
class DetectionRecipe:
    """Transform ADC samples into detector-domain targets."""

    transform: RangeDopplerRecipe
    peak_detection: PeakDetectionSpec | None = None
    detection_method: DetectionMethod = DetectionMethod.THRESHOLD
    cfar_detection: CFARDetectionSpec | RangeDopplerCFARSpec | None = None
    peak_grouping: PeakGroupingSpec | None = None
    quality_filter: DetectionQualitySpec | None = None
    angle_fft: AngleFFTSpec | None = None
    virtual_subarray: VirtualSubarraySpec | None = None
    elevation_subarray: VirtualSubarraySpec | None = None

    def __post_init__(self) -> None:
        method = _normalize_detection_method(self.detection_method)
        object.__setattr__(self, "detection_method", method)
        _validate_detector(self, method=method)
        _validate_azimuth_subarray(self)
        _validate_elevation_subarray(self)


@dataclass(frozen=True)
class PointCloudRecipe:
    """Transform ADC samples into a calibrated Cartesian point cloud."""

    detection: DetectionRecipe
    projection: PointCloudProjectionSpec = PointCloudProjectionSpec()

    def __post_init__(self) -> None:
        angle_fft = self.detection.angle_fft
        if angle_fft is None or angle_fft.virtual_layout is None:
            raise ValueError("PointCloudRecipe requires a calibrated virtual antenna layout.")


def _normalize_detection_method(value: DetectionMethod) -> DetectionMethod:
    return value if isinstance(value, DetectionMethod) else DetectionMethod(value)


def _validate_detector(recipe: DetectionRecipe, *, method: DetectionMethod) -> None:
    if method is DetectionMethod.CFAR and recipe.cfar_detection is None:
        raise ValueError(
            "DetectionRecipe.cfar_detection is required when detection_method is cfar."
        )
    if method is DetectionMethod.THRESHOLD and recipe.peak_detection is None:
        raise ValueError("DetectionRecipe.peak_detection is required for threshold detection.")
    if recipe.peak_grouping is not None and method is not DetectionMethod.CFAR:
        raise ValueError("DetectionRecipe.peak_grouping currently requires CFAR detection.")


def _validate_azimuth_subarray(recipe: DetectionRecipe) -> None:
    subarray = recipe.virtual_subarray
    if subarray is None:
        return
    angle_spec = recipe.angle_fft
    if angle_spec is None:
        raise ValueError("DetectionRecipe.virtual_subarray requires angle_fft.")
    if angle_spec.input_axis != "virtual_rx":
        raise ValueError('Virtual subarray angle FFT must use input_axis="virtual_rx".')
    if angle_spec.virtual_layout != subarray.layout:
        raise ValueError("AngleFFTSpec.virtual_layout must match the virtual subarray layout.")


def _validate_elevation_subarray(recipe: DetectionRecipe) -> None:
    if recipe.elevation_subarray is None:
        return
    if recipe.virtual_subarray is None or recipe.angle_fft is None:
        raise ValueError("DetectionRecipe.elevation_subarray requires calibrated azimuth AoA.")
    if recipe.angle_fft.input_axis != "virtual_rx":
        raise ValueError('Elevation estimation requires angle_fft input_axis="virtual_rx".')
