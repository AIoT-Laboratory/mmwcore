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

    def __post_init__(self) -> None:
        _require_instance(self.adc, ADCFrameSpec, name="ADCDecodeRecipe.adc")
        _require_builtin_bool(
            self.drop_incomplete,
            name="ADCDecodeRecipe.drop_incomplete",
        )


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
        _require_instance(self.decode, ADCDecodeRecipe, name="RangeDopplerRecipe.decode")
        _require_instance(self.range_fft, RangeFFTSpec, name="RangeDopplerRecipe.range_fft")
        _require_instance(
            self.doppler_fft,
            DopplerFFTSpec,
            name="RangeDopplerRecipe.doppler_fft",
        )
        if self.tdm_virtual_array is not None:
            _require_instance(
                self.tdm_virtual_array,
                TDMVirtualArraySpec,
                name="RangeDopplerRecipe.tdm_virtual_array",
            )
        if self.channel_calibration is not None:
            _require_instance(
                self.channel_calibration,
                VirtualChannelCalibration,
                name="RangeDopplerRecipe.channel_calibration",
            )
        _require_builtin_bool(
            self.remove_static_clutter,
            name="RangeDopplerRecipe.remove_static_clutter",
        )
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
        _require_instance(
            self.transform,
            RangeDopplerRecipe,
            name="DetectionRecipe.transform",
        )
        if self.peak_detection is not None:
            _require_instance(
                self.peak_detection,
                PeakDetectionSpec,
                name="DetectionRecipe.peak_detection",
            )
        if self.cfar_detection is not None:
            _require_instance(
                self.cfar_detection,
                (CFARDetectionSpec, RangeDopplerCFARSpec),
                name="DetectionRecipe.cfar_detection",
            )
        if self.peak_grouping is not None:
            _require_instance(
                self.peak_grouping,
                PeakGroupingSpec,
                name="DetectionRecipe.peak_grouping",
            )
        if self.quality_filter is not None:
            _require_instance(
                self.quality_filter,
                DetectionQualitySpec,
                name="DetectionRecipe.quality_filter",
            )
        if self.angle_fft is not None:
            _require_instance(
                self.angle_fft,
                AngleFFTSpec,
                name="DetectionRecipe.angle_fft",
            )
        if self.virtual_subarray is not None:
            _require_instance(
                self.virtual_subarray,
                VirtualSubarraySpec,
                name="DetectionRecipe.virtual_subarray",
            )
        if self.elevation_subarray is not None:
            _require_instance(
                self.elevation_subarray,
                VirtualSubarraySpec,
                name="DetectionRecipe.elevation_subarray",
            )
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
        _require_instance(
            self.detection,
            DetectionRecipe,
            name="PointCloudRecipe.detection",
        )
        _require_instance(
            self.projection,
            PointCloudProjectionSpec,
            name="PointCloudRecipe.projection",
        )
        angle_fft = self.detection.angle_fft
        if angle_fft is None or angle_fft.virtual_layout is None:
            raise ValueError("PointCloudRecipe requires a calibrated virtual antenna layout.")


def _require_builtin_bool(value: object, *, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool.")


def _require_instance(
    value: object,
    expected_type: type[object] | tuple[type[object], ...],
    *,
    name: str,
) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must use the declared contract type.")


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
