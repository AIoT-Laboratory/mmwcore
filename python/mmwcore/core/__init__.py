"""Core data contracts and processing specs for mmwcore."""

from __future__ import annotations

from .calibration import TimeDomainChannelCalibration, VirtualChannelCalibration
from .recipes import ADCDecodeRecipe, DetectionRecipe, PointCloudRecipe, RangeDopplerRecipe
from .spec_adc import (
    ADCFrameSpec,
    AntennaArrayGeometry,
    CascadeADCFrameSpec,
    PlanarApertureLayout,
    TDMVirtualArraySpec,
    VirtualAntennaLayout,
    VirtualSubarraySpec,
)
from .spec_detection import (
    CFAR1DSpec,
    CFARDetectionSpec,
    DetectionQualitySpec,
    PeakDetectionSpec,
    PeakGroupingSpec,
    PointCloudProjectionSpec,
    RangeDopplerCFARSpec,
)
from .spec_enums import ADCComplexLayout, CFARInputScale, CFARMode, DetectionMethod, FFTWindow
from .spec_fft import AngleFFTSpec, DopplerFFTSpec, PlanarAngleFFTSpec, RangeFFTSpec
from .spec_pointcloud import CartesianVolumeSparsificationSpec
from .spec_tracking import (
    DBSCANClusteringSpec,
    TrackAllocationSpec,
    Tracker2DSpec,
    TrackGatingSpec,
    TrackingBox2D,
    TrackLifecycleSpec,
    TrackScenerySpec,
    TrackStatus,
)
from .types import (
    CartesianRadarVolume,
    ClusterFrame,
    DetectionFrame,
    PointCloudFrame,
    RadarCube,
    RawADCFrame,
    TrackFrame,
)
from .vitals import VitalSignQuantity, VitalSignWaveform

__all__ = [
    "ADCComplexLayout",
    "ADCDecodeRecipe",
    "ADCFrameSpec",
    "AntennaArrayGeometry",
    "CascadeADCFrameSpec",
    "CartesianRadarVolume",
    "CartesianVolumeSparsificationSpec",
    "AngleFFTSpec",
    "CFARDetectionSpec",
    "CFAR1DSpec",
    "CFARInputScale",
    "CFARMode",
    "ClusterFrame",
    "Tracker2DSpec",
    "DBSCANClusteringSpec",
    "DetectionFrame",
    "DetectionQualitySpec",
    "DetectionMethod",
    "DopplerFFTSpec",
    "FFTWindow",
    "DetectionRecipe",
    "PeakDetectionSpec",
    "PeakGroupingSpec",
    "PointCloudFrame",
    "PointCloudProjectionSpec",
    "PointCloudRecipe",
    "PlanarAngleFFTSpec",
    "PlanarApertureLayout",
    "RadarCube",
    "RangeFFTSpec",
    "RangeDopplerRecipe",
    "RangeDopplerCFARSpec",
    "RawADCFrame",
    "TDMVirtualArraySpec",
    "TimeDomainChannelCalibration",
    "TrackAllocationSpec",
    "TrackFrame",
    "TrackGatingSpec",
    "TrackLifecycleSpec",
    "TrackScenerySpec",
    "TrackStatus",
    "TrackingBox2D",
    "VirtualAntennaLayout",
    "VirtualChannelCalibration",
    "VirtualSubarraySpec",
    "VitalSignQuantity",
    "VitalSignWaveform",
]
