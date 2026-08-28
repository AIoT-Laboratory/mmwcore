"""Core data contracts and processing specs for mmwcore."""

from __future__ import annotations

from .calibration import TimeDomainChannelCalibration, VirtualChannelCalibration
from .recipes import ADCDecodeSpec, DetectionPipeline, PointCloudPipeline, RangeDopplerPipeline
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
from .spec_pointcloud import SparsifySpec
from .spec_tracking import (
    AllocationSpec,
    Box2D,
    DBSCANSpec,
    GatingSpec,
    LifecycleSpec,
    ScenerySpec,
    Tracker2DSpec,
    TrackStatus,
)
from .types import (
    ADCFrame,
    CartesianVolume,
    ClusterFrame,
    DetectionFrame,
    PointCloudFrame,
    RadarCube,
    TrackFrame,
)

__all__ = [
    "ADCComplexLayout",
    "ADCDecodeSpec",
    "ADCFrameSpec",
    "AntennaArrayGeometry",
    "CascadeADCFrameSpec",
    "CartesianVolume",
    "SparsifySpec",
    "AngleFFTSpec",
    "CFARDetectionSpec",
    "CFAR1DSpec",
    "CFARInputScale",
    "CFARMode",
    "ClusterFrame",
    "Tracker2DSpec",
    "DBSCANSpec",
    "DetectionFrame",
    "DetectionQualitySpec",
    "DetectionMethod",
    "DopplerFFTSpec",
    "FFTWindow",
    "DetectionPipeline",
    "PeakDetectionSpec",
    "PeakGroupingSpec",
    "PointCloudFrame",
    "PointCloudProjectionSpec",
    "PointCloudPipeline",
    "PlanarAngleFFTSpec",
    "PlanarApertureLayout",
    "RadarCube",
    "RangeFFTSpec",
    "RangeDopplerPipeline",
    "RangeDopplerCFARSpec",
    "ADCFrame",
    "TDMVirtualArraySpec",
    "TimeDomainChannelCalibration",
    "AllocationSpec",
    "TrackFrame",
    "GatingSpec",
    "LifecycleSpec",
    "ScenerySpec",
    "TrackStatus",
    "Box2D",
    "VirtualAntennaLayout",
    "VirtualChannelCalibration",
    "VirtualSubarraySpec",
]
