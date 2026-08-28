"""Configuration contracts for mmwcore offline processing."""

from __future__ import annotations

from .capture import RADAR_CAPTURE_SPEC_SCHEMA, RadarCaptureSpec
from .parsers import parse_ti_cli_capture_spec
from .presets import (
    iwr6843_isk_3d_cfar_point_cloud_pipeline,
    iwr6843_isk_3d_point_cloud_pipeline,
    iwr6843_isk_antenna_geometry,
    iwr6843_isk_azimuth_subarray,
    iwr6843_isk_cfar_point_cloud_pipeline,
    iwr6843_isk_detection_pipeline,
    iwr6843_isk_elevation_subarray,
    iwr6843_isk_planar_aperture_layout,
    iwr6843_isk_point_cloud_pipeline,
    iwr6843_isk_range_doppler_pipeline,
    iwr6843_isk_tdm_virtual_array,
    iwr6843_profile,
)
from .profiles import RadarProfile

__all__ = [
    "RADAR_CAPTURE_SPEC_SCHEMA",
    "RadarCaptureSpec",
    "RadarProfile",
    "iwr6843_profile",
    "iwr6843_isk_3d_cfar_point_cloud_pipeline",
    "iwr6843_isk_3d_point_cloud_pipeline",
    "iwr6843_isk_antenna_geometry",
    "iwr6843_isk_azimuth_subarray",
    "iwr6843_isk_cfar_point_cloud_pipeline",
    "iwr6843_isk_detection_pipeline",
    "iwr6843_isk_elevation_subarray",
    "iwr6843_isk_point_cloud_pipeline",
    "iwr6843_isk_planar_aperture_layout",
    "iwr6843_isk_range_doppler_pipeline",
    "iwr6843_isk_tdm_virtual_array",
    "parse_ti_cli_capture_spec",
]
