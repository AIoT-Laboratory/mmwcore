"""Configuration contracts for mmwcore offline processing."""

from __future__ import annotations

from .capture import RADAR_CAPTURE_SPEC_SCHEMA, RadarCaptureSpec
from .parsers import (
    parse_ti_cli_capture_spec,
    parse_ti_cli_capture_spec_file,
)
from .presets import (
    awr1843_aop_antenna_geometry,
    iwr6843_aop_antenna_geometry,
    iwr6843_isk_3d_cfar_point_cloud_recipe,
    iwr6843_isk_3d_point_cloud_recipe,
    iwr6843_isk_antenna_geometry,
    iwr6843_isk_azimuth_subarray,
    iwr6843_isk_cfar_point_cloud_recipe,
    iwr6843_isk_detection_recipe,
    iwr6843_isk_elevation_subarray,
    iwr6843_isk_planar_aperture_layout,
    iwr6843_isk_point_cloud_recipe,
    iwr6843_isk_range_doppler_recipe,
    iwr6843_isk_tdm_virtual_array,
    iwr6843_profile,
    xwr1642_antenna_geometry,
    xwr1843_evm_antenna_geometry,
)
from .profiles import RadarProfile

__all__ = [
    "RADAR_CAPTURE_SPEC_SCHEMA",
    "RadarCaptureSpec",
    "RadarProfile",
    "awr1843_aop_antenna_geometry",
    "iwr6843_aop_antenna_geometry",
    "iwr6843_profile",
    "iwr6843_isk_3d_cfar_point_cloud_recipe",
    "iwr6843_isk_3d_point_cloud_recipe",
    "iwr6843_isk_antenna_geometry",
    "iwr6843_isk_azimuth_subarray",
    "iwr6843_isk_cfar_point_cloud_recipe",
    "iwr6843_isk_detection_recipe",
    "iwr6843_isk_elevation_subarray",
    "iwr6843_isk_point_cloud_recipe",
    "iwr6843_isk_planar_aperture_layout",
    "iwr6843_isk_range_doppler_recipe",
    "iwr6843_isk_tdm_virtual_array",
    "xwr1642_antenna_geometry",
    "xwr1843_evm_antenna_geometry",
    "parse_ti_cli_capture_spec",
    "parse_ti_cli_capture_spec_file",
]
