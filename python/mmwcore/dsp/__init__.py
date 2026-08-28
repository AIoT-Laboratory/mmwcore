"""Pure offline DSP helpers for mmwcore."""

from __future__ import annotations

from .adc import organize_adc_samples
from .aoa import (
    angle_bin_angles,
    angle_fft,
    estimate_candidate_azimuths,
    estimate_candidate_elevations,
    planar_angle_fft,
)
from .calibration import apply_time_domain_channel_calibration, apply_virtual_channel_calibration
from .cartesian_pointcloud import sparsify
from .cartesian_volume import CartesianProjector
from .cfar import CFAR1DResult, detect_cfar, detect_cfar_1d, detect_range_doppler_cfar
from .clustering import cluster_point_cloud
from .clutter import remove_static_clutter
from .detection import detect_peaks
from .doppler import doppler_fft
from .grouping import group_detection_peaks
from .pointcloud import detections_to_point_cloud
from .quality import filter_detection_quality
from .range import range_fft
from .runners import (
    detect,
    point_cloud,
    process_detections_to_point_cloud,
    process_range_doppler_to_calibrated_point_cloud,
    process_range_doppler_to_detections,
    range_doppler,
)
from .virtual_array import (
    compensate_tdm_doppler_phase,
    map_planar_aperture,
    map_tdm_virtual_array,
    select_virtual_subarray,
)

__all__ = [
    "angle_fft",
    "apply_virtual_channel_calibration",
    "apply_time_domain_channel_calibration",
    "angle_bin_angles",
    "estimate_candidate_azimuths",
    "estimate_candidate_elevations",
    "compensate_tdm_doppler_phase",
    "cluster_point_cloud",
    "detect_cfar",
    "detect_cfar_1d",
    "detect_range_doppler_cfar",
    "detect_peaks",
    "detections_to_point_cloud",
    "doppler_fft",
    "group_detection_peaks",
    "filter_detection_quality",
    "map_tdm_virtual_array",
    "map_planar_aperture",
    "planar_angle_fft",
    "CartesianProjector",
    "select_virtual_subarray",
    "sparsify",
    "organize_adc_samples",
    "point_cloud",
    "detect",
    "range_doppler",
    "process_detections_to_point_cloud",
    "process_range_doppler_to_calibrated_point_cloud",
    "process_range_doppler_to_detections",
    "range_fft",
    "remove_static_clutter",
    "CFAR1DResult",
]
