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
from .cartesian_pointcloud import sparsify_cartesian_volume
from .cartesian_volume import PlanarCartesianProjector
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
    process_adc_file_to_calibrated_point_cloud,
    process_adc_file_to_detections,
    process_adc_file_to_range_doppler,
    process_adc_to_calibrated_point_cloud,
    process_adc_to_detections,
    process_adc_to_range_doppler,
    process_detections_to_point_cloud,
    process_range_doppler_to_calibrated_point_cloud,
    process_range_doppler_to_detections,
)
from .virtual_array import (
    compensate_tdm_doppler_phase,
    map_planar_aperture,
    map_tdm_virtual_array,
    select_virtual_subarray,
)
from .vitals import extract_vital_sign_phase, phase_to_displacement

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
    "PlanarCartesianProjector",
    "select_virtual_subarray",
    "sparsify_cartesian_volume",
    "organize_adc_samples",
    "process_adc_file_to_calibrated_point_cloud",
    "process_adc_file_to_detections",
    "process_adc_file_to_range_doppler",
    "process_adc_to_calibrated_point_cloud",
    "process_adc_to_detections",
    "process_adc_to_range_doppler",
    "process_detections_to_point_cloud",
    "process_range_doppler_to_calibrated_point_cloud",
    "process_range_doppler_to_detections",
    "range_fft",
    "remove_static_clutter",
    "CFAR1DResult",
    "extract_vital_sign_phase",
    "phase_to_displacement",
]
