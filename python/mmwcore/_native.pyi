# ruff: noqa: UP040

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

DCA1000PacketResult: TypeAlias = tuple[int, int, NDArray[np.int16]]
DCA1000AssemblyResult: TypeAlias = tuple[
    NDArray[np.int16],
    int,
    int,
    list[int],
    list[int],
    list[int],
]
NativeThresholdDetections: TypeAlias = tuple[NDArray[np.int64], NDArray[np.float32]]
NativeDetectionAxes: TypeAlias = tuple[int, int, int, int]
NativeDetectionIndexColumns: TypeAlias = tuple[int, int, int]
NativePeakGroupingConfig: TypeAlias = tuple[int, int, bool, bool]
NativeCfar1DConfig: TypeAlias = tuple[int, int, float, int, bool, int, int]
NativeCfar2DConfig: TypeAlias = tuple[int, int, float]
NativeCfar1DResult: TypeAlias = tuple[NDArray[np.int64], NDArray[np.float32]]
NativeCfarDetections: TypeAlias = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]
NativePointColumns: TypeAlias = tuple[int, int, int, int | None]
NativeDbscanConfig: TypeAlias = tuple[float, int, float, bool]
NativeClusterResult: TypeAlias = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int64],
]
NativeTrackerDynamicsConfig: TypeAlias = tuple[float, tuple[float, float], float, float, float]
NativeTrackerGatingConfig: TypeAlias = tuple[float, float | None, float | None]
NativeTrackerAllocationConfig: TypeAlias = tuple[int, float, float | None, int | None]
NativeTrackerLifecycleConfig: TypeAlias = tuple[int, int, int]
NativeTrackingBox: TypeAlias = tuple[float, float, float, float]
NativeTrackerSceneryConfig: TypeAlias = tuple[list[NativeTrackingBox], int]
NativeClusterTrackerConfig: TypeAlias = tuple[
    NativeTrackerDynamicsConfig,
    NativeTrackerGatingConfig,
    NativeTrackerAllocationConfig,
    NativeTrackerLifecycleConfig,
    NativeTrackerSceneryConfig,
    int,
]
NativeMeasurementTrackerConfig: TypeAlias = tuple[NativeClusterTrackerConfig, NativeDbscanConfig]
NativeTrackerStepResult: TypeAlias = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.uint8],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]
NativeTrackingMetricsHeader: TypeAlias = tuple[int, int, int, int]
NativeTrackingMetricsInput: TypeAlias = tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.uint8],
]
NativeTrackingMetricsIdentity: TypeAlias = tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]
NativeTrackingMetricsMotion: TypeAlias = tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]
NativeTrackingMetricsIntervals: TypeAlias = tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]
NativeTrackingMetricsResult: TypeAlias = tuple[
    NativeTrackingMetricsHeader,
    NativeTrackingMetricsIdentity,
    NativeTrackingMetricsMotion,
    NativeTrackingMetricsIntervals,
]

def summarize_tracking_metrics(
    arrays: NativeTrackingMetricsInput,
    scenery_boxes: list[NativeTrackingBox] | None,
    frame_index_offset: int,
) -> NativeTrackingMetricsResult: ...
def unwrap_vital_phase(
    samples: NDArray[np.complex64],
    remove_mean: bool,
) -> NDArray[np.float32]: ...
def vital_phase_to_displacement(
    phase_rad: NDArray[np.float32],
    wavelength_m: float,
) -> NDArray[np.float32]: ...

NativeAssignmentResult: TypeAlias = tuple[NDArray[np.int64], NDArray[np.int64]]
NativePlanarCartesianResult: TypeAlias = tuple[
    NDArray[np.float32],
    int,
    int,
    int,
    int,
    int,
    int,
]
NativeDopplerAxis: TypeAlias = tuple[int, float, float]
NativeGridShape: TypeAlias = tuple[int, int, int]
NativeGridCoordinates: TypeAlias = tuple[float, float, float]
NativePlanarAngleConfig: TypeAlias = tuple[int, int, float]
NativePlanarCartesianFfiConfig: TypeAlias = tuple[
    float,
    NativeDopplerAxis,
    NativeDopplerAxis,
    NativeGridShape,
    NativeGridCoordinates,
    NativeGridCoordinates,
    NativePlanarAngleConfig,
]
NativeCartesianAxes: TypeAlias = tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]
NativeCartesianSparsificationThresholdConfig: TypeAlias = tuple[float, int]
NativeCartesianSparsificationPeakConfig: TypeAlias = tuple[int, int, int | None, int]
NativeCartesianSparsificationBackgroundConfig: TypeAlias = tuple[float, float, float, bool]
NativeCartesianSparsificationConfig: TypeAlias = tuple[
    NativeCartesianSparsificationThresholdConfig,
    NativeCartesianSparsificationPeakConfig,
    NativeCartesianSparsificationBackgroundConfig,
]
NativeCartesianSparsificationResult: TypeAlias = tuple[
    NDArray[np.float32],
    tuple[float, float, float],
    tuple[int, int, int, int, int, int, int],
    tuple[bool, int],
]
NativeDetectionPointCloudColumns: TypeAlias = tuple[
    int,
    int,
    int,
    int,
    int,
    tuple[int, int] | None,
    list[int],
]
NativeDetectionPointCloudConfig: TypeAlias = tuple[float, float, bool, int | None, bool]
NativeCandidateCubeAxes: TypeAlias = tuple[int, int, int, int]
NativeCandidateIndexColumns: TypeAlias = tuple[int, int, int]
NativeCandidateElevationColumns: TypeAlias = tuple[int, int, int, int, int]
NativeCandidateAzimuthConfig: TypeAlias = tuple[int, int, bool, int]
NativeCandidateSubarrays: TypeAlias = tuple[
    list[int],
    list[int],
    NDArray[np.float64],
    NDArray[np.float64],
]
NativeCandidateElevationConfig: TypeAlias = tuple[int, int, bool]
NativeCandidateAzimuthResult: TypeAlias = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
]
NativeCandidateElevationResult: TypeAlias = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    tuple[float, float],
]

class NativeClusterTracker2D:
    def __init__(self, config: NativeClusterTrackerConfig) -> None: ...
    def step(
        self,
        centers: NDArray[np.float32],
        extents: NDArray[np.float32],
        mean_velocities: NDArray[np.float32],
        point_counts: NDArray[np.int64],
    ) -> NativeTrackerStepResult: ...

class NativeMeasurementTracker2D:
    def __init__(self, config: NativeMeasurementTrackerConfig) -> None: ...
    def step(
        self,
        coordinates: NDArray[np.float32],
        velocities: NDArray[np.float32],
        snrs: NDArray[np.float32],
    ) -> NativeTrackerStepResult: ...

def decode_adc_i16(
    samples: NDArray[np.int16],
    num_chirps: int,
    num_rx: int,
    num_samples: int,
    layout: int,
    drop_incomplete: bool,
) -> NDArray[np.complex64]: ...
def parse_dca1000_packet(data: bytes) -> DCA1000PacketResult: ...
def reorder_dca1000_packets(
    packet_numbers: NDArray[np.int64],
    payloads: Sequence[NDArray[np.int16]],
    packets_per_frame: int,
    payload_values_per_packet: int | None,
    fill_value: int,
) -> DCA1000AssemblyResult: ...
def assemble_dca1000_frame_bytes(
    packets: Sequence[bytes],
    raw_values_per_frame: int,
    payload_values_per_packet: int,
    fill_value: int,
) -> DCA1000AssemblyResult: ...
def remove_static_clutter_complex(
    data: NDArray[np.complex64],
    axis: int,
) -> NDArray[np.complex64]: ...
def fft_complex_axis(
    data: NDArray[np.complex64],
    axis: int,
    n_fft: int,
    window: int,
    flags: int,
) -> NDArray[np.complex64]: ...
def range_doppler_magnitude_complex(
    data: NDArray[np.complex64],
    axes: tuple[int, int, int, int],
    aggregation: int,
) -> NDArray[np.float32]: ...
def threshold_range_doppler_complex(
    data: NDArray[np.complex64],
    axes: tuple[int, int, int, int],
    aggregation: int,
    threshold: float,
) -> NativeThresholdDetections: ...
def threshold_range_doppler_azimuth_complex(
    data: NDArray[np.complex64],
    axes: tuple[int, int, int, int],
    threshold: float,
    azimuth_peak_radius: int,
    azimuth_peak_strict: bool,
) -> NativeThresholdDetections: ...
def group_range_doppler_candidates(
    data: NDArray[np.complex64],
    axes: NativeDetectionAxes,
    aggregation: int,
    candidates: NDArray[np.float32],
    columns: NativeDetectionIndexColumns,
    config: NativePeakGroupingConfig,
) -> NDArray[np.int64]: ...
def filter_detection_quality_rows(
    candidates: NDArray[np.float32],
    snr_column: int,
    min_snr: float,
) -> NDArray[np.int64]: ...
def detect_cfar_1d(
    power: NDArray[np.float32],
    config: NativeCfar1DConfig,
) -> NativeCfar1DResult: ...
def detect_range_doppler_cfar_complex(
    data: NDArray[np.complex64],
    axes: tuple[int, int, int, int],
    aggregation: int,
    range_config: NativeCfar1DConfig,
    doppler_config: NativeCfar1DConfig | None,
    input_scale: int,
) -> NativeCfarDetections: ...
def detect_cfar_2d_complex(
    data: NDArray[np.complex64],
    axes: tuple[int, int, int, int],
    aggregation: int,
    config: NativeCfar2DConfig,
) -> NativeCfarDetections: ...
def cluster_points(
    points: NDArray[np.float32],
    columns: NativePointColumns,
    config: NativeDbscanConfig,
) -> NativeClusterResult: ...
def linear_sum_assignment(costs: NDArray[np.float64]) -> NativeAssignmentResult: ...

class NativePlanarCartesianProjector:
    def __init__(
        self,
        source_range_bins: int,
        grid_indices: Sequence[tuple[int, int]],
        config: NativePlanarCartesianFfiConfig,
    ) -> None: ...
    def project(
        self,
        data: NDArray[np.complex64],
    ) -> NativePlanarCartesianResult: ...

def sparsify_cartesian_volume(
    magnitude_dzyx: NDArray[np.float32],
    axes: NativeCartesianAxes,
    spatial_mask_zyx: NDArray[np.bool_] | None,
    suppressed_doppler_index: int | None,
    config: NativeCartesianSparsificationConfig,
) -> NativeCartesianSparsificationResult: ...
def project_detection_point_cloud(
    detections: NDArray[np.float32],
    columns: NativeDetectionPointCloudColumns,
    config: NativeDetectionPointCloudConfig,
) -> NDArray[np.float32]: ...
def calibrate_angle_bins(
    positions_wavelengths: NDArray[np.float32],
    num_bins: int,
    angle_axis: int,
    fftshift: bool,
) -> NDArray[np.float32]: ...
def candidate_azimuth_peaks(
    cube: NDArray[np.complex64],
    cube_axes: NativeCandidateCubeAxes,
    candidates: NDArray[np.float32],
    candidate_columns: NativeCandidateIndexColumns,
    positions_wavelengths: NDArray[np.float32],
    config: NativeCandidateAzimuthConfig,
) -> NativeCandidateAzimuthResult: ...
def candidate_elevations(
    cube: NDArray[np.complex64],
    cube_axes: NativeCandidateCubeAxes,
    candidates: NDArray[np.float32],
    candidate_columns: NativeCandidateElevationColumns,
    subarrays: NativeCandidateSubarrays,
    config: NativeCandidateElevationConfig,
) -> NativeCandidateElevationResult: ...
def apply_time_domain_channel_calibration_complex(
    data: NDArray[np.complex64],
    tx_axis: int,
    rx_axis: int,
    sample_axis: int,
    frequencies_rad_per_sample: NDArray[np.float32],
    corrections: NDArray[np.complex64],
) -> NDArray[np.complex64]: ...
def apply_virtual_channel_calibration_complex(
    data: NDArray[np.complex64],
    virtual_axis: int,
    coefficients: NDArray[np.complex64],
) -> NDArray[np.complex64]: ...
def map_tdm_virtual_array_complex(
    data: NDArray[np.complex64],
    chirp_axis: int,
    rx_axis: int,
    num_tx: int,
) -> NDArray[np.complex64]: ...
def compensate_tdm_doppler_phase_complex(
    data: NDArray[np.complex64],
    doppler_axis: int,
    virtual_axis: int,
    num_tx: int,
    num_rx: int,
    fftshift: bool,
) -> NDArray[np.complex64]: ...
def map_planar_aperture_complex(
    data: NDArray[np.complex64],
    virtual_axis: int,
    grid_indices: Sequence[tuple[int, int]],
) -> NDArray[np.complex64]: ...
def select_virtual_subarray_complex(
    data: NDArray[np.complex64],
    virtual_axis: int,
    indices: Sequence[int],
) -> NDArray[np.complex64]: ...
