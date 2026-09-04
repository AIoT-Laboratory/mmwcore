#![forbid(unsafe_code)]

//! Typed mmWave radar data-link and signal-processing primitives.

pub mod adc;
pub mod adc_archive;
pub mod adc_archive_file;
pub mod angle;
pub mod assignment;
pub mod candidate_aoa;
pub mod cartesian;
pub mod cfar;
pub mod clustering;
pub mod cube;
pub mod detection;
pub mod detection_postprocess;
pub mod fft;
pub mod pointcloud;
pub mod sparsification;
pub mod tracking;

#[inline]
pub(crate) fn exact_candidate_index(value: f32, upper_bound: usize) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }

    let index = value as usize;
    (index < upper_bound).then_some(index)
}

pub use adc::{AdcComplexLayout, AdcCube, AdcDecodeError, AdcFrameSpec, decode_adc_i16};
pub use adc_archive::{
    ADC_RICE_BLOCK_SAMPLES, AdcArchiveCodecError, decode_adc_archive_chunk,
    encode_adc_archive_chunk, maximum_adc_archive_chunk_bytes,
};
pub use adc_archive_file::{
    AdcArchiveFile, AdcArchiveFileError, open_adc_archive_file, sha256_from_hex, sha256_to_hex,
    write_adc_archive_file,
};
pub use angle::{
    AngleAxis, AngleBinCalibrationConfig, AngleBinCalibrationInput, AngleCalibrationError,
    calibrate_angle_bins,
};
pub use assignment::{AssignmentError, AssignmentResult, linear_sum_assignment};
pub use candidate_aoa::{
    CandidateAoaError, CandidateAzimuthConfig, CandidateAzimuthInput, CandidateAzimuthPeaks,
    CandidateCubeAxes, CandidateCubeInput, CandidateElevationColumns, CandidateElevationConfig,
    CandidateElevationInput, CandidateElevations, CandidateIndexColumns, CandidateMatrixInput,
    estimate_candidate_azimuths, estimate_candidate_elevations,
};
pub use cartesian::{
    CartesianProjectionError, PlanarCartesianProjection, PlanarCartesianProjectionConfig,
    PlanarCartesianProjectionPlan,
};
pub use cfar::{
    Cfar1DConfig, Cfar1DResult, Cfar2DConfig, CfarDetections, CfarError, CfarInputScale, CfarMode,
    detect_cfar_1d, detect_cfar_2d_complex, detect_range_doppler_cfar_complex,
};
pub use clustering::{ClusterError, ClusterResult, DbscanConfig, PointColumns, cluster_points};
pub use cube::{
    CubeTransformError, apply_time_domain_channel_calibration_complex,
    apply_virtual_channel_calibration_complex, compensate_tdm_doppler_phase_complex,
    map_planar_aperture_complex, map_tdm_virtual_array_complex, remove_static_clutter_complex,
    select_virtual_subarray_complex,
};
pub use detection::{
    DetectionError, RangeDopplerAxes, RangeDopplerAzimuthAxes, ReceiverAggregation,
    ThresholdDetections, range_doppler_magnitude_complex, threshold_range_doppler_azimuth_complex,
    threshold_range_doppler_complex,
};
pub use detection_postprocess::{
    DetectionCandidateInput, DetectionIndexColumns, DetectionPostprocessError,
    DetectionQualityInput, PeakGroupingConfig, PeakGroupingInput, filter_detection_quality,
    group_range_doppler_candidates,
};
pub use fft::{ComplexFftSpec, FftTransformError, FftWindow, fft_complex_axis};
pub use pointcloud::{
    DetectionPointCloudColumns, DetectionPointCloudConfig, DetectionPointCloudError,
    DetectionPointCloudInput, DetectionPointCloudProjection, project_detection_point_cloud,
};
pub use sparsification::{
    CartesianSparsificationConfig, CartesianSparsificationError, CartesianSparsificationInput,
    CartesianSparsificationResult, sparsify,
};
pub use tracking::{
    Box2D, Box3D, ClusterMeasurements, ClusterTracker2D, GTrack2D, GTrack3D, GTrack3DDiagnostics,
    NativeTrackStatus, PointMeasurements, PointTracker2D, TrackAllocationConfig, TrackGatingConfig,
    TrackLifecycleConfig, TrackObservationMetrics, TrackScenery3DConfig, TrackSceneryConfig,
    TrackStepResult, Tracker2DConfig, Tracker3DConfig, TrackerDynamics3DConfig,
    TrackerDynamicsConfig, TrackingError, TrackingMetricsError, TrackingMetricsInput,
    TrackingSequenceMetrics, summarize_tracking_metrics,
};
