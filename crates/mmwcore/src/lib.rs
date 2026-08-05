#![forbid(unsafe_code)]

//! Typed mmWave radar data-link and signal-processing primitives.

pub mod adc;
pub mod angle;
pub mod assignment;
pub mod candidate_aoa;
pub mod cartesian;
pub mod cfar;
pub mod clustering;
pub mod cube;
pub mod dca1000;
pub mod detection;
pub mod detection_postprocess;
pub mod fft;
pub mod pointcloud;
pub mod sparsification;
pub mod tracking;
pub mod vitals;

pub use adc::{AdcComplexLayout, AdcCube, AdcDecodeError, AdcFrameSpec, decode_adc_i16};
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
pub use dca1000::{
    Dca1000Error, Dca1000FrameAssembly, Dca1000Packet, PacketLossStats, assemble_dca1000_frame,
    assemble_dca1000_frame_bytes, parse_dca1000_packet, reorder_dca1000_packets,
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
    CartesianSparsificationResult, sparsify_cartesian_volume,
};
pub use tracking::{
    ClusterMeasurements, ClusterTracker2D, MeasurementTracker2D, NativeTrackStatus,
    PointMeasurements, TrackAllocationConfig, TrackGatingConfig, TrackLifecycleConfig,
    TrackObservationMetrics, TrackSceneryConfig, TrackStepResult, Tracker2DConfig,
    TrackerDynamicsConfig, TrackingBox2D, TrackingError, TrackingMetricsError,
    TrackingMetricsInput, TrackingSequenceMetrics, summarize_tracking_metrics,
};
pub use vitals::{VitalSignError, unwrap_vital_phase_complex, vital_phase_to_displacement};
