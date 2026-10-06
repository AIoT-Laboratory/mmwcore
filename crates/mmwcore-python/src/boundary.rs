//! Internal NumPy boundary contracts, grouped by their owning domain.

mod arrays;
mod capture;
mod cartesian;
mod cube;
mod detection;
mod geometry;

pub(crate) use arrays::{
    bool_cube_input, candidate_indices_array, candidate_matrix_input, complex_cube_array,
    complex_cube_input, contiguous_axis, dzyx_shape, native_indices_array, position_matrix_f32,
    position_matrix_f64, real_cube_array, real_cube_input,
};
pub(crate) use capture::decode_error;
pub(crate) use cartesian::{
    NativeCartesianAxes, NativeCartesianSparsificationConfig, NativeCartesianSparsificationResult,
    NativeDetectionPointCloudColumns, NativeDetectionPointCloudConfig, NativePlanarCartesianConfig,
    NativePlanarCartesianResult, cartesian_axes, cartesian_error, cartesian_sparsification_config,
    detection_point_cloud_config, detection_point_cloud_error, planar_cartesian_config,
    sparsification_error,
};
pub(crate) use cube::{
    FFT_FLAGS_MASK, FFT_ONE_SIDED_FLAG, FFT_REMOVE_DC_FLAG, FFT_SHIFT_FLAG, cube_error, fft_error,
};
pub(crate) use detection::{
    NativeCfar1DConfig, NativeCfar1DResult, NativeCfar2DConfig, NativeCfarDetections,
    NativeDetectionAxes, NativeDetectionIndexColumns, NativePeakGroupingConfig,
    NativeThresholdDetections, cfar_1d_config, cfar_1d_result_array, cfar_2d_config,
    cfar_detections_array, cfar_error, detection_error, detection_postprocess_error,
    threshold_detections_array,
};
pub(crate) use geometry::{
    NativeAssignmentResult, NativeCandidateAzimuthConfig, NativeCandidateAzimuthResult,
    NativeCandidateCubeAxes, NativeCandidateElevationColumns, NativeCandidateElevationConfig,
    NativeCandidateElevationResult, NativeCandidateIndexColumns, NativeCandidateSubarrays,
    NativeClusterResult, NativeDbscanConfig, NativePointColumns, angle_calibration_error,
    assignment_error, assignment_result_array, candidate_aoa_error, cluster_error,
    cluster_result_array, dbscan_config, point_columns,
};
