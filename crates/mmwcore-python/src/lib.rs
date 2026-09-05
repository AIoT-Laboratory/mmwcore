#![forbid(unsafe_code)]

use mmwcore::{
    AdcComplexLayout, AdcDecodeError, AdcFrameSpec, decode_adc_i16 as decode_native_adc_i16,
};
use mmwcore::{
    AngleAxis, AngleBinCalibrationConfig, AngleBinCalibrationInput, AngleCalibrationError,
    calibrate_angle_bins as native_calibrate_angle_bins,
};
use mmwcore::{
    AssignmentError, AssignmentResult, linear_sum_assignment as native_linear_sum_assignment,
};
use mmwcore::{
    CandidateAoaError, CandidateAzimuthConfig, CandidateAzimuthInput, CandidateCubeAxes,
    CandidateCubeInput, CandidateElevationColumns, CandidateElevationConfig,
    CandidateElevationInput, CandidateIndexColumns, CandidateMatrixInput,
    estimate_candidate_azimuths as native_estimate_candidate_azimuths,
    estimate_candidate_elevations as native_estimate_candidate_elevations,
};
use mmwcore::{
    CartesianProjectionError, PlanarCartesianProjectionConfig, PlanarCartesianProjectionPlan,
};
use mmwcore::{
    CartesianSparsificationConfig, CartesianSparsificationError, CartesianSparsificationInput,
    sparsify as native_sparsify,
};
use mmwcore::{
    Cfar1DConfig, Cfar1DResult, Cfar2DConfig, CfarDetections, CfarError, CfarInputScale, CfarMode,
    detect_cfar_1d as native_detect_cfar_1d, detect_cfar_2d_complex as native_detect_cfar_2d,
    detect_range_doppler_cfar_complex as native_detect_range_doppler_cfar,
};
use mmwcore::{
    ClusterError, ClusterResult, DbscanConfig, PointColumns,
    cluster_points as native_cluster_points,
};
use mmwcore::{
    ComplexFftSpec, FftTransformError, FftWindow, fft_complex_axis as fft_native_complex_axis,
};
use mmwcore::{
    CubeTransformError,
    apply_time_domain_channel_calibration_complex as apply_native_time_domain_calibration,
    apply_virtual_channel_calibration_complex as apply_native_virtual_calibration,
    compensate_tdm_doppler_phase_complex as compensate_native_tdm_doppler_phase,
    map_planar_aperture_complex as map_native_planar_aperture,
    map_tdm_virtual_array_complex as map_native_tdm_virtual_array,
    remove_static_clutter_complex as remove_native_static_clutter,
    select_virtual_subarray_complex as select_native_virtual_subarray,
};
use mmwcore::{
    DetectionCandidateInput, DetectionIndexColumns, DetectionPostprocessError,
    DetectionQualityInput, PeakGroupingConfig, PeakGroupingInput,
    filter_detection_quality as native_filter_detection_quality,
    group_range_doppler_candidates as native_group_range_doppler_candidates,
};
use mmwcore::{
    DetectionError, RangeDopplerAxes, RangeDopplerAzimuthAxes, ReceiverAggregation,
    ThresholdDetections, range_doppler_magnitude_complex as native_range_doppler_magnitude,
    threshold_range_doppler_azimuth_complex as native_threshold_range_doppler_azimuth,
    threshold_range_doppler_complex as native_threshold_range_doppler,
};
use mmwcore::{
    DetectionPointCloudColumns, DetectionPointCloudConfig, DetectionPointCloudError,
    DetectionPointCloudInput,
    project_detection_point_cloud as native_project_detection_point_cloud,
};
use numpy::ndarray::{Array1, Array2, ArrayD, IxDyn};
use numpy::{
    Complex32, IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod adc_archive;
mod capture;
mod cartesian;
mod cube;
mod detection;
mod geometry;
mod tracking;

const FFT_REMOVE_DC_FLAG: u8 = 1;
const FFT_SHIFT_FLAG: u8 = 1 << 1;
const FFT_ONE_SIDED_FLAG: u8 = 1 << 2;
const FFT_FLAGS_MASK: u8 = FFT_REMOVE_DC_FLAG | FFT_SHIFT_FLAG | FFT_ONE_SIDED_FLAG;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    capture::register(module)?;
    cartesian::register(module)?;
    cube::register(module)?;
    detection::register(module)?;
    adc_archive::register(module)?;
    geometry::register(module)?;
    tracking::register(module)?;
    Ok(())
}

fn decode_error(error: AdcDecodeError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn complex_cube_input(
    data: PyReadonlyArrayDyn<'_, Complex32>,
) -> PyResult<(Vec<Complex32>, Vec<usize>)> {
    if !data.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Radar cube must be a C-contiguous complex64 array.",
        ));
    }
    let shape = data.shape().to_vec();
    let data = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("Radar cube must be a contiguous complex64 array."))?
        .to_vec();
    Ok((data, shape))
}

fn candidate_matrix_input(data: PyReadonlyArray2<'_, f32>) -> PyResult<(Vec<f32>, [usize; 2])> {
    if !data.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Candidate matrix must be a C-contiguous float32 array.",
        ));
    }
    let shape = data.shape();
    let values = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("Candidate matrix must be a contiguous float32 array."))?
        .to_vec();
    Ok((values, [shape[0], shape[1]]))
}

fn position_matrix_f32(
    positions: PyReadonlyArray2<'_, f32>,
    name: &str,
) -> PyResult<(Vec<f32>, usize)> {
    if !positions.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} positions must be a C-contiguous float32 matrix."
        )));
    }
    let shape = positions.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} positions must have shape (antenna, 3); got ({}, {}).",
            shape[0], shape[1]
        )));
    }
    let values = positions
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "{name} positions must be a contiguous float32 matrix."
            ))
        })?
        .to_vec();
    Ok((values, shape[0]))
}

fn position_matrix_f64(
    positions: PyReadonlyArray2<'_, f64>,
    name: &str,
) -> PyResult<(Vec<f64>, usize)> {
    if !positions.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} positions must be a C-contiguous float64 matrix."
        )));
    }
    let shape = positions.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} positions must have shape (antenna, 3); got ({}, {}).",
            shape[0], shape[1]
        )));
    }
    let values = positions
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "{name} positions must be a contiguous float64 matrix."
            ))
        })?
        .to_vec();
    Ok((values, shape[0]))
}

fn candidate_indices_array(
    py: Python<'_>,
    indices: Vec<usize>,
) -> PyResult<Bound<'_, PyArray1<i64>>> {
    let indices = indices
        .into_iter()
        .map(|index| {
            i64::try_from(index)
                .map_err(|_| PyValueError::new_err("Native candidate index exceeds int64."))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Array1::from_vec(indices).into_pyarray(py))
}

fn complex_cube_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<Complex32>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let cube = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|_| PyValueError::new_err("Native radar cube shape is invalid."))?;
    Ok(cube.into_pyarray(py))
}

fn cube_error(error: CubeTransformError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn fft_error(error: FftTransformError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn detection_error(error: DetectionError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn detection_postprocess_error(error: DetectionPostprocessError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn detection_point_cloud_error(error: DetectionPointCloudError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn angle_calibration_error(error: AngleCalibrationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn candidate_aoa_error(error: CandidateAoaError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn cfar_error(error: CfarError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn cluster_error(error: ClusterError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn assignment_error(error: AssignmentError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn cartesian_error(error: CartesianProjectionError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn sparsification_error(error: CartesianSparsificationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn real_cube_input(data: PyReadonlyArrayDyn<'_, f32>) -> PyResult<(Vec<f32>, Vec<usize>)> {
    if !data.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Cartesian magnitude volume must be a C-contiguous float32 array.",
        ));
    }
    let shape = data.shape().to_vec();
    let data = data.as_slice().map_err(|_| {
        PyValueError::new_err("Cartesian magnitude volume must be a contiguous float32 array.")
    })?;
    Ok((data.to_vec(), shape))
}

fn bool_cube_input(data: PyReadonlyArrayDyn<'_, bool>) -> PyResult<(Vec<bool>, Vec<usize>)> {
    if !data.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Cartesian spatial_mask_zyx must be a C-contiguous bool array.",
        ));
    }
    let shape = data.shape().to_vec();
    let data = data.as_slice().map_err(|_| {
        PyValueError::new_err("Cartesian spatial_mask_zyx must be a contiguous bool array.")
    })?;
    Ok((data.to_vec(), shape))
}

fn dzyx_shape(shape: &[usize]) -> PyResult<[usize; 4]> {
    match shape {
        [doppler, z, y, x] => Ok([*doppler, *z, *y, *x]),
        _ => Err(PyValueError::new_err(format!(
            "Cartesian magnitude volume must have shape (D, Z, Y, X); got {shape:?}."
        ))),
    }
}

fn cartesian_axes(axes: NativeCartesianAxes<'_>) -> PyResult<NativeCartesianAxisValues> {
    let (doppler_velocity_mps, z_m, y_m, x_m) = axes;
    Ok((
        contiguous_axis(doppler_velocity_mps, "doppler_velocity_mps")?,
        contiguous_axis(z_m, "z_m")?,
        contiguous_axis(y_m, "y_m")?,
        contiguous_axis(x_m, "x_m")?,
    ))
}

fn contiguous_axis(values: PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<Vec<f32>> {
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float32 array.")))
        .map(ToOwned::to_owned)
}

fn real_cube_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let cube = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|_| PyValueError::new_err("Native radar magnitude shape is invalid."))?;
    Ok(cube.into_pyarray(py))
}

type NativeThresholdDetections<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<f32>>);
type NativeDetectionAxes = (usize, usize, usize, usize);
type NativeDetectionIndexColumns = (usize, usize, usize);
type NativePeakGroupingConfig = (usize, usize, bool, bool);
type NativeCfar1DConfig = (usize, usize, f32, u8, bool, usize, usize);
type NativeCfar2DConfig = (usize, usize, f32);
type NativeCfar1DResult<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>);
type NativeCfarDetections<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);
type NativePointColumns = (usize, usize, usize, Option<usize>);
type NativeDbscanConfig = (f32, usize, f32, bool);
type NativeClusterResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i64>>,
);
type NativeAssignmentResult<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);
type NativeDopplerAxis = (usize, f32, f32);
type NativeGridShape = (usize, usize, usize);
type NativeGridCoordinates = (f32, f32, f32);
type NativePlanarAngleConfig = (usize, usize, f32);
type NativePlanarCartesianConfig = (
    f32,
    NativeDopplerAxis,
    NativeDopplerAxis,
    NativeGridShape,
    NativeGridCoordinates,
    NativeGridCoordinates,
    (f32, f32),
    NativePlanarAngleConfig,
);
type NativePlanarCartesianResult<'py> = (
    Bound<'py, PyArrayDyn<f32>>,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);
type NativeCartesianAxes<'py> = (
    PyReadonlyArray1<'py, f32>,
    PyReadonlyArray1<'py, f32>,
    PyReadonlyArray1<'py, f32>,
    PyReadonlyArray1<'py, f32>,
);
type NativeCartesianAxisValues = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
type NativeCartesianSparsificationThresholdConfig = (f32, usize);
type NativeCartesianSparsificationPeakConfig = (usize, usize, Option<usize>, usize);
type NativeCartesianSparsificationBackgroundConfig = (f32, f32, f32, bool);
type NativeCartesianSparsificationConfig = (
    NativeCartesianSparsificationThresholdConfig,
    NativeCartesianSparsificationPeakConfig,
    NativeCartesianSparsificationBackgroundConfig,
);
type NativeCartesianSparsificationResult<'py> = (
    Bound<'py, PyArray2<f32>>,
    (f32, f32, f32),
    (usize, usize, usize, usize, usize, usize, usize),
    (bool, usize),
);
type NativeDetectionPointCloudColumns = (
    usize,
    usize,
    usize,
    usize,
    usize,
    Option<(usize, usize)>,
    Vec<usize>,
);
type NativeDetectionPointCloudConfig = (f32, f32, i8, bool, Option<usize>, bool);
type NativeCandidateCubeAxes = (usize, usize, usize, usize);
type NativeCandidateIndexColumns = (usize, usize, usize);
type NativeCandidateElevationColumns = (usize, usize, usize, usize, usize);
type NativeCandidateAzimuthConfig = (usize, u8, bool, u8);
type NativeCandidateSubarrays<'py> = (
    Vec<usize>,
    Vec<usize>,
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, f64>,
);
type NativeCandidateElevationConfig = (usize, u8, bool);
type NativeCandidateAzimuthResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);
type NativeCandidateElevationResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    (f64, f64),
);

fn planar_cartesian_config(config: NativePlanarCartesianConfig) -> PlanarCartesianProjectionConfig {
    let (
        range_resolution_m,
        source_doppler,
        target_doppler,
        grid_shape_zyx,
        grid_origin_xyz_m,
        grid_voxel_size_xyz_m,
        mount,
        angle,
    ) = config;
    let (source_doppler_bins, source_velocity_start_mps, source_velocity_step_mps) = source_doppler;
    let (target_doppler_bins, target_velocity_start_mps, target_velocity_step_mps) = target_doppler;
    let (azimuth_n_fft, elevation_n_fft, aperture_spacing_wavelengths) = angle;
    let (mount_height_m, mount_pitch_deg) = mount;
    PlanarCartesianProjectionConfig {
        range_resolution_m,
        source_doppler_bins,
        source_velocity_start_mps,
        source_velocity_step_mps,
        target_doppler_bins,
        target_velocity_start_mps,
        target_velocity_step_mps,
        grid_shape_zyx: [grid_shape_zyx.0, grid_shape_zyx.1, grid_shape_zyx.2],
        grid_origin_xyz_m: [
            grid_origin_xyz_m.0,
            grid_origin_xyz_m.1,
            grid_origin_xyz_m.2,
        ],
        grid_voxel_size_xyz_m: [
            grid_voxel_size_xyz_m.0,
            grid_voxel_size_xyz_m.1,
            grid_voxel_size_xyz_m.2,
        ],
        mount_height_m,
        mount_pitch_deg,
        azimuth_n_fft,
        elevation_n_fft,
        aperture_spacing_wavelengths,
    }
}

fn cartesian_sparsification_config(
    config: NativeCartesianSparsificationConfig,
) -> CartesianSparsificationConfig {
    let (threshold, peaks, background) = config;
    let (min_snr_db, max_points) = threshold;
    let (
        spatial_peak_radius,
        doppler_peak_radius,
        max_doppler_peaks_per_spatial,
        boundary_margin_voxels,
    ) = peaks;
    let (
        noise_floor_scale,
        static_point_capacity_fraction,
        static_velocity_threshold_mps,
        strongest_point_fallback,
    ) = background;
    CartesianSparsificationConfig {
        min_snr_db,
        max_points,
        spatial_peak_radius,
        doppler_peak_radius,
        max_doppler_peaks_per_spatial,
        boundary_margin_voxels,
        noise_floor_scale,
        static_point_capacity_fraction,
        static_velocity_threshold_mps,
        strongest_point_fallback,
    }
}

fn detection_point_cloud_config(
    config: NativeDetectionPointCloudConfig,
) -> DetectionPointCloudConfig {
    let (
        range_resolution_m,
        doppler_resolution_mps,
        doppler_sign,
        center_doppler,
        doppler_bins,
        doppler_fftshifted,
    ) = config;
    DetectionPointCloudConfig {
        range_resolution_m,
        doppler_resolution_mps,
        doppler_sign,
        center_doppler,
        doppler_bins,
        doppler_fftshifted,
    }
}

fn cfar_1d_config(config: NativeCfar1DConfig) -> PyResult<Cfar1DConfig> {
    let (training_cells, guard_cells, threshold_scale, mode, cyclic, left_skip, right_skip) =
        config;
    let mode = CfarMode::try_from(mode).map_err(cfar_error)?;
    Cfar1DConfig::new(
        training_cells,
        guard_cells,
        threshold_scale,
        mode,
        cyclic,
        left_skip,
        right_skip,
    )
    .map_err(cfar_error)
}

fn cfar_2d_config(config: NativeCfar2DConfig) -> PyResult<Cfar2DConfig> {
    let (training_cells, guard_cells, threshold_scale) = config;
    Cfar2DConfig::new(training_cells, guard_cells, threshold_scale).map_err(cfar_error)
}

fn point_columns(columns: NativePointColumns) -> PointColumns {
    let (x, y, z, velocity) = columns;
    PointColumns { x, y, z, velocity }
}

fn dbscan_config(config: NativeDbscanConfig) -> PyResult<DbscanConfig> {
    let (eps_m, min_samples, velocity_scale_s, use_z) = config;
    DbscanConfig::new(eps_m, min_samples, velocity_scale_s, use_z).map_err(cluster_error)
}

fn cfar_1d_result_array(py: Python<'_>, result: Cfar1DResult) -> PyResult<NativeCfar1DResult<'_>> {
    if result.indices.len() != result.noise.len() {
        return Err(PyValueError::new_err(
            "Native CFAR 1D indices do not match noise values.",
        ));
    }
    let indices = result
        .indices
        .into_iter()
        .map(|index| {
            i64::try_from(index)
                .map_err(|_| PyValueError::new_err("Native CFAR index exceeds int64."))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let indices = Array1::from_vec(indices).into_pyarray(py);
    let noise = Array1::from_vec(result.noise).into_pyarray(py);
    Ok((indices, noise))
}

fn cfar_detections_array(
    py: Python<'_>,
    detections: CfarDetections,
) -> PyResult<NativeCfarDetections<'_>> {
    let count = detections.magnitudes.len();
    if detections.noise.len() != count || detections.snr.len() != count {
        return Err(PyValueError::new_err(
            "Native CFAR candidate channels do not match candidate count.",
        ));
    }
    let indices = native_indices_array(py, detections.indices, count, 3)?;
    Ok((
        indices,
        Array1::from_vec(detections.magnitudes).into_pyarray(py),
        Array1::from_vec(detections.noise).into_pyarray(py),
        Array1::from_vec(detections.snr).into_pyarray(py),
    ))
}

fn cluster_result_array(
    py: Python<'_>,
    result: ClusterResult,
) -> PyResult<NativeClusterResult<'_>> {
    let cluster_count = result.mean_velocities.len();
    let expected_coordinate_count = cluster_count
        .checked_mul(3)
        .ok_or_else(|| PyValueError::new_err("Native cluster coordinate count overflows."))?;
    if result.centers.len() != expected_coordinate_count
        || result.extents.len() != expected_coordinate_count
        || result.point_counts.len() != cluster_count
    {
        return Err(PyValueError::new_err(
            "Native cluster result arrays do not agree on cluster count.",
        ));
    }
    let centers = Array2::from_shape_vec((cluster_count, 3), result.centers)
        .map_err(|_| PyValueError::new_err("Native cluster center shape is invalid."))?
        .into_pyarray(py);
    let extents = Array2::from_shape_vec((cluster_count, 3), result.extents)
        .map_err(|_| PyValueError::new_err("Native cluster extent shape is invalid."))?
        .into_pyarray(py);
    Ok((
        Array1::from_vec(result.labels).into_pyarray(py),
        centers,
        extents,
        Array1::from_vec(result.mean_velocities).into_pyarray(py),
        Array1::from_vec(result.point_counts).into_pyarray(py),
    ))
}

fn assignment_result_array(
    py: Python<'_>,
    result: AssignmentResult,
) -> PyResult<NativeAssignmentResult<'_>> {
    if result.rows.len() != result.columns.len() {
        return Err(PyValueError::new_err(
            "Native assignment rows and columns do not agree on count.",
        ));
    }
    let rows = result
        .rows
        .into_iter()
        .map(|row| {
            i64::try_from(row)
                .map_err(|_| PyValueError::new_err("Native assignment row exceeds int64."))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let columns = result
        .columns
        .into_iter()
        .map(|column| {
            i64::try_from(column)
                .map_err(|_| PyValueError::new_err("Native assignment column exceeds int64."))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        Array1::from_vec(rows).into_pyarray(py),
        Array1::from_vec(columns).into_pyarray(py),
    ))
}

fn native_indices_array(
    py: Python<'_>,
    indices: Vec<usize>,
    count: usize,
    rank: usize,
) -> PyResult<Bound<'_, PyArray2<i64>>> {
    let expected_index_count = count
        .checked_mul(rank)
        .ok_or_else(|| PyValueError::new_err("Native CFAR detection index count overflows."))?;
    if indices.len() != expected_index_count {
        return Err(PyValueError::new_err(
            "Native CFAR detection indices do not match candidate count.",
        ));
    }
    let indices = indices
        .into_iter()
        .map(|index| {
            i64::try_from(index)
                .map_err(|_| PyValueError::new_err("Native CFAR index exceeds int64."))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Array2::from_shape_vec((count, rank), indices)
        .map_err(|_| PyValueError::new_err("Native CFAR detection shape is invalid."))
        .map(|array| array.into_pyarray(py))
}

fn threshold_detections_array(
    py: Python<'_>,
    detections: ThresholdDetections,
) -> PyResult<NativeThresholdDetections<'_>> {
    let count = detections.magnitudes.len();
    let expected_index_count = count.checked_mul(detections.rank).ok_or_else(|| {
        PyValueError::new_err("Native threshold detection index count overflows.")
    })?;
    if detections.indices.len() != expected_index_count {
        return Err(PyValueError::new_err(
            "Native threshold detection indices do not match candidate count.",
        ));
    }
    let indices = detections
        .indices
        .into_iter()
        .map(|index| {
            i64::try_from(index)
                .map_err(|_| PyValueError::new_err("Native detection index exceeds int64."))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let indices = Array2::from_shape_vec((count, detections.rank), indices)
        .map_err(|_| PyValueError::new_err("Native threshold detection shape is invalid."))?
        .into_pyarray(py);
    let magnitudes = Array1::from_vec(detections.magnitudes).into_pyarray(py);
    Ok((indices, magnitudes))
}
