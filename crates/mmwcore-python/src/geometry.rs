//! PyO3 boundary for native clustering, assignment, and angle geometry.

use super::*;

#[pyfunction]
fn cluster_points<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    columns: NativePointColumns,
    config: NativeDbscanConfig,
) -> PyResult<NativeClusterResult<'py>> {
    if !points.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Point matrix must be a C-contiguous float32 array.",
        ));
    }
    let shape = points.shape();
    let (point_count, channel_count) = (shape[0], shape[1]);
    let points = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("Point matrix must be a contiguous float32 array."))?
        .to_vec();
    let columns = point_columns(columns);
    let config = dbscan_config(config)?;
    let result = py
        .detach(move || native_cluster_points(&points, point_count, channel_count, columns, config))
        .map_err(cluster_error)?;
    cluster_result_array(py, result)
}

#[pyfunction]
fn linear_sum_assignment<'py>(
    py: Python<'py>,
    costs: PyReadonlyArray2<'py, f64>,
) -> PyResult<NativeAssignmentResult<'py>> {
    if !costs.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Assignment cost matrix must be a C-contiguous float64 array.",
        ));
    }
    let shape = costs.shape();
    let (row_count, column_count) = (shape[0], shape[1]);
    let costs = costs
        .as_slice()
        .map_err(|_| PyValueError::new_err("Assignment cost matrix must be contiguous."))?
        .to_vec();
    let result = py
        .detach(move || native_linear_sum_assignment(&costs, row_count, column_count))
        .map_err(assignment_error)?;
    assignment_result_array(py, result)
}

#[pyfunction]
fn calibrate_angle_bins<'py>(
    py: Python<'py>,
    positions_wavelengths: PyReadonlyArray2<'py, f32>,
    num_bins: isize,
    angle_axis: u8,
    fftshift: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if !positions_wavelengths.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Angle calibration positions must be a C-contiguous float32 matrix.",
        ));
    }
    let shape = positions_wavelengths.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "Angle calibration positions must have shape (antenna, 3); got ({}, {}).",
            shape[0], shape[1]
        )));
    }
    let position_count = shape[0];
    let positions_wavelengths = positions_wavelengths
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err(
                "Angle calibration positions must be a contiguous float32 matrix.",
            )
        })?
        .to_vec();
    let num_bins = usize::try_from(num_bins).map_err(|_| {
        PyValueError::new_err(format!("num_bins must be positive; got {num_bins}."))
    })?;
    let axis = AngleAxis::try_from(angle_axis).map_err(angle_calibration_error)?;
    let angles = py
        .detach(move || {
            native_calibrate_angle_bins(
                AngleBinCalibrationInput {
                    positions_wavelengths: &positions_wavelengths,
                    position_count,
                },
                AngleBinCalibrationConfig {
                    num_bins,
                    axis,
                    fftshift,
                },
            )
        })
        .map_err(angle_calibration_error)?;
    Ok(Array1::from_vec(angles).into_pyarray(py))
}

#[pyfunction]
fn candidate_azimuth_peaks<'py>(
    py: Python<'py>,
    cube: PyReadonlyArrayDyn<'py, Complex32>,
    cube_axes: NativeCandidateCubeAxes,
    candidates: PyReadonlyArray2<'py, f32>,
    candidate_columns: NativeCandidateIndexColumns,
    positions_wavelengths: PyReadonlyArray2<'py, f32>,
    config: NativeCandidateAzimuthConfig,
) -> PyResult<NativeCandidateAzimuthResult<'py>> {
    let (cube, cube_shape) = complex_cube_input(cube)?;
    let (candidates, candidate_shape) = candidate_matrix_input(candidates)?;
    let (positions_wavelengths, position_count) =
        position_matrix_f32(positions_wavelengths, "Angle calibration")?;
    let (frame, doppler, antenna, range) = cube_axes;
    let (frame_column, range_column, doppler_column) = candidate_columns;
    let (n_fft, window, fftshift, angle_axis) = config;
    let window = FftWindow::try_from(window).map_err(fft_error)?;
    let angle_axis = AngleAxis::try_from(angle_axis).map_err(angle_calibration_error)?;
    let result = py
        .detach(move || {
            native_estimate_candidate_azimuths(
                CandidateAzimuthInput {
                    cube: CandidateCubeInput {
                        data: &cube,
                        shape: &cube_shape,
                        axes: CandidateCubeAxes {
                            frame,
                            doppler,
                            antenna,
                            range,
                        },
                    },
                    candidates: CandidateMatrixInput {
                        values: &candidates,
                        shape: candidate_shape,
                    },
                    columns: CandidateIndexColumns {
                        frame: frame_column,
                        range: range_column,
                        doppler: doppler_column,
                    },
                    positions_wavelengths: &positions_wavelengths,
                    position_count,
                },
                CandidateAzimuthConfig {
                    n_fft,
                    window,
                    fftshift,
                    angle_axis,
                },
            )
        })
        .map_err(candidate_aoa_error)?;
    Ok((
        candidate_indices_array(py, result.peak_bins)?,
        Array1::from_vec(result.angles_rad).into_pyarray(py),
        Array1::from_vec(result.magnitudes).into_pyarray(py),
    ))
}

#[pyfunction]
fn candidate_elevations<'py>(
    py: Python<'py>,
    cube: PyReadonlyArrayDyn<'py, Complex32>,
    cube_axes: NativeCandidateCubeAxes,
    candidates: PyReadonlyArray2<'py, f32>,
    candidate_columns: NativeCandidateElevationColumns,
    subarrays: NativeCandidateSubarrays<'py>,
    config: NativeCandidateElevationConfig,
) -> PyResult<NativeCandidateElevationResult<'py>> {
    let (cube, cube_shape) = complex_cube_input(cube)?;
    let (candidates, candidate_shape) = candidate_matrix_input(candidates)?;
    let (
        azimuth_antenna_indices,
        elevation_antenna_indices,
        azimuth_positions_wavelengths,
        elevation_positions_wavelengths,
    ) = subarrays;
    let (azimuth_positions_wavelengths, azimuth_position_count) =
        position_matrix_f64(azimuth_positions_wavelengths, "Azimuth subarray")?;
    let (elevation_positions_wavelengths, elevation_position_count) =
        position_matrix_f64(elevation_positions_wavelengths, "Elevation subarray")?;
    let (frame, doppler, antenna, range) = cube_axes;
    let (frame_column, range_column, doppler_column, azimuth_bin, azimuth_rad) = candidate_columns;
    let (n_fft, window, fftshift) = config;
    let window = FftWindow::try_from(window).map_err(fft_error)?;
    let result = py
        .detach(move || {
            native_estimate_candidate_elevations(
                CandidateElevationInput {
                    cube: CandidateCubeInput {
                        data: &cube,
                        shape: &cube_shape,
                        axes: CandidateCubeAxes {
                            frame,
                            doppler,
                            antenna,
                            range,
                        },
                    },
                    candidates: CandidateMatrixInput {
                        values: &candidates,
                        shape: candidate_shape,
                    },
                    columns: CandidateElevationColumns {
                        indices: CandidateIndexColumns {
                            frame: frame_column,
                            range: range_column,
                            doppler: doppler_column,
                        },
                        azimuth_bin,
                        azimuth_rad,
                    },
                    azimuth_antenna_indices: &azimuth_antenna_indices,
                    elevation_antenna_indices: &elevation_antenna_indices,
                    azimuth_positions_wavelengths: &azimuth_positions_wavelengths,
                    azimuth_position_count,
                    elevation_positions_wavelengths: &elevation_positions_wavelengths,
                    elevation_position_count,
                },
                CandidateElevationConfig {
                    n_fft,
                    window,
                    fftshift,
                },
            )
        })
        .map_err(candidate_aoa_error)?;
    Ok((
        candidate_indices_array(py, result.valid_candidate_indices)?,
        Array1::from_vec(result.angles_rad).into_pyarray(py),
        Array1::from_vec(result.magnitudes).into_pyarray(py),
        (
            result.row_offsets_wavelengths[0],
            result.row_offsets_wavelengths[1],
        ),
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(cluster_points, module)?)?;
    module.add_function(wrap_pyfunction!(linear_sum_assignment, module)?)?;
    module.add_function(wrap_pyfunction!(calibrate_angle_bins, module)?)?;
    module.add_function(wrap_pyfunction!(candidate_azimuth_peaks, module)?)?;
    module.add_function(wrap_pyfunction!(candidate_elevations, module)?)?;
    Ok(())
}
