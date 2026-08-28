//! PyO3 boundary for native Cartesian RT/RPC and point-cloud projection.

use numpy::ndarray::Array2;
use numpy::{
    Complex32, IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use super::{
    CartesianSparsificationInput, DetectionPointCloudColumns, DetectionPointCloudInput,
    NativeCartesianAxes, NativeCartesianSparsificationConfig, NativeCartesianSparsificationResult,
    NativeDetectionPointCloudColumns, NativeDetectionPointCloudConfig, NativePlanarCartesianConfig,
    NativePlanarCartesianResult, PlanarCartesianProjectionPlan, bool_cube_input, cartesian_axes,
    cartesian_error, cartesian_sparsification_config, complex_cube_input,
    detection_point_cloud_config, detection_point_cloud_error, dzyx_shape,
    native_project_detection_point_cloud, native_sparsify, planar_cartesian_config,
    real_cube_array, real_cube_input, sparsification_error,
};

#[pyclass]
struct NativeCartesianProjector {
    plan: PlanarCartesianProjectionPlan,
}

#[pymethods]
impl NativeCartesianProjector {
    #[new]
    fn new(
        source_range_bins: usize,
        grid_indices: Vec<(usize, usize)>,
        config: NativePlanarCartesianConfig,
    ) -> PyResult<Self> {
        let config = planar_cartesian_config(config);
        let plan = PlanarCartesianProjectionPlan::new(source_range_bins, &grid_indices, config)
            .map_err(cartesian_error)?;
        Ok(Self { plan })
    }

    fn project<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArrayDyn<'py, Complex32>,
    ) -> PyResult<NativePlanarCartesianResult<'py>> {
        let (data, shape) = complex_cube_input(data)?;
        let plan = &self.plan;
        let result = py
            .detach(move || plan.project(&data, &shape))
            .map_err(cartesian_error)?;
        let magnitude = real_cube_array(py, &result.shape_dzyx, result.magnitude_dzyx)?;
        Ok((
            magnitude,
            result.doppler_start,
            result.doppler_stop,
            result.range_start,
            result.range_stop,
            result.spatial_valid_count,
            result.doppler_valid_count,
        ))
    }
}

#[pyfunction]
fn sparsify<'py>(
    py: Python<'py>,
    magnitude_dzyx: PyReadonlyArrayDyn<'py, f32>,
    axes: NativeCartesianAxes<'py>,
    spatial_mask_zyx: Option<PyReadonlyArrayDyn<'py, bool>>,
    suppressed_doppler_index: Option<usize>,
    config: NativeCartesianSparsificationConfig,
) -> PyResult<NativeCartesianSparsificationResult<'py>> {
    let (magnitude_dzyx, shape) = real_cube_input(magnitude_dzyx)?;
    let shape_dzyx = dzyx_shape(&shape)?;
    let (doppler_velocity_mps, z_m, y_m, x_m) = cartesian_axes(axes)?;
    let spatial_mask_zyx = spatial_mask_zyx.map(bool_cube_input).transpose()?;
    if let Some((_, mask_shape)) = &spatial_mask_zyx {
        let expected_shape = vec![shape_dzyx[1], shape_dzyx[2], shape_dzyx[3]];
        if *mask_shape != expected_shape {
            return Err(PyValueError::new_err(format!(
                "Cartesian spatial_mask_zyx must have shape {expected_shape:?}."
            )));
        }
    }
    let config = cartesian_sparsification_config(config);
    let result = py
        .detach(move || {
            native_sparsify(
                CartesianSparsificationInput {
                    magnitude_dzyx: &magnitude_dzyx,
                    shape_dzyx,
                    doppler_velocity_mps: &doppler_velocity_mps,
                    z_m: &z_m,
                    y_m: &y_m,
                    x_m: &x_m,
                    spatial_mask_zyx: spatial_mask_zyx.as_ref().map(|(mask, _)| mask.as_slice()),
                    suppressed_doppler_index,
                },
                config,
            )
        })
        .map_err(sparsification_error)?;
    let expected_point_values = result
        .point_count
        .checked_mul(5)
        .ok_or_else(|| PyValueError::new_err("Native Cartesian point count overflows."))?;
    if result.points.len() != expected_point_values {
        return Err(PyValueError::new_err(
            "Native Cartesian point values do not match point count.",
        ));
    }
    let points = Array2::from_shape_vec((result.point_count, 5), result.points)
        .map_err(|_| PyValueError::new_err("Native Cartesian point shape is invalid."))?
        .into_pyarray(py);
    Ok((
        points,
        (
            result.noise_floor_min,
            result.noise_floor_median,
            result.noise_floor_max,
        ),
        (
            result.valid_spatial_voxels,
            result.positive_volume_voxels,
            result.valid_positive_volume_voxels,
            result.local_peak_voxels,
            result.doppler_peak_voxels,
            result.threshold_peak_voxels,
            result.limited_peak_voxels,
        ),
        (result.fallback_used, result.static_output_points),
    ))
}

#[pyfunction]
fn project_detection_point_cloud<'py>(
    py: Python<'py>,
    detections: PyReadonlyArray2<'py, f32>,
    columns: NativeDetectionPointCloudColumns,
    config: NativeDetectionPointCloudConfig,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if !detections.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Detection point-cloud input must be a C-contiguous float32 matrix.",
        ));
    }
    let shape = detections.shape();
    let (point_count, input_channels) = (shape[0], shape[1]);
    let detections = detections
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err(
                "Detection point-cloud input must be a contiguous float32 matrix.",
            )
        })?
        .to_vec();
    let (range_bin, doppler_bin, magnitude, azimuth_bin, azimuth_rad, elevation, passthrough) =
        columns;
    let config = detection_point_cloud_config(config);
    let result = py
        .detach(move || {
            native_project_detection_point_cloud(
                DetectionPointCloudInput {
                    detections: &detections,
                    shape: [point_count, input_channels],
                    columns: DetectionPointCloudColumns {
                        range_bin,
                        doppler_bin,
                        magnitude,
                        azimuth_bin,
                        azimuth_rad,
                        elevation: elevation.map(|(rad, magnitude)| [rad, magnitude]),
                        passthrough: &passthrough,
                    },
                },
                config,
            )
        })
        .map_err(detection_point_cloud_error)?;
    let expected_values = result
        .point_count
        .checked_mul(result.point_channels)
        .ok_or_else(|| PyValueError::new_err("Native detection point count overflows."))?;
    if result.points.len() != expected_values {
        return Err(PyValueError::new_err(
            "Native detection point values do not match point shape.",
        ));
    }
    Array2::from_shape_vec((result.point_count, result.point_channels), result.points)
        .map_err(|_| PyValueError::new_err("Native detection point shape is invalid."))
        .map(|points| points.into_pyarray(py))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCartesianProjector>()?;
    module.add_function(wrap_pyfunction!(sparsify, module)?)?;
    module.add_function(wrap_pyfunction!(project_detection_point_cloud, module)?)?;
    Ok(())
}
