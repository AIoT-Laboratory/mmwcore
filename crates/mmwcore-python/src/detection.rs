//! PyO3 boundary for native threshold, grouping, quality, and CFAR detection.

use numpy::{
    Complex32, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use super::{
    CfarInputScale, DetectionCandidateInput, DetectionIndexColumns, DetectionQualityInput,
    NativeCfar1DConfig, NativeCfar1DResult, NativeCfar2DConfig, NativeCfarDetections,
    NativeDetectionAxes, NativeDetectionIndexColumns, NativePeakGroupingConfig,
    NativeThresholdDetections, PeakGroupingConfig, PeakGroupingInput, RangeDopplerAxes,
    RangeDopplerAzimuthAxes, ReceiverAggregation, candidate_indices_array, candidate_matrix_input,
    cfar_1d_config, cfar_1d_result_array, cfar_2d_config, cfar_detections_array, cfar_error,
    complex_cube_input, detection_error, detection_postprocess_error, native_detect_cfar_1d,
    native_detect_cfar_2d, native_detect_range_doppler_cfar, native_filter_detection_quality,
    native_group_range_doppler_candidates, native_range_doppler_magnitude,
    native_threshold_range_doppler, native_threshold_range_doppler_azimuth, real_cube_array,
    threshold_detections_array,
};

#[pyfunction]
fn range_doppler_magnitude_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    axes: (usize, usize, usize, usize),
    aggregation: u8,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let aggregation = ReceiverAggregation::try_from(aggregation).map_err(detection_error)?;
    let (frame_axis, doppler_axis, receiver_axis, range_axis) = axes;
    let axes = RangeDopplerAxes {
        frame: frame_axis,
        doppler: doppler_axis,
        receiver: receiver_axis,
        range: range_axis,
    };
    let (output, output_shape) = py
        .detach(move || native_range_doppler_magnitude(&data, &shape, axes, aggregation))
        .map_err(detection_error)?;
    real_cube_array(py, &output_shape, output)
}

#[pyfunction]
fn threshold_range_doppler_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    axes: (usize, usize, usize, usize),
    aggregation: u8,
    threshold: f32,
) -> PyResult<NativeThresholdDetections<'py>> {
    let (data, shape) = complex_cube_input(data)?;
    let aggregation = ReceiverAggregation::try_from(aggregation).map_err(detection_error)?;
    let (frame_axis, doppler_axis, receiver_axis, range_axis) = axes;
    let axes = RangeDopplerAxes {
        frame: frame_axis,
        doppler: doppler_axis,
        receiver: receiver_axis,
        range: range_axis,
    };
    let detections = py
        .detach(move || native_threshold_range_doppler(&data, &shape, axes, aggregation, threshold))
        .map_err(detection_error)?;
    threshold_detections_array(py, detections)
}

#[pyfunction]
fn threshold_range_doppler_azimuth_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    axes: (usize, usize, usize, usize),
    threshold: f32,
    azimuth_peak_radius: usize,
    azimuth_peak_strict: bool,
) -> PyResult<NativeThresholdDetections<'py>> {
    let (data, shape) = complex_cube_input(data)?;
    let (frame_axis, doppler_axis, azimuth_axis, range_axis) = axes;
    let axes = RangeDopplerAzimuthAxes {
        frame: frame_axis,
        doppler: doppler_axis,
        azimuth: azimuth_axis,
        range: range_axis,
    };
    let detections = py
        .detach(move || {
            native_threshold_range_doppler_azimuth(
                &data,
                &shape,
                axes,
                threshold,
                azimuth_peak_radius,
                azimuth_peak_strict,
            )
        })
        .map_err(detection_error)?;
    threshold_detections_array(py, detections)
}

#[pyfunction]
fn group_range_doppler_candidates<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    axes: NativeDetectionAxes,
    aggregation: u8,
    candidates: PyReadonlyArray2<'py, f32>,
    columns: NativeDetectionIndexColumns,
    config: NativePeakGroupingConfig,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (data, shape) = complex_cube_input(data)?;
    let aggregation = ReceiverAggregation::try_from(aggregation).map_err(detection_error)?;
    let (candidates, candidate_shape) = candidate_matrix_input(candidates)?;
    let (frame, doppler, receiver, range) = axes;
    let (frame_column, range_column, doppler_column) = columns;
    let (range_radius, doppler_radius, cyclic_doppler, strict) = config;
    let retained = py
        .detach(move || {
            native_group_range_doppler_candidates(
                PeakGroupingInput {
                    data: &data,
                    shape: &shape,
                    axes: RangeDopplerAxes {
                        frame,
                        doppler,
                        receiver,
                        range,
                    },
                    aggregation,
                    candidates: DetectionCandidateInput {
                        values: &candidates,
                        shape: candidate_shape,
                    },
                    columns: DetectionIndexColumns {
                        frame: frame_column,
                        range: range_column,
                        doppler: doppler_column,
                    },
                },
                PeakGroupingConfig {
                    range_radius,
                    doppler_radius,
                    cyclic_doppler,
                    strict,
                },
            )
        })
        .map_err(detection_postprocess_error)?;
    candidate_indices_array(py, retained)
}

#[pyfunction]
fn filter_detection_quality_rows<'py>(
    py: Python<'py>,
    candidates: PyReadonlyArray2<'py, f32>,
    snr_column: usize,
    min_snr: f32,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (candidates, candidate_shape) = candidate_matrix_input(candidates)?;
    let retained = py
        .detach(move || {
            native_filter_detection_quality(DetectionQualityInput {
                candidates: DetectionCandidateInput {
                    values: &candidates,
                    shape: candidate_shape,
                },
                snr_column,
                min_snr,
            })
        })
        .map_err(detection_postprocess_error)?;
    candidate_indices_array(py, retained)
}

#[pyfunction]
fn detect_cfar_1d<'py>(
    py: Python<'py>,
    power: PyReadonlyArray1<'py, f32>,
    config: NativeCfar1DConfig,
) -> PyResult<NativeCfar1DResult<'py>> {
    let power = power
        .as_slice()
        .map_err(|_| PyValueError::new_err("CFAR power must be a contiguous float32 array."))?
        .to_vec();
    let config = cfar_1d_config(config)?;
    let result = py
        .detach(move || native_detect_cfar_1d(&power, config))
        .map_err(cfar_error)?;
    cfar_1d_result_array(py, result)
}

#[pyfunction]
fn detect_range_doppler_cfar_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    axes: (usize, usize, usize, usize),
    aggregation: u8,
    range_config: NativeCfar1DConfig,
    doppler_config: Option<NativeCfar1DConfig>,
    input_scale: u8,
) -> PyResult<NativeCfarDetections<'py>> {
    let (data, shape) = complex_cube_input(data)?;
    let aggregation = ReceiverAggregation::try_from(aggregation).map_err(detection_error)?;
    let range_config = cfar_1d_config(range_config)?;
    let doppler_config = doppler_config.map(cfar_1d_config).transpose()?;
    let input_scale = CfarInputScale::try_from(input_scale).map_err(cfar_error)?;
    let (frame_axis, doppler_axis, receiver_axis, range_axis) = axes;
    let axes = RangeDopplerAxes {
        frame: frame_axis,
        doppler: doppler_axis,
        receiver: receiver_axis,
        range: range_axis,
    };
    let detections = py
        .detach(move || {
            native_detect_range_doppler_cfar(
                &data,
                &shape,
                axes,
                aggregation,
                range_config,
                doppler_config,
                input_scale,
            )
        })
        .map_err(cfar_error)?;
    cfar_detections_array(py, detections)
}

#[pyfunction]
fn detect_cfar_2d_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    axes: (usize, usize, usize, usize),
    aggregation: u8,
    config: NativeCfar2DConfig,
) -> PyResult<NativeCfarDetections<'py>> {
    let (data, shape) = complex_cube_input(data)?;
    let aggregation = ReceiverAggregation::try_from(aggregation).map_err(detection_error)?;
    let config = cfar_2d_config(config)?;
    let (frame_axis, doppler_axis, receiver_axis, range_axis) = axes;
    let axes = RangeDopplerAxes {
        frame: frame_axis,
        doppler: doppler_axis,
        receiver: receiver_axis,
        range: range_axis,
    };
    let detections = py
        .detach(move || native_detect_cfar_2d(&data, &shape, axes, aggregation, config))
        .map_err(cfar_error)?;
    cfar_detections_array(py, detections)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(range_doppler_magnitude_complex, module)?)?;
    module.add_function(wrap_pyfunction!(threshold_range_doppler_complex, module)?)?;
    module.add_function(wrap_pyfunction!(
        threshold_range_doppler_azimuth_complex,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(group_range_doppler_candidates, module)?)?;
    module.add_function(wrap_pyfunction!(filter_detection_quality_rows, module)?)?;
    module.add_function(wrap_pyfunction!(detect_cfar_1d, module)?)?;
    module.add_function(wrap_pyfunction!(detect_range_doppler_cfar_complex, module)?)?;
    module.add_function(wrap_pyfunction!(detect_cfar_2d_complex, module)?)?;
    Ok(())
}
