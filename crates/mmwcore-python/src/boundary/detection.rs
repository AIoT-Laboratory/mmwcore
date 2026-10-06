//! Detection boundary contracts.
use numpy::IntoPyArray;

use super::native_indices_array;
use mmwcore::{
    Cfar1DConfig, Cfar1DResult, Cfar2DConfig, CfarDetections, CfarError, CfarMode, DetectionError,
    DetectionPostprocessError, ThresholdDetections,
};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn detection_error(error: DetectionError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn detection_postprocess_error(error: DetectionPostprocessError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn cfar_error(error: CfarError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) type NativeThresholdDetections<'py> =
    (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<f32>>);

pub(crate) type NativeDetectionAxes = (usize, usize, usize, usize);

pub(crate) type NativeDetectionIndexColumns = (usize, usize, usize);

pub(crate) type NativePeakGroupingConfig = (usize, usize, bool, bool);

pub(crate) type NativeCfar1DConfig = (usize, usize, f32, u8, bool, usize, usize);

pub(crate) type NativeCfar2DConfig = (usize, usize, f32);

pub(crate) type NativeCfar1DResult<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>);

pub(crate) type NativeCfarDetections<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);

pub(crate) fn cfar_1d_config(config: NativeCfar1DConfig) -> PyResult<Cfar1DConfig> {
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

pub(crate) fn cfar_2d_config(config: NativeCfar2DConfig) -> PyResult<Cfar2DConfig> {
    let (training_cells, guard_cells, threshold_scale) = config;
    Cfar2DConfig::new(training_cells, guard_cells, threshold_scale).map_err(cfar_error)
}

pub(crate) fn cfar_1d_result_array(
    py: Python<'_>,
    result: Cfar1DResult,
) -> PyResult<NativeCfar1DResult<'_>> {
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

pub(crate) fn cfar_detections_array(
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

pub(crate) fn threshold_detections_array(
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
