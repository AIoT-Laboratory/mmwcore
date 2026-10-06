//! Arrays boundary contracts.
use numpy::IntoPyArray;
use numpy::PyUntypedArrayMethods;

use numpy::ndarray::{Array1, Array2, ArrayD, IxDyn};
use numpy::{
    Complex32, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn complex_cube_input(
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

pub(crate) fn candidate_matrix_input(
    data: PyReadonlyArray2<'_, f32>,
) -> PyResult<(Vec<f32>, [usize; 2])> {
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

pub(crate) fn position_matrix_f32(
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

pub(crate) fn position_matrix_f64(
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

pub(crate) fn candidate_indices_array(
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

pub(crate) fn complex_cube_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<Complex32>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let cube = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|_| PyValueError::new_err("Native radar cube shape is invalid."))?;
    Ok(cube.into_pyarray(py))
}

pub(crate) fn real_cube_input(
    data: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(Vec<f32>, Vec<usize>)> {
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

pub(crate) fn bool_cube_input(
    data: PyReadonlyArrayDyn<'_, bool>,
) -> PyResult<(Vec<bool>, Vec<usize>)> {
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

pub(crate) fn dzyx_shape(shape: &[usize]) -> PyResult<[usize; 4]> {
    match shape {
        [doppler, z, y, x] => Ok([*doppler, *z, *y, *x]),
        _ => Err(PyValueError::new_err(format!(
            "Cartesian magnitude volume must have shape (D, Z, Y, X); got {shape:?}."
        ))),
    }
}

pub(crate) fn contiguous_axis(values: PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<Vec<f32>> {
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float32 array.")))
        .map(ToOwned::to_owned)
}

pub(crate) fn real_cube_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let cube = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|_| PyValueError::new_err("Native radar magnitude shape is invalid."))?;
    Ok(cube.into_pyarray(py))
}

pub(crate) fn native_indices_array(
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
