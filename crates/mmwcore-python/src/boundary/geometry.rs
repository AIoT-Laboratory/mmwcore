//! Geometry boundary contracts.
use numpy::IntoPyArray;

use mmwcore::{
    AngleCalibrationError, AssignmentError, AssignmentResult, CandidateAoaError, ClusterError,
    ClusterResult, DbscanConfig, PointColumns,
};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn angle_calibration_error(error: AngleCalibrationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn candidate_aoa_error(error: CandidateAoaError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn cluster_error(error: ClusterError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn assignment_error(error: AssignmentError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) type NativePointColumns = (usize, usize, usize, Option<usize>);

pub(crate) type NativeDbscanConfig = (f32, usize, f32, bool);

pub(crate) type NativeClusterResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i64>>,
);

pub(crate) type NativeAssignmentResult<'py> =
    (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

pub(crate) type NativeCandidateCubeAxes = (usize, usize, usize, usize);

pub(crate) type NativeCandidateIndexColumns = (usize, usize, usize);

pub(crate) type NativeCandidateElevationColumns = (usize, usize, usize, usize, usize);

pub(crate) type NativeCandidateAzimuthConfig = (usize, u8, bool, u8);

pub(crate) type NativeCandidateSubarrays<'py> = (
    Vec<usize>,
    Vec<usize>,
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, f64>,
);

pub(crate) type NativeCandidateElevationConfig = (usize, u8, bool);

pub(crate) type NativeCandidateAzimuthResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);

pub(crate) type NativeCandidateElevationResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    (f64, f64),
);

pub(crate) fn point_columns(columns: NativePointColumns) -> PointColumns {
    let (x, y, z, velocity) = columns;
    PointColumns { x, y, z, velocity }
}

pub(crate) fn dbscan_config(config: NativeDbscanConfig) -> PyResult<DbscanConfig> {
    let (eps_m, min_samples, velocity_scale_s, use_z) = config;
    DbscanConfig::new(eps_m, min_samples, velocity_scale_s, use_z).map_err(cluster_error)
}

pub(crate) fn cluster_result_array(
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

pub(crate) fn assignment_result_array(
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
