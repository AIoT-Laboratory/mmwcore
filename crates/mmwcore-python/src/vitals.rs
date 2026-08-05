//! PyO3 boundary for native vital-sign phase primitives.

use mmwcore::{
    VitalSignError, unwrap_vital_phase_complex,
    vital_phase_to_displacement as native_vital_phase_to_displacement,
};
use numpy::{Complex32, IntoPyArray, PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn unwrap_vital_phase<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'py, Complex32>,
    remove_mean: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if !samples.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Vital-sign samples must be a C-contiguous complex64 vector.",
        ));
    }
    let samples = samples
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("Vital-sign samples must be a contiguous complex64 vector.")
        })?
        .to_vec();
    let phase = py
        .detach(move || unwrap_vital_phase_complex(&samples, remove_mean))
        .map_err(vital_error)?;
    Ok(phase.into_pyarray(py))
}

#[pyfunction]
fn vital_phase_to_displacement<'py>(
    py: Python<'py>,
    phase_rad: PyReadonlyArray1<'py, f32>,
    wavelength_m: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if !phase_rad.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Vital-sign phase must be a C-contiguous float32 vector.",
        ));
    }
    let phase_rad = phase_rad
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("Vital-sign phase must be a contiguous float32 vector.")
        })?
        .to_vec();
    let displacement = py
        .detach(move || native_vital_phase_to_displacement(&phase_rad, wavelength_m))
        .map_err(vital_error)?;
    Ok(displacement.into_pyarray(py))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(unwrap_vital_phase, module)?)?;
    module.add_function(wrap_pyfunction!(vital_phase_to_displacement, module)?)?;
    Ok(())
}

fn vital_error(error: VitalSignError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
