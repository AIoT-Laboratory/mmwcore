//! Checked Python boundary for the supported ISK Capon chain.
use mmwcore::capon::{AZIMUTH_BINS, IskCaponConfig, isk_capon};
use numpy::{Complex32, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
fn isk_capon_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    range_resolution_m: f64,
    velocity_resolution_mps: f64,
    doppler_bins: usize,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, String)> {
    let (data, shape) = crate::boundary::complex_cube_input(data)?;
    if shape.len() != 3 || shape[1] != 12 {
        return Err(PyValueError::new_err(
            "Capon input must have shape (loop, 12, range)",
        ));
    }
    let loops = shape[0];
    let ranges = shape[2];
    let result = py
        .detach(move || {
            isk_capon(
                &data,
                loops,
                ranges,
                IskCaponConfig {
                    range_resolution_m,
                    velocity_resolution_mps,
                    doppler_bins,
                },
            )
        })
        .map_err(PyValueError::new_err)?;
    let report = serde_json::to_string(
        &serde_json::json!({"detections": result.detections, "diagnostics": result.diagnostics}),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        crate::boundary::real_cube_array(py, &[AZIMUTH_BINS, ranges], result.ra_power)?,
        report,
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(isk_capon_complex, module)?)
}
