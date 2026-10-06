//! Checked float64 boundary for the independent multiscale tracking backend.

use mmwcore::tracking::multiscale::{Point, ScatterBodyTracker, ScatterConfig, ScatterState};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
fn scatter_body_step(
    points: PyReadonlyArray2<'_, f64>,
    config_json: &str,
    state_json: &str,
    dt: f64,
) -> PyResult<String> {
    if points.shape()[1] != 5 {
        return Err(PyValueError::new_err("Points must have shape (N, 5)"));
    }
    if !points.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Points must be a C-contiguous float64 matrix",
        ));
    }
    let values = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("Points must be a C-contiguous float64 matrix"))?;
    let points: Vec<Point> = values
        .chunks_exact(5)
        .map(|p| [p[0], p[1], p[2], p[3], p[4]])
        .collect();
    let config: ScatterConfig =
        serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let state: ScatterState =
        serde_json::from_str(state_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut tracker =
        ScatterBodyTracker::new(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
    tracker.state = state;
    let output = tracker
        .step(&points, dt)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&serde_json::json!({"state":tracker.state,"output":output}))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(scatter_body_step, module)?)
}
