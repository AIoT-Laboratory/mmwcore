//! PyO3 delegates optional TI state to its isolated, checked Rust host crate.
use mmwcore_ti_gtrack::{Config, Engine};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Mutex;

#[pyclass]
struct NativeTiGTrack3D {
    engine: Mutex<Option<Engine>>,
}

#[pymethods]
impl NativeTiGTrack3D {
    #[new]
    fn new(manifest_path: &str, config_json: &str) -> PyResult<Self> {
        let config: Config =
            serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let engine =
            Engine::load(Path::new(manifest_path), config).map_err(PyValueError::new_err)?;
        Ok(Self {
            engine: Mutex::new(Some(engine)),
        })
    }

    fn provenance_json(&self) -> PyResult<String> {
        let engine = self
            .engine
            .lock()
            .map_err(|_| PyRuntimeError::new_err("TI tracker lock poisoned"))?;
        let engine = engine
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("TI tracker is closed"))?;
        serde_json::to_string(engine.provenance())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (points, variances=None, *, cartesian=false))]
    fn step(
        &self,
        py: Python<'_>,
        points: PyReadonlyArray2<'_, f32>,
        variances: Option<PyReadonlyArray2<'_, f32>>,
        cartesian: bool,
    ) -> PyResult<String> {
        let points = rows::<5>(points, "points")?;
        let variances = variances.map(|v| rows::<4>(v, "variances")).transpose()?;
        let result = py
            .detach(|| {
                let mut engine = self
                    .engine
                    .lock()
                    .map_err(|_| "TI tracker lock poisoned".to_owned())?;
                let engine = engine.as_mut().ok_or("TI tracker is closed".to_owned())?;
                if cartesian {
                    engine.step_cartesian(&points, variances.as_deref())
                } else {
                    engine.step(&points, variances.as_deref())
                }
            })
            .map_err(PyValueError::new_err)?;
        serde_json::to_string(&result).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn close(&self) -> PyResult<()> {
        self.engine
            .lock()
            .map_err(|_| PyRuntimeError::new_err("TI tracker lock poisoned"))?
            .take();
        Ok(())
    }
}

fn rows<const N: usize>(array: PyReadonlyArray2<'_, f32>, name: &str) -> PyResult<Vec<[f32; N]>> {
    if array.shape()[1] != N {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N,{N})"
        )));
    }
    let view = array.as_array();
    Ok(view
        .rows()
        .into_iter()
        .map(|r| std::array::from_fn(|i| r[i]))
        .collect())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTiGTrack3D>()
}
