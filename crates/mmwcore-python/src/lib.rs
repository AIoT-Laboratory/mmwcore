#![forbid(unsafe_code)]

use pyo3::prelude::*;

mod adc_archive;
mod boundary;
mod capon;
mod capture;
mod cartesian;
mod cube;
mod detection;
mod geometry;
mod multiscale;
mod ti_gtrack;
mod tracking_metrics;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    capture::register(module)?;
    capon::register(module)?;
    cartesian::register(module)?;
    cube::register(module)?;
    detection::register(module)?;
    adc_archive::register(module)?;
    geometry::register(module)?;
    tracking_metrics::register(module)?;
    ti_gtrack::register(module)?;
    multiscale::register(module)?;
    Ok(())
}
