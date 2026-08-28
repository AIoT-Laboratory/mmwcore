//! PyO3 boundary for native ADC decoding.

use numpy::ndarray::Array4;
use numpy::{Complex32, IntoPyArray, PyArray4, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};

use super::{AdcComplexLayout, AdcFrameSpec, decode_error, decode_native_adc_i16};

#[pyfunction]
fn decode_adc_i16<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'py, i16>,
    num_chirps: usize,
    num_rx: usize,
    num_samples: usize,
    layout: u8,
    drop_incomplete: bool,
) -> PyResult<Bound<'py, PyArray4<Complex32>>> {
    let samples = samples
        .as_slice()
        .map_err(|_| PyValueError::new_err("ADC samples must be a contiguous int16 array."))?
        .to_vec();
    let layout = AdcComplexLayout::try_from(layout).map_err(decode_error)?;
    let spec = AdcFrameSpec::new(num_chirps, num_rx, num_samples, layout).map_err(decode_error)?;
    let cube = py
        .detach(move || decode_native_adc_i16(&samples, spec, drop_incomplete))
        .map_err(decode_error)?;
    let [frames, chirps, receivers, samples] = cube.shape();
    let array = Array4::from_shape_vec((frames, chirps, receivers, samples), cube.into_data())
        .map_err(|_| PyValueError::new_err("Native ADC cube shape is invalid."))?;

    Ok(array.into_pyarray(py))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decode_adc_i16, module)?)?;
    Ok(())
}
