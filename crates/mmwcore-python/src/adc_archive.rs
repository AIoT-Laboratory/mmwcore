//! PyO3 boundary for exact single-frame ADC archive compression.

use super::*;
use mmwcore::{
    AdcArchiveCodecError, decode_adc_archive_frame as decode_native_adc_archive_frame,
    encode_adc_archive_frame as encode_native_adc_archive_frame,
};
use pyo3::types::PyBytes;

#[pyfunction]
fn encode_adc_archive_frame<'py>(py: Python<'py>, data: Vec<u8>) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = py
        .detach(move || encode_native_adc_archive_frame(&data))
        .map_err(adc_archive_codec_error)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
fn decode_adc_archive_frame<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    expected_raw_bytes: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let decoded = py
        .detach(move || decode_native_adc_archive_frame(&data, expected_raw_bytes))
        .map_err(adc_archive_codec_error)?;
    Ok(PyBytes::new(py, &decoded))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(encode_adc_archive_frame, module)?)?;
    module.add_function(wrap_pyfunction!(decode_adc_archive_frame, module)?)?;
    Ok(())
}

fn adc_archive_codec_error(error: AdcArchiveCodecError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
