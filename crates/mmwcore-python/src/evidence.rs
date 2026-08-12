//! PyO3 boundary for exact single-frame ADC evidence compression.

use super::*;
use mmwcore::{
    EvidenceCodecError, decode_evidence_frame as decode_native_evidence_frame,
    encode_evidence_frame as encode_native_evidence_frame,
};
use pyo3::types::PyBytes;

#[pyfunction]
fn encode_evidence_frame<'py>(py: Python<'py>, data: Vec<u8>) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = py
        .detach(move || encode_native_evidence_frame(&data))
        .map_err(evidence_codec_error)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
fn decode_evidence_frame<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    expected_raw_bytes: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let decoded = py
        .detach(move || decode_native_evidence_frame(&data, expected_raw_bytes))
        .map_err(evidence_codec_error)?;
    Ok(PyBytes::new(py, &decoded))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(encode_evidence_frame, module)?)?;
    module.add_function(wrap_pyfunction!(decode_evidence_frame, module)?)?;
    Ok(())
}

fn evidence_codec_error(error: EvidenceCodecError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
