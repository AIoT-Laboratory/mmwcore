//! Capture boundary contracts.

use mmwcore::AdcDecodeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn decode_error(error: AdcDecodeError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
