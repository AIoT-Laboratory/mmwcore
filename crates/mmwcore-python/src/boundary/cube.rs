//! Cube boundary contracts.

use mmwcore::{CubeTransformError, FftTransformError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn cube_error(error: CubeTransformError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn fft_error(error: FftTransformError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) const FFT_REMOVE_DC_FLAG: u8 = 1;

pub(crate) const FFT_SHIFT_FLAG: u8 = 1 << 1;

pub(crate) const FFT_ONE_SIDED_FLAG: u8 = 1 << 2;

pub(crate) const FFT_FLAGS_MASK: u8 = FFT_REMOVE_DC_FLAG | FFT_SHIFT_FLAG | FFT_ONE_SIDED_FLAG;
