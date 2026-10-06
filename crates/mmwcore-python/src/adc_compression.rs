//! PyO3 boundary for standardized ADC compression and decompression.

use mmwcore::{
    ADC_RICE_BLOCK_SAMPLES, AdcCompressionError, CompressedAdcFile, CompressedAdcFileError,
    compress_adc_file as compress_native_adc_file,
    compress_adc_frames as compress_native_adc_frames,
    decompress_adc_frames as decompress_native_adc_frames,
    open_compressed_adc as open_native_compressed_adc, sha256_from_hex, sha256_to_hex,
};
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyPermissionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::ErrorKind;
use std::path::Path;

#[pyfunction]
fn decompress_adc_file(py: Python<'_>, source: String, destination: String) -> PyResult<String> {
    py.detach(move || mmwcore::decompress_adc_file(Path::new(&source), Path::new(&destination)))
        .map_err(compressed_adc_file_error)
}

#[pyclass(module = "mmwcore._native", name = "CompressedADCFile")]
struct PyCompressedAdcFile {
    compressed: CompressedAdcFile,
}

#[pymethods]
impl PyCompressedAdcFile {
    #[getter]
    fn path(&self) -> String {
        self.compressed.path().to_string_lossy().into_owned()
    }

    #[getter]
    fn frame_bytes(&self) -> u64 {
        self.compressed.frame_bytes()
    }

    #[getter]
    fn frame_count(&self) -> u64 {
        self.compressed.frame_count()
    }

    #[getter]
    fn block_samples(&self) -> u32 {
        self.compressed.block_samples()
    }

    #[getter]
    fn restart_frames(&self) -> u32 {
        self.compressed.restart_frames()
    }

    #[getter]
    fn capture_json(&self) -> &str {
        self.compressed.capture_json()
    }

    #[getter]
    fn capture_sha256(&self) -> String {
        sha256_to_hex(self.compressed.capture_sha256())
    }

    #[getter]
    fn adc_sha256(&self) -> String {
        sha256_to_hex(self.compressed.adc_sha256())
    }

    #[getter]
    fn archive_size(&self) -> u64 {
        self.compressed.archive_size()
    }

    #[getter]
    fn compressed_size_bytes(&self) -> u64 {
        self.compressed.compressed_size_bytes()
    }

    #[getter]
    fn payload_bytes(&self) -> u64 {
        self.compressed.payload_bytes()
    }

    #[getter]
    fn index_bytes(&self) -> u64 {
        self.compressed.index_bytes()
    }

    #[getter]
    fn header_bytes(&self) -> u64 {
        self.compressed.header_bytes()
    }

    #[getter]
    fn capture_metadata_bytes(&self) -> u64 {
        self.compressed.capture_metadata_bytes()
    }

    #[getter]
    fn container_overhead_bytes(&self) -> u64 {
        self.compressed.container_overhead_bytes()
    }

    #[pyo3(signature = (start, stop, *, verify = true))]
    fn read_frames<'py>(
        &mut self,
        py: Python<'py>,
        start: u64,
        stop: u64,
        verify: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let decoded = py
            .detach(|| self.compressed.read_frames(start, stop, verify))
            .map_err(compressed_adc_file_error)?;
        Ok(PyBytes::new(py, &decoded))
    }

    #[pyo3(signature = (starts, window_frames, *, verify = true))]
    fn read_windows<'py>(
        &mut self,
        py: Python<'py>,
        starts: Vec<u64>,
        window_frames: u64,
        verify: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let decoded = py
            .detach(|| self.compressed.read_windows(&starts, window_frames, verify))
            .map_err(compressed_adc_file_error)?;
        Ok(PyBytes::new(py, &decoded))
    }

    fn verify_all(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.compressed.verify_all())
            .map_err(compressed_adc_file_error)
    }
}

#[pyfunction]
#[pyo3(signature = (data, frame_bytes, block_samples = ADC_RICE_BLOCK_SAMPLES))]
fn compress_adc_frames<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    frame_bytes: usize,
    block_samples: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = py
        .detach(move || compress_native_adc_frames(&data, frame_bytes, block_samples))
        .map_err(adc_compression_error)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
#[pyo3(signature = (data, frame_bytes, frame_count, block_samples = ADC_RICE_BLOCK_SAMPLES))]
fn decompress_adc_frames<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    frame_bytes: usize,
    frame_count: usize,
    block_samples: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let decoded = py
        .detach(move || {
            decompress_native_adc_frames(&data, frame_bytes, frame_count, block_samples)
        })
        .map_err(adc_compression_error)?;
    Ok(PyBytes::new(py, &decoded))
}

#[pyfunction]
fn open_compressed_adc(py: Python<'_>, path: String) -> PyResult<PyCompressedAdcFile> {
    let compressed = py
        .detach(move || open_native_compressed_adc(Path::new(&path)))
        .map_err(compressed_adc_file_error)?;
    Ok(PyCompressedAdcFile { compressed })
}

#[pyfunction]
#[pyo3(signature = (source, destination, capture_json, expected_adc_sha256 = None))]
fn compress_adc_file(
    py: Python<'_>,
    source: String,
    destination: String,
    capture_json: String,
    expected_adc_sha256: Option<String>,
) -> PyResult<PyCompressedAdcFile> {
    let expected = expected_adc_sha256
        .as_deref()
        .map(sha256_from_hex)
        .transpose()
        .map_err(compressed_adc_file_error)?;
    let compressed = py
        .detach(move || {
            compress_native_adc_file(
                Path::new(&source),
                Path::new(&destination),
                &capture_json,
                expected,
            )
        })
        .map_err(compressed_adc_file_error)?;
    Ok(PyCompressedAdcFile { compressed })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decompress_adc_file, module)?)?;
    module.add_class::<PyCompressedAdcFile>()?;
    module.add_function(wrap_pyfunction!(compress_adc_frames, module)?)?;
    module.add_function(wrap_pyfunction!(decompress_adc_frames, module)?)?;
    module.add_function(wrap_pyfunction!(open_compressed_adc, module)?)?;
    module.add_function(wrap_pyfunction!(compress_adc_file, module)?)?;
    for (old, new) in [
        ("ADCArchiveFile", "CompressedADCFile"),
        ("encode_adc_archive_chunk", "compress_adc_frames"),
        ("decode_adc_archive_chunk", "decompress_adc_frames"),
        ("open_adc_archive_file", "open_compressed_adc"),
        ("write_adc_archive_file", "compress_adc_file"),
    ] {
        module.add(old, module.getattr(new)?)?;
    }
    Ok(())
}

fn adc_compression_error(error: AdcCompressionError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn compressed_adc_file_error(error: CompressedAdcFileError) -> PyErr {
    let message = error.to_string();
    match error.io_kind() {
        Some(ErrorKind::NotFound) => PyFileNotFoundError::new_err(message),
        Some(ErrorKind::PermissionDenied) => PyPermissionError::new_err(message),
        Some(_) => PyOSError::new_err(message),
        None => PyValueError::new_err(message),
    }
}
