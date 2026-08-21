//! PyO3 boundary for exact ADC archive chunk compression.

use super::*;
use mmwcore::{
    ADC_RICE_BLOCK_SAMPLES, AdcArchiveCodecError, AdcArchiveFile, AdcArchiveFileError,
    decode_adc_archive_chunk as decode_native_adc_archive_chunk,
    encode_adc_archive_chunk as encode_native_adc_archive_chunk,
    open_adc_archive_file as open_native_adc_archive_file, sha256_from_hex, sha256_to_hex,
    write_adc_archive_file as write_native_adc_archive_file,
};
use pyo3::types::PyBytes;
use std::path::Path;

#[pyclass(module = "mmwcore._native", name = "ADCArchiveFile")]
struct PyAdcArchiveFile {
    archive: AdcArchiveFile,
}

#[pymethods]
impl PyAdcArchiveFile {
    #[getter]
    fn path(&self) -> String {
        self.archive.path().to_string_lossy().into_owned()
    }

    #[getter]
    fn frame_bytes(&self) -> u64 {
        self.archive.frame_bytes()
    }

    #[getter]
    fn frame_count(&self) -> u64 {
        self.archive.frame_count()
    }

    #[getter]
    fn block_samples(&self) -> u32 {
        self.archive.block_samples()
    }

    #[getter]
    fn restart_frames(&self) -> u32 {
        self.archive.restart_frames()
    }

    #[getter]
    fn capture_json(&self) -> &str {
        self.archive.capture_json()
    }

    #[getter]
    fn capture_sha256(&self) -> String {
        sha256_to_hex(self.archive.capture_sha256())
    }

    #[getter]
    fn adc_sha256(&self) -> String {
        sha256_to_hex(self.archive.adc_sha256())
    }

    #[getter]
    fn archive_size(&self) -> u64 {
        self.archive.archive_size()
    }

    #[getter]
    fn payload_bytes(&self) -> u64 {
        self.archive.payload_bytes()
    }

    #[getter]
    fn index_bytes(&self) -> u64 {
        self.archive.index_bytes()
    }

    #[getter]
    fn header_bytes(&self) -> u64 {
        self.archive.header_bytes()
    }

    #[getter]
    fn capture_metadata_bytes(&self) -> u64 {
        self.archive.capture_metadata_bytes()
    }

    #[getter]
    fn container_overhead_bytes(&self) -> u64 {
        self.archive.container_overhead_bytes()
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
            .detach(|| self.archive.read_frames(start, stop, verify))
            .map_err(adc_archive_file_error)?;
        Ok(PyBytes::new(py, &decoded))
    }

    fn verify_all(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.archive.verify_all())
            .map_err(adc_archive_file_error)
    }

    fn revalidate_input(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.archive.revalidate_input())
            .map_err(adc_archive_file_error)
    }
}

#[pyfunction]
#[pyo3(signature = (data, frame_bytes, block_samples = ADC_RICE_BLOCK_SAMPLES))]
fn encode_adc_archive_chunk<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    frame_bytes: usize,
    block_samples: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = py
        .detach(move || encode_native_adc_archive_chunk(&data, frame_bytes, block_samples))
        .map_err(adc_archive_codec_error)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
#[pyo3(signature = (data, frame_bytes, frame_count, block_samples = ADC_RICE_BLOCK_SAMPLES))]
fn decode_adc_archive_chunk<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    frame_bytes: usize,
    frame_count: usize,
    block_samples: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let decoded = py
        .detach(move || {
            decode_native_adc_archive_chunk(&data, frame_bytes, frame_count, block_samples)
        })
        .map_err(adc_archive_codec_error)?;
    Ok(PyBytes::new(py, &decoded))
}

#[pyfunction]
fn open_adc_archive_file(py: Python<'_>, path: String) -> PyResult<PyAdcArchiveFile> {
    let archive = py
        .detach(move || open_native_adc_archive_file(Path::new(&path)))
        .map_err(adc_archive_file_error)?;
    Ok(PyAdcArchiveFile { archive })
}

#[pyfunction]
#[pyo3(signature = (source, destination, capture_json, expected_adc_sha256 = None))]
fn write_adc_archive_file(
    py: Python<'_>,
    source: String,
    destination: String,
    capture_json: String,
    expected_adc_sha256: Option<String>,
) -> PyResult<PyAdcArchiveFile> {
    let expected = expected_adc_sha256
        .as_deref()
        .map(sha256_from_hex)
        .transpose()
        .map_err(adc_archive_file_error)?;
    let archive = py
        .detach(move || {
            write_native_adc_archive_file(
                Path::new(&source),
                Path::new(&destination),
                &capture_json,
                expected,
            )
        })
        .map_err(adc_archive_file_error)?;
    Ok(PyAdcArchiveFile { archive })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAdcArchiveFile>()?;
    module.add_function(wrap_pyfunction!(encode_adc_archive_chunk, module)?)?;
    module.add_function(wrap_pyfunction!(decode_adc_archive_chunk, module)?)?;
    module.add_function(wrap_pyfunction!(open_adc_archive_file, module)?)?;
    module.add_function(wrap_pyfunction!(write_adc_archive_file, module)?)?;
    Ok(())
}

fn adc_archive_codec_error(error: AdcArchiveCodecError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn adc_archive_file_error(error: AdcArchiveFileError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
