//! PyO3 boundary for native complex-cube transforms and FFTs.

use super::*;

#[pyfunction]
fn remove_static_clutter_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    axis: usize,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let output_shape = shape.clone();
    let output = py
        .detach(move || remove_native_static_clutter(&data, &shape, axis))
        .map_err(cube_error)?;
    complex_cube_array(py, &output_shape, output)
}

#[pyfunction]
fn fft_complex_axis<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    axis: usize,
    n_fft: usize,
    window: u8,
    flags: u8,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let window = FftWindow::try_from(window).map_err(fft_error)?;
    if flags & !FFT_FLAGS_MASK != 0 {
        return Err(PyValueError::new_err(format!(
            "Unsupported native FFT flags {flags:#010b}."
        )));
    }
    let spec = ComplexFftSpec::new(
        n_fft,
        window,
        flags & FFT_REMOVE_DC_FLAG != 0,
        flags & FFT_SHIFT_FLAG != 0,
        flags & FFT_ONE_SIDED_FLAG != 0,
    )
    .map_err(fft_error)?;
    let (output, output_shape) = py
        .detach(move || fft_native_complex_axis(&data, &shape, axis, spec))
        .map_err(fft_error)?;
    complex_cube_array(py, &output_shape, output)
}

#[pyfunction]
fn apply_time_domain_channel_calibration_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    tx_axis: usize,
    rx_axis: usize,
    sample_axis: usize,
    frequencies_rad_per_sample: PyReadonlyArray1<'py, f32>,
    corrections: PyReadonlyArray1<'py, Complex32>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let frequencies = frequencies_rad_per_sample
        .as_slice()
        .map_err(|_| PyValueError::new_err("Calibration frequencies must be contiguous float32."))?
        .to_vec();
    let corrections = corrections
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("Calibration corrections must be contiguous complex64.")
        })?
        .to_vec();
    let output_shape = shape.clone();
    let output = py
        .detach(move || {
            apply_native_time_domain_calibration(
                &data,
                &shape,
                tx_axis,
                rx_axis,
                sample_axis,
                &frequencies,
                &corrections,
            )
        })
        .map_err(cube_error)?;
    complex_cube_array(py, &output_shape, output)
}

#[pyfunction]
fn apply_virtual_channel_calibration_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    virtual_axis: usize,
    coefficients: PyReadonlyArray1<'py, Complex32>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let coefficients = coefficients
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("Calibration coefficients must be contiguous complex64.")
        })?
        .to_vec();
    let output_shape = shape.clone();
    let output = py
        .detach(move || {
            apply_native_virtual_calibration(&data, &shape, virtual_axis, &coefficients)
        })
        .map_err(cube_error)?;
    complex_cube_array(py, &output_shape, output)
}

#[pyfunction]
fn map_tdm_virtual_array_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    chirp_axis: usize,
    rx_axis: usize,
    num_tx: usize,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let (output, output_shape) = py
        .detach(move || map_native_tdm_virtual_array(&data, &shape, chirp_axis, rx_axis, num_tx))
        .map_err(cube_error)?;
    complex_cube_array(py, &output_shape, output)
}

#[pyfunction]
fn compensate_tdm_doppler_phase_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    doppler_axis: usize,
    virtual_axis: usize,
    num_tx: usize,
    num_rx: usize,
    fftshift: bool,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let output_shape = shape.clone();
    let output = py
        .detach(move || {
            compensate_native_tdm_doppler_phase(
                &data,
                &shape,
                doppler_axis,
                virtual_axis,
                num_tx,
                num_rx,
                fftshift,
            )
        })
        .map_err(cube_error)?;
    complex_cube_array(py, &output_shape, output)
}

#[pyfunction]
fn map_planar_aperture_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    virtual_axis: usize,
    grid_indices: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let (output, output_shape) = py
        .detach(move || map_native_planar_aperture(&data, &shape, virtual_axis, &grid_indices))
        .map_err(cube_error)?;
    complex_cube_array(py, &output_shape, output)
}

#[pyfunction]
fn select_virtual_subarray_complex<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, Complex32>,
    virtual_axis: usize,
    indices: Vec<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex32>>> {
    let (data, shape) = complex_cube_input(data)?;
    let (output, output_shape) = py
        .detach(move || select_native_virtual_subarray(&data, &shape, virtual_axis, &indices))
        .map_err(cube_error)?;
    complex_cube_array(py, &output_shape, output)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(remove_static_clutter_complex, module)?)?;
    module.add_function(wrap_pyfunction!(fft_complex_axis, module)?)?;
    module.add_function(wrap_pyfunction!(
        apply_time_domain_channel_calibration_complex,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        apply_virtual_channel_calibration_complex,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(map_tdm_virtual_array_complex, module)?)?;
    module.add_function(wrap_pyfunction!(
        compensate_tdm_doppler_phase_complex,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(map_planar_aperture_complex, module)?)?;
    module.add_function(wrap_pyfunction!(select_virtual_subarray_complex, module)?)?;
    Ok(())
}
