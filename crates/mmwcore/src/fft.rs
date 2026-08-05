//! Complex FFT transforms over one contiguous named-axis buffer.

use std::f32::consts::PI;
use std::fmt;

use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::cube::{CubeTransformError, checked_product, validate_axis};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftWindow {
    None,
    Hann,
    Hamming,
}

impl TryFrom<u8> for FftWindow {
    type Error = FftTransformError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Hann),
            2 => Ok(Self::Hamming),
            _ => Err(FftTransformError::UnsupportedWindow { window: value }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComplexFftSpec {
    pub n_fft: usize,
    pub window: FftWindow,
    pub remove_dc: bool,
    pub fftshift: bool,
    pub one_sided: bool,
}

impl ComplexFftSpec {
    pub fn new(
        n_fft: usize,
        window: FftWindow,
        remove_dc: bool,
        fftshift: bool,
        one_sided: bool,
    ) -> Result<Self, FftTransformError> {
        if n_fft == 0 {
            return Err(FftTransformError::InvalidFftLength { n_fft });
        }
        Ok(Self {
            n_fft,
            window,
            remove_dc,
            fftshift,
            one_sided,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FftTransformError {
    Cube(CubeTransformError),
    InvalidFftLength { n_fft: usize },
    UnsupportedWindow { window: u8 },
}

impl fmt::Display for FftTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cube(error) => error.fmt(formatter),
            Self::InvalidFftLength { n_fft } => {
                write!(formatter, "FFT length must be positive; got {n_fft}.")
            }
            Self::UnsupportedWindow { window } => {
                write!(formatter, "Unsupported native FFT window code {window}.")
            }
        }
    }
}

impl std::error::Error for FftTransformError {}

impl From<CubeTransformError> for FftTransformError {
    fn from(error: CubeTransformError) -> Self {
        Self::Cube(error)
    }
}

pub fn fft_complex_axis(
    data: &[Complex32],
    shape: &[usize],
    axis: usize,
    spec: ComplexFftSpec,
) -> Result<(Vec<Complex32>, Vec<usize>), FftTransformError> {
    validate_fft_shape(data, shape, axis)?;

    let input_length = shape[axis];
    let output_axis_length = if spec.one_sided {
        spec.n_fft / 2 + 1
    } else {
        spec.n_fft
    };
    let outer = checked_product(&shape[..axis])?;
    let inner = checked_product(&shape[axis + 1..])?;
    let mut output_shape = shape.to_vec();
    output_shape[axis] = output_axis_length;
    let output_length = checked_product(&output_shape)?;
    let mut output = vec![Complex32::new(0.0, 0.0); output_length];
    let window = window_coefficients(input_length, spec.window);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(spec.n_fft);
    let mut line = vec![Complex32::new(0.0, 0.0); spec.n_fft];
    let copied_length = input_length.min(spec.n_fft);

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let input_base = outer_index * input_length * inner + inner_index;
            let mean = if spec.remove_dc {
                let sum = (0..input_length).fold(Complex32::new(0.0, 0.0), |sum, index| {
                    sum + data[input_base + index * inner]
                });
                sum / input_length as f32
            } else {
                Complex32::new(0.0, 0.0)
            };

            line.fill(Complex32::new(0.0, 0.0));
            for index in 0..copied_length {
                let mut value = data[input_base + index * inner] - mean;
                if let Some(coefficients) = &window {
                    value *= coefficients[index];
                }
                line[index] = value;
            }
            fft.process(&mut line);
            if spec.fftshift {
                line.rotate_left(spec.n_fft.div_ceil(2));
            }

            let output_base = outer_index * output_axis_length * inner + inner_index;
            for index in 0..output_axis_length {
                output[output_base + index * inner] = line[index];
            }
        }
    }

    Ok((output, output_shape))
}

fn validate_fft_shape(
    data: &[Complex32],
    shape: &[usize],
    axis: usize,
) -> Result<(), FftTransformError> {
    if shape.is_empty() {
        return Err(CubeTransformError::EmptyShape.into());
    }
    validate_axis(shape, axis)?;
    if shape[axis] == 0 {
        return Err(CubeTransformError::ZeroDimension { axis }.into());
    }
    let expected = checked_product(shape)?;
    if data.len() != expected {
        return Err(CubeTransformError::ShapeSizeMismatch {
            expected,
            actual: data.len(),
        }
        .into());
    }
    Ok(())
}

fn window_coefficients(size: usize, window: FftWindow) -> Option<Vec<f32>> {
    match window {
        FftWindow::None => None,
        FftWindow::Hann => Some(cosine_window(size, 0.5, 0.5)),
        FftWindow::Hamming => Some(cosine_window(size, 0.54, 0.46)),
    }
}

fn cosine_window(size: usize, offset: f32, scale: f32) -> Vec<f32> {
    if size == 1 {
        return vec![1.0];
    }
    (0..size)
        .map(|index| offset - scale * (2.0 * PI * index as f32 / (size - 1) as f32).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ComplexFftSpec, FftWindow, fft_complex_axis};
    use num_complex::Complex32;

    #[test]
    fn transforms_impulse_and_preserves_axis_shape() {
        let spec = ComplexFftSpec::new(4, FftWindow::None, false, false, false).unwrap();
        let (output, shape) = fft_complex_axis(
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
            ],
            &[1, 1, 1, 4],
            3,
            spec,
        )
        .unwrap();

        assert_eq!(shape, [1, 1, 1, 4]);
        assert_eq!(output, vec![Complex32::new(1.0, 0.0); 4]);
    }

    #[test]
    fn removes_dc_pads_and_keeps_one_sided_output() {
        let spec = ComplexFftSpec::new(8, FftWindow::Hann, true, false, true).unwrap();
        let (output, shape) =
            fft_complex_axis(&[Complex32::new(2.0, 0.0); 4], &[1, 1, 1, 4], 3, spec).unwrap();

        assert_eq!(shape, [1, 1, 1, 5]);
        assert!(output.iter().all(|value| value.norm() < 1e-6));
    }

    #[test]
    fn shifts_odd_length_doppler_bins() {
        let spec = ComplexFftSpec::new(5, FftWindow::None, false, true, false).unwrap();
        let (output, _) = fft_complex_axis(
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
            ],
            &[5],
            0,
            spec,
        )
        .unwrap();

        assert_eq!(output, vec![Complex32::new(1.0, 0.0); 5]);
    }

    #[test]
    fn preserves_empty_nontransform_batch_axes() {
        let spec = ComplexFftSpec::new(4, FftWindow::None, false, true, false).unwrap();
        let (output, shape) = fft_complex_axis(&[], &[0, 4], 1, spec).unwrap();

        assert_eq!(shape, [0, 4]);
        assert!(output.is_empty());
    }
}
