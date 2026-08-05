use std::fmt;

use num_complex::Complex32;

use crate::angle::AngleCalibrationError;
use crate::cube::{CubeTransformError, checked_product, contiguous_strides, validate_axis};
use crate::fft::{ComplexFftSpec, FftTransformError, FftWindow, fft_complex_axis};

/// Named cube axes consumed by candidate-level angle recovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateCubeAxes {
    pub frame: usize,
    pub doppler: usize,
    pub antenna: usize,
    pub range: usize,
}

/// Contiguous complex cube and explicit axis interpretation.
#[derive(Clone, Copy, Debug)]
pub struct CandidateCubeInput<'a> {
    pub data: &'a [Complex32],
    pub shape: &'a [usize],
    pub axes: CandidateCubeAxes,
}

/// Candidate detection matrix in public `float32` storage.
#[derive(Clone, Copy, Debug)]
pub struct CandidateMatrixInput<'a> {
    pub values: &'a [f32],
    pub shape: [usize; 2],
}

/// Frame, range, and Doppler columns in one candidate matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateIndexColumns {
    pub frame: usize,
    pub range: usize,
    pub doppler: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CandidateAoaError {
    Cube(CubeTransformError),
    Fft(FftTransformError),
    AngleCalibration(AngleCalibrationError),
    CubeRankMismatch {
        rank: usize,
    },
    CandidateShapeOverflow,
    CandidateShapeMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteCandidates,
    CandidateColumnOutOfBounds {
        name: &'static str,
        index: usize,
        width: usize,
    },
    CandidateIndexOutOfBounds,
    LayoutAntennaMismatch {
        expected: usize,
        actual: usize,
    },
    SubarrayIndexOutOfBounds,
    SubarrayLayoutMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    AzimuthBinOutOfBounds,
    RowRequiresTwoAntennas {
        name: &'static str,
    },
    NonHorizontalRow {
        name: &'static str,
    },
    UnequalRowSpacing,
    DifferentForwardCoordinates,
    InvalidVerticalSeparation,
}

impl fmt::Display for CandidateAoaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cube(error) => error.fmt(formatter),
            Self::Fft(error) => error.fmt(formatter),
            Self::AngleCalibration(error) => error.fmt(formatter),
            Self::CubeRankMismatch { rank } => {
                write!(
                    formatter,
                    "Candidate angle-estimation cube must have rank 4; got {rank}."
                )
            }
            Self::CandidateShapeOverflow => {
                write!(
                    formatter,
                    "Candidate angle-estimation shape overflows usize."
                )
            }
            Self::CandidateShapeMismatch { expected, actual } => write!(
                formatter,
                "Candidate angle-estimation matrix has {actual} values; expected {expected}."
            ),
            Self::NonFiniteCandidates => {
                write!(
                    formatter,
                    "Candidate angle-estimation matrix contains NaN or Inf values."
                )
            }
            Self::CandidateColumnOutOfBounds { name, index, width } => write!(
                formatter,
                "Candidate angle-estimation column {name}={index} is outside width {width}."
            ),
            Self::CandidateIndexOutOfBounds => {
                write!(
                    formatter,
                    "Detection candidate index is outside the angle-estimation cube."
                )
            }
            Self::LayoutAntennaMismatch { .. } => {
                write!(
                    formatter,
                    "Angle layout antenna count must match the candidate cube."
                )
            }
            Self::SubarrayIndexOutOfBounds => {
                write!(
                    formatter,
                    "Elevation subarray index exceeds the candidate cube."
                )
            }
            Self::SubarrayLayoutMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "The {name} subarray has {actual} positions; expected {expected}."
            ),
            Self::AzimuthBinOutOfBounds => {
                write!(
                    formatter,
                    "Detection azimuth_bin is outside the configured angle FFT."
                )
            }
            Self::RowRequiresTwoAntennas { name } => {
                write!(formatter, "The {name} row requires at least two antennas.")
            }
            Self::NonHorizontalRow { name } => {
                write!(
                    formatter,
                    "The {name} subarray must be one horizontal antenna row."
                )
            }
            Self::UnequalRowSpacing => {
                write!(
                    formatter,
                    "Paired antenna rows must use the same uniform horizontal spacing."
                )
            }
            Self::DifferentForwardCoordinates => {
                write!(
                    formatter,
                    "Paired antenna rows must share the same forward coordinate."
                )
            }
            Self::InvalidVerticalSeparation => write!(
                formatter,
                "Paired antenna rows require a non-zero vertical separation no larger than half a wavelength."
            ),
        }
    }
}

impl std::error::Error for CandidateAoaError {}

impl From<CubeTransformError> for CandidateAoaError {
    fn from(error: CubeTransformError) -> Self {
        Self::Cube(error)
    }
}

impl From<FftTransformError> for CandidateAoaError {
    fn from(error: FftTransformError) -> Self {
        Self::Fft(error)
    }
}

impl From<AngleCalibrationError> for CandidateAoaError {
    fn from(error: AngleCalibrationError) -> Self {
        Self::AngleCalibration(error)
    }
}

pub(super) struct CandidateCubeLayout<'a> {
    data: &'a [Complex32],
    shape: &'a [usize],
    strides: Vec<usize>,
    axes: CandidateCubeAxes,
}

impl<'a> CandidateCubeLayout<'a> {
    pub(super) fn new(input: CandidateCubeInput<'a>) -> Result<Self, CandidateAoaError> {
        if input.shape.len() != 4 {
            return Err(CandidateAoaError::CubeRankMismatch {
                rank: input.shape.len(),
            });
        }
        let expected = checked_product(input.shape)?;
        if input.data.len() != expected {
            return Err(CubeTransformError::ShapeSizeMismatch {
                expected,
                actual: input.data.len(),
            }
            .into());
        }
        let axes = [
            input.axes.frame,
            input.axes.doppler,
            input.axes.antenna,
            input.axes.range,
        ];
        for &axis in &axes {
            validate_axis(input.shape, axis)?;
        }
        for (index, &first) in axes.iter().enumerate() {
            if let Some(&second) = axes[index + 1..].iter().find(|&&second| second == first) {
                return Err(CubeTransformError::DuplicateAxes { first, second }.into());
            }
        }
        if input.shape[input.axes.antenna] == 0 {
            return Err(CubeTransformError::ZeroDimension {
                axis: input.axes.antenna,
            }
            .into());
        }
        let strides = contiguous_strides(input.shape)?;
        Ok(Self {
            data: input.data,
            shape: input.shape,
            strides,
            axes: input.axes,
        })
    }

    pub(super) fn antenna_count(&self) -> usize {
        self.shape[self.axes.antenna]
    }

    fn value(&self, frame: usize, doppler: usize, antenna: usize, range: usize) -> Complex32 {
        self.data[frame * self.strides[self.axes.frame]
            + doppler * self.strides[self.axes.doppler]
            + antenna * self.strides[self.axes.antenna]
            + range * self.strides[self.axes.range]]
    }
}

pub(super) fn validate_candidates(
    input: CandidateMatrixInput<'_>,
) -> Result<(), CandidateAoaError> {
    let expected = input.shape.iter().try_fold(1_usize, |size, &dimension| {
        size.checked_mul(dimension)
            .ok_or(CandidateAoaError::CandidateShapeOverflow)
    })?;
    if input.values.len() != expected {
        return Err(CandidateAoaError::CandidateShapeMismatch {
            expected,
            actual: input.values.len(),
        });
    }
    if input.values.iter().any(|value| !value.is_finite()) {
        return Err(CandidateAoaError::NonFiniteCandidates);
    }
    Ok(())
}

pub(super) fn validate_index_columns(
    width: usize,
    columns: CandidateIndexColumns,
) -> Result<(), CandidateAoaError> {
    for (name, index) in [
        ("frame", columns.frame),
        ("range_bin", columns.range),
        ("doppler_bin", columns.doppler),
    ] {
        validate_column(width, name, index)?;
    }
    Ok(())
}

pub(super) fn validate_column(
    width: usize,
    name: &'static str,
    index: usize,
) -> Result<(), CandidateAoaError> {
    if index >= width {
        return Err(CandidateAoaError::CandidateColumnOutOfBounds { name, index, width });
    }
    Ok(())
}

pub(super) fn extract_candidate_vectors(
    cube: &CandidateCubeLayout<'_>,
    candidates: CandidateMatrixInput<'_>,
    columns: CandidateIndexColumns,
    antenna_indices: &[usize],
) -> Result<Vec<Complex32>, CandidateAoaError> {
    let capacity = candidates.shape[0]
        .checked_mul(antenna_indices.len())
        .ok_or(CandidateAoaError::CandidateShapeOverflow)?;
    let mut vectors = Vec::with_capacity(capacity);
    for candidate_index in 0..candidates.shape[0] {
        let row = candidate_row(candidates, candidate_index);
        let frame = candidate_integer(row[columns.frame])
            .ok_or(CandidateAoaError::CandidateIndexOutOfBounds)?;
        let doppler = candidate_integer(row[columns.doppler])
            .ok_or(CandidateAoaError::CandidateIndexOutOfBounds)?;
        let range = candidate_integer(row[columns.range])
            .ok_or(CandidateAoaError::CandidateIndexOutOfBounds)?;
        if frame >= cube.shape[cube.axes.frame]
            || doppler >= cube.shape[cube.axes.doppler]
            || range >= cube.shape[cube.axes.range]
        {
            return Err(CandidateAoaError::CandidateIndexOutOfBounds);
        }
        for &antenna in antenna_indices {
            vectors.push(cube.value(frame, doppler, antenna, range));
        }
    }
    Ok(vectors)
}

pub(super) fn candidate_row(input: CandidateMatrixInput<'_>, index: usize) -> &[f32] {
    let width = input.shape[1];
    &input.values[index * width..(index + 1) * width]
}

pub(super) fn candidate_integer(value: f32) -> Option<usize> {
    let value = value.trunc();
    if value < 0.0 || value > usize::MAX as f32 {
        None
    } else {
        Some(value as usize)
    }
}

pub(super) fn row_spectra(
    vectors: &[Complex32],
    candidate_count: usize,
    antenna_count: usize,
    n_fft: usize,
    window: FftWindow,
    fftshift: bool,
) -> Result<Vec<Complex32>, CandidateAoaError> {
    let (spectra, shape) = fft_complex_axis(
        vectors,
        &[candidate_count, antenna_count],
        1,
        ComplexFftSpec::new(n_fft, window, false, fftshift, false)?,
    )?;
    debug_assert_eq!(shape, [candidate_count, n_fft]);
    Ok(spectra)
}

pub(super) fn first_peak(spectrum: &[Complex32]) -> (usize, f32) {
    let mut peak_bin = 0;
    let mut peak_magnitude = spectrum[0].norm();
    for (bin, value) in spectrum.iter().enumerate().skip(1) {
        let magnitude = value.norm();
        if !peak_magnitude.is_nan() && (magnitude.is_nan() || magnitude > peak_magnitude) {
            peak_bin = bin;
            peak_magnitude = magnitude;
        }
    }
    (peak_bin, peak_magnitude)
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::first_peak;

    #[test]
    fn keeps_the_first_nan_peak_like_numpy_argmax() {
        let spectrum = [
            Complex32::new(1.0, 0.0),
            Complex32::new(f32::NAN, 0.0),
            Complex32::new(f32::NAN, 0.0),
        ];

        assert_eq!(first_peak(&spectrum).0, 1);
    }
}
