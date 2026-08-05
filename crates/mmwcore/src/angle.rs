//! Calibrated physical angles for uniform linear virtual arrays.

use std::fmt;

/// Physical axis represented by a virtual linear array.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AngleAxis {
    Azimuth,
    Elevation,
}

impl TryFrom<u8> for AngleAxis {
    type Error = AngleCalibrationError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Azimuth),
            1 => Ok(Self::Elevation),
            _ => Err(AngleCalibrationError::UnsupportedAxis { axis: value }),
        }
    }
}

impl AngleAxis {
    fn coordinate_index(self) -> usize {
        match self {
            Self::Azimuth => 0,
            Self::Elevation => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Azimuth => "Azimuth",
            Self::Elevation => "Elevation",
        }
    }

    fn coordinate_name(self) -> &'static str {
        match self {
            Self::Azimuth => "x",
            Self::Elevation => "z",
        }
    }
}

/// Flat `(antenna, xyz)` position matrix in wavelengths.
#[derive(Clone, Copy, Debug)]
pub struct AngleBinCalibrationInput<'a> {
    pub positions_wavelengths: &'a [f32],
    pub position_count: usize,
}

/// Calibration configuration for one angle FFT axis.
#[derive(Clone, Copy, Debug)]
pub struct AngleBinCalibrationConfig {
    pub num_bins: usize,
    pub axis: AngleAxis,
    pub fftshift: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AngleCalibrationError {
    PositionShapeOverflow,
    PositionShapeMismatch { expected: usize, actual: usize },
    NonFinitePosition,
    InvalidBinCount { num_bins: usize },
    InsufficientAntennas,
    NonLinearArray { axis: AngleAxis },
    NonUniformArray { axis: AngleAxis },
    InvisibleBins,
    UnsupportedAxis { axis: u8 },
}

impl fmt::Display for AngleCalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionShapeOverflow => {
                write!(
                    formatter,
                    "Angle calibration position shape overflows usize."
                )
            }
            Self::PositionShapeMismatch { expected, actual } => write!(
                formatter,
                "Angle calibration position buffer has {actual} values; expected {expected}."
            ),
            Self::NonFinitePosition => {
                write!(formatter, "Angle calibration positions must be finite.")
            }
            Self::InvalidBinCount { num_bins } => {
                write!(formatter, "num_bins must be positive; got {num_bins}.")
            }
            Self::InsufficientAntennas => {
                write!(
                    formatter,
                    "At least two virtual antennas are required for angle calibration."
                )
            }
            Self::NonLinearArray { axis } => write!(
                formatter,
                "{} calibration requires a linear array along the {}-axis.",
                axis.name(),
                axis.coordinate_name(),
            ),
            Self::NonUniformArray { axis } => write!(
                formatter,
                "{} calibration requires uniformly spaced positions.",
                axis.name(),
            ),
            Self::InvisibleBins => write!(
                formatter,
                "Angle bins exceed visible azimuth range for the virtual antenna spacing."
            ),
            Self::UnsupportedAxis { axis } => {
                write!(formatter, "Unsupported native angle axis code {axis}.")
            }
        }
    }
}

impl std::error::Error for AngleCalibrationError {}

/// Return physical angle radians for a calibrated uniform linear array.
pub fn calibrate_angle_bins(
    input: AngleBinCalibrationInput<'_>,
    config: AngleBinCalibrationConfig,
) -> Result<Vec<f32>, AngleCalibrationError> {
    if config.num_bins == 0 {
        return Err(AngleCalibrationError::InvalidBinCount { num_bins: 0 });
    }
    validate_positions(input)?;
    let spacing = uniform_linear_spacing(input, config.axis)?;
    let mut angles = Vec::with_capacity(config.num_bins);

    for bin in 0..config.num_bins {
        let spatial_frequency = fft_frequency(bin, config.num_bins, config.fftshift);
        let direction_cosine = spatial_frequency / spacing;
        if direction_cosine.abs() > 1.0 {
            return Err(AngleCalibrationError::InvisibleBins);
        }
        angles.push(direction_cosine.asin() as f32);
    }

    Ok(angles)
}

fn validate_positions(input: AngleBinCalibrationInput<'_>) -> Result<(), AngleCalibrationError> {
    let expected = input
        .position_count
        .checked_mul(3)
        .ok_or(AngleCalibrationError::PositionShapeOverflow)?;
    if input.positions_wavelengths.len() != expected {
        return Err(AngleCalibrationError::PositionShapeMismatch {
            expected,
            actual: input.positions_wavelengths.len(),
        });
    }
    if input.position_count < 2 {
        return Err(AngleCalibrationError::InsufficientAntennas);
    }
    if input
        .positions_wavelengths
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(AngleCalibrationError::NonFinitePosition);
    }
    Ok(())
}

fn uniform_linear_spacing(
    input: AngleBinCalibrationInput<'_>,
    axis: AngleAxis,
) -> Result<f64, AngleCalibrationError> {
    let coordinate_index = axis.coordinate_index();
    for fixed_index in 0..3 {
        if fixed_index == coordinate_index {
            continue;
        }
        let reference = coordinate(input, 0, fixed_index);
        if (1..input.position_count)
            .any(|antenna| !all_close(coordinate(input, antenna, fixed_index), reference))
        {
            return Err(AngleCalibrationError::NonLinearArray { axis });
        }
    }

    let reference_spacing =
        coordinate(input, 1, coordinate_index) - coordinate(input, 0, coordinate_index);
    if reference_spacing <= 0.0
        || (2..input.position_count).any(|antenna| {
            let spacing = coordinate(input, antenna, coordinate_index)
                - coordinate(input, antenna - 1, coordinate_index);
            spacing <= 0.0 || !all_close(spacing, reference_spacing)
        })
    {
        return Err(AngleCalibrationError::NonUniformArray { axis });
    }
    Ok(reference_spacing)
}

fn coordinate(input: AngleBinCalibrationInput<'_>, antenna: usize, axis: usize) -> f64 {
    f64::from(input.positions_wavelengths[antenna * 3 + axis])
}

fn all_close(value: f64, reference: f64) -> bool {
    (value - reference).abs() <= 1e-8 + 1e-5 * reference.abs()
}

fn fft_frequency(bin: usize, bin_count: usize, fftshift: bool) -> f64 {
    let bin_count_f64 = bin_count as f64;
    if fftshift {
        return (bin as f64 - (bin_count / 2) as f64) / bin_count_f64;
    }
    if bin < bin_count.div_ceil(2) {
        return bin as f64 / bin_count_f64;
    }
    -((bin_count - bin) as f64) / bin_count_f64
}

#[cfg(test)]
mod tests {
    use super::{
        AngleAxis, AngleBinCalibrationConfig, AngleBinCalibrationInput, AngleCalibrationError,
        calibrate_angle_bins,
    };

    fn input(positions: &[f32]) -> AngleBinCalibrationInput<'_> {
        AngleBinCalibrationInput {
            positions_wavelengths: positions,
            position_count: positions.len() / 3,
        }
    }

    #[test]
    fn calibrates_shifted_and_unshifted_ula_bins() {
        let positions = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let shifted = calibrate_angle_bins(
            input(&positions),
            AngleBinCalibrationConfig {
                num_bins: 4,
                axis: AngleAxis::Azimuth,
                fftshift: true,
            },
        )
        .unwrap();
        let unshifted = calibrate_angle_bins(
            input(&positions),
            AngleBinCalibrationConfig {
                num_bins: 4,
                axis: AngleAxis::Azimuth,
                fftshift: false,
            },
        )
        .unwrap();

        let expected = [
            -core::f32::consts::FRAC_PI_2,
            -core::f32::consts::FRAC_PI_6,
            0.0,
            core::f32::consts::FRAC_PI_6,
        ];
        for (actual, expected) in shifted.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!((unshifted[0] - 0.0).abs() < 1e-6);
        assert!((unshifted[1] - core::f32::consts::FRAC_PI_6).abs() < 1e-6);
        assert!((unshifted[2] + core::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((unshifted[3] + core::f32::consts::FRAC_PI_6).abs() < 1e-6);
    }

    #[test]
    fn validates_ula_geometry_and_visible_bins() {
        let non_uniform = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.2, 0.0, 0.0];
        let non_linear = [0.0, 0.0, 0.0, 0.5, 0.1, 0.0];
        let compact = [0.0, 0.0, 0.0, 0.25, 0.0, 0.0];
        let config = AngleBinCalibrationConfig {
            num_bins: 4,
            axis: AngleAxis::Azimuth,
            fftshift: true,
        };

        assert!(matches!(
            calibrate_angle_bins(input(&non_uniform), config),
            Err(AngleCalibrationError::NonUniformArray { .. })
        ));
        assert!(matches!(
            calibrate_angle_bins(input(&non_linear), config),
            Err(AngleCalibrationError::NonLinearArray { .. })
        ));
        assert!(matches!(
            calibrate_angle_bins(input(&compact), config),
            Err(AngleCalibrationError::InvisibleBins)
        ));
    }
}
