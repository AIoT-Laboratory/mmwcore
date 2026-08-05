//! Native slow-time phase primitives for vital-sign sensing.

use std::fmt;

use num_complex::Complex32;

/// Unwrap one complex slow-time phase sequence.
pub fn unwrap_vital_phase_complex(
    samples: &[Complex32],
    remove_mean: bool,
) -> Result<Vec<f32>, VitalSignError> {
    if samples.is_empty() {
        return Err(VitalSignError::EmptySamples);
    }

    let mut phase = Vec::with_capacity(samples.len());
    let mut previous_angle = 0.0_f32;
    let mut phase_offset = 0.0_f32;
    for (index, sample) in samples.iter().enumerate() {
        if !sample.re.is_finite() || !sample.im.is_finite() {
            return Err(VitalSignError::NonFiniteComplexSample { index });
        }
        let angle = sample.im.atan2(sample.re);
        if index != 0 {
            let difference = angle - previous_angle;
            if difference > std::f32::consts::PI {
                phase_offset -= std::f32::consts::TAU;
            } else if difference < -std::f32::consts::PI {
                phase_offset += std::f32::consts::TAU;
            }
        }
        phase.push(angle + phase_offset);
        previous_angle = angle;
    }

    if remove_mean {
        let mean = phase.iter().map(|&value| f64::from(value)).sum::<f64>() / phase.len() as f64;
        for value in &mut phase {
            *value = (f64::from(*value) - mean) as f32;
        }
    }
    Ok(phase)
}

/// Convert monostatic round-trip phase to relative displacement.
pub fn vital_phase_to_displacement(
    phase_rad: &[f32],
    wavelength_m: f32,
) -> Result<Vec<f32>, VitalSignError> {
    if !wavelength_m.is_finite() || wavelength_m <= 0.0 {
        return Err(VitalSignError::InvalidWavelength);
    }
    let scale = wavelength_m / (4.0 * std::f32::consts::PI);
    let mut displacement = Vec::with_capacity(phase_rad.len());
    for (index, &phase) in phase_rad.iter().enumerate() {
        if !phase.is_finite() {
            return Err(VitalSignError::NonFinitePhase { index });
        }
        displacement.push(phase * scale);
    }
    Ok(displacement)
}

/// Vital-sign numerical validation errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VitalSignError {
    EmptySamples,
    NonFiniteComplexSample { index: usize },
    NonFinitePhase { index: usize },
    InvalidWavelength,
}

impl fmt::Display for VitalSignError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySamples => {
                write!(formatter, "Vital-sign phase requires at least one sample.")
            }
            Self::NonFiniteComplexSample { index } => {
                write!(
                    formatter,
                    "Vital-sign complex sample {index} contains NaN or Inf."
                )
            }
            Self::NonFinitePhase { index } => {
                write!(
                    formatter,
                    "Vital-sign phase sample {index} contains NaN or Inf."
                )
            }
            Self::InvalidWavelength => {
                write!(
                    formatter,
                    "Vital-sign wavelength must be finite and positive."
                )
            }
        }
    }
}

impl std::error::Error for VitalSignError {}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::{unwrap_vital_phase_complex, vital_phase_to_displacement};

    #[test]
    fn unwraps_the_phase_boundary_without_mean_removal() {
        let samples = [
            Complex32::from_polar(1.0, 2.8),
            Complex32::from_polar(1.0, -2.9),
        ];

        let phase = unwrap_vital_phase_complex(&samples, false).unwrap();

        assert!((phase[0] - 2.8).abs() < 1.0e-6);
        assert!((phase[1] - (std::f32::consts::TAU - 2.9)).abs() < 1.0e-6);
    }

    #[test]
    fn uses_monostatic_round_trip_displacement() {
        let displacement = vital_phase_to_displacement(&[-std::f32::consts::PI], 0.004).unwrap();

        assert!((displacement[0] + 0.001).abs() < 1.0e-7);
    }
}
