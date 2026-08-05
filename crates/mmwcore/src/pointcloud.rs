//! Calibrated detection-to-point-cloud projection.

use std::fmt;

/// Detection-matrix columns consumed by calibrated point-cloud projection.
#[derive(Clone, Copy, Debug)]
pub struct DetectionPointCloudColumns<'a> {
    pub range_bin: usize,
    pub doppler_bin: usize,
    pub magnitude: usize,
    pub azimuth_bin: usize,
    pub azimuth_rad: usize,
    pub elevation: Option<[usize; 2]>,
    pub passthrough: &'a [usize],
}

/// Numeric input for calibrated detection-to-point-cloud projection.
#[derive(Clone, Copy, Debug)]
pub struct DetectionPointCloudInput<'a> {
    pub detections: &'a [f32],
    pub shape: [usize; 2],
    pub columns: DetectionPointCloudColumns<'a>,
}

/// Physical sampling contract for calibrated point-cloud projection.
#[derive(Clone, Copy, Debug)]
pub struct DetectionPointCloudConfig {
    pub range_resolution_m: f32,
    pub doppler_resolution_mps: f32,
    pub center_doppler: bool,
    pub doppler_bins: Option<usize>,
    pub doppler_fftshifted: bool,
}

impl DetectionPointCloudConfig {
    pub fn validate(self) -> Result<(), DetectionPointCloudError> {
        if !self.range_resolution_m.is_finite() || self.range_resolution_m <= 0.0 {
            return Err(DetectionPointCloudError::InvalidRangeResolution);
        }
        if !self.doppler_resolution_mps.is_finite() || self.doppler_resolution_mps <= 0.0 {
            return Err(DetectionPointCloudError::InvalidDopplerResolution);
        }
        if self.doppler_bins == Some(0) {
            return Err(DetectionPointCloudError::InvalidDopplerBins);
        }
        if self.center_doppler && self.doppler_bins.is_none() {
            return Err(DetectionPointCloudError::CenteredDopplerRequiresBins);
        }
        Ok(())
    }
}

/// Native point matrix in the public detection-point-cloud channel order.
#[derive(Debug)]
pub struct DetectionPointCloudProjection {
    pub points: Vec<f32>,
    pub point_count: usize,
    pub point_channels: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DetectionPointCloudError {
    ShapeOverflow,
    DetectionSizeMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteDetection,
    ColumnOutOfBounds {
        name: &'static str,
        index: usize,
        width: usize,
    },
    PassthroughColumnOutOfBounds {
        index: usize,
        width: usize,
    },
    InvalidRangeResolution,
    InvalidDopplerResolution,
    InvalidDopplerBins,
    CenteredDopplerRequiresBins,
    NonPhysicalDirection {
        row: usize,
        forward_squared: f32,
    },
}

impl fmt::Display for DetectionPointCloudError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => {
                write!(formatter, "Detection point-cloud shape overflows usize.")
            }
            Self::DetectionSizeMismatch { expected, actual } => write!(
                formatter,
                "Detection point-cloud buffer size {actual} does not match expected shape size {expected}."
            ),
            Self::NonFiniteDetection => {
                write!(
                    formatter,
                    "Detection point-cloud input contains NaN or Inf values."
                )
            }
            Self::ColumnOutOfBounds { name, index, width } => write!(
                formatter,
                "Detection point-cloud column {name}={index} is outside input width {width}."
            ),
            Self::PassthroughColumnOutOfBounds { index, width } => write!(
                formatter,
                "Detection point-cloud passthrough column {index} is outside input width {width}."
            ),
            Self::InvalidRangeResolution => write!(
                formatter,
                "Detection point-cloud range_resolution_m must be finite and positive."
            ),
            Self::InvalidDopplerResolution => write!(
                formatter,
                "Detection point-cloud doppler_resolution_mps must be finite and positive."
            ),
            Self::InvalidDopplerBins => write!(
                formatter,
                "Detection point-cloud doppler_bins must be positive when provided."
            ),
            Self::CenteredDopplerRequiresBins => write!(
                formatter,
                "Detection point-cloud doppler_bins is required when center_doppler=True."
            ),
            Self::NonPhysicalDirection {
                row,
                forward_squared,
            } => write!(
                formatter,
                "Detection direction cosines do not define a physical point at row {row} (forward_squared={forward_squared})."
            ),
        }
    }
}

impl std::error::Error for DetectionPointCloudError {}

/// Project calibrated range-Doppler-angle detections into radar Cartesian space.
pub fn project_detection_point_cloud(
    input: DetectionPointCloudInput<'_>,
    config: DetectionPointCloudConfig,
) -> Result<DetectionPointCloudProjection, DetectionPointCloudError> {
    config.validate()?;
    validate_input(input)?;
    let [point_count, input_channels] = input.shape;
    let point_channels = output_channels(input.columns);
    let output_values = point_count
        .checked_mul(point_channels)
        .ok_or(DetectionPointCloudError::ShapeOverflow)?;
    let mut points = Vec::with_capacity(output_values);

    for row_index in 0..point_count {
        let row_start = row_index * input_channels;
        let row = &input.detections[row_start..row_start + input_channels];
        let range_bin = row[input.columns.range_bin];
        let doppler_bin = row[input.columns.doppler_bin];
        let azimuth_rad = row[input.columns.azimuth_rad];
        // Calibrated AoA values are arcsines of axis direction cosines, not yaw/pitch.
        let lateral_direction = azimuth_rad.sin();
        let vertical_direction = input
            .columns
            .elevation
            .map_or(0.0, |columns| row[columns[0]].sin());
        let forward_squared = 1.0 - lateral_direction.powi(2) - vertical_direction.powi(2);
        if forward_squared < -1e-5 {
            return Err(DetectionPointCloudError::NonPhysicalDirection {
                row: row_index,
                forward_squared,
            });
        }
        let range_m = range_bin * config.range_resolution_m;
        points.extend_from_slice(&[
            range_m * lateral_direction,
            range_m * forward_squared.max(0.0).sqrt(),
            range_m * vertical_direction,
            doppler_velocity(doppler_bin, config),
            row[input.columns.magnitude],
            range_bin,
            doppler_bin,
            row[input.columns.azimuth_bin],
            azimuth_rad,
        ]);
        if let Some([elevation_rad, elevation_magnitude]) = input.columns.elevation {
            points.extend_from_slice(&[row[elevation_rad], row[elevation_magnitude]]);
        }
        for &column in input.columns.passthrough {
            points.push(row[column]);
        }
    }

    Ok(DetectionPointCloudProjection {
        points,
        point_count,
        point_channels,
    })
}

fn validate_input(input: DetectionPointCloudInput<'_>) -> Result<(), DetectionPointCloudError> {
    let expected_size = checked_product(&input.shape)?;
    if input.detections.len() != expected_size {
        return Err(DetectionPointCloudError::DetectionSizeMismatch {
            expected: expected_size,
            actual: input.detections.len(),
        });
    }
    if input.detections.iter().any(|value| !value.is_finite()) {
        return Err(DetectionPointCloudError::NonFiniteDetection);
    }
    let input_width = input.shape[1];
    for (name, index) in [
        ("range_bin", input.columns.range_bin),
        ("doppler_bin", input.columns.doppler_bin),
        ("magnitude", input.columns.magnitude),
        ("azimuth_bin", input.columns.azimuth_bin),
        ("azimuth_rad", input.columns.azimuth_rad),
    ] {
        validate_column(name, index, input_width)?;
    }
    if let Some([elevation_rad, elevation_magnitude]) = input.columns.elevation {
        validate_column("elevation_rad", elevation_rad, input_width)?;
        validate_column("elevation_magnitude", elevation_magnitude, input_width)?;
    }
    for &column in input.columns.passthrough {
        if column >= input_width {
            return Err(DetectionPointCloudError::PassthroughColumnOutOfBounds {
                index: column,
                width: input_width,
            });
        }
    }
    Ok(())
}

fn checked_product(shape: &[usize]) -> Result<usize, DetectionPointCloudError> {
    shape.iter().try_fold(1_usize, |product, &size| {
        product
            .checked_mul(size)
            .ok_or(DetectionPointCloudError::ShapeOverflow)
    })
}

fn validate_column(
    name: &'static str,
    index: usize,
    input_width: usize,
) -> Result<(), DetectionPointCloudError> {
    if index >= input_width {
        return Err(DetectionPointCloudError::ColumnOutOfBounds {
            name,
            index,
            width: input_width,
        });
    }
    Ok(())
}

fn output_channels(columns: DetectionPointCloudColumns<'_>) -> usize {
    9 + usize::from(columns.elevation.is_some()) * 2 + columns.passthrough.len()
}

fn doppler_velocity(doppler_bin: f32, config: DetectionPointCloudConfig) -> f32 {
    let centered_bin = if !config.center_doppler {
        doppler_bin
    } else {
        let doppler_bins = config
            .doppler_bins
            .expect("validated centered Doppler configuration");
        let doppler_bins = doppler_bins as f32;
        if config.doppler_fftshifted {
            doppler_bin - (doppler_bins as usize / 2) as f32
        } else {
            (doppler_bin + doppler_bins / 2.0).rem_euclid(doppler_bins) - doppler_bins / 2.0
        }
    };
    centered_bin * config.doppler_resolution_mps
}

#[cfg(test)]
mod tests {
    use super::{
        DetectionPointCloudColumns, DetectionPointCloudConfig, DetectionPointCloudError,
        DetectionPointCloudInput, project_detection_point_cloud,
    };

    fn config() -> DetectionPointCloudConfig {
        DetectionPointCloudConfig {
            range_resolution_m: 0.5,
            doppler_resolution_mps: 0.25,
            center_doppler: false,
            doppler_bins: None,
            doppler_fftshifted: false,
        }
    }

    #[test]
    fn projects_calibrated_detection_rows_and_passthrough_values() {
        let input = [0.0, 4.0, 2.0, 3.0, 0.5, 10.0, 0.25, 12.0, 6.0];
        let result = project_detection_point_cloud(
            DetectionPointCloudInput {
                detections: &input,
                shape: [1, 9],
                columns: DetectionPointCloudColumns {
                    range_bin: 1,
                    doppler_bin: 2,
                    magnitude: 5,
                    azimuth_bin: 3,
                    azimuth_rad: 4,
                    elevation: Some([6, 7]),
                    passthrough: &[8],
                },
            },
            config(),
        )
        .unwrap();

        assert_eq!(result.point_count, 1);
        assert_eq!(result.point_channels, 12);
        assert!((result.points[0] - 2.0 * 0.5_f32.sin()).abs() < 1e-6);
        assert!((result.points[2] - 2.0 * 0.25_f32.sin()).abs() < 1e-6);
        assert_eq!(result.points[3], 0.5);
        assert_eq!(&result.points[9..], &[0.25, 12.0, 6.0]);
    }

    #[test]
    fn centers_unshifted_doppler_bins() {
        let input = [0.0, 1.0, 7.0, 0.0, 0.0, 3.0];
        let result = project_detection_point_cloud(
            DetectionPointCloudInput {
                detections: &input,
                shape: [1, 6],
                columns: DetectionPointCloudColumns {
                    range_bin: 1,
                    doppler_bin: 2,
                    magnitude: 5,
                    azimuth_bin: 3,
                    azimuth_rad: 4,
                    elevation: None,
                    passthrough: &[],
                },
            },
            DetectionPointCloudConfig {
                center_doppler: true,
                doppler_bins: Some(8),
                ..config()
            },
        )
        .unwrap();

        assert_eq!(result.points[3], -0.25);
    }

    #[test]
    fn rejects_nonphysical_direction_cosines() {
        let input = [0.0, 1.0, 0.0, 0.0, core::f32::consts::FRAC_PI_2, 1.0];
        let error = project_detection_point_cloud(
            DetectionPointCloudInput {
                detections: &input,
                shape: [1, 6],
                columns: DetectionPointCloudColumns {
                    range_bin: 1,
                    doppler_bin: 2,
                    magnitude: 5,
                    azimuth_bin: 3,
                    azimuth_rad: 4,
                    elevation: Some([4, 5]),
                    passthrough: &[],
                },
            },
            config(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            DetectionPointCloudError::NonPhysicalDirection { .. }
        ));
    }
}
