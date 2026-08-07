//! Candidate-level range-Doppler grouping and quality filtering.

use std::fmt;

use num_complex::Complex32;

use crate::detection::{
    DetectionError, RangeDopplerAxes, ReceiverAggregation, range_doppler_magnitude_complex,
};
use crate::exact_candidate_index;

/// Contiguous public `float32` detection-candidate matrix.
#[derive(Clone, Copy, Debug)]
pub struct DetectionCandidateInput<'a> {
    pub values: &'a [f32],
    pub shape: [usize; 2],
}

/// Frame, range, and Doppler columns in a detection-candidate matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectionIndexColumns {
    pub frame: usize,
    pub range: usize,
    pub doppler: usize,
}

/// Complete range-Doppler grouping input.
#[derive(Clone, Copy, Debug)]
pub struct PeakGroupingInput<'a> {
    pub data: &'a [Complex32],
    pub shape: &'a [usize],
    pub axes: RangeDopplerAxes,
    pub aggregation: ReceiverAggregation,
    pub candidates: DetectionCandidateInput<'a>,
    pub columns: DetectionIndexColumns,
}

/// Local range-Doppler neighborhood policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeakGroupingConfig {
    pub range_radius: usize,
    pub doppler_radius: usize,
    pub cyclic_doppler: bool,
    pub strict: bool,
}

/// Complete SNR quality-filter input.
#[derive(Clone, Copy, Debug)]
pub struct DetectionQualityInput<'a> {
    pub candidates: DetectionCandidateInput<'a>,
    pub snr_column: usize,
    pub min_snr: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DetectionPostprocessError {
    Detection(DetectionError),
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
    CandidateIndexOutOfBounds {
        axis: &'static str,
        value: f32,
        shape: [usize; 3],
    },
    EmptyGroupingNeighborhood,
    InvalidMinimumSnr {
        min_snr: f32,
    },
}

impl fmt::Display for DetectionPostprocessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Detection(error) => error.fmt(formatter),
            Self::CandidateShapeOverflow => {
                write!(
                    formatter,
                    "Detection candidate matrix shape overflows usize."
                )
            }
            Self::CandidateShapeMismatch { expected, actual } => write!(
                formatter,
                "Detection candidate matrix has {actual} values; expected {expected}."
            ),
            Self::NonFiniteCandidates => {
                write!(
                    formatter,
                    "Detection candidate matrix contains NaN or Inf values."
                )
            }
            Self::CandidateColumnOutOfBounds { name, index, width } => write!(
                formatter,
                "Detection candidate column {name}={index} is outside width {width}."
            ),
            Self::CandidateIndexOutOfBounds { axis, value, shape } => write!(
                formatter,
                "Detection {axis} index {value} is outside shape ({}, {}, {}).",
                shape[0], shape[1], shape[2]
            ),
            Self::EmptyGroupingNeighborhood => {
                write!(
                    formatter,
                    "Peak grouping requires at least one non-zero radius."
                )
            }
            Self::InvalidMinimumSnr { min_snr } => {
                write!(
                    formatter,
                    "Minimum SNR must be finite and positive; got {min_snr}."
                )
            }
        }
    }
}

impl std::error::Error for DetectionPostprocessError {}

impl From<DetectionError> for DetectionPostprocessError {
    fn from(error: DetectionError) -> Self {
        Self::Detection(error)
    }
}

/// Retain candidates that are local range-Doppler magnitude maxima.
pub fn group_range_doppler_candidates(
    input: PeakGroupingInput<'_>,
    config: PeakGroupingConfig,
) -> Result<Vec<usize>, DetectionPostprocessError> {
    validate_candidates(input.candidates)?;
    validate_index_columns(input.candidates.shape[1], input.columns)?;
    if config.range_radius == 0 && config.doppler_radius == 0 {
        return Err(DetectionPostprocessError::EmptyGroupingNeighborhood);
    }
    let (magnitude, shape) =
        range_doppler_magnitude_complex(input.data, input.shape, input.axes, input.aggregation)?;
    let mut retained = Vec::with_capacity(input.candidates.shape[0]);
    for candidate_index in 0..input.candidates.shape[0] {
        let candidate = candidate_row(input.candidates, candidate_index);
        let frame = candidate_axis_index(candidate, input.columns.frame, "frame", shape, shape[0])?;
        let range = candidate_axis_index(candidate, input.columns.range, "range", shape, shape[2])?;
        let doppler =
            candidate_axis_index(candidate, input.columns.doppler, "Doppler", shape, shape[1])?;
        if is_local_peak(&magnitude, shape, frame, doppler, range, config) {
            retained.push(candidate_index);
        }
    }
    Ok(retained)
}

/// Retain candidates whose linear SNR meets the configured threshold.
pub fn filter_detection_quality(
    input: DetectionQualityInput<'_>,
) -> Result<Vec<usize>, DetectionPostprocessError> {
    validate_candidates(input.candidates)?;
    validate_column(input.candidates.shape[1], "snr", input.snr_column)?;
    if !input.min_snr.is_finite() || input.min_snr <= 0.0 {
        return Err(DetectionPostprocessError::InvalidMinimumSnr {
            min_snr: input.min_snr,
        });
    }

    Ok((0..input.candidates.shape[0])
        .filter(|&candidate_index| {
            candidate_row(input.candidates, candidate_index)[input.snr_column] >= input.min_snr
        })
        .collect())
}

fn validate_candidates(
    input: DetectionCandidateInput<'_>,
) -> Result<(), DetectionPostprocessError> {
    let expected = input.shape.iter().try_fold(1_usize, |size, &dimension| {
        size.checked_mul(dimension)
            .ok_or(DetectionPostprocessError::CandidateShapeOverflow)
    })?;
    if input.values.len() != expected {
        return Err(DetectionPostprocessError::CandidateShapeMismatch {
            expected,
            actual: input.values.len(),
        });
    }
    if input.values.iter().any(|value| !value.is_finite()) {
        return Err(DetectionPostprocessError::NonFiniteCandidates);
    }
    Ok(())
}

fn validate_index_columns(
    width: usize,
    columns: DetectionIndexColumns,
) -> Result<(), DetectionPostprocessError> {
    for (name, index) in [
        ("frame", columns.frame),
        ("range_bin", columns.range),
        ("doppler_bin", columns.doppler),
    ] {
        validate_column(width, name, index)?;
    }
    Ok(())
}

fn validate_column(
    width: usize,
    name: &'static str,
    index: usize,
) -> Result<(), DetectionPostprocessError> {
    if index >= width {
        return Err(DetectionPostprocessError::CandidateColumnOutOfBounds { name, index, width });
    }
    Ok(())
}

fn candidate_axis_index(
    candidate: &[f32],
    column: usize,
    axis: &'static str,
    shape: [usize; 3],
    axis_length: usize,
) -> Result<usize, DetectionPostprocessError> {
    let value = candidate[column];
    let index = exact_candidate_index(value, axis_length)
        .ok_or(DetectionPostprocessError::CandidateIndexOutOfBounds { axis, value, shape })?;
    Ok(index)
}

fn candidate_row(input: DetectionCandidateInput<'_>, index: usize) -> &[f32] {
    let width = input.shape[1];
    &input.values[index * width..(index + 1) * width]
}

fn is_local_peak(
    magnitude: &[f32],
    shape: [usize; 3],
    frame: usize,
    doppler: usize,
    range: usize,
    config: PeakGroupingConfig,
) -> bool {
    let center = magnitude[range_doppler_index(frame, doppler, range, shape)];
    let mut maximum = None;
    for doppler_distance in 0..=config.doppler_radius {
        let directions = if doppler_distance == 0 { 1 } else { 2 };
        for direction in 0..directions {
            let Some(neighbor_doppler) = neighbor_index(
                doppler,
                doppler_distance,
                direction == 1,
                shape[1],
                config.cyclic_doppler,
            ) else {
                continue;
            };
            for range_distance in 0..=config.range_radius {
                let directions = if range_distance == 0 { 1 } else { 2 };
                for direction in 0..directions {
                    if doppler_distance == 0 && range_distance == 0 {
                        continue;
                    }
                    let Some(neighbor_range) =
                        neighbor_index(range, range_distance, direction == 1, shape[2], false)
                    else {
                        continue;
                    };
                    let value = magnitude
                        [range_doppler_index(frame, neighbor_doppler, neighbor_range, shape)];
                    if value.is_nan() {
                        return false;
                    }
                    maximum = Some(maximum.map_or(value, |current: f32| current.max(value)));
                }
            }
        }
    }
    match maximum {
        None => true,
        Some(value) if config.strict => center > value,
        Some(value) => center >= value,
    }
}

fn neighbor_index(
    index: usize,
    distance: usize,
    positive: bool,
    length: usize,
    cyclic: bool,
) -> Option<usize> {
    if distance == 0 {
        return Some(index);
    }
    if !cyclic {
        return if positive {
            index
                .checked_add(distance)
                .filter(|&candidate| candidate < length)
        } else {
            index.checked_sub(distance)
        };
    }

    let offset = distance % length;
    if offset == 0 {
        return Some(index);
    }
    if positive {
        Some(if index >= length - offset {
            index - (length - offset)
        } else {
            index + offset
        })
    } else {
        Some(if index >= offset {
            index - offset
        } else {
            length - (offset - index)
        })
    }
}

fn range_doppler_index(frame: usize, doppler: usize, range: usize, shape: [usize; 3]) -> usize {
    (frame * shape[1] + doppler) * shape[2] + range
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::{
        DetectionCandidateInput, DetectionIndexColumns, DetectionPostprocessError,
        DetectionQualityInput, PeakGroupingConfig, PeakGroupingInput, filter_detection_quality,
        group_range_doppler_candidates,
    };
    use crate::detection::{RangeDopplerAxes, ReceiverAggregation};

    #[test]
    fn groups_cyclic_doppler_peaks_in_candidate_order() {
        let mut data = vec![Complex32::new(1.0, 0.0); 15];
        data[1] = Complex32::new(8.0, 0.0);
        data[13] = Complex32::new(10.0, 0.0);
        let candidates = [0.0, 1.0, 0.0, 8.0, 0.0, 1.0, 4.0, 10.0];

        let retained = group_range_doppler_candidates(
            PeakGroupingInput {
                data: &data,
                shape: &[1, 5, 1, 3],
                axes: RangeDopplerAxes {
                    frame: 0,
                    doppler: 1,
                    receiver: 2,
                    range: 3,
                },
                aggregation: ReceiverAggregation::Sum,
                candidates: DetectionCandidateInput {
                    values: &candidates,
                    shape: [2, 4],
                },
                columns: DetectionIndexColumns {
                    frame: 0,
                    range: 1,
                    doppler: 2,
                },
            },
            PeakGroupingConfig {
                range_radius: 0,
                doppler_radius: 1,
                cyclic_doppler: true,
                strict: true,
            },
        )
        .unwrap();

        assert_eq!(retained, [1]);
    }

    #[test]
    fn filters_snr_at_the_inclusive_threshold() {
        let candidates = [5.0, 10.0, 15.0];
        let retained = filter_detection_quality(DetectionQualityInput {
            candidates: DetectionCandidateInput {
                values: &candidates,
                shape: [3, 1],
            },
            snr_column: 0,
            min_snr: 10.0,
        })
        .unwrap();

        assert_eq!(retained, [1, 2]);
    }

    #[test]
    fn rejects_inexact_or_non_finite_candidate_indices() {
        let data = [Complex32::new(1.0, 0.0); 2];
        for invalid in [1.9, -0.5, f32::NAN, f32::INFINITY] {
            let candidates = [0.0, invalid, 0.0];
            let result = group_range_doppler_candidates(
                PeakGroupingInput {
                    data: &data,
                    shape: &[1, 1, 1, 2],
                    axes: RangeDopplerAxes {
                        frame: 0,
                        doppler: 1,
                        receiver: 2,
                        range: 3,
                    },
                    aggregation: ReceiverAggregation::Sum,
                    candidates: DetectionCandidateInput {
                        values: &candidates,
                        shape: [1, 3],
                    },
                    columns: DetectionIndexColumns {
                        frame: 0,
                        range: 1,
                        doppler: 2,
                    },
                },
                PeakGroupingConfig {
                    range_radius: 1,
                    doppler_radius: 0,
                    cyclic_doppler: false,
                    strict: true,
                },
            );

            if invalid.is_finite() {
                assert!(matches!(
                    result,
                    Err(DetectionPostprocessError::CandidateIndexOutOfBounds { .. })
                ));
            } else {
                assert!(matches!(
                    result,
                    Err(DetectionPostprocessError::NonFiniteCandidates)
                ));
            }
        }
    }
}
