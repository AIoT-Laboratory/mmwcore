//! Threshold detection over contiguous range-Doppler radar cubes.

use std::fmt;

use num_complex::Complex32;

use crate::cube::{
    CubeTransformError, checked_product, contiguous_strides, validate_axis, validate_shape,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ReceiverAggregation {
    Max = 0,
    Sum = 1,
    Mean = 2,
}

impl TryFrom<u8> for ReceiverAggregation {
    type Error = DetectionError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Max),
            1 => Ok(Self::Sum),
            2 => Ok(Self::Mean),
            _ => Err(DetectionError::UnsupportedReceiverAggregation { aggregation: value }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeDopplerAxes {
    pub frame: usize,
    pub doppler: usize,
    pub receiver: usize,
    pub range: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeDopplerAzimuthAxes {
    pub frame: usize,
    pub doppler: usize,
    pub azimuth: usize,
    pub range: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdDetections {
    pub indices: Vec<usize>,
    pub magnitudes: Vec<f32>,
    pub rank: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DetectionError {
    Cube(CubeTransformError),
    UnexpectedRank { expected: usize, actual: usize },
    DuplicateAxes { first: usize, second: usize },
    UnsupportedReceiverAggregation { aggregation: u8 },
}

impl fmt::Display for DetectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cube(error) => error.fmt(formatter),
            Self::UnexpectedRank { expected, actual } => write!(
                formatter,
                "Range-Doppler detection requires rank {expected}; got rank {actual}."
            ),
            Self::DuplicateAxes { first, second } => write!(
                formatter,
                "Range-Doppler detection axes {first} and {second} must be distinct."
            ),
            Self::UnsupportedReceiverAggregation { aggregation } => write!(
                formatter,
                "Unsupported native receiver aggregation code {aggregation}."
            ),
        }
    }
}

impl std::error::Error for DetectionError {}

impl From<CubeTransformError> for DetectionError {
    fn from(error: CubeTransformError) -> Self {
        Self::Cube(error)
    }
}

pub fn range_doppler_magnitude_complex(
    data: &[Complex32],
    shape: &[usize],
    axes: RangeDopplerAxes,
    aggregation: ReceiverAggregation,
) -> Result<(Vec<f32>, [usize; 3]), DetectionError> {
    let layout = CanonicalFourAxisLayout::new(
        data,
        shape,
        [axes.frame, axes.doppler, axes.receiver, axes.range],
    )?;
    let output_shape = [layout.lengths[0], layout.lengths[1], layout.lengths[3]];
    let mut magnitude = Vec::with_capacity(checked_product(&output_shape)?);

    for frame in 0..layout.lengths[0] {
        for doppler in 0..layout.lengths[1] {
            for range in 0..layout.lengths[3] {
                magnitude.push(aggregate_receiver_magnitude(
                    data,
                    layout,
                    frame,
                    doppler,
                    range,
                    aggregation,
                ));
            }
        }
    }

    Ok((magnitude, output_shape))
}

pub fn threshold_range_doppler_complex(
    data: &[Complex32],
    shape: &[usize],
    axes: RangeDopplerAxes,
    aggregation: ReceiverAggregation,
    threshold: f32,
) -> Result<ThresholdDetections, DetectionError> {
    let (magnitude, [frames, doppler_bins, range_bins]) =
        range_doppler_magnitude_complex(data, shape, axes, aggregation)?;
    let mut indices = Vec::new();
    let mut magnitudes = Vec::new();

    for frame in 0..frames {
        for doppler in 0..doppler_bins {
            for range in 0..range_bins {
                let value =
                    magnitude[range_doppler_index(frame, doppler, range, doppler_bins, range_bins)];
                if value >= threshold {
                    indices.extend([frame, doppler, range]);
                    magnitudes.push(value);
                }
            }
        }
    }

    Ok(ThresholdDetections {
        indices,
        magnitudes,
        rank: 3,
    })
}

pub fn threshold_range_doppler_azimuth_complex(
    data: &[Complex32],
    shape: &[usize],
    axes: RangeDopplerAzimuthAxes,
    threshold: f32,
    azimuth_peak_radius: usize,
    azimuth_peak_strict: bool,
) -> Result<ThresholdDetections, DetectionError> {
    let layout = CanonicalFourAxisLayout::new(
        data,
        shape,
        [axes.frame, axes.doppler, axes.azimuth, axes.range],
    )?;
    let magnitude = canonical_magnitudes(data, layout)?;
    let [frames, doppler_bins, azimuth_bins, range_bins] = layout.lengths;
    let mut indices = Vec::new();
    let mut magnitudes = Vec::new();

    for frame in 0..frames {
        for doppler in 0..doppler_bins {
            for azimuth in 0..azimuth_bins {
                for range in 0..range_bins {
                    let value = magnitude[range_doppler_azimuth_index(
                        frame,
                        doppler,
                        azimuth,
                        range,
                        doppler_bins,
                        azimuth_bins,
                        range_bins,
                    )];
                    if value >= threshold
                        && is_azimuth_local_maximum(
                            &magnitude,
                            [frame, doppler, azimuth, range],
                            [doppler_bins, azimuth_bins, range_bins],
                            azimuth_peak_radius,
                            azimuth_peak_strict,
                        )
                    {
                        indices.extend([frame, doppler, azimuth, range]);
                        magnitudes.push(value);
                    }
                }
            }
        }
    }

    Ok(ThresholdDetections {
        indices,
        magnitudes,
        rank: 4,
    })
}

#[derive(Clone, Copy)]
struct CanonicalFourAxisLayout {
    lengths: [usize; 4],
    strides: [usize; 4],
}

impl CanonicalFourAxisLayout {
    fn new(data: &[Complex32], shape: &[usize], axes: [usize; 4]) -> Result<Self, DetectionError> {
        validate_shape(data, shape)?;
        if shape.len() != 4 {
            return Err(DetectionError::UnexpectedRank {
                expected: 4,
                actual: shape.len(),
            });
        }
        for &axis in &axes {
            validate_axis(shape, axis)?;
        }
        for (index, &first) in axes.iter().enumerate() {
            if let Some(&second) = axes[index + 1..].iter().find(|&&second| second == first) {
                return Err(DetectionError::DuplicateAxes { first, second });
            }
        }

        let input_strides = contiguous_strides(shape)?;
        Ok(Self {
            lengths: [
                shape[axes[0]],
                shape[axes[1]],
                shape[axes[2]],
                shape[axes[3]],
            ],
            strides: [
                input_strides[axes[0]],
                input_strides[axes[1]],
                input_strides[axes[2]],
                input_strides[axes[3]],
            ],
        })
    }

    fn input_index(self, frame: usize, doppler: usize, third: usize, range: usize) -> usize {
        frame * self.strides[0]
            + doppler * self.strides[1]
            + third * self.strides[2]
            + range * self.strides[3]
    }
}

fn aggregate_receiver_magnitude(
    data: &[Complex32],
    layout: CanonicalFourAxisLayout,
    frame: usize,
    doppler: usize,
    range: usize,
    aggregation: ReceiverAggregation,
) -> f32 {
    match aggregation {
        ReceiverAggregation::Max => {
            let mut maximum = 0.0_f32;
            for receiver in 0..layout.lengths[2] {
                let value =
                    complex_magnitude(data[layout.input_index(frame, doppler, receiver, range)]);
                if value.is_nan() {
                    return f32::NAN;
                }
                maximum = maximum.max(value);
            }
            maximum
        }
        ReceiverAggregation::Sum => (0..layout.lengths[2])
            .map(|receiver| {
                complex_magnitude(data[layout.input_index(frame, doppler, receiver, range)])
            })
            .sum(),
        ReceiverAggregation::Mean => {
            let sum: f32 = (0..layout.lengths[2])
                .map(|receiver| {
                    complex_magnitude(data[layout.input_index(frame, doppler, receiver, range)])
                })
                .sum();
            sum / layout.lengths[2] as f32
        }
    }
}

fn canonical_magnitudes(
    data: &[Complex32],
    layout: CanonicalFourAxisLayout,
) -> Result<Vec<f32>, DetectionError> {
    let mut magnitude = Vec::with_capacity(checked_product(&layout.lengths)?);
    for frame in 0..layout.lengths[0] {
        for doppler in 0..layout.lengths[1] {
            for third in 0..layout.lengths[2] {
                for range in 0..layout.lengths[3] {
                    magnitude.push(complex_magnitude(
                        data[layout.input_index(frame, doppler, third, range)],
                    ));
                }
            }
        }
    }
    Ok(magnitude)
}

fn complex_magnitude(value: Complex32) -> f32 {
    value.re.hypot(value.im)
}

fn range_doppler_index(
    frame: usize,
    doppler: usize,
    range: usize,
    doppler_bins: usize,
    range_bins: usize,
) -> usize {
    (frame * doppler_bins + doppler) * range_bins + range
}

fn range_doppler_azimuth_index(
    frame: usize,
    doppler: usize,
    azimuth: usize,
    range: usize,
    doppler_bins: usize,
    azimuth_bins: usize,
    range_bins: usize,
) -> usize {
    ((frame * doppler_bins + doppler) * azimuth_bins + azimuth) * range_bins + range
}

fn is_azimuth_local_maximum(
    magnitude: &[f32],
    candidate: [usize; 4],
    dimensions: [usize; 3],
    radius: usize,
    strict: bool,
) -> bool {
    let [frame, doppler, azimuth, range] = candidate;
    let [doppler_bins, azimuth_bins, range_bins] = dimensions;
    let center = magnitude[range_doppler_azimuth_index(
        frame,
        doppler,
        azimuth,
        range,
        doppler_bins,
        azimuth_bins,
        range_bins,
    )];
    for offset in 1..=radius {
        if offset >= azimuth_bins {
            break;
        }
        if azimuth >= offset {
            let left = magnitude[range_doppler_azimuth_index(
                frame,
                doppler,
                azimuth - offset,
                range,
                doppler_bins,
                azimuth_bins,
                range_bins,
            )];
            if !dominates(center, left, strict) {
                return false;
            }
        }
        if azimuth + offset < azimuth_bins {
            let right = magnitude[range_doppler_azimuth_index(
                frame,
                doppler,
                azimuth + offset,
                range,
                doppler_bins,
                azimuth_bins,
                range_bins,
            )];
            if !dominates(center, right, strict) {
                return false;
            }
        }
    }
    true
}

fn dominates(center: f32, neighbor: f32, strict: bool) -> bool {
    if strict {
        center > neighbor
    } else {
        center >= neighbor
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::{
        RangeDopplerAxes, RangeDopplerAzimuthAxes, ReceiverAggregation,
        range_doppler_magnitude_complex, threshold_range_doppler_azimuth_complex,
        threshold_range_doppler_complex,
    };

    #[test]
    fn aggregates_receivers_in_canonical_range_doppler_order() {
        let data = [
            Complex32::new(3.0, 4.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(6.0, 8.0),
        ];
        let axes = RangeDopplerAxes {
            frame: 0,
            doppler: 1,
            receiver: 2,
            range: 3,
        };

        let (maximum, shape) =
            range_doppler_magnitude_complex(&data, &[1, 1, 2, 2], axes, ReceiverAggregation::Max)
                .unwrap();
        let (mean, _) =
            range_doppler_magnitude_complex(&data, &[1, 1, 2, 2], axes, ReceiverAggregation::Mean)
                .unwrap();

        assert_eq!(shape, [1, 1, 2]);
        assert_eq!(maximum, [5.0, 10.0]);
        assert_eq!(mean, [3.0, 6.0]);
    }

    #[test]
    fn threshold_detection_uses_canonical_c_order() {
        let data = [
            Complex32::new(1.0, 0.0),
            Complex32::new(7.0, 0.0),
            Complex32::new(8.0, 0.0),
            Complex32::new(2.0, 0.0),
        ];
        let detections = threshold_range_doppler_complex(
            &data,
            &[1, 2, 1, 2],
            RangeDopplerAxes {
                frame: 0,
                doppler: 1,
                receiver: 2,
                range: 3,
            },
            ReceiverAggregation::Max,
            5.0,
        )
        .unwrap();

        assert_eq!(detections.indices, [0, 0, 1, 0, 1, 0]);
        assert_eq!(detections.magnitudes, [7.0, 8.0]);
        assert_eq!(detections.rank, 3);
    }

    #[test]
    fn azimuth_threshold_respects_strict_plateau_policy() {
        let data = [
            Complex32::new(2.0, 0.0),
            Complex32::new(6.0, 0.0),
            Complex32::new(6.0, 0.0),
        ];
        let axes = RangeDopplerAzimuthAxes {
            frame: 0,
            doppler: 1,
            azimuth: 2,
            range: 3,
        };

        let strict =
            threshold_range_doppler_azimuth_complex(&data, &[1, 1, 3, 1], axes, 5.0, 1, true)
                .unwrap();
        let non_strict =
            threshold_range_doppler_azimuth_complex(&data, &[1, 1, 3, 1], axes, 5.0, 1, false)
                .unwrap();

        assert!(strict.indices.is_empty());
        assert_eq!(non_strict.indices, [0, 0, 1, 0, 0, 0, 2, 0]);
        assert_eq!(non_strict.magnitudes, [6.0, 6.0]);
        assert_eq!(non_strict.rank, 4);
    }
}
