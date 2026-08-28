//! Native sequence-level tracking metrics.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use super::Box2D;

/// Packed observations from ordered tracker frames.
#[derive(Clone, Copy, Debug)]
pub struct TrackingMetricsInput<'a> {
    /// Cumulative observation counts, with one entry per frame plus the final count.
    pub frame_offsets: &'a [usize],
    pub track_ids: &'a [i64],
    /// Packed XYZ positions with shape `(observations, 3)`.
    pub positions: &'a [f32],
    /// Packed XYZ velocities with shape `(observations, 3)`.
    pub velocities: &'a [f32],
    /// Native lifecycle status codes aligned with observations.
    pub status_codes: &'a [u8],
    /// `None` means scenery metrics are disabled; `Some([])` means every point is in scenery.
    pub scenery_boxes: Option<&'a [Box2D]>,
    pub frame_index_offset: usize,
}

/// One track's observed motion and lifecycle summary.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackObservationMetrics {
    pub track_id: i64,
    pub observed_frames: usize,
    pub confirmed_frames: usize,
    pub first_frame_index: usize,
    pub last_frame_index: usize,
    pub first_position_m: [f32; 3],
    pub last_position_m: [f32; 3],
    pub median_position_m: [f32; 3],
    pub displacement_m: f32,
    pub path_length_m: f32,
    pub median_speed_mps: f32,
    pub max_speed_mps: f32,
    pub confirmed_intervals: Vec<[usize; 2]>,
    pub in_scenery_frames: Option<usize>,
    pub outside_scenery_frames: Option<usize>,
}

/// Aggregated tracker coverage over an ordered frame sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackingSequenceMetrics {
    pub num_frames: usize,
    pub frames_with_tracks: usize,
    pub frames_with_confirmed_tracks: usize,
    pub max_concurrent_tracks: usize,
    pub tracks: Vec<TrackObservationMetrics>,
}

/// Summarize ordered tracker reports in a single native call.
pub fn summarize_tracking_metrics(
    input: TrackingMetricsInput<'_>,
) -> Result<TrackingSequenceMetrics, TrackingMetricsError> {
    let num_frames = validate_input(input)?;
    let mut observed = BTreeMap::<i64, Vec<Observation>>::new();
    let mut frames_with_tracks = 0_usize;
    let mut frames_with_confirmed_tracks = 0_usize;
    let mut max_concurrent_tracks = 0_usize;

    for relative_frame_index in 0..num_frames {
        let start = input.frame_offsets[relative_frame_index];
        let stop = input.frame_offsets[relative_frame_index + 1];
        let track_count = stop - start;
        frames_with_tracks += usize::from(track_count > 0);
        max_concurrent_tracks = max_concurrent_tracks.max(track_count);
        let frame_index = input
            .frame_index_offset
            .checked_add(relative_frame_index)
            .ok_or(TrackingMetricsError::FrameIndexOverflow)?;
        let mut seen_track_ids = HashSet::with_capacity(track_count);
        let mut confirmed_in_frame = false;
        for observation_index in start..stop {
            let track_id = input.track_ids[observation_index];
            if !seen_track_ids.insert(track_id) {
                return Err(TrackingMetricsError::DuplicateTrackId {
                    frame_index,
                    track_id,
                });
            }
            let status_code = input.status_codes[observation_index];
            let confirmed = status_code == 1;
            confirmed_in_frame |= confirmed;
            observed.entry(track_id).or_default().push(Observation {
                frame_index,
                position: vector3(input.positions, observation_index),
                velocity: vector3(input.velocities, observation_index),
                confirmed,
            });
        }
        frames_with_confirmed_tracks += usize::from(confirmed_in_frame);
    }

    let tracks = observed
        .into_iter()
        .map(|(track_id, observations)| {
            summarize_track(track_id, &observations, input.scenery_boxes)
        })
        .collect();
    Ok(TrackingSequenceMetrics {
        num_frames,
        frames_with_tracks,
        frames_with_confirmed_tracks,
        max_concurrent_tracks,
        tracks,
    })
}

#[derive(Clone, Copy, Debug)]
struct Observation {
    frame_index: usize,
    position: [f32; 3],
    velocity: [f32; 3],
    confirmed: bool,
}

fn validate_input(input: TrackingMetricsInput<'_>) -> Result<usize, TrackingMetricsError> {
    let Some((&first_offset, remaining_offsets)) = input.frame_offsets.split_first() else {
        return Err(TrackingMetricsError::EmptyFrameOffsets);
    };
    if first_offset != 0 {
        return Err(TrackingMetricsError::InvalidFirstFrameOffset {
            actual: first_offset,
        });
    }
    let observation_count = input.track_ids.len();
    for (index, &offset) in remaining_offsets.iter().enumerate() {
        let previous = input.frame_offsets[index];
        if offset < previous {
            return Err(TrackingMetricsError::NonMonotonicFrameOffset {
                index: index + 1,
                previous,
                actual: offset,
            });
        }
        if offset > observation_count {
            return Err(TrackingMetricsError::FrameOffsetOutOfBounds {
                index: index + 1,
                value: offset,
                observation_count,
            });
        }
    }
    let final_offset = *input.frame_offsets.last().expect("non-empty is checked");
    if final_offset != observation_count {
        return Err(TrackingMetricsError::FinalFrameOffset {
            expected: observation_count,
            actual: final_offset,
        });
    }
    validate_vector_length("positions", input.positions.len(), observation_count)?;
    validate_vector_length("velocities", input.velocities.len(), observation_count)?;
    if input.status_codes.len() != observation_count {
        return Err(TrackingMetricsError::InputLength {
            name: "status codes",
            expected: observation_count,
            actual: input.status_codes.len(),
        });
    }
    for (index, &track_id) in input.track_ids.iter().enumerate() {
        if track_id < 0 {
            return Err(TrackingMetricsError::NegativeTrackId { index, track_id });
        }
    }
    for (index, &status_code) in input.status_codes.iter().enumerate() {
        if status_code > 2 {
            return Err(TrackingMetricsError::InvalidStatusCode { index, status_code });
        }
    }
    for (name, values) in [
        ("positions", input.positions),
        ("velocities", input.velocities),
    ] {
        if let Some((index, _)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(TrackingMetricsError::NonFiniteInput { name, index });
        }
    }
    Ok(input.frame_offsets.len() - 1)
}

fn validate_vector_length(
    name: &'static str,
    actual: usize,
    observation_count: usize,
) -> Result<(), TrackingMetricsError> {
    let expected = observation_count
        .checked_mul(3)
        .ok_or(TrackingMetricsError::InputLengthOverflow { name })?;
    if actual != expected {
        return Err(TrackingMetricsError::InputLength {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn vector3(values: &[f32], index: usize) -> [f32; 3] {
    let offset = index * 3;
    [values[offset], values[offset + 1], values[offset + 2]]
}

fn summarize_track(
    track_id: i64,
    observations: &[Observation],
    scenery_boxes: Option<&[Box2D]>,
) -> TrackObservationMetrics {
    let first = observations
        .first()
        .expect("tracked observations are non-empty");
    let last = observations
        .last()
        .expect("tracked observations are non-empty");
    let confirmed_frames = observations
        .iter()
        .filter_map(|observation| observation.confirmed.then_some(observation.frame_index))
        .collect::<Vec<_>>();
    let median_position_m = [
        median(
            observations
                .iter()
                .map(|observation| observation.position[0])
                .collect(),
        ),
        median(
            observations
                .iter()
                .map(|observation| observation.position[1])
                .collect(),
        ),
        median(
            observations
                .iter()
                .map(|observation| observation.position[2])
                .collect(),
        ),
    ];
    let speeds = observations
        .iter()
        .map(|observation| observation.velocity[0].hypot(observation.velocity[1]))
        .collect::<Vec<_>>();
    let path_length_m = observations
        .windows(2)
        .map(|window| {
            (window[1].position[0] - window[0].position[0])
                .hypot(window[1].position[1] - window[0].position[1])
        })
        .sum();
    let in_scenery_frames = scenery_boxes.map(|boxes| {
        observations
            .iter()
            .filter(|observation| scenery_contains(boxes, observation.position))
            .count()
    });
    TrackObservationMetrics {
        track_id,
        observed_frames: observations.len(),
        confirmed_frames: confirmed_frames.len(),
        first_frame_index: first.frame_index,
        last_frame_index: last.frame_index,
        first_position_m: first.position,
        last_position_m: last.position,
        median_position_m,
        displacement_m: (last.position[0] - first.position[0])
            .hypot(last.position[1] - first.position[1]),
        path_length_m,
        median_speed_mps: median(speeds.clone()),
        max_speed_mps: speeds.into_iter().fold(f32::NEG_INFINITY, f32::max),
        confirmed_intervals: contiguous_intervals(&confirmed_frames),
        in_scenery_frames,
        outside_scenery_frames: in_scenery_frames.map(|count| observations.len() - count),
    }
}

fn median(mut values: Vec<f32>) -> f32 {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) * 0.5
    }
}

fn scenery_contains(boxes: &[Box2D], position: [f32; 3]) -> bool {
    boxes.is_empty()
        || boxes
            .iter()
            .copied()
            .any(|boundary| boundary.contains(f64::from(position[0]), f64::from(position[1])))
}

fn contiguous_intervals(indices: &[usize]) -> Vec<[usize; 2]> {
    let Some((&first, remaining)) = indices.split_first() else {
        return Vec::new();
    };
    let mut intervals = Vec::new();
    let mut start = first;
    let mut previous = first;
    for &index in remaining {
        if index != previous + 1 {
            intervals.push([start, previous]);
            start = index;
        }
        previous = index;
    }
    intervals.push([start, previous]);
    intervals
}

/// Native input and numerical validation errors for tracking metrics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrackingMetricsError {
    EmptyFrameOffsets,
    InvalidFirstFrameOffset {
        actual: usize,
    },
    NonMonotonicFrameOffset {
        index: usize,
        previous: usize,
        actual: usize,
    },
    FrameOffsetOutOfBounds {
        index: usize,
        value: usize,
        observation_count: usize,
    },
    FinalFrameOffset {
        expected: usize,
        actual: usize,
    },
    InputLengthOverflow {
        name: &'static str,
    },
    InputLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    NegativeTrackId {
        index: usize,
        track_id: i64,
    },
    InvalidStatusCode {
        index: usize,
        status_code: u8,
    },
    NonFiniteInput {
        name: &'static str,
        index: usize,
    },
    DuplicateTrackId {
        frame_index: usize,
        track_id: i64,
    },
    FrameIndexOverflow,
}

impl fmt::Display for TrackingMetricsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrameOffsets => write!(formatter, "Tracking metrics require frame offsets."),
            Self::InvalidFirstFrameOffset { actual } => {
                write!(
                    formatter,
                    "Tracking metrics first frame offset must be zero; got {actual}."
                )
            }
            Self::NonMonotonicFrameOffset {
                index,
                previous,
                actual,
            } => write!(
                formatter,
                "Tracking metrics frame offset {index} must be at least {previous}; got {actual}."
            ),
            Self::FrameOffsetOutOfBounds {
                index,
                value,
                observation_count,
            } => write!(
                formatter,
                "Tracking metrics frame offset {index} is {value}; only {observation_count} observations exist."
            ),
            Self::FinalFrameOffset { expected, actual } => write!(
                formatter,
                "Tracking metrics final frame offset must be {expected}; got {actual}."
            ),
            Self::InputLengthOverflow { name } => {
                write!(formatter, "Tracking metrics {name} length overflows.")
            }
            Self::InputLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "Tracking metrics {name} has {actual} values; expected {expected}."
            ),
            Self::NegativeTrackId { index, track_id } => write!(
                formatter,
                "Tracking metrics track ID at observation {index} must be non-negative; got {track_id}."
            ),
            Self::InvalidStatusCode { index, status_code } => write!(
                formatter,
                "Tracking metrics status at observation {index} is invalid: {status_code}."
            ),
            Self::NonFiniteInput { name, index } => write!(
                formatter,
                "Tracking metrics {name} value {index} contains NaN or Inf."
            ),
            Self::DuplicateTrackId {
                frame_index,
                track_id,
            } => write!(
                formatter,
                "Tracking metrics frame {frame_index} repeats track ID {track_id}."
            ),
            Self::FrameIndexOverflow => {
                write!(formatter, "Tracking metrics frame index overflows.")
            }
        }
    }
}

impl std::error::Error for TrackingMetricsError {}

#[cfg(test)]
mod tests {
    use super::{TrackingMetricsInput, summarize_tracking_metrics};

    #[test]
    fn summarizes_identity_motion_and_confirmed_intervals() {
        let summary = summarize_tracking_metrics(TrackingMetricsInput {
            frame_offsets: &[0, 1, 3, 4],
            track_ids: &[2, 2, 7, 2],
            positions: &[0.0, 1.0, 0.0, 0.3, 1.4, 0.0, 1.0, 2.0, 0.0, 1.2, 1.8, 0.0],
            velocities: &[0.0, 0.0, 0.0, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            status_codes: &[0, 1, 0, 1],
            scenery_boxes: None,
            frame_index_offset: 10,
        })
        .unwrap();

        let track = &summary.tracks[0];
        assert_eq!(summary.frames_with_tracks, 3);
        assert_eq!(summary.frames_with_confirmed_tracks, 2);
        assert_eq!(track.first_frame_index, 10);
        assert_eq!(track.last_frame_index, 12);
        assert_eq!(track.confirmed_intervals, vec![[11, 12]]);
        assert!((track.path_length_m - (0.5 + 0.9_f32.hypot(0.4))).abs() < 1.0e-6);
    }
}
