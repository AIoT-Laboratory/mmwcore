//! Native cluster-level constant-velocity target tracker.

use std::cmp::Reverse;

use crate::assignment::linear_sum_assignment;

use super::kalman::radial_velocity;
use super::state::TrackerState2D;
use super::{ClusterMeasurements, TrackStepResult, Tracker2DConfig, TrackingError};

/// Stateful global-assignment tracker over Cartesian cluster summaries.
#[derive(Clone, Debug)]
pub struct ClusterTracker2D {
    state: TrackerState2D,
}

impl ClusterTracker2D {
    /// Construct one native cluster tracker.
    pub fn new(config: Tracker2DConfig) -> Result<Self, TrackingError> {
        if config.allocation.min_total_snr.is_some() {
            return Err(TrackingError::ClusterTrackingDoesNotSupportSnrAllocation);
        }
        Ok(Self {
            state: TrackerState2D::new(config),
        })
    }

    /// Advance the tracker by one cluster-summary frame.
    pub fn step(
        &mut self,
        measurements: ClusterMeasurements<'_>,
    ) -> Result<TrackStepResult, TrackingError> {
        let cluster_count = validate_measurements(measurements)?;
        self.state.predict();

        let mut associations = vec![-1_i64; cluster_count];
        let (matched_tracks, matched_clusters) = self.associate(measurements, cluster_count)?;
        let mut updated_tracks = Vec::with_capacity(matched_tracks.len());
        for (&track_index, &cluster_index) in matched_tracks.iter().zip(&matched_clusters) {
            let center = cluster_center(measurements.centers, cluster_index);
            let extent_covariance = cluster_extent_covariance(measurements.extents, cluster_index);
            let point_count = cluster_point_count(measurements.point_counts, cluster_index)?;
            if !self.state.supports_update(point_count) {
                continue;
            }
            self.state
                .update_track(track_index, center, extent_covariance, point_count)?;
            associations[cluster_index] = self.state.tracks[track_index].track_id;
            updated_tracks.push(track_index);
        }

        self.state.miss_unmatched(&updated_tracks);
        let expired_ids = self.state.delete_expired();
        if !expired_ids.is_empty() {
            for association in &mut associations {
                if expired_ids.contains(association) {
                    *association = -1;
                }
            }
        }

        let mut assigned_clusters = vec![false; cluster_count];
        for &cluster_index in &matched_clusters {
            assigned_clusters[cluster_index] = true;
        }
        let mut allocation_order = (0..cluster_count).collect::<Vec<_>>();
        allocation_order.sort_by_key(|&index| Reverse(measurements.point_counts[index]));
        let mut allocated_count = 0_usize;
        for cluster_index in allocation_order {
            if self.state.allocation_limit_reached(allocated_count) {
                break;
            }
            if assigned_clusters[cluster_index] {
                continue;
            }
            let center = cluster_center(measurements.centers, cluster_index);
            let point_count = cluster_point_count(measurements.point_counts, cluster_index)?;
            if !self.state.can_allocate(
                center,
                point_count,
                f64::from(measurements.mean_velocities[cluster_index]),
                None,
            )? {
                continue;
            }
            let track_id = self.state.allocate(
                center,
                cluster_extent_covariance(measurements.extents, cluster_index),
            )?;
            associations[cluster_index] = track_id;
            allocated_count += 1;
        }

        Ok(self.state.report(associations))
    }

    fn associate(
        &self,
        measurements: ClusterMeasurements<'_>,
        cluster_count: usize,
    ) -> Result<(Vec<usize>, Vec<usize>), TrackingError> {
        if self.state.tracks.is_empty() || cluster_count == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let track_count = self.state.tracks.len();
        let mut valid = vec![false; track_count * cluster_count];
        let mut costs = vec![1e12_f64; track_count * cluster_count];
        let mut has_valid_match = false;
        for (track_index, track) in self.state.tracks.iter().enumerate() {
            let predicted_radial_velocity = radial_velocity(track.state);
            for cluster_index in 0..cluster_count {
                let center = cluster_center(measurements.centers, cluster_index);
                let distance = (track.state[0] - center[0]).hypot(track.state[1] - center[1]);
                let mut accepted = distance <= self.state.config.gating.max_distance_m
                    && self.state.config.scenery.contains(center[0], center[1]);
                let mut score = distance;
                if accepted && let Some(limit) = self.state.config.gating.max_mahalanobis_distance {
                    let mahalanobis = self
                        .state
                        .filter
                        .mahalanobis_distance(track, [center[0], center[1]])?;
                    accepted = mahalanobis <= limit;
                    score = mahalanobis;
                }
                if accepted
                    && let Some(limit) = self.state.config.gating.max_radial_velocity_difference_mps
                {
                    accepted = (predicted_radial_velocity
                        - f64::from(measurements.mean_velocities[cluster_index]))
                    .abs()
                        <= limit;
                }
                let index = track_index * cluster_count + cluster_index;
                valid[index] = accepted;
                if accepted {
                    costs[index] = score;
                    has_valid_match = true;
                }
            }
        }
        if !has_valid_match {
            return Ok((Vec::new(), Vec::new()));
        }
        let assigned = linear_sum_assignment(&costs, track_count, cluster_count)?;
        let accepted = assigned
            .rows
            .into_iter()
            .zip(assigned.columns)
            .filter(|(track_index, cluster_index)| {
                valid[*track_index * cluster_count + *cluster_index]
            })
            .collect::<Vec<_>>();
        Ok((
            accepted
                .iter()
                .map(|(track_index, _)| *track_index)
                .collect(),
            accepted
                .into_iter()
                .map(|(_, cluster_index)| cluster_index)
                .collect(),
        ))
    }
}

fn validate_measurements(measurements: ClusterMeasurements<'_>) -> Result<usize, TrackingError> {
    let cluster_count = measurements.mean_velocities.len();
    let expected_matrix_length =
        cluster_count
            .checked_mul(3)
            .ok_or(TrackingError::ClusterMatrixLength {
                name: "centers",
                expected: usize::MAX,
                actual: measurements.centers.len(),
            })?;
    for (name, values) in [
        ("centers", measurements.centers),
        ("extents", measurements.extents),
    ] {
        if values.len() != expected_matrix_length {
            return Err(TrackingError::ClusterMatrixLength {
                name,
                expected: expected_matrix_length,
                actual: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(TrackingError::NonFiniteClusterValues { name });
        }
    }
    if let Some((index, &value)) = measurements
        .extents
        .iter()
        .enumerate()
        .find(|(_, value)| **value < 0.0)
    {
        return Err(TrackingError::NegativeClusterExtent { index, value });
    }
    if measurements.point_counts.len() != cluster_count {
        return Err(TrackingError::ClusterVectorLength {
            name: "point_counts",
            expected: cluster_count,
            actual: measurements.point_counts.len(),
        });
    }
    if measurements
        .mean_velocities
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(TrackingError::NonFiniteClusterValues {
            name: "mean_velocities",
        });
    }
    for (index, &point_count) in measurements.point_counts.iter().enumerate() {
        if point_count <= 0 {
            return Err(TrackingError::NonPositiveClusterPointCount {
                index,
                value: point_count,
            });
        }
    }
    Ok(cluster_count)
}

fn cluster_center(centers: &[f32], index: usize) -> [f64; 3] {
    let offset = index * 3;
    [
        f64::from(centers[offset]),
        f64::from(centers[offset + 1]),
        f64::from(centers[offset + 2]),
    ]
}

fn cluster_extent_covariance(extents: &[f32], index: usize) -> [[f64; 2]; 2] {
    let offset = index * 3;
    [
        [f64::from(extents[offset]).powi(2) / 12.0, 0.0],
        [0.0, f64::from(extents[offset + 1]).powi(2) / 12.0],
    ]
}

fn cluster_point_count(point_counts: &[i64], index: usize) -> Result<usize, TrackingError> {
    usize::try_from(point_counts[index]).map_err(|_| TrackingError::NonPositiveClusterPointCount {
        index,
        value: point_counts[index],
    })
}

#[cfg(test)]
mod tests {
    use super::{ClusterMeasurements, ClusterTracker2D};
    use crate::tracking::{
        TrackAllocationConfig, TrackGatingConfig, TrackLifecycleConfig, TrackSceneryConfig,
        Tracker2DConfig, TrackerDynamicsConfig,
    };

    fn config() -> Tracker2DConfig {
        Tracker2DConfig::new(
            TrackerDynamicsConfig::new(0.1, [2.0, 2.0], 0.2, 2.0, 0.2).unwrap(),
            TrackGatingConfig::new(0.5, None, None).unwrap(),
            TrackAllocationConfig::new(1, 0.0, None, None, None).unwrap(),
            TrackLifecycleConfig::new(3, 2, 3, 1).unwrap(),
            TrackSceneryConfig::new(Vec::new(), 5).unwrap(),
            200,
        )
        .unwrap()
    }

    #[test]
    fn confirms_a_native_track_and_preserves_its_id() {
        let mut tracker = ClusterTracker2D::new(config()).unwrap();
        let first = tracker
            .step(ClusterMeasurements {
                centers: &[0.0, 1.0, 0.0],
                extents: &[0.0, 0.0, 0.0],
                mean_velocities: &[0.0],
                point_counts: &[1],
            })
            .unwrap();
        tracker
            .step(ClusterMeasurements {
                centers: &[0.1, 1.0, 0.0],
                extents: &[0.0, 0.0, 0.0],
                mean_velocities: &[0.0],
                point_counts: &[1],
            })
            .unwrap();
        let third = tracker
            .step(ClusterMeasurements {
                centers: &[0.2, 1.0, 0.0],
                extents: &[0.0, 0.0, 0.0],
                mean_velocities: &[0.0],
                point_counts: &[1],
            })
            .unwrap();

        assert_eq!(first.statuses, [0]);
        assert_eq!(third.track_ids, [0]);
        assert_eq!(third.statuses, [1]);
        assert_eq!(third.observation_track_ids, [0]);
        assert!(third.velocities[0] > 0.0);
    }
}
