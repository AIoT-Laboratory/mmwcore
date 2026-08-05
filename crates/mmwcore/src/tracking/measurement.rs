//! Native measurement-level tracking with DBSCAN track births.

use std::cmp::Reverse;

use crate::clustering::{DbscanConfig, PointColumns, cluster_points};

use super::kalman::radial_velocity;
use super::state::TrackerState2D;
use super::{PointMeasurements, TrackStepResult, Tracker2DConfig, TrackingError};

/// Stateful tracker that associates individual Cartesian radar points.
#[derive(Clone, Debug)]
pub struct MeasurementTracker2D {
    state: TrackerState2D,
    allocation_clustering: DbscanConfig,
}

impl MeasurementTracker2D {
    /// Construct one native measurement-level tracker.
    pub fn new(config: Tracker2DConfig, allocation_clustering: DbscanConfig) -> Self {
        Self {
            state: TrackerState2D::new(config),
            allocation_clustering,
        }
    }

    /// Advance tracking for one packed Cartesian point cloud.
    pub fn step(
        &mut self,
        measurements: PointMeasurements<'_>,
    ) -> Result<TrackStepResult, TrackingError> {
        let point_count = validate_measurements(measurements)?;
        self.state.predict();
        let mut associations = self.associate_points(measurements, point_count)?;
        let matched_tracks = self.update_matched_tracks(measurements, &associations)?;
        self.state.miss_unmatched(&matched_tracks);
        let expired_ids = self.state.delete_expired();
        if !expired_ids.is_empty() {
            for association in &mut associations {
                if expired_ids.contains(association) {
                    *association = -1;
                }
            }
        }
        self.allocate_from_unassigned(measurements, &mut associations)?;
        Ok(self.state.report(associations))
    }

    fn associate_points(
        &self,
        measurements: PointMeasurements<'_>,
        point_count: usize,
    ) -> Result<Vec<i64>, TrackingError> {
        let mut associations = vec![-1_i64; point_count];
        if self.state.tracks.is_empty() || point_count == 0 {
            return Ok(associations);
        }

        for (point_index, association) in associations.iter_mut().enumerate() {
            let point = point_coordinate(measurements.coordinates, point_index);
            if !self.state.config.scenery.contains(point[0], point[1]) {
                continue;
            }
            let mut best_match = None;
            let mut best_distance = f64::INFINITY;
            for (track_index, track) in self.state.tracks.iter().enumerate() {
                let distance = (track.state[0] - point[0]).hypot(track.state[1] - point[1]);
                if distance > self.state.config.gating.max_distance_m {
                    continue;
                }
                if let Some(limit) = self.state.config.gating.max_mahalanobis_distance
                    && self
                        .state
                        .filter
                        .mahalanobis_distance(track, [point[0], point[1]])?
                        > limit
                {
                    continue;
                }
                if let Some(limit) = self.state.config.gating.max_radial_velocity_difference_mps
                    && (radial_velocity(track.state)
                        - f64::from(measurements.velocities[point_index]))
                    .abs()
                        > limit
                {
                    continue;
                }
                if distance < best_distance {
                    best_distance = distance;
                    best_match = Some(track_index);
                }
            }
            if let Some(track_index) = best_match {
                *association = self.state.tracks[track_index].track_id;
            }
        }
        Ok(associations)
    }

    fn update_matched_tracks(
        &mut self,
        measurements: PointMeasurements<'_>,
        associations: &[i64],
    ) -> Result<Vec<usize>, TrackingError> {
        let mut matched_tracks = Vec::new();
        for track_index in 0..self.state.tracks.len() {
            let track_id = self.state.tracks[track_index].track_id;
            let indices = associations
                .iter()
                .enumerate()
                .filter_map(|(index, &association)| (association == track_id).then_some(index))
                .collect::<Vec<_>>();
            if indices.is_empty() {
                continue;
            }
            let summary = summarize_points(measurements.coordinates, &indices);
            self.state.update_track(
                track_index,
                summary.center,
                summary.extent_covariance,
                indices.len(),
            )?;
            matched_tracks.push(track_index);
        }
        Ok(matched_tracks)
    }

    fn allocate_from_unassigned(
        &mut self,
        measurements: PointMeasurements<'_>,
        associations: &mut [i64],
    ) -> Result<(), TrackingError> {
        if self.state.allocation_limit_reached(0) {
            return Ok(());
        }
        let unassigned_indices = associations
            .iter()
            .enumerate()
            .filter_map(|(index, &association)| (association < 0).then_some(index))
            .collect::<Vec<_>>();
        if unassigned_indices.is_empty() {
            return Ok(());
        }

        let mut candidate_points = Vec::with_capacity(unassigned_indices.len() * 4);
        for &index in &unassigned_indices {
            let coordinate = point_coordinate(measurements.coordinates, index);
            candidate_points.extend([
                coordinate[0] as f32,
                coordinate[1] as f32,
                coordinate[2] as f32,
                measurements.velocities[index],
            ]);
        }
        let clusters = cluster_points(
            &candidate_points,
            unassigned_indices.len(),
            4,
            PointColumns {
                x: 0,
                y: 1,
                z: 2,
                velocity: Some(3),
            },
            self.allocation_clustering,
        )?;
        let mut allocation_order = (0..clusters.point_counts.len()).collect::<Vec<_>>();
        allocation_order.sort_by_key(|&index| Reverse(clusters.point_counts[index]));
        let mut allocated_count = 0_usize;
        for cluster_index in allocation_order {
            if self.state.allocation_limit_reached(allocated_count) {
                break;
            }
            let member_indices =
                cluster_member_indices(&clusters.labels, cluster_index, &unassigned_indices);
            let point_count =
                usize::try_from(clusters.point_counts[cluster_index]).map_err(|_| {
                    TrackingError::NonPositiveClusterPointCount {
                        index: cluster_index,
                        value: clusters.point_counts[cluster_index],
                    }
                })?;
            let center = cluster_center(&clusters.centers, cluster_index);
            let total_snr = member_indices
                .iter()
                .map(|&index| f64::from(measurements.snrs[index]))
                .sum();
            if !self.state.can_allocate(
                center,
                point_count,
                f64::from(clusters.mean_velocities[cluster_index]),
                Some(total_snr),
            )? {
                continue;
            }
            let summary = summarize_points(measurements.coordinates, &member_indices);
            let track_id = self.state.allocate(center, summary.extent_covariance)?;
            for index in member_indices {
                associations[index] = track_id;
            }
            allocated_count += 1;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct PointSummary {
    center: [f64; 3],
    extent_covariance: [[f64; 2]; 2],
}

fn validate_measurements(measurements: PointMeasurements<'_>) -> Result<usize, TrackingError> {
    let point_count = measurements.velocities.len();
    let expected_coordinates =
        point_count
            .checked_mul(3)
            .ok_or(TrackingError::PointMatrixLength {
                expected: usize::MAX,
                actual: measurements.coordinates.len(),
            })?;
    if measurements.coordinates.len() != expected_coordinates {
        return Err(TrackingError::PointMatrixLength {
            expected: expected_coordinates,
            actual: measurements.coordinates.len(),
        });
    }
    if measurements.snrs.len() != point_count {
        return Err(TrackingError::PointVectorLength {
            name: "snr",
            expected: point_count,
            actual: measurements.snrs.len(),
        });
    }
    for (name, values) in [
        ("coordinates", measurements.coordinates),
        ("velocity", measurements.velocities),
        ("snr", measurements.snrs),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(TrackingError::NonFinitePointValues { name });
        }
    }
    Ok(point_count)
}

fn point_coordinate(coordinates: &[f32], index: usize) -> [f64; 3] {
    let offset = index * 3;
    [
        f64::from(coordinates[offset]),
        f64::from(coordinates[offset + 1]),
        f64::from(coordinates[offset + 2]),
    ]
}

fn cluster_center(centers: &[f32], index: usize) -> [f64; 3] {
    point_coordinate(centers, index)
}

fn cluster_member_indices(
    labels: &[i64],
    cluster_index: usize,
    unassigned_indices: &[usize],
) -> Vec<usize> {
    labels
        .iter()
        .enumerate()
        .filter_map(|(candidate_index, &label)| {
            (label == cluster_index as i64).then_some(unassigned_indices[candidate_index])
        })
        .collect()
}

fn summarize_points(coordinates: &[f32], indices: &[usize]) -> PointSummary {
    let count = indices.len() as f64;
    let mut center = [0.0; 3];
    for &index in indices {
        let point = point_coordinate(coordinates, index);
        for axis in 0..3 {
            center[axis] += point[axis];
        }
    }
    for coordinate in &mut center {
        *coordinate /= count;
    }
    let mut extent_covariance = [[0.0; 2]; 2];
    for &index in indices {
        let point = point_coordinate(coordinates, index);
        let dx = point[0] - center[0];
        let dy = point[1] - center[1];
        extent_covariance[0][0] += dx * dx;
        extent_covariance[0][1] += dx * dy;
        extent_covariance[1][0] += dy * dx;
        extent_covariance[1][1] += dy * dy;
    }
    for row in &mut extent_covariance {
        for value in row {
            *value /= count;
        }
    }
    PointSummary {
        center,
        extent_covariance,
    }
}

#[cfg(test)]
mod tests {
    use super::{MeasurementTracker2D, PointMeasurements};
    use crate::clustering::DbscanConfig;
    use crate::tracking::{
        TrackAllocationConfig, TrackGatingConfig, TrackLifecycleConfig, TrackSceneryConfig,
        Tracker2DConfig, TrackerDynamicsConfig,
    };

    fn config() -> Tracker2DConfig {
        Tracker2DConfig::new(
            TrackerDynamicsConfig::new(0.1, [2.0, 2.0], 0.2, 2.0, 0.2).unwrap(),
            TrackGatingConfig::new(0.5, None, None).unwrap(),
            TrackAllocationConfig::new(1, 0.0, None, None).unwrap(),
            TrackLifecycleConfig::new(1, 2, 3).unwrap(),
            TrackSceneryConfig::new(Vec::new(), 5).unwrap(),
            200,
        )
        .unwrap()
    }

    #[test]
    fn partitions_points_between_native_tracks() {
        let mut tracker =
            MeasurementTracker2D::new(config(), DbscanConfig::new(0.2, 2, 0.0, false).unwrap());
        tracker
            .step(PointMeasurements {
                coordinates: &[
                    -1.05, 1.0, 0.0, -0.95, 1.0, 0.0, 0.95, 1.0, 0.0, 1.05, 1.0, 0.0,
                ],
                velocities: &[0.0; 4],
                snrs: &[0.0; 4],
            })
            .unwrap();
        let frame = tracker
            .step(PointMeasurements {
                coordinates: &[-0.9, 1.0, 0.0, -0.8, 1.0, 0.0, 0.8, 1.0, 0.0, 0.9, 1.0, 0.0],
                velocities: &[0.0; 4],
                snrs: &[0.0; 4],
            })
            .unwrap();

        assert_eq!(frame.track_ids, [0, 1]);
        assert_eq!(frame.observation_track_ids, [0, 0, 1, 1]);
    }
}
