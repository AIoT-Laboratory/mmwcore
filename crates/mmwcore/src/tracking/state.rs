//! Shared state, lifecycle, and reporting for native two-dimensional trackers.

use super::kalman::{ConstantVelocity2DFilter, CvTrackState};
use super::{NativeTrackStatus, TrackStepResult, Tracker2DConfig, TrackingError};

/// Mutable state shared by cluster-level and measurement-level trackers.
#[derive(Clone, Debug)]
pub(crate) struct TrackerState2D {
    pub(crate) config: Tracker2DConfig,
    pub(crate) filter: ConstantVelocity2DFilter,
    pub(crate) tracks: Vec<CvTrackState>,
    next_track_id: i64,
}

impl TrackerState2D {
    pub(crate) fn new(config: Tracker2DConfig) -> Self {
        Self {
            filter: ConstantVelocity2DFilter::new(config.dynamics),
            config,
            tracks: Vec::new(),
            next_track_id: 0,
        }
    }

    pub(crate) fn predict(&mut self) {
        for track in &mut self.tracks {
            self.filter.predict(track);
        }
    }

    pub(crate) fn update_track(
        &mut self,
        track_index: usize,
        center: [f64; 3],
        extent_covariance: [[f64; 2]; 2],
        point_count: usize,
    ) -> Result<(), TrackingError> {
        let alpha = self.config.dynamics.extent_covariance_smoothing;
        let track = &mut self.tracks[track_index];
        let smoothed_extent = if point_count == 1 {
            track.extent_covariance
        } else {
            matrix_blend(track.extent_covariance, extent_covariance, alpha)
        };
        let centroid_covariance = centroid_covariance(
            smoothed_extent,
            point_count,
            self.config.allocation.min_points,
        );
        self.filter
            .update(track, [center[0], center[1]], centroid_covariance)?;
        track.extent_covariance = smoothed_extent;
        track.z = center[2];
        track.age += 1;
        if track.status == NativeTrackStatus::Tentative {
            track.hits += 1;
            track.missed = 0;
            if track.hits >= self.config.lifecycle.confirmation_hits {
                track.status = NativeTrackStatus::Confirmed;
            }
        } else if point_count >= self.config.allocation.min_points {
            track.missed = 0;
            track.status = NativeTrackStatus::Confirmed;
        } else {
            track.missed += 1;
            if track.status == NativeTrackStatus::Confirmed {
                track.status = NativeTrackStatus::Coasting;
            }
        }
        update_outside_count(&self.config, track);
        Ok(())
    }

    pub(crate) fn supports_update(&self, point_count: usize) -> bool {
        point_count >= self.config.lifecycle.min_update_points
    }

    pub(crate) fn miss_unmatched(&mut self, matched_tracks: &[usize]) {
        for (index, track) in self.tracks.iter_mut().enumerate() {
            if matched_tracks.contains(&index) {
                continue;
            }
            track.age += 1;
            track.missed += 1;
            match track.status {
                NativeTrackStatus::Tentative => track.hits = 0,
                NativeTrackStatus::Confirmed => track.status = NativeTrackStatus::Coasting,
                NativeTrackStatus::Coasting => {}
            }
            update_outside_count(&self.config, track);
        }
    }

    pub(crate) fn can_allocate(
        &self,
        center: [f64; 3],
        point_count: usize,
        mean_radial_velocity: f64,
        total_snr: Option<f64>,
    ) -> Result<bool, TrackingError> {
        let allocation = self.config.allocation;
        let snr_sufficient = match allocation.min_total_snr {
            Some(threshold) => total_snr.ok_or(TrackingError::MissingAllocationSnr)? >= threshold,
            None => true,
        };
        let separated = allocation.min_separation_m.is_none_or(|minimum| {
            self.tracks.iter().all(|track| {
                (track.state[0] - center[0]).hypot(track.state[1] - center[1]) >= minimum
            })
        });
        Ok(self.config.scenery.contains(center[0], center[1])
            && point_count >= allocation.min_points
            && mean_radial_velocity.abs() >= allocation.min_abs_radial_velocity_mps
            && snr_sufficient
            && separated)
    }

    pub(crate) fn allocate(
        &mut self,
        center: [f64; 3],
        extent_covariance: [[f64; 2]; 2],
    ) -> Result<i64, TrackingError> {
        let track_id = self.next_track_id;
        self.next_track_id = self
            .next_track_id
            .checked_add(1)
            .ok_or(TrackingError::TrackIdOverflow)?;
        let track = self.filter.allocate(
            track_id,
            center,
            extent_covariance,
            self.config.lifecycle.confirmation_hits == 1,
        );
        self.tracks.push(track);
        Ok(track_id)
    }

    pub(crate) fn allocation_limit_reached(&self, allocated_count: usize) -> bool {
        self.tracks.len() >= self.config.max_tracks
            || self
                .config
                .allocation
                .max_new_tracks_per_frame
                .is_some_and(|limit| allocated_count >= limit)
    }

    pub(crate) fn delete_expired(&mut self) -> Vec<i64> {
        let mut expired_ids = Vec::new();
        self.tracks.retain(|track| {
            let coast_limit = coasting_miss_limit(&self.config, track);
            let expired = (track.status == NativeTrackStatus::Tentative
                && track.missed >= self.config.lifecycle.tentative_max_misses)
                || (track.status == NativeTrackStatus::Coasting && track.missed >= coast_limit)
                || track.outside >= self.config.scenery.outside_max_frames;
            if expired {
                expired_ids.push(track.track_id);
            }
            !expired
        });
        expired_ids
    }

    pub(crate) fn report(&self, observation_track_ids: Vec<i64>) -> TrackStepResult {
        let track_count = self.tracks.len();
        let mut track_ids = Vec::with_capacity(track_count);
        let mut positions = Vec::with_capacity(track_count * 3);
        let mut velocities = Vec::with_capacity(track_count * 3);
        let mut position_covariances = Vec::with_capacity(track_count * 4);
        let mut extent_covariances = Vec::with_capacity(track_count * 4);
        let mut statuses = Vec::with_capacity(track_count);
        let mut ages = Vec::with_capacity(track_count);
        let mut missed_counts = Vec::with_capacity(track_count);
        for track in &self.tracks {
            track_ids.push(track.track_id);
            positions.extend([track.state[0] as f32, track.state[1] as f32, track.z as f32]);
            velocities.extend([track.state[2] as f32, track.state[3] as f32, 0.0]);
            position_covariances.extend([
                track.covariance[0][0] as f32,
                track.covariance[0][1] as f32,
                track.covariance[1][0] as f32,
                track.covariance[1][1] as f32,
            ]);
            extent_covariances.extend([
                track.extent_covariance[0][0] as f32,
                track.extent_covariance[0][1] as f32,
                track.extent_covariance[1][0] as f32,
                track.extent_covariance[1][1] as f32,
            ]);
            statuses.push(track.status.code());
            ages.push(track.age as i64);
            missed_counts.push(track.missed as i64);
        }
        TrackStepResult {
            track_ids,
            positions,
            velocities,
            position_covariances,
            extent_covariances,
            statuses,
            ages,
            missed_counts,
            observation_track_ids,
        }
    }
}

fn coasting_miss_limit(config: &Tracker2DConfig, track: &CvTrackState) -> usize {
    let scenery = &config.scenery;
    let lifecycle = config.lifecycle;
    if let Some(limit) = lifecycle.static_max_misses
        && scenery.contains_static(track.state[0], track.state[1])
        && track.state[2].hypot(track.state[3]) <= lifecycle.static_speed_threshold_mps
    {
        return limit;
    }
    if let Some(limit) = lifecycle.exit_max_misses
        && !scenery.static_boxes.is_empty()
        && scenery.contains(track.state[0], track.state[1])
        && !scenery.contains_static(track.state[0], track.state[1])
    {
        return limit;
    }
    lifecycle.confirmed_max_misses
}

fn centroid_covariance(
    extent: [[f64; 2]; 2],
    point_count: usize,
    expected_point_count: usize,
) -> [[f64; 2]; 2] {
    let count = point_count as f64;
    let expected = expected_point_count.max(point_count) as f64;
    let missing_fraction = 1.0 - count / expected;
    scale_matrix(extent, count.recip() + missing_fraction.powi(2))
}

fn update_outside_count(config: &Tracker2DConfig, track: &mut CvTrackState) {
    if config.scenery.contains(track.state[0], track.state[1]) {
        track.outside = 0;
    } else {
        track.outside += 1;
    }
}

fn scale_matrix(matrix: [[f64; 2]; 2], factor: f64) -> [[f64; 2]; 2] {
    [
        [matrix[0][0] * factor, matrix[0][1] * factor],
        [matrix[1][0] * factor, matrix[1][1] * factor],
    ]
}

fn matrix_blend(previous: [[f64; 2]; 2], update: [[f64; 2]; 2], alpha: f64) -> [[f64; 2]; 2] {
    let previous_weight = 1.0 - alpha;
    [
        [
            previous_weight * previous[0][0] + alpha * update[0][0],
            previous_weight * previous[0][1] + alpha * update[0][1],
        ],
        [
            previous_weight * previous[1][0] + alpha * update[1][0],
            previous_weight * previous[1][1] + alpha * update[1][1],
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::{TrackerState2D, centroid_covariance};
    use crate::tracking::{
        Box2D, NativeTrackStatus, TrackAllocationConfig, TrackGatingConfig, TrackLifecycleConfig,
        TrackSceneryConfig, Tracker2DConfig, TrackerDynamicsConfig,
    };

    #[test]
    fn missing_reflections_keep_centroid_uncertainty_conservative() {
        let extent = [[1.0, 0.0], [0.0, 0.25]];

        let partial = centroid_covariance(extent, 2, 4);
        let complete = centroid_covariance(extent, 4, 4);

        assert_eq!(partial, [[0.75, 0.0], [0.0, 0.1875]]);
        assert_eq!(complete, [[0.25, 0.0], [0.0, 0.0625]]);
    }

    #[test]
    fn slow_interior_tracks_outlive_edge_and_moving_tracks() {
        let lifecycle = TrackLifecycleConfig::new(1, 2, 2, 1)
            .unwrap()
            .with_scene_miss_limits(Some(5), Some(1), 0.1)
            .unwrap();
        let scenery = TrackSceneryConfig::new(vec![Box2D::new(-2.0, 2.0, 0.0, 4.0).unwrap()], 2)
            .unwrap()
            .with_static_boxes(vec![Box2D::new(-1.0, 1.0, 0.0, 3.0).unwrap()]);
        let config = Tracker2DConfig::new(
            TrackerDynamicsConfig::new(0.1, [1.0, 1.0], 0.2, 1.0, 0.2).unwrap(),
            TrackGatingConfig::new(1.0, None, None).unwrap(),
            TrackAllocationConfig::new(1, 0.0, None, None, None).unwrap(),
            lifecycle,
            scenery,
            4,
        )
        .unwrap();
        let mut state = TrackerState2D::new(config);
        state
            .allocate([0.0, 1.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])
            .unwrap();
        state
            .allocate([1.5, 1.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])
            .unwrap();
        state
            .allocate([0.0, 2.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])
            .unwrap();
        for track in &mut state.tracks {
            track.status = NativeTrackStatus::Coasting;
        }
        state.tracks[0].missed = 2;
        state.tracks[1].missed = 1;
        state.tracks[2].missed = 2;
        state.tracks[2].state[2] = 0.2;

        assert_eq!(state.delete_expired(), [1, 2]);
        assert_eq!(state.tracks[0].track_id, 0);
    }
}
