//! Three-dimensional extended-target tracking over spherical radar measurements.

use std::cmp::Ordering;
use std::f64::consts::PI;

use crate::clustering::DbscanConfig;

use super::{
    NativeTrackStatus, PointMeasurements, TrackStepResult, Tracker3DConfig, TrackingError,
};

type Matrix3 = [[f64; 3]; 3];
type Matrix4 = [[f64; 4]; 4];
type Matrix6 = [[f64; 6]; 6];

#[derive(Clone, Debug)]
struct Unit {
    id: i64,
    state: [f64; 6],
    covariance: Matrix6,
    dispersion: Matrix4,
    extent: Matrix3,
    expected_points: f64,
    status: NativeTrackStatus,
    hits: usize,
    age: usize,
    missed: usize,
    outside: usize,
}

impl Unit {
    fn position(&self) -> [f64; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

    fn speed(&self) -> f64 {
        self.state[3].hypot(self.state[4]).hypot(self.state[5])
    }
}

#[derive(Clone, Copy, Debug)]
struct SphericalSummary {
    centroid: [f64; 4],
    dispersion: Matrix4,
    extent: Matrix3,
}

/// Cumulative evidence explaining GTrack3D association and lifecycle decisions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GTrack3DDiagnostics {
    pub frames: u64,
    pub points: u64,
    pub outside_points: u64,
    pub distance_gate_misses: u64,
    pub doppler_gate_misses: u64,
    pub mahalanobis_gate_misses: u64,
    pub empty_updates: u64,
    pub partial_updates: u64,
    pub allocations: u64,
    pub confirmations: u64,
    pub reactivations: u64,
    pub tentative_deletions: u64,
    pub coasting_deletions: u64,
    pub outside_deletions: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct AssociationDiagnostics {
    outside_points: u64,
    distance_gate_misses: u64,
    doppler_gate_misses: u64,
    mahalanobis_gate_misses: u64,
}

#[derive(Clone, Debug, Default)]
struct UpdateDiagnostics {
    observed: Vec<usize>,
    empty_updates: u64,
    partial_updates: u64,
    confirmations: u64,
    reactivations: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct DeletionDiagnostics {
    tentative: u64,
    coasting: u64,
    outside: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct AllocationDiagnostics {
    allocations: u64,
    confirmations: u64,
}

#[derive(Clone, Debug)]
struct GTrackFilter3D {
    transition: Matrix6,
    process_noise: Matrix6,
    point_noise: Matrix4,
    initial_velocity_variance: f64,
    max_velocity_mps: f64,
}

impl GTrackFilter3D {
    fn new(config: &Tracker3DConfig) -> Self {
        let dynamics = config.dynamics;
        let dt = dynamics.frame_period_s;
        let mut transition = identity::<6>();
        for axis in 0..3 {
            transition[axis][axis + 3] = dt;
        }
        let mut gain = [[0.0; 3]; 6];
        for axis in 0..3 {
            gain[axis][axis] = 0.5 * dt.powi(2);
            gain[axis + 3][axis] = dt;
        }
        let acceleration = diagonal([
            dynamics.max_acceleration_mps2[0].powi(2),
            dynamics.max_acceleration_mps2[1].powi(2),
            dynamics.max_acceleration_mps2[2].powi(2),
        ]);
        Self {
            transition,
            process_noise: product(&product(&gain, &acceleration), &transpose(&gain)),
            point_noise: diagonal([
                dynamics.measurement_noise_m.powi(2),
                dynamics.angle_noise_rad.powi(2),
                dynamics.elevation_noise_rad.powi(2),
                dynamics.doppler_noise_mps.powi(2),
            ]),
            initial_velocity_variance: dynamics.initial_velocity_std_mps.powi(2),
            max_velocity_mps: dynamics.max_velocity_mps,
        }
    }

    fn predict(&self, unit: &mut Unit) {
        unit.state = matrix_vector_product(&self.transition, unit.state);
        unit.covariance = matrix_add(
            product(
                &product(&self.transition, &unit.covariance),
                &transpose(&self.transition),
            ),
            self.process_noise,
        );
    }

    fn association(&self, unit: &Unit, measured: [f64; 4]) -> Result<(f64, f64), TrackingError> {
        let (predicted, jacobian) = measurement_model(unit.state);
        let measured = aligned_measurement(measured, predicted, self.max_velocity_mps);
        let residual = vector_subtract(measured, predicted);
        let covariance = matrix_add(
            matrix_add(
                product(&product(&jacobian, &unit.covariance), &transpose(&jacobian)),
                self.point_noise,
            ),
            unit.dispersion,
        );
        let (inverse, log_determinant) = inverse_spd(covariance)?;
        let solved = matrix_vector_product(&inverse, residual);
        let distance_squared = dot(residual, solved).max(0.0);
        Ok((distance_squared.sqrt(), log_determinant + distance_squared))
    }

    fn update(
        &self,
        unit: &mut Unit,
        measured: [f64; 4],
        measurement_covariance: Matrix4,
    ) -> Result<(), TrackingError> {
        let (predicted, jacobian) = measurement_model(unit.state);
        let measured = aligned_measurement(measured, predicted, self.max_velocity_mps);
        let residual = vector_subtract(measured, predicted);
        let innovation_covariance = matrix_add(
            product(&product(&jacobian, &unit.covariance), &transpose(&jacobian)),
            measurement_covariance,
        );
        let (inverse, _) = inverse_spd(innovation_covariance)?;
        let cross_covariance = product(&unit.covariance, &transpose(&jacobian));
        let gain = product(&cross_covariance, &inverse);
        let correction = matrix_vector_product(&gain, residual);
        for (state, correction) in unit.state.iter_mut().zip(correction) {
            *state += correction;
        }

        let residual_gain = matrix_subtract(identity::<6>(), product(&gain, &jacobian));
        let posterior = product(
            &product(&residual_gain, &unit.covariance),
            &transpose(&residual_gain),
        );
        let measurement_term = product(&product(&gain, &measurement_covariance), &transpose(&gain));
        unit.covariance = symmetrize(matrix_add(posterior, measurement_term));
        Ok(())
    }

    fn allocate(&self, id: i64, summary: SphericalSummary, confirmed: bool, points: usize) -> Unit {
        let [range, azimuth, elevation, radial_velocity] = summary.centroid;
        let position = spherical_position(summary.centroid);
        let line_of_sight = if range > 1e-9 {
            [
                position[0] / range,
                position[1] / range,
                position[2] / range,
            ]
        } else {
            [1.0, 0.0, 0.0]
        };
        let position_jacobian = spherical_position_jacobian(range, azimuth, elevation);
        let position_noise = diagonal([
            self.point_noise[0][0],
            self.point_noise[1][1],
            self.point_noise[2][2],
        ]);
        let position_covariance = product(
            &product(&position_jacobian, &position_noise),
            &transpose(&position_jacobian),
        );
        let mut covariance = [[0.0; 6]; 6];
        for row in 0..3 {
            for column in 0..3 {
                covariance[row][column] = position_covariance[row][column];
            }
            covariance[row + 3][row + 3] = self.initial_velocity_variance;
        }
        Unit {
            id,
            state: [
                position[0],
                position[1],
                position[2],
                radial_velocity * line_of_sight[0],
                radial_velocity * line_of_sight[1],
                radial_velocity * line_of_sight[2],
            ],
            covariance,
            dispersion: summary.dispersion,
            extent: summary.extent,
            expected_points: points as f64,
            status: if confirmed {
                NativeTrackStatus::Confirmed
            } else {
                NativeTrackStatus::Tentative
            },
            hits: 1,
            age: 1,
            missed: 0,
            outside: 0,
        }
    }
}

/// Stateful 3D GTRACK implementation in sensor forward/lateral/up coordinates.
#[derive(Clone, Debug)]
pub struct GTrack3D {
    config: Tracker3DConfig,
    allocation: DbscanConfig,
    filter: GTrackFilter3D,
    units: Vec<Unit>,
    next_id: i64,
    diagnostics: GTrack3DDiagnostics,
}

impl GTrack3D {
    /// Construct a tracker. The clustering policy supplies allocation proximity.
    pub fn new(config: Tracker3DConfig, allocation: DbscanConfig) -> Self {
        let filter = GTrackFilter3D::new(&config);
        Self {
            config,
            allocation,
            filter,
            units: Vec::new(),
            next_id: 0,
            diagnostics: GTrack3DDiagnostics::default(),
        }
    }

    /// Return cumulative counters since this tracker was constructed.
    pub fn diagnostics(&self) -> GTrack3DDiagnostics {
        self.diagnostics
    }

    /// Predict, bid for points, update units, allocate, and manage lifecycles.
    pub fn step(
        &mut self,
        measurements: PointMeasurements<'_>,
    ) -> Result<TrackStepResult, TrackingError> {
        let point_count = validate_measurements(measurements)?;
        for unit in &mut self.units {
            self.filter.predict(unit);
        }

        let (mut associations, association) = self.associate(measurements, point_count)?;
        let update = self.update_units(measurements, &mut associations)?;
        self.miss_unobserved(&update.observed);
        let (expired, deletion) = self.delete_expired();
        for association in &mut associations {
            if expired.contains(association) {
                *association = -1;
            }
        }
        let allocation = self.allocate(measurements, &mut associations)?;
        self.record_diagnostics(point_count, association, &update, deletion, allocation);
        Ok(self.report(associations))
    }

    fn associate(
        &self,
        measurements: PointMeasurements<'_>,
        point_count: usize,
    ) -> Result<(Vec<i64>, AssociationDiagnostics), TrackingError> {
        let mut associations = vec![-1; point_count];
        let mut diagnostics = AssociationDiagnostics::default();
        let mahalanobis_limit = self.config.gating.max_mahalanobis_distance.unwrap_or(4.0);
        for (index, association) in associations.iter_mut().enumerate() {
            let point = point_coordinate(measurements.coordinates, index);
            if !self.config.scenery.contains(point[0], point[1], point[2]) {
                diagnostics.outside_points += 1;
                continue;
            }
            let measurement = point_measurement(point, f64::from(measurements.velocities[index]));
            let mut best: Option<(i64, f64)> = None;
            let mut passed_distance = false;
            let mut passed_doppler = false;
            for unit in &self.units {
                let distance = distance3(unit.position(), point);
                if distance > self.config.gating.max_distance_m {
                    continue;
                }
                passed_distance = true;
                let expected_velocity = measurement_model(unit.state).0[3];
                let velocity = unwrap_doppler(
                    measurement[3],
                    expected_velocity,
                    self.filter.max_velocity_mps,
                );
                if self
                    .config
                    .gating
                    .max_radial_velocity_difference_mps
                    .is_some_and(|limit| (velocity - expected_velocity).abs() > limit)
                {
                    continue;
                }
                passed_doppler = true;
                let (mahalanobis, bid) = self.filter.association(unit, measurement)?;
                if mahalanobis > mahalanobis_limit {
                    continue;
                }
                if best.is_none_or(|(_, best_bid)| bid < best_bid) {
                    best = Some((unit.id, bid));
                }
            }
            if let Some((id, _)) = best {
                *association = id;
            } else if !self.units.is_empty() {
                if !passed_distance {
                    diagnostics.distance_gate_misses += 1;
                } else if !passed_doppler {
                    diagnostics.doppler_gate_misses += 1;
                } else {
                    diagnostics.mahalanobis_gate_misses += 1;
                }
            }
        }
        Ok((associations, diagnostics))
    }

    fn update_units(
        &mut self,
        measurements: PointMeasurements<'_>,
        associations: &mut [i64],
    ) -> Result<UpdateDiagnostics, TrackingError> {
        let mut diagnostics = UpdateDiagnostics::default();
        for unit_index in 0..self.units.len() {
            let id = self.units[unit_index].id;
            let indices = associations
                .iter()
                .enumerate()
                .filter_map(|(index, &association)| (association == id).then_some(index))
                .collect::<Vec<_>>();
            let status = self.units[unit_index].status;
            let lifecycle_hit = indices.len() >= self.config.lifecycle.min_update_points;
            let update_allowed = status != NativeTrackStatus::Tentative || lifecycle_hit;
            if indices.is_empty() || !update_allowed {
                if indices.is_empty() {
                    diagnostics.empty_updates += 1;
                } else {
                    diagnostics.partial_updates += 1;
                }
                for index in indices {
                    associations[index] = -1;
                }
                continue;
            }
            if !lifecycle_hit {
                diagnostics.partial_updates += 1;
            }

            let reference = measurement_model(self.units[unit_index].state).0;
            let summary = summarize(
                measurements,
                &indices,
                reference,
                self.filter.max_velocity_mps,
            );
            let static_reactivation = status == NativeTrackStatus::Coasting
                && self.config.lifecycle.static_speed_threshold_mps > 0.0
                && summary.centroid[3].abs() < self.config.lifecycle.static_speed_threshold_mps;
            if static_reactivation {
                diagnostics.partial_updates += 1;
            }
            let alpha = self.config.dynamics.extent_covariance_smoothing;
            let unit = &mut self.units[unit_index];
            unit.dispersion = blend(unit.dispersion, summary.dispersion, alpha);
            if indices.len() > 1 {
                unit.extent = blend(unit.extent, summary.extent, alpha);
            }
            unit.expected_points = (unit.expected_points * 0.95).max(indices.len() as f64);
            let measurement_covariance = centroid_covariance(
                self.filter.point_noise,
                unit.dispersion,
                indices.len(),
                unit.expected_points,
            );
            self.filter
                .update(unit, summary.centroid, measurement_covariance)?;
            if !lifecycle_hit || static_reactivation {
                continue;
            }
            let previous_status = unit.status;
            observe(&self.config, unit);
            if previous_status == NativeTrackStatus::Tentative
                && unit.status == NativeTrackStatus::Confirmed
            {
                diagnostics.confirmations += 1;
            } else if previous_status == NativeTrackStatus::Coasting {
                diagnostics.reactivations += 1;
            }
            diagnostics.observed.push(unit_index);
        }
        Ok(diagnostics)
    }

    fn miss_unobserved(&mut self, observed: &[usize]) {
        for (index, unit) in self.units.iter_mut().enumerate() {
            if observed.contains(&index) {
                continue;
            }
            unit.age += 1;
            unit.missed += 1;
            match unit.status {
                NativeTrackStatus::Tentative => unit.hits = 0,
                NativeTrackStatus::Confirmed => unit.status = NativeTrackStatus::Coasting,
                NativeTrackStatus::Coasting => {}
            }
            update_outside(&self.config, unit);
        }
    }

    fn delete_expired(&mut self) -> (Vec<i64>, DeletionDiagnostics) {
        let config = &self.config;
        let mut expired = Vec::new();
        let mut diagnostics = DeletionDiagnostics::default();
        self.units.retain(|unit| {
            let miss_limit = coast_limit(config, unit);
            let outside = unit.outside >= config.scenery.outside_max_frames;
            let tentative = unit.status == NativeTrackStatus::Tentative
                && unit.missed >= config.lifecycle.tentative_max_misses;
            let coasting = unit.status == NativeTrackStatus::Coasting && unit.missed >= miss_limit;
            let delete = outside || tentative || coasting;
            if delete {
                expired.push(unit.id);
                if outside {
                    diagnostics.outside += 1;
                } else if tentative {
                    diagnostics.tentative += 1;
                } else {
                    diagnostics.coasting += 1;
                }
            }
            !delete
        });
        (expired, diagnostics)
    }

    fn allocate(
        &mut self,
        measurements: PointMeasurements<'_>,
        associations: &mut [i64],
    ) -> Result<AllocationDiagnostics, TrackingError> {
        let mut candidates = allocation_groups(
            measurements,
            associations,
            self.allocation,
            &self.config,
            self.filter.max_velocity_mps,
        );
        candidates.sort_by(|left, right| {
            right
                .len()
                .cmp(&left.len())
                .then_with(|| {
                    total_snr(measurements, right).total_cmp(&total_snr(measurements, left))
                })
                .then_with(|| left[0].cmp(&right[0]))
        });

        let mut allocated = 0;
        let mut diagnostics = AllocationDiagnostics::default();
        for indices in candidates {
            if self.units.len() >= self.config.max_tracks
                || self
                    .config
                    .allocation
                    .max_new_tracks_per_frame
                    .is_some_and(|limit| allocated >= limit)
            {
                break;
            }
            if indices.len() < self.config.allocation.min_points
                || indices.len() < self.allocation.min_samples()
            {
                continue;
            }
            let seed = point_measurement(
                point_coordinate(measurements.coordinates, indices[0]),
                f64::from(measurements.velocities[indices[0]]),
            );
            let summary = summarize(measurements, &indices, seed, self.filter.max_velocity_mps);
            let center = spherical_position(summary.centroid);
            if !self.can_allocate(
                center,
                summary.centroid[3],
                total_snr(measurements, &indices),
            ) {
                continue;
            }
            let id = self.next_id;
            self.next_id = self
                .next_id
                .checked_add(1)
                .ok_or(TrackingError::TrackIdOverflow)?;
            self.units.push(self.filter.allocate(
                id,
                summary,
                self.config.lifecycle.confirmation_hits == 1,
                indices.len(),
            ));
            diagnostics.allocations += 1;
            if self.config.lifecycle.confirmation_hits == 1 {
                diagnostics.confirmations += 1;
            }
            for index in indices {
                associations[index] = id;
            }
            allocated += 1;
        }
        Ok(diagnostics)
    }

    fn record_diagnostics(
        &mut self,
        point_count: usize,
        association: AssociationDiagnostics,
        update: &UpdateDiagnostics,
        deletion: DeletionDiagnostics,
        allocation: AllocationDiagnostics,
    ) {
        let totals = &mut self.diagnostics;
        totals.frames = totals.frames.saturating_add(1);
        totals.points = totals
            .points
            .saturating_add(u64::try_from(point_count).unwrap_or(u64::MAX));
        totals.outside_points = totals
            .outside_points
            .saturating_add(association.outside_points);
        totals.distance_gate_misses = totals
            .distance_gate_misses
            .saturating_add(association.distance_gate_misses);
        totals.doppler_gate_misses = totals
            .doppler_gate_misses
            .saturating_add(association.doppler_gate_misses);
        totals.mahalanobis_gate_misses = totals
            .mahalanobis_gate_misses
            .saturating_add(association.mahalanobis_gate_misses);
        totals.empty_updates = totals.empty_updates.saturating_add(update.empty_updates);
        totals.partial_updates = totals
            .partial_updates
            .saturating_add(update.partial_updates);
        totals.allocations = totals.allocations.saturating_add(allocation.allocations);
        totals.confirmations = totals
            .confirmations
            .saturating_add(update.confirmations)
            .saturating_add(allocation.confirmations);
        totals.reactivations = totals.reactivations.saturating_add(update.reactivations);
        totals.tentative_deletions = totals
            .tentative_deletions
            .saturating_add(deletion.tentative);
        totals.coasting_deletions = totals.coasting_deletions.saturating_add(deletion.coasting);
        totals.outside_deletions = totals.outside_deletions.saturating_add(deletion.outside);
    }

    fn can_allocate(&self, center: [f64; 3], radial_velocity: f64, snr: f64) -> bool {
        let allocation = self.config.allocation;
        let snr_sufficient = allocation
            .min_total_snr
            .is_none_or(|threshold| snr >= threshold);
        let separated = allocation.min_separation_m.is_none_or(|minimum| {
            self.units
                .iter()
                .all(|unit| distance3(unit.position(), center) >= minimum)
        });
        self.config
            .scenery
            .contains(center[0], center[1], center[2])
            && radial_velocity.abs() >= allocation.min_abs_radial_velocity_mps
            && snr_sufficient
            && separated
    }

    fn report(&self, observation_track_ids: Vec<i64>) -> TrackStepResult {
        let mut result = TrackStepResult {
            track_ids: Vec::with_capacity(self.units.len()),
            positions: Vec::with_capacity(self.units.len() * 3),
            velocities: Vec::with_capacity(self.units.len() * 3),
            position_covariances: Vec::with_capacity(self.units.len() * 9),
            extent_covariances: Vec::with_capacity(self.units.len() * 9),
            statuses: Vec::with_capacity(self.units.len()),
            ages: Vec::with_capacity(self.units.len()),
            missed_counts: Vec::with_capacity(self.units.len()),
            observation_track_ids,
        };
        for unit in &self.units {
            result.track_ids.push(unit.id);
            result
                .positions
                .extend(unit.state[..3].iter().map(|value| *value as f32));
            result
                .velocities
                .extend(unit.state[3..].iter().map(|value| *value as f32));
            for row in 0..3 {
                result
                    .position_covariances
                    .extend(unit.covariance[row][..3].iter().map(|value| *value as f32));
                result
                    .extent_covariances
                    .extend(unit.extent[row].iter().map(|value| *value as f32));
            }
            result.statuses.push(unit.status.code());
            result.ages.push(unit.age as i64);
            result.missed_counts.push(unit.missed as i64);
        }
        result
    }
}

fn observe(config: &Tracker3DConfig, unit: &mut Unit) {
    unit.age += 1;
    unit.missed = 0;
    match unit.status {
        NativeTrackStatus::Tentative => {
            unit.hits += 1;
            if unit.hits >= config.lifecycle.confirmation_hits {
                unit.status = NativeTrackStatus::Confirmed;
            }
        }
        NativeTrackStatus::Confirmed | NativeTrackStatus::Coasting => {
            unit.status = NativeTrackStatus::Confirmed;
        }
    }
    update_outside(config, unit);
}

fn update_outside(config: &Tracker3DConfig, unit: &mut Unit) {
    let [x, y, z] = unit.position();
    if config.scenery.contains(x, y, z) {
        unit.outside = 0;
    } else {
        unit.outside += 1;
    }
}

fn coast_limit(config: &Tracker3DConfig, unit: &Unit) -> usize {
    let lifecycle = config.lifecycle;
    let [x, y, z] = unit.position();
    if let Some(limit) = lifecycle.static_max_misses
        && config.scenery.contains_static(x, y, z)
        && unit.speed() <= lifecycle.static_speed_threshold_mps
    {
        return limit;
    }
    if let Some(limit) = lifecycle.exit_max_misses
        && !config.scenery.contains(x, y, z)
    {
        return limit;
    }
    lifecycle.confirmed_max_misses
}

fn allocation_groups(
    measurements: PointMeasurements<'_>,
    associations: &[i64],
    config: DbscanConfig,
    tracker: &Tracker3DConfig,
    max_velocity_mps: f64,
) -> Vec<Vec<usize>> {
    let mut remaining = associations
        .iter()
        .enumerate()
        .filter_map(|(index, &association)| {
            let point = point_coordinate(measurements.coordinates, index);
            (association < 0 && tracker.scenery.contains(point[0], point[1], point[2]))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    remaining.sort_by(|&left, &right| {
        f64::from(measurements.snrs[right])
            .partial_cmp(&f64::from(measurements.snrs[left]))
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });

    let mut groups = Vec::new();
    while let Some(&lead) = remaining.first() {
        let lead_point = point_coordinate(measurements.coordinates, lead);
        let lead_velocity = f64::from(measurements.velocities[lead]);
        let mut group = Vec::new();
        remaining.retain(|&index| {
            let point = point_coordinate(measurements.coordinates, index);
            let velocity = unwrap_doppler(
                f64::from(measurements.velocities[index]),
                lead_velocity,
                max_velocity_mps,
            );
            let mut distance_squared =
                (point[0] - lead_point[0]).powi(2) + (point[1] - lead_point[1]).powi(2);
            if config.use_z() {
                distance_squared += (point[2] - lead_point[2]).powi(2);
            }
            distance_squared +=
                ((velocity - lead_velocity) * f64::from(config.velocity_scale_s())).powi(2);
            if distance_squared.sqrt() <= f64::from(config.eps_m()) {
                group.push(index);
                false
            } else {
                true
            }
        });
        groups.push(group);
    }
    groups
}

fn summarize(
    measurements: PointMeasurements<'_>,
    indices: &[usize],
    reference: [f64; 4],
    max_velocity_mps: f64,
) -> SphericalSummary {
    let count = indices.len() as f64;
    let mut centroid = [0.0; 4];
    let mut xyz_center = [0.0; 3];
    let mut values = Vec::with_capacity(indices.len());
    for &index in indices {
        let point = point_coordinate(measurements.coordinates, index);
        let measured = aligned_measurement(
            point_measurement(point, f64::from(measurements.velocities[index])),
            reference,
            max_velocity_mps,
        );
        for axis in 0..4 {
            centroid[axis] += measured[axis];
        }
        for axis in 0..3 {
            xyz_center[axis] += point[axis];
        }
        values.push((measured, point));
    }
    for value in &mut centroid {
        *value /= count;
    }
    for value in &mut xyz_center {
        *value /= count;
    }
    centroid[1] = wrap_angle(centroid[1]);

    let mut dispersion = [[0.0; 4]; 4];
    let mut extent = [[0.0; 3]; 3];
    for (measured, point) in values {
        let residual = [
            measured[0] - centroid[0],
            wrap_angle(measured[1] - centroid[1]),
            measured[2] - centroid[2],
            measured[3] - centroid[3],
        ];
        for row in 0..4 {
            for column in 0..4 {
                dispersion[row][column] += residual[row] * residual[column] / count;
            }
        }
        let delta = [
            point[0] - xyz_center[0],
            point[1] - xyz_center[1],
            point[2] - xyz_center[2],
        ];
        for row in 0..3 {
            for column in 0..3 {
                extent[row][column] += delta[row] * delta[column] / count;
            }
        }
    }
    SphericalSummary {
        centroid,
        dispersion,
        extent,
    }
}

fn centroid_covariance(
    point_noise: Matrix4,
    dispersion: Matrix4,
    point_count: usize,
    expected_points: f64,
) -> Matrix4 {
    let count = point_count as f64;
    let missing = (1.0 - count / expected_points.max(count)).max(0.0);
    matrix_add(
        scale(point_noise, count.recip()),
        scale(dispersion, missing.powi(2)),
    )
}

fn measurement_model(state: [f64; 6]) -> ([f64; 4], [[f64; 6]; 4]) {
    let [x, y, z, vx, vy, vz] = state;
    let horizontal_squared = (x * x + y * y).max(1e-12);
    let horizontal = horizontal_squared.sqrt();
    let range_squared = (horizontal_squared + z * z).max(1e-12);
    let range = range_squared.sqrt();
    let radial_velocity = (x * vx + y * vy + z * vz) / range;
    let jacobian = [
        [x / range, y / range, z / range, 0.0, 0.0, 0.0],
        [
            -y / horizontal_squared,
            x / horizontal_squared,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -x * z / (horizontal * range_squared),
            -y * z / (horizontal * range_squared),
            horizontal / range_squared,
            0.0,
            0.0,
            0.0,
        ],
        [
            vx / range - x * radial_velocity / range_squared,
            vy / range - y * radial_velocity / range_squared,
            vz / range - z * radial_velocity / range_squared,
            x / range,
            y / range,
            z / range,
        ],
    ];
    (
        [range, y.atan2(x), z.atan2(horizontal), radial_velocity],
        jacobian,
    )
}

fn point_measurement(point: [f64; 3], velocity: f64) -> [f64; 4] {
    let horizontal = point[0].hypot(point[1]);
    [
        horizontal.hypot(point[2]),
        point[1].atan2(point[0]),
        point[2].atan2(horizontal),
        velocity,
    ]
}

fn aligned_measurement(measured: [f64; 4], reference: [f64; 4], max_velocity_mps: f64) -> [f64; 4] {
    [
        measured[0],
        reference[1] + wrap_angle(measured[1] - reference[1]),
        measured[2],
        unwrap_doppler(measured[3], reference[3], max_velocity_mps),
    ]
}

fn spherical_position(measurement: [f64; 4]) -> [f64; 3] {
    let [range, azimuth, elevation, _] = measurement;
    let horizontal = range * elevation.cos();
    [
        horizontal * azimuth.cos(),
        horizontal * azimuth.sin(),
        range * elevation.sin(),
    ]
}

fn spherical_position_jacobian(range: f64, azimuth: f64, elevation: f64) -> Matrix3 {
    let (sin_azimuth, cos_azimuth) = azimuth.sin_cos();
    let (sin_elevation, cos_elevation) = elevation.sin_cos();
    [
        [
            cos_elevation * cos_azimuth,
            -range * cos_elevation * sin_azimuth,
            -range * sin_elevation * cos_azimuth,
        ],
        [
            cos_elevation * sin_azimuth,
            range * cos_elevation * cos_azimuth,
            -range * sin_elevation * sin_azimuth,
        ],
        [sin_elevation, 0.0, range * cos_elevation],
    ]
}

fn unwrap_doppler(measured: f64, reference: f64, max_velocity_mps: f64) -> f64 {
    let period = 2.0 * max_velocity_mps;
    measured + ((reference - measured) / period).round() * period
}

fn wrap_angle(value: f64) -> f64 {
    (value + PI).rem_euclid(2.0 * PI) - PI
}

fn distance3(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}

fn total_snr(measurements: PointMeasurements<'_>, indices: &[usize]) -> f64 {
    indices
        .iter()
        .map(|&index| f64::from(measurements.snrs[index]))
        .sum()
}

fn validate_measurements(measurements: PointMeasurements<'_>) -> Result<usize, TrackingError> {
    let point_count = measurements.velocities.len();
    let expected = point_count
        .checked_mul(3)
        .ok_or(TrackingError::PointMatrixLength {
            expected: usize::MAX,
            actual: measurements.coordinates.len(),
        })?;
    if measurements.coordinates.len() != expected {
        return Err(TrackingError::PointMatrixLength {
            expected,
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

fn inverse_spd<const SIZE: usize>(
    matrix: [[f64; SIZE]; SIZE],
) -> Result<([[f64; SIZE]; SIZE], f64), TrackingError> {
    let mut lower = [[0.0; SIZE]; SIZE];
    for row in 0..SIZE {
        for column in 0..=row {
            let mut value = matrix[row][column];
            for (&row_value, &column_value) in
                lower[row][..column].iter().zip(&lower[column][..column])
            {
                value -= row_value * column_value;
            }
            if row == column {
                if !value.is_finite() || value <= 1e-12 {
                    return Err(TrackingError::SingularInnovationCovariance);
                }
                lower[row][column] = value.sqrt();
            } else {
                lower[row][column] = value / lower[column][column];
            }
        }
    }
    let log_determinant = 2.0 * (0..SIZE).map(|axis| lower[axis][axis].ln()).sum::<f64>();
    let mut inverse_columns = [[0.0; SIZE]; SIZE];
    for (column, inverse_column) in inverse_columns.iter_mut().enumerate() {
        let mut forward = [0.0; SIZE];
        for row in 0..SIZE {
            let rhs = if row == column { 1.0 } else { 0.0 };
            let solved = (0..row)
                .map(|inner| lower[row][inner] * forward[inner])
                .sum::<f64>();
            forward[row] = (rhs - solved) / lower[row][row];
        }
        let mut backward = [0.0; SIZE];
        for row in (0..SIZE).rev() {
            let solved = ((row + 1)..SIZE)
                .map(|inner| lower[inner][row] * backward[inner])
                .sum::<f64>();
            backward[row] = (forward[row] - solved) / lower[row][row];
        }
        *inverse_column = backward;
    }
    let inverse = transpose(&inverse_columns);
    Ok((symmetrize(inverse), log_determinant))
}

fn dot<const SIZE: usize>(left: [f64; SIZE], right: [f64; SIZE]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn matrix_vector_product<const ROWS: usize, const COLUMNS: usize>(
    matrix: &[[f64; COLUMNS]; ROWS],
    vector: [f64; COLUMNS],
) -> [f64; ROWS] {
    let mut output = [0.0; ROWS];
    for (value, row) in output.iter_mut().zip(matrix) {
        *value = dot(*row, vector);
    }
    output
}

fn product<const ROWS: usize, const INNER: usize, const COLUMNS: usize>(
    left: &[[f64; INNER]; ROWS],
    right: &[[f64; COLUMNS]; INNER],
) -> [[f64; COLUMNS]; ROWS] {
    let mut output = [[0.0; COLUMNS]; ROWS];
    for (row_index, row) in left.iter().enumerate() {
        for column in 0..COLUMNS {
            output[row_index][column] = row
                .iter()
                .zip(right)
                .map(|(left, right_row)| left * right_row[column])
                .sum();
        }
    }
    output
}

fn transpose<const ROWS: usize, const COLUMNS: usize>(
    matrix: &[[f64; COLUMNS]; ROWS],
) -> [[f64; ROWS]; COLUMNS] {
    let mut output = [[0.0; ROWS]; COLUMNS];
    for (row, values) in matrix.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            output[column][row] = *value;
        }
    }
    output
}

fn matrix_add<const ROWS: usize, const COLUMNS: usize>(
    left: [[f64; COLUMNS]; ROWS],
    right: [[f64; COLUMNS]; ROWS],
) -> [[f64; COLUMNS]; ROWS] {
    combine(left, right, |left, right| left + right)
}

fn matrix_subtract<const ROWS: usize, const COLUMNS: usize>(
    left: [[f64; COLUMNS]; ROWS],
    right: [[f64; COLUMNS]; ROWS],
) -> [[f64; COLUMNS]; ROWS] {
    combine(left, right, |left, right| left - right)
}

fn combine<const ROWS: usize, const COLUMNS: usize>(
    left: [[f64; COLUMNS]; ROWS],
    right: [[f64; COLUMNS]; ROWS],
    operation: impl Fn(f64, f64) -> f64,
) -> [[f64; COLUMNS]; ROWS] {
    let mut output = [[0.0; COLUMNS]; ROWS];
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            output[row][column] = operation(left[row][column], right[row][column]);
        }
    }
    output
}

fn scale<const ROWS: usize, const COLUMNS: usize>(
    matrix: [[f64; COLUMNS]; ROWS],
    factor: f64,
) -> [[f64; COLUMNS]; ROWS] {
    let mut output = matrix;
    for row in &mut output {
        for value in row {
            *value *= factor;
        }
    }
    output
}

fn blend<const SIZE: usize>(
    previous: [[f64; SIZE]; SIZE],
    update: [[f64; SIZE]; SIZE],
    alpha: f64,
) -> [[f64; SIZE]; SIZE] {
    matrix_add(scale(previous, 1.0 - alpha), scale(update, alpha))
}

fn vector_subtract<const SIZE: usize>(left: [f64; SIZE], right: [f64; SIZE]) -> [f64; SIZE] {
    let mut output = [0.0; SIZE];
    for index in 0..SIZE {
        output[index] = left[index] - right[index];
    }
    output
}

fn identity<const SIZE: usize>() -> [[f64; SIZE]; SIZE] {
    let mut output = [[0.0; SIZE]; SIZE];
    for (axis, row) in output.iter_mut().enumerate() {
        row[axis] = 1.0;
    }
    output
}

fn diagonal<const SIZE: usize>(values: [f64; SIZE]) -> [[f64; SIZE]; SIZE] {
    let mut output = [[0.0; SIZE]; SIZE];
    for (axis, value) in values.into_iter().enumerate() {
        output[axis][axis] = value;
    }
    output
}

fn symmetrize<const SIZE: usize>(matrix: [[f64; SIZE]; SIZE]) -> [[f64; SIZE]; SIZE] {
    let mut output = matrix;
    for row in 0..SIZE {
        for column in 0..row {
            let value = 0.5 * (matrix[row][column] + matrix[column][row]);
            output[row][column] = value;
            output[column][row] = value;
        }
        output[row][row] = output[row][row].max(1e-12);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{GTrack3D, measurement_model};
    use crate::clustering::DbscanConfig;
    use crate::tracking::{
        NativeTrackStatus, PointMeasurements, TrackAllocationConfig, TrackGatingConfig,
        TrackLifecycleConfig, TrackScenery3DConfig, Tracker3DConfig, TrackerDynamics3DConfig,
    };

    fn config_with(gating: TrackGatingConfig, lifecycle: TrackLifecycleConfig) -> Tracker3DConfig {
        Tracker3DConfig::new(
            TrackerDynamics3DConfig::new(0.1, [2.0; 3], 0.1, 2.0, 0.2)
                .unwrap()
                .with_spherical_measurement(0.03, 0.04, 0.1, 4.0)
                .unwrap(),
            gating,
            TrackAllocationConfig::new(2, 0.0, None, None, None).unwrap(),
            lifecycle,
            TrackScenery3DConfig::new(Vec::new(), 5).unwrap(),
            8,
        )
        .unwrap()
    }

    fn config() -> Tracker3DConfig {
        config_with(
            TrackGatingConfig::new(0.8, Some(1.0), Some(5.0)).unwrap(),
            TrackLifecycleConfig::new(2, 2, 3, 1).unwrap(),
        )
    }

    fn tracker() -> GTrack3D {
        GTrack3D::new(config(), DbscanConfig::new(0.7, 2, 0.5, true).unwrap())
    }

    #[test]
    fn spherical_measurement_uses_forward_lateral_up_axes() {
        let (forward, _) = measurement_model([2.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(forward, [2.0, 0.0, 0.0, 1.0]);

        let (lateral, _) = measurement_model([0.0, 2.0, 0.0, 0.0, 1.0, 0.0]);
        assert!((lateral[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(lateral[3], 1.0);

        let (up, _) = measurement_model([2.0, 0.0, 2.0, 0.0, 0.0, 1.0]);
        assert!((up[2] - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((up[3] - 2.0_f64.sqrt() / 2.0).abs() < 1e-12);
    }

    #[test]
    fn spherical_measurement_jacobian_matches_finite_differences() {
        let state = [3.0, 0.4, -0.6, 0.7, -0.2, 0.1];
        let (_, jacobian) = measurement_model(state);
        let epsilon = 1e-6;
        for column in 0..6 {
            let mut lower = state;
            let mut upper = state;
            lower[column] -= epsilon;
            upper[column] += epsilon;
            let (lower, _) = measurement_model(lower);
            let (upper, _) = measurement_model(upper);
            for row in 0..4 {
                let numerical = (upper[row] - lower[row]) / (2.0 * epsilon);
                assert!((jacobian[row][column] - numerical).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn persistent_3d_group_confirms_one_unit() {
        let mut tracker = tracker();
        let first = tracker
            .step(PointMeasurements {
                coordinates: &[1.95, -0.05, 0.75, 2.05, 0.05, 1.25],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        assert_eq!(first.statuses, [NativeTrackStatus::Tentative.code()]);

        let second = tracker
            .step(PointMeasurements {
                coordinates: &[1.97, -0.05, 0.77, 2.07, 0.05, 1.27],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        assert_eq!(second.track_ids, [0]);
        assert_eq!(second.observation_track_ids, [0, 0]);
        assert_eq!(second.statuses, [NativeTrackStatus::Confirmed.code()]);
        assert_eq!(second.position_covariances.len(), 9);
        assert_eq!(second.extent_covariances.len(), 9);
        assert!(second.positions[2] > 0.7 && second.positions[2] < 1.3);
        assert!(
            second
                .position_covariances
                .iter()
                .all(|value| value.is_finite())
        );
        let diagnostics = tracker.diagnostics();
        assert_eq!(diagnostics.frames, 2);
        assert_eq!(diagnostics.points, 4);
        assert_eq!(diagnostics.allocations, 1);
        assert_eq!(diagnostics.confirmations, 1);
        assert_eq!(diagnostics.empty_updates, 0);
    }

    #[test]
    fn diagnostics_expose_association_loss_before_rebirth() {
        let mut tracker = tracker();
        for coordinates in [
            [1.95, -0.05, 0.75, 2.05, 0.05, 1.25],
            [3.45, -0.05, 0.75, 3.55, 0.05, 1.25],
        ] {
            tracker
                .step(PointMeasurements {
                    coordinates: &coordinates,
                    velocities: &[0.2, 0.2],
                    snrs: &[10.0, 10.0],
                })
                .unwrap();
        }

        let diagnostics = tracker.diagnostics();
        assert_eq!(diagnostics.distance_gate_misses, 2);
        assert_eq!(diagnostics.empty_updates, 1);
        assert_eq!(diagnostics.allocations, 2);
        assert_eq!(diagnostics.confirmations, 0);
    }

    #[test]
    fn distance_gate_bounds_a_confirmed_unit() {
        let config = config_with(
            TrackGatingConfig::new(0.2, Some(1.0), Some(100.0)).unwrap(),
            TrackLifecycleConfig::new(1, 2, 3, 2).unwrap(),
        );
        let mut tracker = GTrack3D::new(config, DbscanConfig::new(0.7, 2, 0.5, true).unwrap());
        tracker
            .step(PointMeasurements {
                coordinates: &[1.95, -0.05, 0.75, 2.05, 0.05, 1.25],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        let frame = tracker
            .step(PointMeasurements {
                coordinates: &[2.45, -0.05, 0.75, 2.55, 0.05, 1.25],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();

        assert_eq!(frame.track_ids, [0, 1]);
        assert_eq!(frame.observation_track_ids, [1, 1]);
        assert_eq!(
            frame.statuses,
            [
                NativeTrackStatus::Coasting.code(),
                NativeTrackStatus::Confirmed.code()
            ]
        );
        assert_eq!(tracker.diagnostics().distance_gate_misses, 2);
        assert_eq!(tracker.diagnostics().allocations, 2);
    }

    #[test]
    fn one_measurement_does_not_keep_a_confirmed_unit_alive() {
        let config = config_with(
            TrackGatingConfig::new(0.8, Some(1.0), Some(5.0)).unwrap(),
            TrackLifecycleConfig::new(2, 2, 3, 2).unwrap(),
        );
        let mut tracker = GTrack3D::new(config, DbscanConfig::new(0.7, 2, 0.5, true).unwrap());
        for coordinates in [
            [1.95, -0.05, 0.75, 2.05, 0.05, 1.25],
            [1.97, -0.05, 0.77, 2.07, 0.05, 1.27],
        ] {
            tracker
                .step(PointMeasurements {
                    coordinates: &coordinates,
                    velocities: &[0.2, 0.2],
                    snrs: &[10.0, 10.0],
                })
                .unwrap();
        }
        let frame = tracker
            .step(PointMeasurements {
                coordinates: &[2.20, 0.0, 1.0],
                velocities: &[0.2],
                snrs: &[10.0],
            })
            .unwrap();

        assert_eq!(frame.observation_track_ids, [0]);
        assert_eq!(frame.statuses, [NativeTrackStatus::Coasting.code()]);
        assert_eq!(frame.missed_counts, [1]);
        assert_eq!(tracker.diagnostics().partial_updates, 1);
        assert_eq!(tracker.diagnostics().reactivations, 0);
    }

    #[test]
    fn one_measurement_does_not_reactivate_a_coasting_unit() {
        let config = config_with(
            TrackGatingConfig::new(0.8, Some(1.0), Some(5.0)).unwrap(),
            TrackLifecycleConfig::new(1, 2, 3, 2).unwrap(),
        );
        let mut tracker = GTrack3D::new(config, DbscanConfig::new(0.7, 2, 0.5, true).unwrap());
        tracker
            .step(PointMeasurements {
                coordinates: &[1.95, -0.05, 0.75, 2.05, 0.05, 1.25],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        tracker
            .step(PointMeasurements {
                coordinates: &[],
                velocities: &[],
                snrs: &[],
            })
            .unwrap();
        let frame = tracker
            .step(PointMeasurements {
                coordinates: &[2.02, 0.0, 1.0],
                velocities: &[0.2],
                snrs: &[10.0],
            })
            .unwrap();

        assert_eq!(frame.observation_track_ids, [0]);
        assert_eq!(frame.statuses, [NativeTrackStatus::Coasting.code()]);
        assert_eq!(frame.missed_counts, [2]);
        assert_eq!(tracker.diagnostics().partial_updates, 1);
        assert_eq!(tracker.diagnostics().reactivations, 0);
    }

    #[test]
    fn static_group_does_not_reactivate_a_coasting_unit() {
        let lifecycle = TrackLifecycleConfig::new(1, 2, 5, 2)
            .unwrap()
            .with_scene_miss_limits(Some(5), None, 0.15)
            .unwrap();
        let config = config_with(
            TrackGatingConfig::new(0.8, Some(1.0), Some(5.0)).unwrap(),
            lifecycle,
        );
        let mut tracker = GTrack3D::new(config, DbscanConfig::new(0.7, 2, 0.5, true).unwrap());
        tracker
            .step(PointMeasurements {
                coordinates: &[1.95, -0.05, 0.75, 2.05, 0.05, 1.25],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        tracker
            .step(PointMeasurements {
                coordinates: &[],
                velocities: &[],
                snrs: &[],
            })
            .unwrap();
        let static_frame = tracker
            .step(PointMeasurements {
                coordinates: &[1.97, -0.05, 0.77, 2.07, 0.05, 1.27],
                velocities: &[0.05, 0.05],
                snrs: &[10.0, 10.0],
            })
            .unwrap();

        assert_eq!(static_frame.statuses, [NativeTrackStatus::Coasting.code()]);
        assert_eq!(static_frame.missed_counts, [2]);
        assert_eq!(tracker.diagnostics().partial_updates, 1);
        assert_eq!(tracker.diagnostics().reactivations, 0);

        let moving_frame = tracker
            .step(PointMeasurements {
                coordinates: &[1.99, -0.05, 0.79, 2.09, 0.05, 1.29],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        assert_eq!(moving_frame.statuses, [NativeTrackStatus::Confirmed.code()]);
        assert_eq!(moving_frame.missed_counts, [0]);
        assert_eq!(tracker.diagnostics().reactivations, 1);
    }

    #[test]
    fn one_measurement_does_not_confirm_a_tentative_unit() {
        let config = config_with(
            TrackGatingConfig::new(0.8, Some(1.0), Some(5.0)).unwrap(),
            TrackLifecycleConfig::new(2, 2, 3, 2).unwrap(),
        );
        let mut tracker = GTrack3D::new(config, DbscanConfig::new(0.7, 2, 0.5, true).unwrap());
        tracker
            .step(PointMeasurements {
                coordinates: &[1.95, -0.05, 0.75, 2.05, 0.05, 1.25],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        let frame = tracker
            .step(PointMeasurements {
                coordinates: &[2.02, 0.0, 1.0],
                velocities: &[0.2],
                snrs: &[10.0],
            })
            .unwrap();

        assert_eq!(frame.observation_track_ids, [-1]);
        assert_eq!(frame.statuses, [NativeTrackStatus::Tentative.code()]);
        assert_eq!(frame.missed_counts, [1]);
        assert_eq!(tracker.diagnostics().partial_updates, 1);
        assert_eq!(tracker.diagnostics().confirmations, 0);
    }
}
