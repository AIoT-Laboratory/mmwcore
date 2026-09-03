//! TI GTRACK-style 2D extended-target tracking for radar point measurements.

use std::cmp::Ordering;
use std::f64::consts::PI;

use crate::clustering::DbscanConfig;

use super::{
    NativeTrackStatus, PointMeasurements, TrackStepResult, Tracker2DConfig, TrackingError,
};

type Matrix2 = [[f64; 2]; 2];
type Matrix3 = [[f64; 3]; 3];
type Matrix4 = [[f64; 4]; 4];

#[derive(Clone, Debug)]
struct Unit {
    id: i64,
    state: [f64; 4],
    covariance: Matrix4,
    dispersion: Matrix3,
    extent: Matrix2,
    expected_points: f64,
    z: f64,
    status: NativeTrackStatus,
    hits: usize,
    age: usize,
    missed: usize,
    outside: usize,
}

#[derive(Clone, Copy, Debug)]
struct PolarSummary {
    centroid: [f64; 3],
    dispersion: Matrix3,
    extent: Matrix2,
    z: f64,
}

#[derive(Clone, Debug)]
struct GTrackFilter {
    transition: Matrix4,
    process_noise: Matrix4,
    point_noise: Matrix3,
    initial_velocity_variance: f64,
    max_velocity_mps: f64,
}

impl GTrackFilter {
    fn new(config: &Tracker2DConfig) -> Self {
        let dynamics = config.dynamics;
        let dt = dynamics.frame_period_s;
        let transition = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let gain = [
            [0.5 * dt.powi(2), 0.0],
            [0.0, 0.5 * dt.powi(2)],
            [dt, 0.0],
            [0.0, dt],
        ];
        let acceleration = [
            [dynamics.max_acceleration_mps2[0].powi(2), 0.0],
            [0.0, dynamics.max_acceleration_mps2[1].powi(2)],
        ];
        Self {
            transition,
            process_noise: product(&product(&gain, &acceleration), &transpose(&gain)),
            point_noise: [
                [dynamics.measurement_noise_m.powi(2), 0.0, 0.0],
                [0.0, dynamics.angle_noise_rad.powi(2), 0.0],
                [0.0, 0.0, dynamics.doppler_noise_mps.powi(2)],
            ],
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

    fn association(&self, unit: &Unit, measurement: [f64; 3]) -> Result<(f64, f64), TrackingError> {
        let (predicted, jacobian) = measurement_model(unit.state);
        let measurement = [
            measurement[0],
            predicted[1] + wrap_angle(measurement[1] - predicted[1]),
            unwrap_doppler(measurement[2], predicted[2], self.max_velocity_mps),
        ];
        let residual = vector_subtract(measurement, predicted);
        let covariance = matrix_add(
            matrix_add(
                product(&product(&jacobian, &unit.covariance), &transpose(&jacobian)),
                self.point_noise,
            ),
            unit.dispersion,
        );
        let (inverse, determinant) = inverse_3x3(covariance)?;
        let solved = matrix_vector_product(&inverse, residual);
        let distance_squared = dot(residual, solved).max(0.0);
        Ok((distance_squared.sqrt(), determinant.ln() + distance_squared))
    }

    fn update(
        &self,
        unit: &mut Unit,
        measurement: [f64; 3],
        measurement_covariance: Matrix3,
    ) -> Result<(), TrackingError> {
        let (predicted, jacobian) = measurement_model(unit.state);
        let measurement = [
            measurement[0],
            predicted[1] + wrap_angle(measurement[1] - predicted[1]),
            unwrap_doppler(measurement[2], predicted[2], self.max_velocity_mps),
        ];
        let residual = vector_subtract(measurement, predicted);
        let innovation_covariance = matrix_add(
            product(&product(&jacobian, &unit.covariance), &transpose(&jacobian)),
            measurement_covariance,
        );
        let (inverse, _) = inverse_3x3(innovation_covariance)?;
        let cross_covariance = product(&unit.covariance, &transpose(&jacobian));
        let gain = product(&cross_covariance, &inverse);
        let correction = matrix_vector_product(&gain, residual);
        for (state, correction) in unit.state.iter_mut().zip(correction) {
            *state += correction;
        }

        let residual_gain = matrix_subtract(identity_4x4(), product(&gain, &jacobian));
        let posterior = product(
            &product(&residual_gain, &unit.covariance),
            &transpose(&residual_gain),
        );
        let measurement_term = product(&product(&gain, &measurement_covariance), &transpose(&gain));
        unit.covariance = symmetrize(matrix_add(posterior, measurement_term));
        Ok(())
    }

    fn allocate(&self, id: i64, summary: PolarSummary, confirmed: bool, points: usize) -> Unit {
        let range = summary.centroid[0];
        let angle = summary.centroid[1];
        let radial_velocity = summary.centroid[2];
        let cosine = angle.cos();
        let sine = angle.sin();
        let polar_position_covariance =
            [[self.point_noise[0][0], 0.0], [0.0, self.point_noise[1][1]]];
        let position_jacobian = [[cosine, -range * sine], [sine, range * cosine]];
        let position_covariance = product(
            &product(&position_jacobian, &polar_position_covariance),
            &transpose(&position_jacobian),
        );
        Unit {
            id,
            state: [
                range * cosine,
                range * sine,
                radial_velocity * cosine,
                radial_velocity * sine,
            ],
            covariance: [
                [
                    position_covariance[0][0],
                    position_covariance[0][1],
                    0.0,
                    0.0,
                ],
                [
                    position_covariance[1][0],
                    position_covariance[1][1],
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, self.initial_velocity_variance, 0.0],
                [0.0, 0.0, 0.0, self.initial_velocity_variance],
            ],
            dispersion: summary.dispersion,
            extent: summary.extent,
            expected_points: points as f64,
            z: summary.z,
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

/// Stateful 2D GTRACK implementation over Cartesian points with radial-Doppler measurements.
#[derive(Clone, Debug)]
pub struct GTrack2D {
    config: Tracker2DConfig,
    allocation: DbscanConfig,
    filter: GTrackFilter,
    units: Vec<Unit>,
    next_id: i64,
}

impl GTrack2D {
    /// Construct a tracker. The clustering policy supplies allocation proximity, not DBSCAN births.
    pub fn new(config: Tracker2DConfig, allocation: DbscanConfig) -> Self {
        let filter = GTrackFilter::new(&config);
        Self {
            config,
            allocation,
            filter,
            units: Vec::new(),
            next_id: 0,
        }
    }

    /// Predict, bid for points, update extended-target units, allocate, and manage lifecycles.
    pub fn step(
        &mut self,
        measurements: PointMeasurements<'_>,
    ) -> Result<TrackStepResult, TrackingError> {
        let point_count = validate_measurements(measurements)?;
        for unit in &mut self.units {
            self.filter.predict(unit);
        }

        let mut associations = self.associate(measurements, point_count)?;
        let observed = self.update_units(measurements, &mut associations)?;
        self.miss_unobserved(&observed);
        let expired = self.delete_expired();
        for association in &mut associations {
            if expired.contains(association) {
                *association = -1;
            }
        }
        self.allocate(measurements, &mut associations)?;
        Ok(self.report(associations))
    }

    fn associate(
        &self,
        measurements: PointMeasurements<'_>,
        point_count: usize,
    ) -> Result<Vec<i64>, TrackingError> {
        let mut associations = vec![-1; point_count];
        let mahalanobis_limit = self.config.gating.max_mahalanobis_distance.unwrap_or(3.0);
        for (index, association) in associations.iter_mut().enumerate() {
            let point = point_coordinate(measurements.coordinates, index);
            if !self.config.scenery.contains(point[0], point[1]) {
                continue;
            }
            let measurement = point_measurement(point, f64::from(measurements.velocities[index]));
            let mut best: Option<(i64, f64)> = None;
            for unit in &self.units {
                let distance = (unit.state[0] - point[0]).hypot(unit.state[1] - point[1]);
                if distance > self.config.gating.max_distance_m {
                    continue;
                }
                let expected_velocity = measurement_model(unit.state).0[2];
                let velocity = unwrap_doppler(
                    measurement[2],
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
            }
        }
        Ok(associations)
    }

    fn update_units(
        &mut self,
        measurements: PointMeasurements<'_>,
        associations: &mut [i64],
    ) -> Result<Vec<usize>, TrackingError> {
        let mut observed = Vec::new();
        for unit_index in 0..self.units.len() {
            let id = self.units[unit_index].id;
            let indices = associations
                .iter()
                .enumerate()
                .filter_map(|(index, &association)| (association == id).then_some(index))
                .collect::<Vec<_>>();
            if indices.len() < self.config.lifecycle.min_update_points {
                for index in indices {
                    associations[index] = -1;
                }
                continue;
            }

            let reference = measurement_model(self.units[unit_index].state).0;
            let summary = summarize(
                measurements,
                &indices,
                reference,
                self.filter.max_velocity_mps,
            );
            let alpha = self.config.dynamics.extent_covariance_smoothing;
            let unit = &mut self.units[unit_index];
            unit.dispersion = blend_3x3(unit.dispersion, summary.dispersion, alpha);
            if indices.len() > 1 {
                unit.extent = blend_2x2(unit.extent, summary.extent, alpha);
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
            unit.z = (1.0 - alpha) * unit.z + alpha * summary.z;
            observe(&self.config, unit);
            observed.push(unit_index);
        }
        Ok(observed)
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

    fn delete_expired(&mut self) -> Vec<i64> {
        let config = &self.config;
        let mut expired = Vec::new();
        self.units.retain(|unit| {
            let miss_limit = coast_limit(config, unit);
            let delete = (unit.status == NativeTrackStatus::Tentative
                && unit.missed >= config.lifecycle.tentative_max_misses)
                || (unit.status == NativeTrackStatus::Coasting && unit.missed >= miss_limit)
                || unit.outside >= config.scenery.outside_max_frames;
            if delete {
                expired.push(unit.id);
            }
            !delete
        });
        expired
    }

    fn allocate(
        &mut self,
        measurements: PointMeasurements<'_>,
        associations: &mut [i64],
    ) -> Result<(), TrackingError> {
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
            let [x, y] = polar_position(summary.centroid);
            if !self.can_allocate(
                [x, y],
                summary.centroid[2],
                total_snr(measurements, &indices),
            )? {
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
            for index in indices {
                associations[index] = id;
            }
            allocated += 1;
        }
        Ok(())
    }

    fn can_allocate(
        &self,
        center: [f64; 2],
        radial_velocity: f64,
        snr: f64,
    ) -> Result<bool, TrackingError> {
        let allocation = self.config.allocation;
        let snr_sufficient = match allocation.min_total_snr {
            Some(threshold) => snr >= threshold,
            None => true,
        };
        let separated = allocation.min_separation_m.is_none_or(|minimum| {
            self.units
                .iter()
                .all(|unit| (unit.state[0] - center[0]).hypot(unit.state[1] - center[1]) >= minimum)
        });
        Ok(self.config.scenery.contains(center[0], center[1])
            && radial_velocity.abs() >= allocation.min_abs_radial_velocity_mps
            && snr_sufficient
            && separated)
    }

    fn report(&self, observation_track_ids: Vec<i64>) -> TrackStepResult {
        let mut result = TrackStepResult {
            track_ids: Vec::with_capacity(self.units.len()),
            positions: Vec::with_capacity(self.units.len() * 3),
            velocities: Vec::with_capacity(self.units.len() * 3),
            position_covariances: Vec::with_capacity(self.units.len() * 4),
            extent_covariances: Vec::with_capacity(self.units.len() * 4),
            statuses: Vec::with_capacity(self.units.len()),
            ages: Vec::with_capacity(self.units.len()),
            missed_counts: Vec::with_capacity(self.units.len()),
            observation_track_ids,
        };
        for unit in &self.units {
            result.track_ids.push(unit.id);
            result
                .positions
                .extend([unit.state[0] as f32, unit.state[1] as f32, unit.z as f32]);
            result
                .velocities
                .extend([unit.state[2] as f32, unit.state[3] as f32, 0.0]);
            result.position_covariances.extend([
                unit.covariance[0][0] as f32,
                unit.covariance[0][1] as f32,
                unit.covariance[1][0] as f32,
                unit.covariance[1][1] as f32,
            ]);
            result.extent_covariances.extend([
                unit.extent[0][0] as f32,
                unit.extent[0][1] as f32,
                unit.extent[1][0] as f32,
                unit.extent[1][1] as f32,
            ]);
            result.statuses.push(unit.status.code());
            result.ages.push(unit.age as i64);
            result.missed_counts.push(unit.missed as i64);
        }
        result
    }
}

/// Compatibility alias for the earlier generic measurement-tracker name.
pub type PointTracker2D = GTrack2D;

fn observe(config: &Tracker2DConfig, unit: &mut Unit) {
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

fn update_outside(config: &Tracker2DConfig, unit: &mut Unit) {
    if config.scenery.contains(unit.state[0], unit.state[1]) {
        unit.outside = 0;
    } else {
        unit.outside += 1;
    }
}

fn coast_limit(config: &Tracker2DConfig, unit: &Unit) -> usize {
    let lifecycle = config.lifecycle;
    if let Some(limit) = lifecycle.static_max_misses
        && config.scenery.contains_static(unit.state[0], unit.state[1])
        && unit.state[2].hypot(unit.state[3]) <= lifecycle.static_speed_threshold_mps
    {
        return limit;
    }
    if let Some(limit) = lifecycle.exit_max_misses
        && !config.scenery.contains(unit.state[0], unit.state[1])
    {
        return limit;
    }
    lifecycle.confirmed_max_misses
}

fn allocation_groups(
    measurements: PointMeasurements<'_>,
    associations: &[i64],
    config: DbscanConfig,
    tracker: &Tracker2DConfig,
    max_velocity_mps: f64,
) -> Vec<Vec<usize>> {
    let mut remaining = associations
        .iter()
        .enumerate()
        .filter_map(|(index, &association)| {
            let point = point_coordinate(measurements.coordinates, index);
            (association < 0 && tracker.scenery.contains(point[0], point[1])).then_some(index)
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
    reference: [f64; 3],
    max_velocity_mps: f64,
) -> PolarSummary {
    let count = indices.len() as f64;
    let mut centroid = [0.0; 3];
    let mut xyz_center = [0.0; 3];
    let mut values = Vec::with_capacity(indices.len());
    for &index in indices {
        let point = point_coordinate(measurements.coordinates, index);
        let mut measurement = point_measurement(point, f64::from(measurements.velocities[index]));
        measurement[1] = reference[1] + wrap_angle(measurement[1] - reference[1]);
        measurement[2] = unwrap_doppler(measurement[2], reference[2], max_velocity_mps);
        for axis in 0..3 {
            centroid[axis] += measurement[axis];
            xyz_center[axis] += point[axis];
        }
        values.push((measurement, point));
    }
    for axis in 0..3 {
        centroid[axis] /= count;
        xyz_center[axis] /= count;
    }
    centroid[1] = wrap_angle(centroid[1]);

    let mut dispersion = [[0.0; 3]; 3];
    let mut extent = [[0.0; 2]; 2];
    for (measurement, point) in values {
        let residual = [
            measurement[0] - centroid[0],
            wrap_angle(measurement[1] - centroid[1]),
            measurement[2] - centroid[2],
        ];
        for row in 0..3 {
            for column in 0..3 {
                dispersion[row][column] += residual[row] * residual[column] / count;
            }
        }
        let dx = point[0] - xyz_center[0];
        let dy = point[1] - xyz_center[1];
        extent[0][0] += dx * dx / count;
        extent[0][1] += dx * dy / count;
        extent[1][0] += dy * dx / count;
        extent[1][1] += dy * dy / count;
    }
    PolarSummary {
        centroid,
        dispersion,
        extent,
        z: xyz_center[2],
    }
}

fn centroid_covariance(
    point_noise: Matrix3,
    dispersion: Matrix3,
    point_count: usize,
    expected_points: f64,
) -> Matrix3 {
    let count = point_count as f64;
    let missing = (1.0 - count / expected_points.max(count)).max(0.0);
    matrix_add(
        scale(point_noise, count.recip()),
        scale(dispersion, missing.powi(2)),
    )
}

fn measurement_model(state: [f64; 4]) -> ([f64; 3], [[f64; 4]; 3]) {
    let [x, y, vx, vy] = state;
    let range_squared = (x * x + y * y).max(1e-12);
    let range = range_squared.sqrt();
    let radial_velocity = (x * vx + y * vy) / range;
    let jacobian = [
        [x / range, y / range, 0.0, 0.0],
        [-y / range_squared, x / range_squared, 0.0, 0.0],
        [
            vx / range - x * radial_velocity / range_squared,
            vy / range - y * radial_velocity / range_squared,
            x / range,
            y / range,
        ],
    ];
    ([range, y.atan2(x), radial_velocity], jacobian)
}

fn point_measurement(point: [f64; 3], velocity: f64) -> [f64; 3] {
    [point[0].hypot(point[1]), point[1].atan2(point[0]), velocity]
}

fn polar_position(measurement: [f64; 3]) -> [f64; 2] {
    [
        measurement[0] * measurement[1].cos(),
        measurement[0] * measurement[1].sin(),
    ]
}

fn unwrap_doppler(measured: f64, reference: f64, max_velocity_mps: f64) -> f64 {
    let period = 2.0 * max_velocity_mps;
    measured + ((reference - measured) / period).round() * period
}

fn wrap_angle(value: f64) -> f64 {
    (value + PI).rem_euclid(2.0 * PI) - PI
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

fn inverse_3x3(matrix: Matrix3) -> Result<(Matrix3, f64), TrackingError> {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if !determinant.is_finite() || determinant <= f64::EPSILON {
        return Err(TrackingError::SingularInnovationCovariance);
    }
    let inverse_determinant = determinant.recip();
    let inverse = [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inverse_determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inverse_determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inverse_determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inverse_determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inverse_determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inverse_determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inverse_determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inverse_determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inverse_determinant,
        ],
    ];
    Ok((inverse, determinant))
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

fn vector_subtract<const SIZE: usize>(left: [f64; SIZE], right: [f64; SIZE]) -> [f64; SIZE] {
    let mut output = [0.0; SIZE];
    for index in 0..SIZE {
        output[index] = left[index] - right[index];
    }
    output
}

fn identity_4x4() -> Matrix4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn symmetrize(matrix: Matrix4) -> Matrix4 {
    let mut output = matrix;
    for row in 0..4 {
        for column in 0..row {
            let value = 0.5 * (matrix[row][column] + matrix[column][row]);
            output[row][column] = value;
            output[column][row] = value;
        }
        output[row][row] = output[row][row].max(1e-12);
    }
    output
}

fn blend_2x2(previous: Matrix2, update: Matrix2, alpha: f64) -> Matrix2 {
    matrix_add(scale(previous, 1.0 - alpha), scale(update, alpha))
}

fn blend_3x3(previous: Matrix3, update: Matrix3, alpha: f64) -> Matrix3 {
    matrix_add(scale(previous, 1.0 - alpha), scale(update, alpha))
}

#[cfg(test)]
mod tests {
    use super::{GTrack2D, GTrackFilter, measurement_model, unwrap_doppler};
    use crate::clustering::DbscanConfig;
    use crate::tracking::{
        NativeTrackStatus, PointMeasurements, TrackAllocationConfig, TrackGatingConfig,
        TrackLifecycleConfig, TrackSceneryConfig, Tracker2DConfig, TrackerDynamicsConfig,
    };

    fn config() -> Tracker2DConfig {
        Tracker2DConfig::new(
            TrackerDynamicsConfig::new(0.1, [2.0, 2.0], 0.1, 2.0, 0.2)
                .unwrap()
                .with_polar_measurement(0.03, 0.1, 4.0)
                .unwrap(),
            TrackGatingConfig::new(0.8, Some(1.0), Some(4.0)).unwrap(),
            TrackAllocationConfig::new(2, 0.0, None, None, None).unwrap(),
            TrackLifecycleConfig::new(2, 2, 3, 1).unwrap(),
            TrackSceneryConfig::new(Vec::new(), 5).unwrap(),
            8,
        )
        .unwrap()
    }

    fn tracker() -> GTrack2D {
        GTrack2D::new(config(), DbscanConfig::new(0.25, 2, 0.5, false).unwrap())
    }

    #[test]
    fn measurement_model_uses_forward_x_and_lateral_y() {
        let (measurement, _) = measurement_model([2.0, 0.0, 1.0, 0.0]);
        assert_eq!(measurement, [2.0, 0.0, 1.0]);

        let (measurement, _) = measurement_model([0.0, 2.0, 0.0, 1.0]);
        assert!((measurement[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(measurement[2], 1.0);
    }

    #[test]
    fn doppler_is_unwrapped_nearest_the_prediction() {
        assert!((unwrap_doppler(-3.8, 4.1, 4.0) - 4.2).abs() < 1e-12);
        assert!((unwrap_doppler(3.8, -4.1, 4.0) + 4.2).abs() < 1e-12);
    }

    #[test]
    fn persistent_group_confirms_one_unit() {
        let mut tracker = tracker();
        let first = tracker
            .step(PointMeasurements {
                coordinates: &[1.95, -0.05, 0.8, 2.05, 0.05, 1.2],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        assert_eq!(first.statuses, [NativeTrackStatus::Tentative.code()]);

        let second = tracker
            .step(PointMeasurements {
                coordinates: &[1.97, -0.05, 0.8, 2.07, 0.05, 1.2],
                velocities: &[0.2, 0.2],
                snrs: &[10.0, 10.0],
            })
            .unwrap();
        assert_eq!(second.track_ids, [0]);
        assert_eq!(second.observation_track_ids, [0, 0]);
        assert_eq!(second.statuses, [NativeTrackStatus::Confirmed.code()]);
        assert!(second.velocities[0] > 0.0);
    }

    #[test]
    fn competing_units_assign_each_point_to_one_bidder() {
        let mut tracker = tracker();
        tracker
            .step(PointMeasurements {
                coordinates: &[
                    2.0, -1.05, 0.0, 2.0, -0.95, 0.0, 2.0, 0.95, 0.0, 2.0, 1.05, 0.0,
                ],
                velocities: &[0.0; 4],
                snrs: &[1.0; 4],
            })
            .unwrap();
        let result = tracker
            .step(PointMeasurements {
                coordinates: &[2.0, -0.9, 0.0, 2.0, -0.8, 0.0, 2.0, 0.8, 0.0, 2.0, 0.9, 0.0],
                velocities: &[0.0; 4],
                snrs: &[1.0; 4],
            })
            .unwrap();
        assert_eq!(result.track_ids, [0, 1]);
        assert_eq!(result.observation_track_ids, [0, 0, 1, 1]);
    }

    #[test]
    fn polar_ekf_covariance_stays_finite() {
        let filter = GTrackFilter::new(&config());
        let mut tracker = tracker();
        tracker
            .step(PointMeasurements {
                coordinates: &[2.0, -0.05, 0.0, 2.0, 0.05, 0.0],
                velocities: &[0.3, 0.3],
                snrs: &[1.0, 1.0],
            })
            .unwrap();
        let unit = &mut tracker.units[0];
        filter.predict(unit);
        let association = filter.association(unit, [2.04, 0.0, 0.3]).unwrap();
        assert!(association.0.is_finite());
        assert!(association.1.is_finite());
    }
}
