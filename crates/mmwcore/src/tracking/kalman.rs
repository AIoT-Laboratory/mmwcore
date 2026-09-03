//! Constant-velocity state estimation used by native target trackers.

use super::{NativeTrackStatus, TrackerDynamicsConfig, TrackingError};

#[derive(Clone, Debug)]
pub(crate) struct CvTrackState {
    pub(crate) track_id: i64,
    pub(crate) state: [f64; 4],
    pub(crate) covariance: [[f64; 4]; 4],
    pub(crate) extent_covariance: [[f64; 2]; 2],
    pub(crate) z: f64,
    pub(crate) status: NativeTrackStatus,
    pub(crate) hits: usize,
    pub(crate) age: usize,
    pub(crate) missed: usize,
    pub(crate) outside: usize,
}

/// Cartesian `[x, y, vx, vy]` constant-velocity filter.
#[derive(Clone, Debug)]
pub(crate) struct ConstantVelocity2DFilter {
    transition: [[f64; 4]; 4],
    process_noise: [[f64; 4]; 4],
    measurement_noise: [[f64; 2]; 2],
    initial_velocity_variance: f64,
}

impl ConstantVelocity2DFilter {
    pub(crate) fn new(config: TrackerDynamicsConfig) -> Self {
        let dt = config.frame_period_s;
        let transition = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let noise_gain = [
            [0.5 * dt.powi(2), 0.0],
            [0.0, 0.5 * dt.powi(2)],
            [dt, 0.0],
            [0.0, dt],
        ];
        let acceleration_variance = [
            [config.max_acceleration_mps2[0].powi(2), 0.0],
            [0.0, config.max_acceleration_mps2[1].powi(2)],
        ];
        let process_noise = product(
            &product(&noise_gain, &acceleration_variance),
            &transpose(&noise_gain),
        );
        let measurement_variance = config.measurement_noise_m.powi(2);
        Self {
            transition,
            process_noise,
            measurement_noise: [[measurement_variance, 0.0], [0.0, measurement_variance]],
            initial_velocity_variance: config.initial_velocity_std_mps.powi(2),
        }
    }

    pub(crate) fn allocate(
        &self,
        track_id: i64,
        center: [f64; 3],
        extent_covariance: [[f64; 2]; 2],
        confirmed: bool,
    ) -> CvTrackState {
        let position_variance = self.measurement_noise[0][0];
        CvTrackState {
            track_id,
            state: [center[0], center[1], 0.0, 0.0],
            covariance: [
                [position_variance, 0.0, 0.0, 0.0],
                [0.0, position_variance, 0.0, 0.0],
                [0.0, 0.0, self.initial_velocity_variance, 0.0],
                [0.0, 0.0, 0.0, self.initial_velocity_variance],
            ],
            extent_covariance,
            z: center[2],
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

    pub(crate) fn predict(&self, track: &mut CvTrackState) {
        track.state = matrix_vector_product(&self.transition, track.state);
        track.covariance = matrix_add(
            product(
                &product(&self.transition, &track.covariance),
                &transpose(&self.transition),
            ),
            self.process_noise,
        );
    }

    pub(crate) fn update(
        &self,
        track: &mut CvTrackState,
        measurement: [f64; 2],
        centroid_covariance: [[f64; 2]; 2],
    ) -> Result<(), TrackingError> {
        let innovation = [
            measurement[0] - track.state[0],
            measurement[1] - track.state[1],
        ];
        let measurement_covariance = matrix_add(self.measurement_noise, centroid_covariance);
        let innovation_covariance =
            matrix_add(top_left_2x2(track.covariance), measurement_covariance);
        let gain = product(
            &position_cross_covariance(track.covariance),
            &inverse_2x2(innovation_covariance)?,
        );
        for (state_index, gain_row) in gain.iter().enumerate() {
            track.state[state_index] += gain_row[0] * innovation[0] + gain_row[1] * innovation[1];
        }

        let mut residual = identity_4x4();
        for row in 0..4 {
            for column in 0..2 {
                residual[row][column] -= gain[row][column];
            }
        }
        let posterior = product(
            &product(&residual, &track.covariance),
            &transpose(&residual),
        );
        let measurement_term = product(&product(&gain, &measurement_covariance), &transpose(&gain));
        track.covariance = matrix_add(posterior, measurement_term);
        Ok(())
    }

    pub(crate) fn mahalanobis_distance(
        &self,
        track: &CvTrackState,
        measurement: [f64; 2],
    ) -> Result<f64, TrackingError> {
        Ok(self.position_association(track, measurement)?.0)
    }

    pub(crate) fn position_association(
        &self,
        track: &CvTrackState,
        measurement: [f64; 2],
    ) -> Result<(f64, f64), TrackingError> {
        let covariance = matrix_add(
            matrix_add(top_left_2x2(track.covariance), self.measurement_noise),
            track.extent_covariance,
        );
        let innovation = [
            measurement[0] - track.state[0],
            measurement[1] - track.state[1],
        ];
        let solved = matrix_vector_product(&inverse_2x2(covariance)?, innovation);
        let squared_distance = innovation[0] * solved[0] + innovation[1] * solved[1];
        let squared_distance = squared_distance.max(0.0);
        let determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
        if !determinant.is_finite() || determinant <= f64::EPSILON {
            return Err(TrackingError::SingularInnovationCovariance);
        }
        Ok((squared_distance.sqrt(), determinant.ln() + squared_distance))
    }
}

pub(crate) fn radial_velocity(state: [f64; 4]) -> f64 {
    let distance = state[0].hypot(state[1]);
    if distance == 0.0 {
        return 0.0;
    }
    (state[0] * state[2] + state[1] * state[3]) / distance
}

fn inverse_2x2(matrix: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], TrackingError> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return Err(TrackingError::SingularInnovationCovariance);
    }
    let inverse_determinant = determinant.recip();
    Ok([
        [
            matrix[1][1] * inverse_determinant,
            -matrix[0][1] * inverse_determinant,
        ],
        [
            -matrix[1][0] * inverse_determinant,
            matrix[0][0] * inverse_determinant,
        ],
    ])
}

fn top_left_2x2(matrix: [[f64; 4]; 4]) -> [[f64; 2]; 2] {
    [[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]]
}

fn position_cross_covariance(matrix: [[f64; 4]; 4]) -> [[f64; 2]; 4] {
    [
        [matrix[0][0], matrix[0][1]],
        [matrix[1][0], matrix[1][1]],
        [matrix[2][0], matrix[2][1]],
        [matrix[3][0], matrix[3][1]],
    ]
}

fn identity_4x4() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn matrix_add<const ROWS: usize, const COLUMNS: usize>(
    left: [[f64; COLUMNS]; ROWS],
    right: [[f64; COLUMNS]; ROWS],
) -> [[f64; COLUMNS]; ROWS] {
    let mut values = [[0.0; COLUMNS]; ROWS];
    for ((values_row, left_row), right_row) in values.iter_mut().zip(left).zip(right) {
        for ((value, left_value), right_value) in values_row.iter_mut().zip(left_row).zip(right_row)
        {
            *value = left_value + right_value;
        }
    }
    values
}

fn product<const ROWS: usize, const INNER: usize, const COLUMNS: usize>(
    left: &[[f64; INNER]; ROWS],
    right: &[[f64; COLUMNS]; INNER],
) -> [[f64; COLUMNS]; ROWS] {
    let mut values = [[0.0; COLUMNS]; ROWS];
    for (values_row, left_row) in values.iter_mut().zip(left) {
        for (column, value) in values_row.iter_mut().enumerate() {
            *value = left_row
                .iter()
                .zip(right)
                .map(|(left_value, right_row)| left_value * right_row[column])
                .sum();
        }
    }
    values
}

fn transpose<const ROWS: usize, const COLUMNS: usize>(
    matrix: &[[f64; COLUMNS]; ROWS],
) -> [[f64; ROWS]; COLUMNS] {
    let mut values = [[0.0; ROWS]; COLUMNS];
    for (row, matrix_row) in matrix.iter().enumerate() {
        for (column, value) in matrix_row.iter().enumerate() {
            values[column][row] = *value;
        }
    }
    values
}

fn matrix_vector_product<const ROWS: usize, const COLUMNS: usize>(
    matrix: &[[f64; COLUMNS]; ROWS],
    vector: [f64; COLUMNS],
) -> [f64; ROWS] {
    let mut values = [0.0; ROWS];
    for (value, matrix_row) in values.iter_mut().zip(matrix) {
        *value = matrix_row
            .iter()
            .zip(vector)
            .map(|(matrix_value, vector_value)| matrix_value * vector_value)
            .sum();
    }
    values
}
