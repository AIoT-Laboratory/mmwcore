// Copyright (C) 2024 Texas Instruments Incorporated.
// Rust adaptation of GTRACK 3DA; see ../TI-LICENSE.txt.
use super::{
    Config, Target,
    association::{Assignment, Association, Gate},
    math::*,
};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum TrackState {
    #[default]
    Free,
    Detection,
    Active,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum VelocityState {
    #[default]
    Initial,
    RangeRate,
    Tracking,
    Locked,
}

/// Names replace the C counter array; u16 wrapping is part of TI lifecycle semantics.
#[derive(Default)]
struct Counters {
    detect_hits: u16,
    detect_misses: u16,
    active_misses: u16,
    sleep: u16,
    outside: u16,
    static_history: u16,
}

#[derive(Default)]
pub struct Unit {
    uid: usize,
    tid: u32,
    pub state: TrackState,
    velocity_state: VelocityState,
    is_static: bool,
    counters: Counters,
    age: u64,
    state_vector: [f32; 9],
    state_covariance: Matrix<9, 9>,
    pub predicted_state: [f32; 9],
    predicted_covariance: Matrix<9, 9>,
    predicted_measurement: [f32; 4],
    ec: Matrix<4, 4>,
    group_covariance: Matrix<4, 4>,
    group_dispersion: Matrix<4, 4>,
    gain: f32,
    dimensions: [f32; 4],
    pub measurement_center: [f32; 4],
    confidence: f32,
    expected_points: f32,
    range_rate: f32,
    pub spread: [f32; 4],
    limits: [f32; 4],
    inverse: Matrix<4, 4>,
    determinant: f32,
    associated: usize,
    allocation_range: f32,
    pub center_position: [f32; 3],
}

impl Unit {
    /// Numeric state codes and C layout belong to the report, not mutable state.
    pub fn report(&self, c: &Config) -> Target {
        Target {
            uid: self.uid as u32,
            tid: self.tid,
            state: match self.state {
                TrackState::Free => 0,
                TrackState::Detection => 2,
                TrackState::Active => 3,
            },
            velocity_state: match self.velocity_state {
                VelocityState::Initial => 0,
                VelocityState::RangeRate => 1,
                VelocityState::Tracking => 2,
                VelocityState::Locked => 3,
            },
            is_static: u32::from(self.is_static),
            snr_weighting: u32::from(!ceiling(c)),
            height_ignore: u32::from(ceiling(c)),
            point_number_estimation: u32::from(!ceiling(c)),
            counters: [
                self.counters.detect_hits,
                self.counters.detect_misses,
                self.counters.active_misses,
                self.counters.sleep,
                self.counters.outside,
                self.counters.static_history,
            ]
            .map(u32::from),
            age: self.age,
            state_vector: self.state_vector,
            state_covariance: self.state_covariance,
            predicted_state: self.predicted_state,
            predicted_covariance: self.predicted_covariance,
            predicted_measurement: self.predicted_measurement,
            ec: self.ec,
            group_covariance: self.group_covariance,
            group_dispersion: self.group_dispersion,
            gain: self.gain,
            dimensions: self.dimensions,
            measurement_center: self.measurement_center,
            confidence: self.confidence,
            expected_points: self.expected_points,
            range_rate: self.range_rate,
        }
    }

    pub fn start(
        &mut self,
        uid: usize,
        tid: u32,
        center: [f32; 4],
        count: usize,
        behind: bool,
        c: &Config,
    ) {
        self.uid = uid;
        self.tid = tid;
        self.age = 1;
        self.state = TrackState::Detection;
        self.velocity_state = VelocityState::Initial;
        self.is_static = false;
        // TI retains outside/static history across slot reuse.
        self.counters.detect_hits = 0;
        self.counters.detect_misses = 0;
        self.counters.active_misses = 0;
        self.counters.sleep = 0;
        self.confidence = if behind { 0.5 } else { 1.0 };
        self.expected_points = if ceiling(c) { 100.0 } else { count as f32 };
        self.allocation_range = center[0];
        let mut u = center;
        u[3] = unroll(c.max_velocity, c.initial_velocity, u[3]);
        self.range_rate = u[3];
        self.predicted_state[..3].copy_from_slice(&position(&u));
        let velocity = position(&[u[3], u[1], u[2], 0.0]);
        self.predicted_state[3..6].copy_from_slice(&velocity);
        // Pinned 3DA start clears ax/ay but leaves az in a reused slot.
        self.predicted_state[6..8].fill(0.0);
        self.predicted_measurement = u;
        self.limits = limits(u[0], c);
        self.spread = spread(u[0], c);
        self.predicted_covariance = [[0.0; 9]; 9];
        for i in 3..9 {
            self.predicted_covariance[i][i] = if i < 6 { 0.5 } else { 1.0 };
        }
        self.group_dispersion = [[0.0; 4]; 4];
        self.gain = c.gating_gain;
    }

    pub fn predict(&mut self, c: &Config, f: &Matrix<9, 9>, q: &Matrix<9, 9>) {
        self.age += 1;
        if self.is_static {
            self.predicted_state = self.state_vector;
            self.predicted_covariance = self.state_covariance;
        } else {
            let state = self.state_vector.map(|v| [v]);
            self.predicted_state = mul(f, &state).map(|v| v[0]);
            let fp = mul(f, &self.state_covariance);
            let mut p = mul_transpose(&fp, f);
            for i in 0..9 {
                for j in 0..9 {
                    p[i][j] += q[i][j];
                }
            }
            self.predicted_covariance =
                std::array::from_fn(|i| std::array::from_fn(|j| (p[i][j] + p[j][i]) / 2.0));
        }
        self.predicted_measurement = measurement(&self.predicted_state);
        self.limits = limits(self.predicted_measurement[0], c);
    }

    pub fn gate(&self, c: &Config) -> Gate<'_> {
        let mut bounds = self.limits;
        bounds[3] = (2.0 * self.limits[3]).min(2.0 * self.spread[3]);
        let velocity = world(&self.predicted_state[3..6], c);
        Gate {
            predicted: &self.predicted_measurement,
            inverse: &self.inverse,
            bounds,
            gain: self.gain,
            logdet: if ceiling(c) {
                0.0
            } else {
                self.determinant.ln()
            },
            slow: (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt() < 0.1,
        }
    }

    pub fn score(&mut self, points: &mut [[f32; 5]], associations: &mut [Association], c: &Config) {
        self.associated = self.gate(c).associate(self.uid, points, associations, c);
        self.ec = self.inverse;
    }

    pub fn update(
        &mut self,
        points: &[[f32; 5]],
        variances: Option<&[[f32; 4]]>,
        associations: &[Association],
        c: &Config,
        static_hit: bool,
        good: &mut Vec<usize>,
    ) {
        let mut count = 0;
        let mut dynamic = 0;
        good.clear();
        let mut sum = [0.0; 4];
        let mut var_sum = [0.0; 4];
        let mut low = [f32::MAX; 4];
        let mut high = [-f32::MAX; 4];
        let mut snr_sum = 0.0;
        let mut pilot = 0.0;
        for (n, (point, association)) in points.iter().zip(associations).enumerate() {
            if association.assignment != Assignment::Track(self.uid) {
                continue;
            }
            count += 1;
            if let Some(v) = variances {
                for i in 0..4 {
                    var_sum[i] += v[n][i];
                }
            }
            if point[3].abs() <= f32::EPSILON {
                continue;
            }
            dynamic += 1;
            if !association.unique {
                continue;
            }
            let [range, azimuth, elevation, doppler, _] = *point;
            let mut u = [range, azimuth, elevation, doppler];
            if good.is_empty() {
                pilot = u[3];
            } else {
                u[3] = unroll(c.max_velocity, pilot, u[3]);
            }
            good.push(n);
            let weight = if ceiling(c) { 1.0 } else { point[4] };
            snr_sum += weight;
            for i in 0..4 {
                low[i] = low[i].min(u[i]);
                high[i] = high[i].max(u[i]);
                sum[i] += u[i] * weight;
            }
        }
        let number = good.len() as f32;
        if !ceiling(c) {
            if number > 0.0 {
                self.expected_points = if number > self.expected_points {
                    number
                } else {
                    0.9 * self.expected_points + 0.1 * number
                };
                self.expected_points = self.expected_points.max(c.allocation_points as f32);
            }
        } else {
            self.expected_points = if number > self.expected_points {
                number
            } else {
                100.0
            };
        }
        let mut input_velocity = 0.0;
        let mut variance = [0.0; 4];
        if !good.is_empty() {
            let reciprocal = 1.0 / snr_sum;
            self.measurement_center = sum.map(|v| v * reciprocal);
            input_velocity = self.measurement_center[3];
            self.velocity(c);
            self.center_position = position(&self.measurement_center);
            variance = var_sum.map(|v| v * (1.0 / count as f32));
        }
        if good.len() > 1 {
            for i in 0..4 {
                let spread = ((high[i] - low[i]) * (number + 1.0) / (number - 1.0))
                    .min(2.0 * self.limits[i])
                    .max(self.limits[i]);
                self.spread[i] = if spread > self.spread[i] {
                    spread
                } else {
                    (1.0 - 0.01) * self.spread[i] + 0.01 * spread
                };
            }
            self.dimensions = [
                self.spread[0],
                2.0 * self.measurement_center[0] * (self.spread[1] / 2.0).tan(),
                2.0 * self.measurement_center[0] * (self.spread[2] / 2.0).tan(),
                self.spread[3],
            ];
        }
        let velocity = world(&self.predicted_state[3..6], c);
        let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
        let stop = if ceiling(c) { 1.0 } else { 0.5 };
        if count == 0 {
            if !self.is_static {
                self.predicted_state[6..9].fill(0.0);
                if speed < stop {
                    self.predicted_state[3..6].fill(0.0);
                    self.is_static = true;
                } else {
                    self.confidence *= 0.99;
                }
            }
        } else if dynamic == 0 {
            if self.is_static {
                self.confidence = 0.99 * self.confidence + 0.01;
            } else if speed < stop {
                self.predicted_state[3..9].fill(0.0);
                self.is_static = true;
                if count > 3 {
                    self.confidence = 0.95 * self.confidence + 0.05;
                }
            } else if speed < stop * 2.0 {
                for value in &mut self.predicted_state[3..6] {
                    *value *= 0.5;
                }
                self.predicted_state[6..9].fill(0.0);
            } else {
                self.confidence *= 0.99;
            }
        }
        if good.len() > 3 {
            if self.is_static && input_velocity.abs() < stop {
                self.is_static = false;
            }
            // TI intentionally uses integer division for ambiguous dynamic points.
            let confidence = if self.associated != 0 {
                (good.len() + (dynamic - good.len()) / 2) as f32 / self.associated as f32
            } else {
                0.0
            };
            self.confidence = (1.0 - 0.1) * self.confidence + 0.1 * confidence;
        }
        if !good.is_empty() {
            if variances.is_none() {
                variance = self.spread.map(|s| (s / 2.0) * (s / 2.0));
            }
            if good.len() > 3 {
                let mut dispersion = [[0.0; 4]; 4];
                for &n in good.iter() {
                    for i in 0..4 {
                        for j in i..4 {
                            // This is TI's absolute-product dispersion, not a signed covariance.
                            dispersion[i][j] += ((points[n][i] - self.measurement_center[i])
                                * (points[n][j] - self.measurement_center[j]))
                                .abs();
                        }
                    }
                }
                let alpha = number / self.expected_points;
                for (i, row) in dispersion.iter().enumerate() {
                    for (j, entry) in row.iter().enumerate().skip(i) {
                        let value =
                            (1.0 - alpha) * self.group_dispersion[i][j] + alpha * (entry / number);
                        self.group_dispersion[i][j] = value;
                        self.group_dispersion[j][i] = value;
                    }
                }
            }
            let j = jacobian(&self.predicted_state);
            let pj = mul_transpose(&self.predicted_covariance, &j);
            let jpj = mul(&j, &pj);
            let alpha = (self.expected_points - number) / ((self.expected_points - 1.0) * number);
            let mut covariance = jpj;
            for i in 0..4 {
                covariance[i][i] += variance[i] / number + self.group_dispersion[i][i] * alpha;
            }
            let (inv, _) = inverse(&covariance);
            let k = mul(&pj, &inv);
            let residual = std::array::from_fn(|i| {
                [self.measurement_center[i] - self.predicted_measurement[i]]
            });
            let correction = mul(&k, &residual);
            let reduction = mul_transpose(&k, &pj);
            for i in 0..9 {
                self.state_vector[i] = self.predicted_state[i] + correction[i][0];
                for (j, value) in reduction[i].iter().enumerate() {
                    self.state_covariance[i][j] = self.predicted_covariance[i][j] - value;
                }
            }
            for i in 0..4 {
                for (j, value) in jpj[i].iter().enumerate() {
                    self.group_covariance[i][j] = (value + if i == j { variance[i] } else { 0.0 })
                        + self.group_dispersion[i][j];
                }
            }
            (self.inverse, self.determinant) = inverse(&self.group_covariance);
        } else {
            self.state_vector = self.predicted_state;
            self.state_covariance = self.predicted_covariance;
            self.center_position
                .copy_from_slice(&self.predicted_state[..3]);
        }
        // Additional static detections may hold an ACTIVE ID, never create one.
        if static_hit
            && self.state == TrackState::Active
            && self.is_static
            && count != 0
            && dynamic == 0
        {
            self.counters.sleep = 0;
        }
        self.event(count, good.len(), dynamic, c);
    }

    fn velocity(&mut self, c: &Config) {
        let u = &mut self.measurement_center;
        match self.velocity_state {
            VelocityState::Initial => {
                u[3] = self.range_rate;
                self.velocity_state = VelocityState::RangeRate;
            }
            VelocityState::RangeRate | VelocityState::Tracking => {
                let rate = (u[0] - self.allocation_range) / ((self.age - 1) as f32 * c.delta_t);
                self.range_rate = 0.5 * self.range_rate + (1.0 - 0.5) * rate;
                u[3] = unroll(c.max_velocity, self.range_rate, u[3]);
                if self.velocity_state == VelocityState::RangeRate {
                    if ((rate - self.range_rate) / self.range_rate).abs() < 0.1 {
                        self.velocity_state = VelocityState::Tracking;
                    }
                } else if ((self.predicted_measurement[3] - u[3]) / u[3]).abs() < 0.1 {
                    self.velocity_state = VelocityState::Locked;
                }
            }
            VelocityState::Locked => {
                u[3] = unroll(c.max_velocity, self.predicted_measurement[3], u[3]);
            }
        }
    }

    fn event(&mut self, count: usize, reliable: usize, dynamic: usize, c: &Config) {
        let world = world(&self.state_vector[..3], c);
        let [
            det_active,
            det_free,
            active_free,
            static_free,
            exit_free,
            sleep_free,
        ] = c.state_thresholds.map(|threshold| threshold as u16);
        if inside(&world, &c.boundary_boxes, c.boundary_count) {
            self.counters.outside = 0;
        } else {
            self.counters.outside = self.counters.outside.wrapping_add(1);
            if self.counters.outside >= exit_free {
                self.state = TrackState::Free;
            }
        }
        match self.state {
            TrackState::Free => {}
            TrackState::Detection => {
                if reliable > 3 {
                    self.counters.detect_misses = 0;
                    self.counters.detect_hits = self.counters.detect_hits.wrapping_add(1);
                    if self.counters.detect_hits > det_active {
                        self.state = TrackState::Active;
                    }
                } else if reliable == 0 {
                    self.counters.detect_misses = self.counters.detect_misses.wrapping_add(1);
                    self.counters.detect_hits = self.counters.detect_hits.saturating_sub(1);
                    if self.counters.detect_misses > det_free {
                        self.state = TrackState::Free;
                    }
                } else {
                    self.counters.detect_misses = 0;
                }
            }
            TrackState::Active => {
                let in_static = inside(&world, &c.static_boxes, c.static_count);
                if sleep_free != 0 {
                    if self.is_static {
                        self.counters.sleep = self.counters.sleep.wrapping_add(1);
                        self.counters.static_history = ((1.0 - 0.05)
                            * self.counters.static_history as f32
                            + 0.05 * (count - dynamic) as f32 * 15.0)
                            as u16;
                        if self.counters.static_history > 3 {
                            self.counters.sleep = 0;
                        }
                        let threshold =
                            if in_static && self.confidence >= if ceiling(c) { 0.3 } else { 0.5 } {
                                sleep_free
                            } else {
                                exit_free
                            };
                        if self.counters.sleep > threshold {
                            self.state = TrackState::Free;
                            return;
                        }
                    } else {
                        self.counters.sleep = 0;
                    }
                }
                if (self.is_static && count != 0) || (!self.is_static && dynamic != 0) {
                    self.counters.active_misses = 0;
                } else {
                    self.counters.active_misses = self.counters.active_misses.wrapping_add(1);
                    let threshold = if !in_static {
                        exit_free
                    } else if self.is_static {
                        static_free
                    } else {
                        active_free
                    };
                    if self.counters.active_misses > threshold {
                        self.state = TrackState::Free;
                    }
                }
            }
        }
    }
}
