//! Built-in safe Rust implementation of the pinned TI 6843 GTRACK 3DA.
//! See TI-LICENSE.txt: this component and its derivatives are for TI devices.
//! The C plugin is an optional development oracle, never a runtime requirement.
#![cfg_attr(not(feature = "reference-plugin"), forbid(unsafe_code))]
mod association;
mod math;
#[cfg(feature = "reference-plugin")]
mod reference;
mod tracker;
mod unit;

use std::path::Path;

use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "reference-plugin", repr(C))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub max_points: u32,
    pub max_tracks: u32,
    pub delta_t: f32,
    pub initial_velocity: f32,
    pub max_velocity: f32,
    pub velocity_resolution: f32,
    pub max_acceleration: [f32; 3],
    pub boresight_filtering: u32,
    pub gating_gain: f32,
    pub gating_limits: [f32; 4],
    pub allocation_snr: f32,
    pub allocation_obscured_snr: f32,
    pub allocation_velocity: f32,
    pub allocation_points: u32,
    pub allocation_distance: f32,
    pub allocation_max_velocity: f32,
    pub state_thresholds: [u32; 6],
    pub sensor_position: [f32; 3],
    pub sensor_orientation: [f32; 2],
    pub boundary_count: u32,
    pub static_count: u32,
    pub occupancy_count: u32,
    pub boundary_boxes: [f32; 12],
    pub static_boxes: [f32; 12],
    pub occupancy_boxes: [f32; 12],
    pub presence_points: u32,
    pub presence_on_to_off: u32,
    pub presence_velocity: f32,
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=1000).contains(&self.max_points) || !(1..=200).contains(&self.max_tracks) {
            return Err("TI capacity requires 1..1000 points and 1..200 tracks".into());
        }
        if self.boundary_count == 0 {
            return Err("At least one world boundary box is required; zero boxes mean always outside in pinned TI".into());
        }
        for (name, value) in [
            ("delta_t", self.delta_t),
            ("max_velocity", self.max_velocity),
            ("velocity_resolution", self.velocity_resolution),
            ("gating_gain", self.gating_gain),
            ("allocation_distance", self.allocation_distance),
            ("allocation_max_velocity", self.allocation_max_velocity),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        if self
            .max_acceleration
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err("max_acceleration must be finite and positive".into());
        }
        if !self.initial_velocity.is_finite()
            || self
                .sensor_position
                .iter()
                .chain(&self.sensor_orientation)
                .any(|x| !x.is_finite())
        {
            return Err("initial velocity and installation must be finite".into());
        }
        if self.sensor_position[0] != 0.0 || self.sensor_position[1] != 0.0 {
            return Err("Pinned TI transform supports height only; scene horizontal origin must be the sensor".into());
        }
        for value in self.gating_limits.iter().chain([
            &self.allocation_snr,
            &self.allocation_obscured_snr,
            &self.allocation_velocity,
            &self.presence_velocity,
        ]) {
            if !value.is_finite() || *value < 0.0 {
                return Err("TI limits and thresholds must be finite and non-negative".into());
            }
        }
        if self.boresight_filtering > 1
            || self.allocation_points == 0
            || self.allocation_points > self.max_points
            || self.presence_points > self.max_points
            || self.presence_on_to_off > 65535
            || self.state_thresholds.iter().any(|x| *x > 65535)
        {
            return Err("Invalid TI integer threshold or flag".into());
        }
        for (count, boxes) in [
            (self.boundary_count, self.boundary_boxes),
            (self.static_count, self.static_boxes),
            (self.occupancy_count, self.occupancy_boxes),
        ] {
            if count > 2 {
                return Err("TI supports at most two boxes of each kind".into());
            }
            if boxes.iter().any(|x| !x.is_finite()) {
                return Err("Scene boxes must be finite".into());
            }
            for b in boxes[..count as usize * 6].chunks_exact(6) {
                if b[0] >= b[1] || b[2] >= b[3] || b[4] >= b[5] {
                    return Err("Scene box bounds must increase".into());
                }
            }
        }
        Ok(())
    }
}

/// Original TI axes: right, forward, up; EC is inverse group covariance.
#[cfg_attr(feature = "reference-plugin", repr(C))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Target {
    pub uid: u32,
    pub tid: u32,
    pub state: u32,
    pub velocity_state: u32,
    pub is_static: u32,
    pub snr_weighting: u32,
    pub height_ignore: u32,
    pub point_number_estimation: u32,
    pub counters: [u32; 6],
    pub age: u64,
    pub state_vector: [f32; 9],
    pub state_covariance: [[f32; 9]; 9],
    #[serde(rename = "apriori_state_after_step")]
    pub predicted_state: [f32; 9],
    #[serde(rename = "apriori_covariance_after_step")]
    pub predicted_covariance: [[f32; 9]; 9],
    pub predicted_measurement: [f32; 4],
    pub ec: [[f32; 4]; 4],
    pub group_covariance: [[f32; 4]; 4],
    pub group_dispersion: [[f32; 4]; 4],
    pub gain: f32,
    pub dimensions: [f32; 4],
    pub measurement_center: [f32; 4],
    pub confidence: f32,
    pub expected_points: f32,
    pub range_rate: f32,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub targets: Vec<Target>,
    pub sensor_targets: Vec<SensorTarget>,
    pub point_uid: Vec<u8>,
    pub point_tid: Vec<i64>,
    pub point_unique: Vec<u8>,
    pub point_static: Vec<u8>,
    pub point_score: Vec<f32>,
    pub updated_doppler: Vec<f32>,
    pub presence: u32,
    pub benchmark_ticks: [u32; 7],
}

/// Workspace view in sensor forward/right/up, retaining the full TI report above.
#[derive(Debug, Serialize)]
pub struct SensorTarget {
    pub state_vector: [f32; 9],
    pub position_covariance: [[f32; 3]; 3],
    pub extent_covariance: [[f32; 3]; 3],
}

fn sensor_target(target: &Target) -> SensorTarget {
    let axes = [1, 0, 2, 4, 3, 5, 7, 6, 8];
    let state_vector = std::array::from_fn(|i| target.state_vector[axes[i]]);
    let position_covariance =
        std::array::from_fn(|i| std::array::from_fn(|j| target.state_covariance[axes[i]][axes[j]]));
    let [r, a, e, _] = target.measurement_center;
    let (sa, ca) = a.sin_cos();
    let (se, ce) = e.sin_cos();
    // Project measured spherical group dispersion, not TI EC (which is an inverse).
    let j = [
        [ce * ca, -r * ce * sa, -r * se * ca],
        [ce * sa, r * ce * ca, -r * se * sa],
        [se, 0.0, r * ce],
    ];
    let mut extent_covariance = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    extent_covariance[row][col] +=
                        j[row][k] * target.group_dispersion[k][l] * j[col][l];
                }
            }
        }
    }
    SensorTarget {
        state_vector,
        position_covariance,
        extent_covariance,
    }
}

pub struct Engine {
    tracker: tracker::Tracker,
    #[cfg(feature = "reference-plugin")]
    reference: Option<reference::Reference>,
    config: Config,
    provenance: serde_json::Value,
    poisoned: bool,
}

impl Engine {
    pub fn new(config: Config) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            tracker: tracker::Tracker::new(&config),
            #[cfg(feature = "reference-plugin")]
            reference: None,
            config,
            provenance: serde_json::json!({
                "implementation": "mmwcore-rust-gtrack-3da-v1",
                "source_version": "4.00.00.05/custom_sdk_files/sdk3/trackerproc_overhead/3DA",
                "capabilities": ["static_support_v1"],
                "license": "TI device-only; see TI-LICENSE.txt"
            }),
            poisoned: false,
        })
    }

    /// Explicit C oracle selection is available only in development builds.
    pub fn load(path: &Path, config: Config) -> Result<Self, String> {
        config.validate()?;
        #[cfg(feature = "reference-plugin")]
        {
            let reference = reference::Reference::load(path, config.clone())?;
            let mut engine = Self::new(config)?;
            engine.provenance = reference.provenance.clone();
            engine.reference = Some(reference);
            Ok(engine)
        }
        #[cfg(not(feature = "reference-plugin"))]
        {
            let _ = path;
            Err("C oracle requires the reference-plugin build feature; omit plugin_manifest for built-in Rust GTRACK".into())
        }
    }

    pub fn provenance(&self) -> &serde_json::Value {
        &self.provenance
    }

    /// Cartesian contract is sensor forward/right/up, with velocity and linear SNR.
    pub fn step_cartesian(
        &mut self,
        points: &[[f32; 5]],
        variances: Option<&[[f32; 4]]>,
    ) -> Result<Report, String> {
        self.step_cartesian_with_static(points, variances, None)
    }

    pub fn step_cartesian_with_static(
        &mut self,
        points: &[[f32; 5]],
        variances: Option<&[[f32; 4]]>,
        static_start: Option<usize>,
    ) -> Result<Report, String> {
        let spherical: Vec<_> = points
            .iter()
            .map(|p| {
                let range = p[0].hypot(p[1]).hypot(p[2]);
                [
                    range,
                    p[1].atan2(p[0]),
                    p[2].atan2(p[0].hypot(p[1])),
                    p[3],
                    p[4],
                ]
            })
            .collect();
        self.step_with_static(&spherical, variances, static_start)
    }

    /// Spherical input rows: range, azimuth right, elevation up, radial velocity,
    /// linear SNR. Variance rows have the same four measurement dimensions.
    pub fn step(
        &mut self,
        points: &[[f32; 5]],
        variances: Option<&[[f32; 4]]>,
    ) -> Result<Report, String> {
        self.step_with_static(points, variances, None)
    }

    pub fn step_with_static(
        &mut self,
        points: &[[f32; 5]],
        variances: Option<&[[f32; 4]]>,
        static_start: Option<usize>,
    ) -> Result<Report, String> {
        if self.poisoned {
            return Err(
                "TI tracker encountered non-finite state; reset before another step".into(),
            );
        }
        if points.len() > self.config.max_points as usize {
            return Err("Frame exceeds configured TI max_points; no points were truncated".into());
        }
        for p in points {
            if p.iter().any(|x| !x.is_finite())
                || p[0] <= 0.0
                || p[4] <= 0.0
                || p[1].abs() >= std::f32::consts::FRAC_PI_2
                || p[2].abs() >= std::f32::consts::FRAC_PI_2
            {
                return Err("TI measurements need positive range/SNR and finite forward-hemisphere angles/velocity".into());
            }
        }
        if let Some(v) = variances
            && (v.len() != points.len() || v.iter().flatten().any(|x| !x.is_finite() || *x <= 0.0))
        {
            return Err("Explicit measurement variances must be positive finite (N,4) values; omit them if unknown".into());
        }
        if let Some(start) = static_start
            && (start > points.len() || points[start..].iter().any(|p| p[3] != 0.0))
        {
            return Err(
                "Static support requires a valid RPC prefix and zero Doppler suffix".into(),
            );
        }
        #[cfg(feature = "reference-plugin")]
        let mut result = if let Some(reference) = &mut self.reference {
            reference.step_raw(points, variances, static_start)?
        } else {
            self.tracker
                .step(points, variances, static_start, &self.config)
        };
        #[cfg(not(feature = "reference-plugin"))]
        let mut result = self
            .tracker
            .step(points, variances, static_start, &self.config);
        result.sensor_targets = result.targets.iter().map(sensor_target).collect();
        if result.targets.iter().any(|t| !target_finite(t))
            || result
                .updated_doppler
                .iter()
                .chain(&result.point_score)
                .any(|x| !x.is_finite())
            || result
                .sensor_targets
                .iter()
                .any(|t| t.extent_covariance.iter().flatten().any(|x| !x.is_finite()))
        {
            self.poisoned = true;
            return Err(
                "TI gtrack_step produced non-finite state; no report emitted, reset required"
                    .into(),
            );
        }
        let mut tid_by_uid = [-1_i64; 256];
        for target in &result.targets {
            if let Some(tid) = tid_by_uid.get_mut(target.uid as usize)
                && *tid == -1
            {
                *tid = i64::from(target.tid);
            }
        }
        for (uid, tid) in result.point_uid.iter().zip(&mut result.point_tid) {
            *tid = tid_by_uid[usize::from(*uid)];
        }
        Ok(result)
    }
}

fn target_finite(t: &Target) -> bool {
    t.state_vector
        .iter()
        .chain(&t.predicted_state)
        .chain(&t.predicted_measurement)
        .chain(t.state_covariance.iter().flatten())
        .chain(t.predicted_covariance.iter().flatten())
        .chain(t.ec.iter().flatten())
        .chain(t.group_covariance.iter().flatten())
        .chain(t.group_dispersion.iter().flatten())
        .chain(&t.dimensions)
        .chain(&t.measurement_center)
        .chain([&t.gain, &t.confidence, &t.expected_points, &t.range_rate])
        .all(|x| x.is_finite())
}
