//! Multiscale scatter-component tracking and causal body-level readout.
//!
//! This is the independent Apache-2.0 backend, not TI GTRACK. Coordinates are
//! level-world forward/right/up; point SNR is in dB and Doppler is in m/s.

mod cluster;
mod lineage;
mod motion;
mod state;

pub use cluster::{ClusterConfig, ClusterOutput, cluster_points};
pub use state::{Body, Component, Observation, ScatterConfig, ScatterState, Track};

use std::fmt;

pub type Point = [f64; 5];
pub type XY = [f64; 2];

#[derive(Debug)]
pub struct MultiscaleError(pub(crate) &'static str);

impl fmt::Display for MultiscaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for MultiscaleError {}

#[derive(serde::Serialize)]
pub struct ScatterOutput {
    pub labels: Vec<i64>,
    pub weights: Vec<f64>,
    pub observations: Vec<Observation>,
    pub bodies: Vec<Body>,
}

pub struct ScatterBodyTracker {
    pub config: ScatterConfig,
    pub state: ScatterState,
}

impl ScatterBodyTracker {
    pub fn new(config: ScatterConfig) -> Result<Self, MultiscaleError> {
        config.validate()?;
        Ok(Self {
            config,
            state: ScatterState::default(),
        })
    }

    pub fn step(&mut self, points: &[Point], dt: f64) -> Result<ScatterOutput, MultiscaleError> {
        if !dt.is_finite() || dt <= 0.0 || points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(MultiscaleError("Points must be finite and dt positive"));
        }
        self.config.validate()?;
        self.state.validate()?;
        let old_ids = self.state.tracks.keys().copied().collect();
        let previous_roots = self
            .state
            .tracks
            .keys()
            .filter(|id| !self.state.parents.contains_key(id))
            .copied()
            .collect();
        self.state.time_s += dt;
        let clustered = cluster_points(
            points,
            &self.config.clustering_method,
            ClusterConfig::default(),
        )?;
        let mut observations = clustered.observations;
        self.state.update(&observations, &self.config, dt)?;
        let components = self.state.motion(points, &mut observations, &self.config);
        self.state.lineage_events.clear();
        if self.config.lineage {
            self.state.update_lineage(points, &old_ids, dt)?;
        }
        self.state
            .limit_bodies(&previous_roots, self.config.max_bodies);
        self.state.previous_clouds = self
            .state
            .last_matches
            .iter()
            .map(|(&id, o)| {
                (
                    id,
                    (
                        o.members
                            .iter()
                            .map(|&i| [points[i][0], points[i][1]])
                            .collect(),
                        self.state.tracks[&id].association_velocity,
                    ),
                )
            })
            .collect();
        let bodies = self.state.bodies(components);
        Ok(ScatterOutput {
            labels: clustered.labels,
            weights: clustered.weights,
            observations,
            bodies,
        })
    }
}

pub(crate) fn distance(a: XY, b: XY) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}
pub(crate) fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}
