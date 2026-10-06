use super::{MultiscaleError, XY, distance};
use crate::linear_sum_assignment;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Serialize, Deserialize)]
pub struct ScatterConfig {
    pub lineage: bool,
    pub prefer_recent: bool,
    pub temporal: bool,
    pub doppler: bool,
    pub association_guard: bool,
    pub max_components: Option<usize>,
    pub max_bodies: Option<usize>,
    pub height_m: f64,
    pub velocity_scale_mps: f64,
    pub clustering_method: String,
}
impl Default for ScatterConfig {
    fn default() -> Self {
        Self {
            lineage: true,
            prefer_recent: true,
            temporal: true,
            doppler: true,
            association_guard: true,
            max_components: None,
            max_bodies: None,
            height_m: 1.5,
            velocity_scale_mps: super::ClusterConfig::default().velocity_scale_mps,
            clustering_method: "power_split".into(),
        }
    }
}
impl ScatterConfig {
    pub fn validate(&self) -> Result<(), MultiscaleError> {
        if self.max_components == Some(0) || self.max_bodies == Some(0) {
            return Err(MultiscaleError("Capacity must be positive or None"));
        }
        if !self.height_m.is_finite()
            || !self.velocity_scale_mps.is_finite()
            || self.height_m <= 0.0
            || self.velocity_scale_mps <= 0.0
        {
            return Err(MultiscaleError(
                "Sensor height and Doppler resolution must be positive and finite",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Observation {
    pub label: i64,
    pub members: Vec<usize>,
    pub center: XY,
    pub radial_velocity: f64,
    pub snr_sum: f64,
    pub rms_extent_m: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bulk_velocity_correction: Option<XY>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Track {
    pub position: XY,
    pub velocity: XY,
    pub association_position: XY,
    pub association_velocity: XY,
    pub misses: usize,
    pub hits: usize,
    pub age: usize,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Component {
    pub id: usize,
    pub xy: XY,
    pub coasting: bool,
    pub bulk_velocity_xy_mps: XY,
    pub measurement_members: Vec<usize>,
    pub temporal_samples: usize,
    pub ambiguous_neighbour: bool,
}
#[derive(Serialize)]
pub struct Body {
    #[serde(flatten)]
    pub component: Component,
    pub component_ids: Vec<usize>,
    pub components: Vec<Component>,
    pub body_measurement_members: Vec<usize>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct LineageEvent {
    pub child: usize,
    pub parent: usize,
    pub matched_child_points: usize,
    pub matched_parent_points: usize,
}
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ScatterState {
    pub tracks: BTreeMap<usize, Track>,
    pub next_id: usize,
    pub last_matches: BTreeMap<usize, Observation>,
    pub position_history: BTreeMap<usize, Vec<(f64, XY)>>,
    pub time_s: f64,
    pub previous_clouds: BTreeMap<usize, (Vec<XY>, XY)>,
    pub parents: BTreeMap<usize, usize>,
    pub lineage_events: Vec<LineageEvent>,
    pub component_support: BTreeMap<usize, (f64, f64)>,
}

pub(crate) fn assign(
    cost: &[f64],
    rows: usize,
    columns: usize,
) -> Result<Vec<(usize, usize)>, MultiscaleError> {
    let result = linear_sum_assignment(cost, rows, columns)
        .map_err(|_| MultiscaleError("Invalid assignment costs"))?;
    Ok(result.rows.into_iter().zip(result.columns).collect())
}

impl ScatterState {
    pub(crate) fn validate(&self) -> Result<(), MultiscaleError> {
        if !self.time_s.is_finite()
            || self.tracks.values().any(|track| {
                track
                    .position
                    .iter()
                    .chain(&track.velocity)
                    .chain(&track.association_position)
                    .chain(&track.association_velocity)
                    .any(|v| !v.is_finite())
            })
        {
            return Err(MultiscaleError("Tracking state must be finite"));
        }
        for &child in self.parents.keys() {
            let mut current = child;
            let mut visited = BTreeSet::new();
            while let Some(&parent) = self.parents.get(&current) {
                if !visited.insert(current) || !self.tracks.contains_key(&parent) {
                    return Err(MultiscaleError("Invalid scatter-component parent graph"));
                }
                current = parent;
            }
        }
        Ok(())
    }

    fn assignment(
        &self,
        keys: &[usize],
        observations: &[Observation],
        recent: bool,
    ) -> Result<Vec<(usize, usize)>, MultiscaleError> {
        let rows = keys.len();
        let columns = observations.len();
        let distances: Vec<f64> = keys
            .iter()
            .flat_map(|id| {
                observations.iter().map(move |observation| {
                    distance(self.tracks[id].association_position, observation.center)
                })
            })
            .collect();
        let base: Vec<f64> = (0..rows)
            .flat_map(|i| {
                (0..columns + rows).map({
                    let distances = &distances;
                    move |j| {
                        if j >= columns {
                            0.81
                        } else if distances[i * columns + j] <= 0.8 {
                            distances[i * columns + j]
                        } else {
                            1e6
                        }
                    }
                })
            })
            .collect();
        let original = assign(&base, rows, columns + rows)?;
        if !recent {
            return Ok(original);
        }
        let count = original
            .iter()
            .filter(|&&(i, j)| j < columns && distances[i * columns + j] <= 0.8)
            .count();
        let width = columns + rows - count;
        let height = rows + columns - count;
        let mut cost = vec![1e6; width * height];
        for i in 0..height {
            for j in 0..width {
                cost[i * width + j] = if i < rows && j < columns {
                    let d = distances[i * columns + j];
                    let track = &self.tracks[&keys[i]];
                    if d <= 0.8 {
                        d - if track.hits >= 3 && track.misses == 1 {
                            count as f64 * 0.8 + 1.0
                        } else {
                            0.0
                        }
                    } else {
                        1e6
                    }
                } else if i < rows || j < columns {
                    0.0
                } else {
                    1e6
                };
            }
        }
        Ok(assign(&cost, height, width)?
            .into_iter()
            .filter(|&(i, _)| i < rows)
            .collect())
    }

    pub(crate) fn update(
        &mut self,
        observations: &[Observation],
        config: &ScatterConfig,
        dt: f64,
    ) -> Result<(), MultiscaleError> {
        self.last_matches.clear();
        let keys: Vec<usize> = self.tracks.keys().copied().collect();
        for track in self.tracks.values_mut() {
            track.age += 1;
            track.misses += 1;
            for a in 0..2 {
                track.position[a] += track.velocity[a] * dt;
                track.association_position[a] += track.association_velocity[a] * dt;
            }
        }
        let mut matched = BTreeSet::new();
        if !keys.is_empty() && !observations.is_empty() {
            for (i, j) in self.assignment(&keys, observations, config.prefer_recent)? {
                if j >= observations.len() {
                    continue;
                }
                let track = self.tracks.get_mut(&keys[i]).unwrap();
                let observation = &observations[j];
                if distance(track.association_position, observation.center) > 0.8 {
                    continue;
                }
                for a in 0..2 {
                    let residual = observation.center[a] - track.position[a];
                    track.position[a] += 0.65 * residual;
                    track.velocity[a] += 0.1 * residual / dt;
                    let residual = observation.center[a] - track.association_position[a];
                    track.association_position[a] += 0.65 * residual;
                    track.association_velocity[a] += 0.1 * residual / dt;
                }
                track.misses = 0;
                track.hits += 1;
                self.last_matches.insert(keys[i], observation.clone());
                matched.insert(j);
            }
        }
        let mut order: Vec<usize> = (0..observations.len()).collect();
        if config.max_components.is_some() {
            order.sort_by(|&a, &b| {
                observations[b]
                    .members
                    .len()
                    .cmp(&observations[a].members.len())
                    .then(observations[b].snr_sum.total_cmp(&observations[a].snr_sum))
                    .then(a.cmp(&b))
            });
        }
        for j in order {
            if config
                .max_components
                .is_some_and(|max| self.tracks.len() >= max)
            {
                break;
            }
            if matched.contains(&j) {
                continue;
            }
            let observation = &observations[j];
            self.tracks.insert(
                self.next_id,
                Track {
                    position: observation.center,
                    velocity: [0.0; 2],
                    association_position: observation.center,
                    association_velocity: [0.0; 2],
                    misses: 0,
                    hits: 1,
                    age: 1,
                },
            );
            self.last_matches.insert(self.next_id, observation.clone());
            self.next_id += 1;
        }
        self.tracks.retain(|_, track| track.misses <= 3);
        Ok(())
    }
}
