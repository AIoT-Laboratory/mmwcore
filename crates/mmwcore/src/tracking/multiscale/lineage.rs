use super::state::{LineageEvent, assign};
use super::{Body, ClusterConfig, Component, MultiscaleError, Point, ScatterState, XY, distance};
use std::collections::{BTreeMap, BTreeSet};

pub fn support(points: &[Point]) -> (f64, f64) {
    let peak = points
        .iter()
        .map(|p| p[4])
        .fold(f64::NEG_INFINITY, f64::max);
    let signal: Vec<f64> = points
        .iter()
        .map(|p| 10.0_f64.powf((p[4] - peak) / 10.0))
        .collect();
    let sum: f64 = signal.iter().sum();
    (
        peak + 10.0 * sum.log10(),
        sum * sum / signal.iter().map(|v| v * v).sum::<f64>(),
    )
}
pub fn independently_supported(child: (f64, f64), parent: (f64, f64)) -> bool {
    child.0 >= parent.0 - 10.0 * ClusterConfig::default().core_power_ratio.log10() - 1e-10
        || child.1 >= parent.1 / 2.0 - 1e-10
}
pub fn split_support(
    previous: &[XY],
    child: &[XY],
    parent: &[XY],
) -> Result<(usize, usize), MultiscaleError> {
    let current: Vec<XY> = child.iter().chain(parent).copied().collect();
    let rows = current.len();
    let columns = previous.len();
    let width = rows + columns;
    let distances: Vec<f64> = current
        .iter()
        .flat_map(|&a| previous.iter().map(move |&b| distance(a, b)))
        .collect();
    let mut cost = vec![rows as f64 + 1.0; rows * width];
    for i in 0..rows {
        for j in 0..columns {
            cost[i * width + j] =
                if distances[i * columns + j] <= ClusterConfig::default().fine_radius_m {
                    distances[i * columns + j]
                } else {
                    1e6
                };
        }
    }
    let mut counts = (0, 0);
    for (i, j) in assign(&cost, rows, width)? {
        if j < columns && distances[i * columns + j] <= ClusterConfig::default().fine_radius_m {
            if i < child.len() {
                counts.0 += 1;
            } else {
                counts.1 += 1;
            }
        }
    }
    Ok(counts)
}

impl ScatterState {
    pub(crate) fn update_lineage(
        &mut self,
        points: &[Point],
        old_ids: &BTreeSet<usize>,
        dt: f64,
    ) -> Result<(), MultiscaleError> {
        let mut evidence = self.component_support.clone();
        for (&id, observation) in &self.last_matches {
            evidence.insert(
                id,
                support(
                    &observation
                        .members
                        .iter()
                        .map(|&i| points[i])
                        .collect::<Vec<_>>(),
                ),
            );
        }
        for &child in self.tracks.keys().filter(|id| !old_ids.contains(id)) {
            let child_points: Vec<XY> = self.last_matches[&child]
                .members
                .iter()
                .map(|&i| [points[i][0], points[i][1]])
                .collect();
            let mut candidates = Vec::new();
            for (&parent, observation) in &self.last_matches {
                if !self.previous_clouds.contains_key(&parent) || self.tracks[&parent].hits < 3 {
                    continue;
                }
                if independently_supported(evidence[&child], evidence[&parent]) {
                    continue;
                }
                let (previous, velocity) = &self.previous_clouds[&parent];
                let transported: Vec<XY> = previous
                    .iter()
                    .map(|p| [p[0] + dt * velocity[0], p[1] + dt * velocity[1]])
                    .collect();
                let parent_points: Vec<XY> = observation
                    .members
                    .iter()
                    .map(|&i| [points[i][0], points[i][1]])
                    .collect();
                let (a, b) = split_support(&transported, &child_points, &parent_points)?;
                if a as f64
                    >= (ClusterConfig::default().min_points as f64)
                        .max(child_points.len() as f64 / 2.0)
                    && b as f64
                        >= (ClusterConfig::default().min_points as f64)
                            .max(parent_points.len() as f64 / 2.0)
                {
                    candidates.push((parent, a, b));
                }
            }
            if candidates.len() == 1 {
                let (parent, a, b) = candidates[0];
                self.parents.insert(child, parent);
                self.lineage_events.push(LineageEvent {
                    child,
                    parent,
                    matched_child_points: a,
                    matched_parent_points: b,
                });
            }
        }
        self.parents.retain(|child, parent| {
            self.tracks.contains_key(child)
                && self.tracks.contains_key(parent)
                && !independently_supported(evidence[child], evidence[parent])
                && distance(
                    self.tracks[child].association_position,
                    self.tracks[parent].association_position,
                ) <= ClusterConfig::default().outer_radius_m
        });
        self.component_support = evidence
            .into_iter()
            .filter(|(id, _)| self.tracks.contains_key(id))
            .collect();
        Ok(())
    }

    pub(crate) fn root(&self, mut id: usize) -> usize {
        while let Some(&parent) = self.parents.get(&id) {
            id = parent;
        }
        id
    }
    pub(crate) fn limit_bodies(&mut self, previous: &BTreeSet<usize>, maximum: Option<usize>) {
        let Some(maximum) = maximum else {
            return;
        };
        let roots: BTreeSet<usize> = self.tracks.keys().map(|&id| self.root(id)).collect();
        if roots.len() <= maximum {
            return;
        }
        let mut order: Vec<usize> = roots.into_iter().collect();
        order.sort_by(|a, b| {
            let oa = self.last_matches.get(a);
            let ob = self.last_matches.get(b);
            (!previous.contains(a))
                .cmp(&!previous.contains(b))
                .then(
                    ob.map_or(0, |observation| observation.members.len())
                        .cmp(&oa.map_or(0, |observation| observation.members.len())),
                )
                .then(
                    ob.map_or(0.0, |observation| observation.snr_sum)
                        .total_cmp(&oa.map_or(0.0, |observation| observation.snr_sum)),
                )
                .then(a.cmp(b))
        });
        let admitted: BTreeSet<usize> = order.into_iter().take(maximum).collect();
        let rejected: BTreeSet<usize> = self
            .tracks
            .keys()
            .copied()
            .filter(|&id| !admitted.contains(&self.root(id)))
            .collect();
        self.tracks.retain(|id, _| !rejected.contains(id));
        self.last_matches.retain(|id, _| !rejected.contains(id));
        self.position_history.retain(|id, _| !rejected.contains(id));
        self.component_support
            .retain(|id, _| !rejected.contains(id));
        self.parents.retain(|id, _| !rejected.contains(id));
        self.lineage_events.retain(|e| !rejected.contains(&e.child));
    }
    pub(crate) fn bodies(&self, components: Vec<Component>) -> Vec<Body> {
        let mut groups: BTreeMap<usize, Vec<Component>> = BTreeMap::new();
        for c in components
            .into_iter()
            .filter(|c| self.tracks.contains_key(&c.id))
        {
            groups.entry(self.root(c.id)).or_default().push(c);
        }
        groups
            .into_iter()
            .map(|(root, group)| Body {
                component: group.iter().find(|c| c.id == root).unwrap().clone(),
                component_ids: group.iter().map(|c| c.id).collect(),
                body_measurement_members: group
                    .iter()
                    .flat_map(|c| c.measurement_members.iter().copied())
                    .collect(),
                components: group,
            })
            .collect()
    }
}
