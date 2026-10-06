use super::{MultiscaleError, Observation, Point, XY, distance, median};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub fine_radius_m: f64,
    pub outer_radius_m: f64,
    pub min_points: usize,
    pub velocity_scale_mps: f64,
    pub core_power_ratio: f64,
}
impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            fine_radius_m: 0.35,
            outer_radius_m: 0.8,
            min_points: 3,
            velocity_scale_mps: 0.2710796965422454,
            core_power_ratio: 4.0,
        }
    }
}
pub struct ClusterOutput {
    pub labels: Vec<i64>,
    pub weights: Vec<f64>,
    pub observations: Vec<Observation>,
}

pub fn density_labels(xy: &[XY], radius: f64, minimum: usize) -> Vec<i64> {
    let n = xy.len();
    let neighbours: Vec<Vec<bool>> = xy
        .iter()
        .map(|&a| xy.iter().map(|&b| distance(a, b) <= radius).collect())
        .collect();
    let core: Vec<bool> = neighbours
        .iter()
        .map(|row| row.iter().filter(|&&v| v).count() >= minimum)
        .collect();
    let mut labels = vec![-1; n];
    let mut label = 0;
    for seed in 0..n {
        if !core[seed] || labels[seed] >= 0 {
            continue;
        }
        let mut queue = vec![seed];
        labels[seed] = label;
        while let Some(current) = queue.pop() {
            for other in 0..n {
                if neighbours[current][other] && labels[other] < 0 {
                    labels[other] = label;
                    if core[other] {
                        queue.push(other);
                    }
                }
            }
        }
        label += 1;
    }
    labels
}

pub fn split_supported_cores(points: &[Point], labels: &mut [i64], config: ClusterConfig) {
    let n = points.len();
    let xy: Vec<XY> = points.iter().map(|p| [p[0], p[1]]).collect();
    let distances: Vec<Vec<f64>> = xy
        .iter()
        .map(|&a| xy.iter().map(|&b| distance(a, b)).collect())
        .collect();
    let reliable: Vec<bool> = (0..n)
        .map(|i| {
            let neighbours: Vec<usize> = (0..n)
                .filter(|&j| distances[i][j] <= config.fine_radius_m)
                .collect();
            let peak = neighbours
                .iter()
                .map(|&j| points[j][4])
                .fold(f64::NEG_INFINITY, f64::max);
            neighbours.len() >= config.min_points
                && points[i][4] >= peak - 10.0 * config.core_power_ratio.log10()
        })
        .collect();
    let mut next = labels.iter().max().copied().unwrap_or(-1) + 1;
    let ids: BTreeSet<i64> = labels.iter().copied().filter(|&l| l >= 0).collect();
    for label in ids {
        let members: Vec<usize> = (0..n).filter(|&i| labels[i] == label).collect();
        let mut remaining: BTreeSet<usize> =
            members.iter().copied().filter(|&i| reliable[i]).collect();
        let mut groups = Vec::new();
        while let Some(seed) = remaining.pop_first() {
            let mut group = vec![seed];
            let mut queue = vec![seed];
            while let Some(current) = queue.pop() {
                let more: Vec<usize> = remaining
                    .iter()
                    .copied()
                    .filter(|&i| distances[current][i] <= config.fine_radius_m)
                    .collect();
                for &i in &more {
                    remaining.remove(&i);
                }
                group.extend(&more);
                queue.extend(more);
            }
            if group.len() >= config.min_points {
                groups.push(group);
            }
        }
        if groups.len() < 2 {
            continue;
        }
        for &i in &members {
            let nearest = groups
                .iter()
                .enumerate()
                .map(|(g, group)| {
                    (
                        g,
                        group
                            .iter()
                            .map(|&j| distances[i][j])
                            .fold(f64::INFINITY, f64::min),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .unwrap()
                .0;
            labels[i] = if nearest == 0 {
                label
            } else {
                next + nearest as i64 - 1
            };
        }
        next += (groups.len() - 1) as i64;
    }
}

pub fn cluster_points(
    points: &[Point],
    method: &str,
    config: ClusterConfig,
) -> Result<ClusterOutput, MultiscaleError> {
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(MultiscaleError("Points must be finite"));
    }
    if !["small", "large", "two_scale", "weighted", "power_split"].contains(&method) {
        return Err(MultiscaleError("Unknown clustering method"));
    }
    let xy: Vec<XY> = points.iter().map(|p| [p[0], p[1]]).collect();
    let mut labels = density_labels(
        &xy,
        if method == "large" {
            config.outer_radius_m
        } else {
            config.fine_radius_m
        },
        config.min_points,
    );
    if method == "power_split" {
        split_supported_cores(points, &mut labels, config);
    }
    let initial = labels.clone();
    let members: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|&(_, l)| *l >= 0)
        .map(|(i, _)| i)
        .collect();
    if ["two_scale", "weighted", "power_split"].contains(&method) && !members.is_empty() {
        for i in 0..points.len() {
            if initial[i] >= 0 {
                continue;
            }
            let (j, d) = members
                .iter()
                .map(|&j| (j, distance(xy[i], xy[j])))
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .unwrap();
            if d <= config.outer_radius_m {
                labels[i] = initial[j];
            }
        }
    }
    let mut weights = vec![0.0; points.len()];
    let mut observations = Vec::new();
    let ids: BTreeSet<i64> = labels.iter().copied().filter(|&l| l >= 0).collect();
    for label in ids {
        let indices: Vec<usize> = (0..points.len()).filter(|&i| labels[i] == label).collect();
        let density: Vec<f64> = indices
            .iter()
            .map(|&i| {
                indices
                    .iter()
                    .filter(|&&j| distance(xy[i], xy[j]) <= config.fine_radius_m)
                    .count() as f64
            })
            .collect();
        let mut local = vec![1.0; indices.len()];
        if method == "weighted" {
            let bulk = median(
                indices
                    .iter()
                    .filter(|&&i| initial[i] == label)
                    .map(|&i| points[i][3])
                    .collect(),
            );
            let peak = density.iter().copied().fold(0.0, f64::max);
            for (k, &i) in indices.iter().enumerate() {
                local[k] = 0.05
                    + density[k]
                        / peak
                        / (1.0 + ((points[i][3] - bulk) / config.velocity_scale_mps).powi(2));
            }
        }
        let sum: f64 = local.iter().sum();
        for w in &mut local {
            *w /= sum;
        }
        let mut center = [0.0; 2];
        let mut radial = 0.0;
        let mut snr = 0.0;
        for (k, &i) in indices.iter().enumerate() {
            weights[i] = local[k];
            for axis in 0..2 {
                center[axis] += xy[i][axis] * local[k];
            }
            radial += local[k] * points[i][3];
            snr += 10.0_f64.powf(points[i][4] / 10.0);
        }
        let extent = indices
            .iter()
            .enumerate()
            .map(|(k, &i)| local[k] * distance(xy[i], center).powi(2))
            .sum::<f64>()
            .sqrt();
        observations.push(Observation {
            label,
            members: indices,
            center,
            radial_velocity: radial,
            snr_sum: snr,
            rms_extent_m: extent,
            bulk_velocity_correction: None,
        });
    }
    Ok(ClusterOutput {
        labels,
        weights,
        observations,
    })
}
