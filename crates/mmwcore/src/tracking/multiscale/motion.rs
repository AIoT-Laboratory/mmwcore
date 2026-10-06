use super::{Component, Observation, Point, ScatterConfig, ScatterState, XY, distance, median};

pub fn geometric_median(positions: &[XY]) -> XY {
    let mut center = [0.0; 2];
    for p in positions {
        for a in 0..2 {
            center[a] += p[a];
        }
    }
    for v in &mut center {
        *v /= positions.len() as f64;
    }
    for _ in 0..30 {
        let weights: Vec<f64> = positions
            .iter()
            .map(|&p| 1.0 / distance(p, center).max(1e-8))
            .collect();
        let sum: f64 = weights.iter().sum();
        center = [0.0; 2];
        for (p, w) in positions.iter().zip(weights) {
            for a in 0..2 {
                center[a] += w * p[a];
            }
        }
        for v in &mut center {
            *v /= sum;
        }
    }
    center
}

impl ScatterState {
    pub(crate) fn motion(
        &mut self,
        points: &[Point],
        observations: &mut [Observation],
        config: &ScatterConfig,
    ) -> Vec<Component> {
        let mut output = Vec::new();
        for (&id, track) in &self.tracks {
            if track.hits < 3 {
                continue;
            }
            let mut velocity = track.velocity;
            if config.doppler
                && let Some(measurement) = self.last_matches.get_mut(&id)
            {
                let subset: Vec<Point> = measurement.members.iter().map(|&i| points[i]).collect();
                let los: Vec<XY> = subset
                    .iter()
                    .map(|p| {
                        let radius = (p[0] * p[0] + p[1] * p[1] + (p[2] - config.height_m).powi(2))
                            .sqrt()
                            .max(1e-8);
                        [p[0] / radius, p[1] / radius]
                    })
                    .collect();
                let residual: Vec<f64> = subset
                    .iter()
                    .zip(&los)
                    .map(|(p, h)| p[3] - (h[0] * velocity[0] + h[1] * velocity[1]))
                    .collect();
                let middle = median(residual.clone());
                let scale = (config.velocity_scale_mps / 3.0)
                    .max(1.4826 * median(residual.iter().map(|v| (v - middle).abs()).collect()));
                let noise = scale * scale + config.velocity_scale_mps * config.velocity_scale_mps;
                let mut h = [0.0; 2];
                for row in &los {
                    for a in 0..2 {
                        h[a] += row[a];
                    }
                }
                for v in &mut h {
                    *v /= los.len() as f64;
                }
                let denominator = 0.25 * (h[0] * h[0] + h[1] * h[1]) + noise;
                let correction = [
                    0.25 * h[0] / denominator * middle,
                    0.25 * h[1] / denominator * middle,
                ];
                for a in 0..2 {
                    velocity[a] += correction[a];
                }
                measurement.bulk_velocity_correction = Some(correction);
                observations
                    .iter_mut()
                    .find(|observation| observation.label == measurement.label)
                    .unwrap()
                    .bulk_velocity_correction = Some(correction);
            }
            let measurement = self.last_matches.get(&id);
            let ambiguous = measurement.is_some_and(|m| {
                observations.iter().any(|observation| {
                    observation.label != m.label && distance(observation.center, m.center) <= 0.8
                })
            });
            let history = self.position_history.entry(id).or_default();
            if config.association_guard && ambiguous {
                history.clear();
            }
            history.push((self.time_s, track.position));
            if history.len() > 3 {
                history.remove(0);
            }
            let xy = if config.temporal {
                geometric_median(
                    &history
                        .iter()
                        .map(|(time, p)| {
                            [
                                p[0] + (self.time_s - time) * velocity[0],
                                p[1] + (self.time_s - time) * velocity[1],
                            ]
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                track.position
            };
            output.push(Component {
                id,
                xy,
                coasting: track.misses > 0,
                bulk_velocity_xy_mps: velocity,
                measurement_members: measurement.map_or_else(Vec::new, |m| m.members.clone()),
                temporal_samples: if config.temporal { history.len() } else { 1 },
                ambiguous_neighbour: ambiguous,
            });
        }
        self.position_history
            .retain(|id, _| self.tracks.contains_key(id));
        output
    }
}
