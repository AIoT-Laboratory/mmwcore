//! Deterministic DBSCAN clustering for Cartesian radar point clouds.

use std::collections::VecDeque;
use std::fmt;

const UNASSIGNED_LABEL: i64 = -2;
const NOISE_LABEL: i64 = -1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DbscanConfig {
    pub eps_m: f32,
    pub min_samples: usize,
    pub velocity_scale_s: f32,
    pub use_z: bool,
}

impl DbscanConfig {
    pub fn new(
        eps_m: f32,
        min_samples: usize,
        velocity_scale_s: f32,
        use_z: bool,
    ) -> Result<Self, ClusterError> {
        if !eps_m.is_finite() || eps_m <= 0.0 {
            return Err(ClusterError::InvalidEpsilon);
        }
        if min_samples == 0 {
            return Err(ClusterError::ZeroMinSamples);
        }
        if !velocity_scale_s.is_finite() || velocity_scale_s < 0.0 {
            return Err(ClusterError::InvalidVelocityScale);
        }
        Ok(Self {
            eps_m,
            min_samples,
            velocity_scale_s,
            use_z,
        })
    }

    fn uses_velocity(self) -> bool {
        self.velocity_scale_s > 0.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointColumns {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub velocity: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterResult {
    pub labels: Vec<i64>,
    pub centers: Vec<f32>,
    pub extents: Vec<f32>,
    pub mean_velocities: Vec<f32>,
    pub point_counts: Vec<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClusterError {
    InvalidPointShape {
        point_count: usize,
        channel_count: usize,
        data_length: usize,
    },
    InvalidColumnIndex {
        column: usize,
        channel_count: usize,
    },
    DuplicateCoordinateColumns,
    MissingVelocityColumn,
    NonFinitePointValue,
    InvalidEpsilon,
    ZeroMinSamples,
    InvalidVelocityScale,
}

impl fmt::Display for ClusterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPointShape {
                point_count,
                channel_count,
                data_length,
            } => write!(
                formatter,
                "Point matrix shape ({point_count}, {channel_count}) does not match {data_length} values."
            ),
            Self::InvalidColumnIndex {
                column,
                channel_count,
            } => write!(
                formatter,
                "Point column {column} is outside the available {channel_count} channels."
            ),
            Self::DuplicateCoordinateColumns => {
                write!(formatter, "Point coordinate columns must be distinct.")
            }
            Self::MissingVelocityColumn => write!(
                formatter,
                "A velocity column is required when velocity_scale_s is positive."
            ),
            Self::NonFinitePointValue => {
                write!(formatter, "Point values must be finite.")
            }
            Self::InvalidEpsilon => {
                write!(formatter, "DBSCAN epsilon must be finite and positive.")
            }
            Self::ZeroMinSamples => write!(formatter, "DBSCAN min_samples must be positive."),
            Self::InvalidVelocityScale => write!(
                formatter,
                "DBSCAN velocity scale must be finite and non-negative."
            ),
        }
    }
}

impl std::error::Error for ClusterError {}

pub fn cluster_points(
    points: &[f32],
    point_count: usize,
    channel_count: usize,
    columns: PointColumns,
    config: DbscanConfig,
) -> Result<ClusterResult, ClusterError> {
    validate_input(points, point_count, channel_count, columns, config)?;
    if point_count == 0 {
        return Ok(empty_result());
    }

    let neighborhoods = neighborhoods(points, point_count, channel_count, columns, config);
    let labels = dbscan_labels(&neighborhoods, config.min_samples);
    summarize_clusters(points, channel_count, columns, &labels)
}

fn validate_input(
    points: &[f32],
    point_count: usize,
    channel_count: usize,
    columns: PointColumns,
    config: DbscanConfig,
) -> Result<(), ClusterError> {
    let expected_length =
        point_count
            .checked_mul(channel_count)
            .ok_or(ClusterError::InvalidPointShape {
                point_count,
                channel_count,
                data_length: points.len(),
            })?;
    if points.len() != expected_length {
        return Err(ClusterError::InvalidPointShape {
            point_count,
            channel_count,
            data_length: points.len(),
        });
    }
    for column in [columns.x, columns.y, columns.z] {
        validate_column(column, channel_count)?;
    }
    if columns.x == columns.y || columns.x == columns.z || columns.y == columns.z {
        return Err(ClusterError::DuplicateCoordinateColumns);
    }
    if let Some(velocity) = columns.velocity {
        validate_column(velocity, channel_count)?;
    } else if config.uses_velocity() {
        return Err(ClusterError::MissingVelocityColumn);
    }
    if points.iter().any(|value| !value.is_finite()) {
        return Err(ClusterError::NonFinitePointValue);
    }
    Ok(())
}

fn validate_column(column: usize, channel_count: usize) -> Result<(), ClusterError> {
    if column >= channel_count {
        return Err(ClusterError::InvalidColumnIndex {
            column,
            channel_count,
        });
    }
    Ok(())
}

fn neighborhoods(
    points: &[f32],
    point_count: usize,
    channel_count: usize,
    columns: PointColumns,
    config: DbscanConfig,
) -> Vec<Vec<usize>> {
    let mut values = vec![Vec::new(); point_count];
    let eps_squared = config.eps_m * config.eps_m;
    for first in 0..point_count {
        for second in first..point_count {
            if feature_distance_squared(points, first, second, channel_count, columns, config)
                <= eps_squared
            {
                values[first].push(second);
                if first != second {
                    values[second].push(first);
                }
            }
        }
    }
    values
}

fn feature_distance_squared(
    points: &[f32],
    first: usize,
    second: usize,
    channel_count: usize,
    columns: PointColumns,
    config: DbscanConfig,
) -> f32 {
    let x = point_value(points, first, channel_count, columns.x)
        - point_value(points, second, channel_count, columns.x);
    let y = point_value(points, first, channel_count, columns.y)
        - point_value(points, second, channel_count, columns.y);
    let mut distance_squared = x.mul_add(x, y * y);
    if config.use_z {
        let z = point_value(points, first, channel_count, columns.z)
            - point_value(points, second, channel_count, columns.z);
        distance_squared = z.mul_add(z, distance_squared);
    }
    if let Some(velocity_column) = columns.velocity.filter(|_| config.uses_velocity()) {
        let velocity = (point_value(points, first, channel_count, velocity_column)
            - point_value(points, second, channel_count, velocity_column))
            * config.velocity_scale_s;
        distance_squared = velocity.mul_add(velocity, distance_squared);
    }
    distance_squared
}

fn dbscan_labels(neighborhoods: &[Vec<usize>], min_samples: usize) -> Vec<i64> {
    let mut labels = vec![UNASSIGNED_LABEL; neighborhoods.len()];
    let mut cluster_id = 0_i64;
    for start in 0..neighborhoods.len() {
        if labels[start] != UNASSIGNED_LABEL {
            continue;
        }
        if neighborhoods[start].len() < min_samples {
            labels[start] = NOISE_LABEL;
            continue;
        }

        labels[start] = cluster_id;
        let mut queue = VecDeque::new();
        let mut queued = vec![false; neighborhoods.len()];
        enqueue_neighbors(
            &mut queue,
            &mut queued,
            &labels,
            &neighborhoods[start],
            start,
        );
        while let Some(current) = queue.pop_front() {
            queued[current] = false;
            if labels[current] == NOISE_LABEL {
                labels[current] = cluster_id;
                continue;
            }
            if labels[current] != UNASSIGNED_LABEL {
                continue;
            }
            labels[current] = cluster_id;
            if neighborhoods[current].len() >= min_samples {
                enqueue_neighbors(
                    &mut queue,
                    &mut queued,
                    &labels,
                    &neighborhoods[current],
                    current,
                );
            }
        }
        cluster_id += 1;
    }
    labels
}

fn enqueue_neighbors(
    queue: &mut VecDeque<usize>,
    queued: &mut [bool],
    labels: &[i64],
    neighbors: &[usize],
    origin: usize,
) {
    for &candidate in neighbors {
        if candidate != origin
            && (labels[candidate] == UNASSIGNED_LABEL || labels[candidate] == NOISE_LABEL)
            && !queued[candidate]
        {
            queue.push_back(candidate);
            queued[candidate] = true;
        }
    }
}

fn summarize_clusters(
    points: &[f32],
    channel_count: usize,
    columns: PointColumns,
    labels: &[i64],
) -> Result<ClusterResult, ClusterError> {
    let cluster_count = labels
        .iter()
        .filter_map(|&label| usize::try_from(label).ok())
        .max()
        .map_or(0, |label| label + 1);
    if cluster_count == 0 {
        return Ok(ClusterResult {
            labels: labels.to_vec(),
            ..empty_result()
        });
    }

    let mut sums = vec![[0.0_f64; 3]; cluster_count];
    let mut minimums = vec![[f32::INFINITY; 3]; cluster_count];
    let mut maximums = vec![[f32::NEG_INFINITY; 3]; cluster_count];
    let mut velocity_sums = vec![0.0_f64; cluster_count];
    let mut point_counts = vec![0_i64; cluster_count];

    for (point_index, &label) in labels.iter().enumerate() {
        let Ok(cluster) = usize::try_from(label) else {
            continue;
        };
        let coordinates = [
            point_value(points, point_index, channel_count, columns.x),
            point_value(points, point_index, channel_count, columns.y),
            point_value(points, point_index, channel_count, columns.z),
        ];
        for (axis, value) in coordinates.into_iter().enumerate() {
            sums[cluster][axis] += f64::from(value);
            minimums[cluster][axis] = minimums[cluster][axis].min(value);
            maximums[cluster][axis] = maximums[cluster][axis].max(value);
        }
        if let Some(velocity_column) = columns.velocity {
            velocity_sums[cluster] += f64::from(point_value(
                points,
                point_index,
                channel_count,
                velocity_column,
            ));
        }
        point_counts[cluster] += 1;
    }

    let mut centers = Vec::with_capacity(cluster_count * 3);
    let mut extents = Vec::with_capacity(cluster_count * 3);
    let mut mean_velocities = Vec::with_capacity(cluster_count);
    for cluster in 0..cluster_count {
        let count = point_counts[cluster] as f64;
        for axis in 0..3 {
            centers.push((sums[cluster][axis] / count) as f32);
            extents.push(maximums[cluster][axis] - minimums[cluster][axis]);
        }
        mean_velocities.push((velocity_sums[cluster] / count) as f32);
    }

    Ok(ClusterResult {
        labels: labels.to_vec(),
        centers,
        extents,
        mean_velocities,
        point_counts,
    })
}

fn point_value(points: &[f32], point: usize, channel_count: usize, column: usize) -> f32 {
    points[point * channel_count + column]
}

fn empty_result() -> ClusterResult {
    ClusterResult {
        labels: Vec::new(),
        centers: Vec::new(),
        extents: Vec::new(),
        mean_velocities: Vec::new(),
        point_counts: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{DbscanConfig, PointColumns, cluster_points};

    #[test]
    fn dbscan_clusters_spatial_velocity_features_and_summarizes_members() {
        let points = [
            0.0, 1.0, 0.0, 0.2, 0.1, 1.1, 0.0, 0.4, 3.0, 4.0, 0.5, -1.0, 3.1, 4.1, 0.5, -1.2, 9.0,
            9.0, 9.0, 5.0,
        ];
        let result = cluster_points(
            &points,
            5,
            4,
            PointColumns {
                x: 0,
                y: 1,
                z: 2,
                velocity: Some(3),
            },
            DbscanConfig::new(0.3, 2, 0.2, true).unwrap(),
        )
        .unwrap();

        assert_eq!(result.labels, [0, 0, 1, 1, -1]);
        assert_close(&result.centers, &[0.05, 1.05, 0.0, 3.05, 4.05, 0.5]);
        assert_close(&result.extents, &[0.1, 0.1, 0.0, 0.1, 0.1, 0.0]);
        assert_close(&result.mean_velocities, &[0.3, -1.1]);
        assert_eq!(result.point_counts, [2, 2]);
    }

    #[test]
    fn dbscan_requires_velocity_for_weighted_features() {
        let error = cluster_points(
            &[0.0, 1.0, 0.0],
            1,
            3,
            PointColumns {
                x: 0,
                y: 1,
                z: 2,
                velocity: None,
            },
            DbscanConfig::new(1.0, 1, 1.0, true).unwrap(),
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "A velocity column is required when velocity_scale_s is positive."
        );
    }

    #[test]
    fn dbscan_promotes_prior_noise_and_supports_xy_mode() {
        let points = [0.0, 0.0, 0.0, 1.0, 0.0, 7.0, 2.0, 0.0, -7.0];
        let result = cluster_points(
            &points,
            3,
            3,
            PointColumns {
                x: 0,
                y: 1,
                z: 2,
                velocity: None,
            },
            DbscanConfig::new(1.1, 3, 0.0, false).unwrap(),
        )
        .unwrap();

        assert_eq!(result.labels, [0, 0, 0]);
        assert_close(&result.centers, &[1.0, 0.0, 0.0]);
        assert_close(&result.extents, &[2.0, 0.0, 14.0]);
        assert_close(&result.mean_velocities, &[0.0]);
        assert_eq!(result.point_counts, [3]);
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }
}
