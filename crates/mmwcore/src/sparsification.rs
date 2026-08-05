//! Deterministic sparse RPC extraction from a Cartesian Doppler volume.

mod scan;

use std::cmp::Ordering;
use std::fmt;

use scan::analyze_doppler_slabs;

/// Numeric input for Cartesian RT sparsification.
#[derive(Clone, Copy, Debug)]
pub struct CartesianSparsificationInput<'a> {
    pub magnitude_dzyx: &'a [f32],
    pub shape_dzyx: [usize; 4],
    pub doppler_velocity_mps: &'a [f32],
    pub z_m: &'a [f32],
    pub y_m: &'a [f32],
    pub x_m: &'a [f32],
    pub spatial_mask_zyx: Option<&'a [bool]>,
    pub suppressed_doppler_index: Option<usize>,
}

/// Deterministic peak-selection policy for one Cartesian radar volume.
#[derive(Clone, Copy, Debug)]
pub struct CartesianSparsificationConfig {
    pub min_snr_db: f32,
    pub max_points: usize,
    pub spatial_peak_radius: usize,
    pub doppler_peak_radius: usize,
    pub max_doppler_peaks_per_spatial: Option<usize>,
    pub boundary_margin_voxels: usize,
    pub noise_floor_scale: f32,
    pub static_point_capacity_fraction: f32,
    pub static_velocity_threshold_mps: f32,
    pub strongest_point_fallback: bool,
}

impl CartesianSparsificationConfig {
    pub fn validate(self) -> Result<(), CartesianSparsificationError> {
        if !self.min_snr_db.is_finite() {
            return Err(CartesianSparsificationError::InvalidMinSnr);
        }
        if self.max_points == 0 {
            return Err(CartesianSparsificationError::InvalidMaxPoints);
        }
        if self.max_doppler_peaks_per_spatial == Some(0) {
            return Err(CartesianSparsificationError::InvalidDopplerPeakLimit);
        }
        if !self.noise_floor_scale.is_finite() || self.noise_floor_scale <= 0.0 {
            return Err(CartesianSparsificationError::InvalidNoiseFloorScale);
        }
        if !self.static_point_capacity_fraction.is_finite()
            || self.static_point_capacity_fraction <= 0.0
            || self.static_point_capacity_fraction > 1.0
        {
            return Err(CartesianSparsificationError::InvalidStaticCapacity);
        }
        if !self.static_velocity_threshold_mps.is_finite()
            || self.static_velocity_threshold_mps < 0.0
        {
            return Err(CartesianSparsificationError::InvalidStaticVelocityThreshold);
        }
        Ok(())
    }
}

/// Sparse point matrix and diagnostics for one Cartesian radar volume.
#[derive(Debug, PartialEq)]
pub struct CartesianSparsificationResult {
    /// C-order `(point, [x, y, z, velocity, snr_db])` values.
    pub points: Vec<f32>,
    pub point_count: usize,
    pub noise_floor_min: f32,
    pub noise_floor_median: f32,
    pub noise_floor_max: f32,
    pub valid_spatial_voxels: usize,
    pub positive_volume_voxels: usize,
    pub valid_positive_volume_voxels: usize,
    pub local_peak_voxels: usize,
    pub doppler_peak_voxels: usize,
    pub threshold_peak_voxels: usize,
    pub limited_peak_voxels: usize,
    pub fallback_used: bool,
    pub static_output_points: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CartesianSparsificationError {
    ShapeOverflow,
    MagnitudeSizeMismatch {
        expected: usize,
        actual: usize,
    },
    AxisLengthMismatch {
        axis: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteMagnitude,
    NegativeMagnitude,
    NonFiniteAxis {
        axis: &'static str,
    },
    NonIncreasingAxis {
        axis: &'static str,
    },
    SpatialMaskSizeMismatch {
        expected: usize,
        actual: usize,
    },
    BoundaryMarginLeavesNoDomain {
        shape_zyx: [usize; 3],
    },
    SpatialMaskLeavesNoDomain,
    SuppressedDopplerIndexOutOfBounds {
        index: usize,
        bins: usize,
    },
    InvalidMinSnr,
    InvalidMaxPoints,
    InvalidDopplerPeakLimit,
    InvalidNoiseFloorScale,
    InvalidStaticCapacity,
    InvalidStaticVelocityThreshold,
    WorkerPanicked,
}

impl fmt::Display for CartesianSparsificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => write!(formatter, "Cartesian volume shape overflows usize."),
            Self::MagnitudeSizeMismatch { expected, actual } => write!(
                formatter,
                "Cartesian magnitude buffer size {actual} does not match expected shape size {expected}."
            ),
            Self::AxisLengthMismatch {
                axis,
                expected,
                actual,
            } => write!(
                formatter,
                "{axis} must have shape ({expected},); got ({actual},)."
            ),
            Self::NonFiniteMagnitude => {
                write!(
                    formatter,
                    "Cartesian magnitude volume contains NaN or Inf values."
                )
            }
            Self::NegativeMagnitude => {
                write!(
                    formatter,
                    "Cartesian magnitude volume must be non-negative."
                )
            }
            Self::NonFiniteAxis { axis } => write!(formatter, "{axis} contains NaN or Inf values."),
            Self::NonIncreasingAxis { axis } => {
                write!(formatter, "{axis} must be strictly increasing.")
            }
            Self::SpatialMaskSizeMismatch { expected, actual } => write!(
                formatter,
                "Cartesian spatial mask size {actual} does not match expected spatial size {expected}."
            ),
            Self::BoundaryMarginLeavesNoDomain { shape_zyx } => write!(
                formatter,
                "Cartesian volume boundary_margin_voxels leaves no valid spatial domain for shape {shape_zyx:?}."
            ),
            Self::SpatialMaskLeavesNoDomain => {
                write!(
                    formatter,
                    "Cartesian spatial_mask_zyx leaves no valid spatial voxels."
                )
            }
            Self::SuppressedDopplerIndexOutOfBounds { index, bins } => write!(
                formatter,
                "Suppressed Cartesian Doppler index {index} is outside {bins} bins."
            ),
            Self::InvalidMinSnr => write!(formatter, "Cartesian volume min_snr_db must be finite."),
            Self::InvalidMaxPoints => {
                write!(formatter, "Cartesian volume max_points must be positive.")
            }
            Self::InvalidDopplerPeakLimit => write!(
                formatter,
                "Cartesian volume max_doppler_peaks_per_spatial must be positive when provided."
            ),
            Self::InvalidNoiseFloorScale => write!(
                formatter,
                "Cartesian volume noise_floor_scale must be finite and positive."
            ),
            Self::InvalidStaticCapacity => write!(
                formatter,
                "Cartesian volume static_point_capacity_fraction must be within (0, 1]."
            ),
            Self::InvalidStaticVelocityThreshold => write!(
                formatter,
                "Cartesian volume static_velocity_threshold_mps must be finite and non-negative."
            ),
            Self::WorkerPanicked => write!(formatter, "Cartesian sparsification worker panicked."),
        }
    }
}

impl std::error::Error for CartesianSparsificationError {}

/// Extract deterministic Cartesian radar points from one DZYX magnitude volume.
pub fn sparsify_cartesian_volume(
    input: CartesianSparsificationInput<'_>,
    config: CartesianSparsificationConfig,
) -> Result<CartesianSparsificationResult, CartesianSparsificationError> {
    config.validate()?;
    validate_input(input)?;
    let worker_count = std::thread::available_parallelism()
        .map_or(1, |parallelism| parallelism.get())
        .min(4)
        .min(input.shape_dzyx[0].max(1));
    sparsify_cartesian_volume_with_workers(input, config, worker_count)
}

fn sparsify_cartesian_volume_with_workers(
    input: CartesianSparsificationInput<'_>,
    config: CartesianSparsificationConfig,
    worker_count: usize,
) -> Result<CartesianSparsificationResult, CartesianSparsificationError> {
    let [_, z_size, y_size, x_size] = input.shape_dzyx;
    let spatial_shape = [z_size, y_size, x_size];
    let spatial_size = checked_product(&spatial_shape)?;
    let valid_spatial = valid_spatial_domain(
        spatial_shape,
        config.boundary_margin_voxels,
        input.spatial_mask_zyx,
    )?;
    let valid_spatial_voxels = valid_spatial.iter().filter(|&&valid| valid).count();
    let worker_count = worker_count.clamp(1, input.shape_dzyx[0].max(1));
    let (noise_floors, scan) =
        analyze_doppler_slabs(input, config, spatial_shape, &valid_spatial, worker_count)?;
    let noise_stats = noise_floor_stats(&noise_floors);

    let mut candidates = limit_doppler_peaks_per_spatial(
        scan.candidates,
        spatial_size,
        config.max_doppler_peaks_per_spatial,
    );
    let limited_peak_voxels = candidates.len();
    candidates = rank_candidates(candidates, input, spatial_size, config);
    let reported_limited_peak_voxels = if candidates.is_empty() {
        0
    } else {
        limited_peak_voxels
    };
    let (candidates, fallback_used) = if candidates.is_empty()
        && config.strongest_point_fallback
        && scan.valid_positive_volume_voxels > 0
    {
        (
            vec![strongest_valid_candidate(
                input,
                &valid_spatial,
                spatial_size,
                &noise_floors,
            )],
            true,
        )
    } else {
        (candidates, false)
    };
    let static_output_points = static_candidate_count(
        &candidates,
        input.doppler_velocity_mps,
        spatial_size,
        config.static_velocity_threshold_mps,
    );
    let points = materialize_points(candidates, input, spatial_size);

    Ok(CartesianSparsificationResult {
        point_count: points.len() / 5,
        points,
        noise_floor_min: noise_stats.min,
        noise_floor_median: noise_stats.median,
        noise_floor_max: noise_stats.max,
        valid_spatial_voxels,
        positive_volume_voxels: scan.positive_volume_voxels,
        valid_positive_volume_voxels: scan.valid_positive_volume_voxels,
        local_peak_voxels: scan.local_peak_voxels,
        doppler_peak_voxels: scan.doppler_peak_voxels,
        threshold_peak_voxels: scan.threshold_peak_voxels,
        limited_peak_voxels: reported_limited_peak_voxels,
        fallback_used,
        static_output_points,
    })
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    index: usize,
    snr_db: f32,
}

#[derive(Clone, Copy, Debug)]
struct NoiseFloorStats {
    min: f32,
    median: f32,
    max: f32,
}

#[derive(Debug, Default)]
struct CandidateScan {
    candidates: Vec<Candidate>,
    positive_volume_voxels: usize,
    valid_positive_volume_voxels: usize,
    local_peak_voxels: usize,
    doppler_peak_voxels: usize,
    threshold_peak_voxels: usize,
}

impl CandidateScan {
    fn extend(&mut self, mut other: Self) {
        self.candidates.append(&mut other.candidates);
        self.positive_volume_voxels += other.positive_volume_voxels;
        self.valid_positive_volume_voxels += other.valid_positive_volume_voxels;
        self.local_peak_voxels += other.local_peak_voxels;
        self.doppler_peak_voxels += other.doppler_peak_voxels;
        self.threshold_peak_voxels += other.threshold_peak_voxels;
    }
}

fn validate_input(
    input: CartesianSparsificationInput<'_>,
) -> Result<(), CartesianSparsificationError> {
    let expected_size = checked_product(&input.shape_dzyx)?;
    if input.magnitude_dzyx.len() != expected_size {
        return Err(CartesianSparsificationError::MagnitudeSizeMismatch {
            expected: expected_size,
            actual: input.magnitude_dzyx.len(),
        });
    }
    if input
        .magnitude_dzyx
        .iter()
        .any(|magnitude| !magnitude.is_finite())
    {
        return Err(CartesianSparsificationError::NonFiniteMagnitude);
    }
    if input
        .magnitude_dzyx
        .iter()
        .any(|&magnitude| magnitude < 0.0)
    {
        return Err(CartesianSparsificationError::NegativeMagnitude);
    }
    for (axis_name, axis, expected_size) in [
        (
            "doppler_velocity_mps",
            input.doppler_velocity_mps,
            input.shape_dzyx[0],
        ),
        ("z_m", input.z_m, input.shape_dzyx[1]),
        ("y_m", input.y_m, input.shape_dzyx[2]),
        ("x_m", input.x_m, input.shape_dzyx[3]),
    ] {
        validate_axis(axis_name, axis, expected_size)?;
    }
    if let Some(mask) = input.spatial_mask_zyx {
        let expected_size = checked_product(&input.shape_dzyx[1..])?;
        if mask.len() != expected_size {
            return Err(CartesianSparsificationError::SpatialMaskSizeMismatch {
                expected: expected_size,
                actual: mask.len(),
            });
        }
    }
    if let Some(index) = input.suppressed_doppler_index {
        if index >= input.shape_dzyx[0] {
            return Err(
                CartesianSparsificationError::SuppressedDopplerIndexOutOfBounds {
                    index,
                    bins: input.shape_dzyx[0],
                },
            );
        }
    }
    Ok(())
}

fn validate_axis(
    name: &'static str,
    axis: &[f32],
    expected_size: usize,
) -> Result<(), CartesianSparsificationError> {
    if axis.len() != expected_size {
        return Err(CartesianSparsificationError::AxisLengthMismatch {
            axis: name,
            expected: expected_size,
            actual: axis.len(),
        });
    }
    if axis.iter().any(|value| !value.is_finite()) {
        return Err(CartesianSparsificationError::NonFiniteAxis { axis: name });
    }
    if axis.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(CartesianSparsificationError::NonIncreasingAxis { axis: name });
    }
    Ok(())
}

fn checked_product(shape: &[usize]) -> Result<usize, CartesianSparsificationError> {
    shape.iter().try_fold(1_usize, |product, &size| {
        product
            .checked_mul(size)
            .ok_or(CartesianSparsificationError::ShapeOverflow)
    })
}

fn valid_spatial_domain(
    shape_zyx: [usize; 3],
    margin: usize,
    spatial_mask: Option<&[bool]>,
) -> Result<Vec<bool>, CartesianSparsificationError> {
    if shape_zyx
        .into_iter()
        .any(|size| margin.saturating_mul(2) >= size)
    {
        return Err(CartesianSparsificationError::BoundaryMarginLeavesNoDomain { shape_zyx });
    }
    let [z_size, y_size, x_size] = shape_zyx;
    let mut valid = Vec::with_capacity(checked_product(&shape_zyx)?);
    for z in 0..z_size {
        for y in 0..y_size {
            for x in 0..x_size {
                valid.push(
                    z >= margin
                        && z < z_size - margin
                        && y >= margin
                        && y < y_size - margin
                        && x >= margin
                        && x < x_size - margin,
                );
            }
        }
    }
    if let Some(mask) = spatial_mask {
        for (valid, &masked) in valid.iter_mut().zip(mask) {
            *valid &= masked;
        }
        if !valid.iter().any(|&value| value) {
            return Err(CartesianSparsificationError::SpatialMaskLeavesNoDomain);
        }
    }
    Ok(valid)
}

fn noise_floor_stats(noise_floors: &[f32]) -> NoiseFloorStats {
    let mut positive = noise_floors
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .collect::<Vec<_>>();
    if positive.is_empty() {
        return NoiseFloorStats {
            min: 0.0,
            median: 0.0,
            max: 0.0,
        };
    }
    positive.sort_unstable_by(f32::total_cmp);
    let middle = positive.len() / 2;
    let median = if positive.len() % 2 == 0 {
        (positive[middle - 1] + positive[middle]) / 2.0
    } else {
        positive[middle]
    };
    NoiseFloorStats {
        min: positive[0],
        median,
        max: positive[positive.len() - 1],
    }
}

fn snr_db(magnitude: f32, noise_floor: f32) -> f32 {
    if magnitude <= 0.0 || noise_floor <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * (magnitude / noise_floor).log10()
    }
}

fn limit_doppler_peaks_per_spatial(
    mut candidates: Vec<Candidate>,
    spatial_size: usize,
    limit: Option<usize>,
) -> Vec<Candidate> {
    let Some(limit) = limit else {
        return candidates;
    };
    candidates.sort_unstable_by(|left, right| {
        (left.index % spatial_size)
            .cmp(&(right.index % spatial_size))
            .then_with(|| score_order(left.snr_db, right.snr_db))
            .then_with(|| left.index.cmp(&right.index))
    });
    let mut limited = Vec::with_capacity(candidates.len());
    let mut current_spatial = None;
    let mut retained = 0;
    for candidate in candidates {
        let spatial = candidate.index % spatial_size;
        if current_spatial != Some(spatial) {
            current_spatial = Some(spatial);
            retained = 0;
        }
        if retained < limit {
            limited.push(candidate);
            retained += 1;
        }
    }
    limited
}

fn rank_candidates(
    candidates: Vec<Candidate>,
    input: CartesianSparsificationInput<'_>,
    spatial_size: usize,
    config: CartesianSparsificationConfig,
) -> Vec<Candidate> {
    if config.static_point_capacity_fraction == 1.0 {
        return rank_group(candidates, config.max_points);
    }
    let tolerance = velocity_tolerance(&candidates, input.doppler_velocity_mps, spatial_size);
    let mut static_candidates = Vec::new();
    let mut dynamic_candidates = Vec::new();
    for candidate in candidates {
        let velocity = input.doppler_velocity_mps[candidate.index / spatial_size];
        if velocity.abs() <= config.static_velocity_threshold_mps + tolerance {
            static_candidates.push(candidate);
        } else {
            dynamic_candidates.push(candidate);
        }
    }
    let static_limit = (config.max_points as f64 * f64::from(config.static_point_capacity_fraction))
        .floor() as usize;
    let mut selected = rank_group(static_candidates, static_limit);
    selected.extend(rank_group(
        dynamic_candidates,
        config.max_points - selected.len(),
    ));
    selected.sort_unstable_by(candidate_order);
    selected
}

fn rank_group(mut candidates: Vec<Candidate>, limit: usize) -> Vec<Candidate> {
    candidates.sort_unstable_by(candidate_order);
    candidates.truncate(limit);
    candidates
}

fn candidate_order(left: &Candidate, right: &Candidate) -> Ordering {
    score_order(left.snr_db, right.snr_db).then_with(|| left.index.cmp(&right.index))
}

fn score_order(left: f32, right: f32) -> Ordering {
    right.total_cmp(&left)
}

fn velocity_tolerance(candidates: &[Candidate], velocities: &[f32], spatial_size: usize) -> f32 {
    let max_abs_velocity = candidates.iter().fold(0.0_f32, |maximum, candidate| {
        maximum.max(velocities[candidate.index / spatial_size].abs())
    });
    f32::EPSILON * max_abs_velocity.max(1.0)
}

fn static_candidate_count(
    candidates: &[Candidate],
    velocities: &[f32],
    spatial_size: usize,
    threshold_mps: f32,
) -> usize {
    let tolerance = velocity_tolerance(candidates, velocities, spatial_size);
    candidates
        .iter()
        .filter(|candidate| {
            velocities[candidate.index / spatial_size].abs() <= threshold_mps + tolerance
        })
        .count()
}

fn strongest_valid_candidate(
    input: CartesianSparsificationInput<'_>,
    valid_spatial: &[bool],
    spatial_size: usize,
    noise_floors: &[f32],
) -> Candidate {
    let mut strongest = Candidate {
        index: 0,
        snr_db: f32::NEG_INFINITY,
    };
    let mut strongest_magnitude = f32::NEG_INFINITY;
    for (index, &magnitude) in input.magnitude_dzyx.iter().enumerate() {
        let doppler = index / spatial_size;
        if input.suppressed_doppler_index == Some(doppler) {
            continue;
        }
        if !valid_spatial[index % spatial_size] || magnitude <= strongest_magnitude {
            continue;
        }
        strongest_magnitude = magnitude;
        strongest.index = index;
    }
    let doppler = strongest.index / spatial_size;
    strongest.snr_db = snr_db(strongest_magnitude, noise_floors[doppler]);
    strongest
}

fn materialize_points(
    candidates: Vec<Candidate>,
    input: CartesianSparsificationInput<'_>,
    spatial_size: usize,
) -> Vec<f32> {
    let [_, z_size, y_size, x_size] = input.shape_dzyx;
    let mut points = Vec::with_capacity(candidates.len() * 5);
    for candidate in candidates {
        let doppler = candidate.index / spatial_size;
        let spatial = candidate.index % spatial_size;
        let z = spatial / (y_size * x_size);
        let remaining = spatial % (y_size * x_size);
        let y = remaining / x_size;
        let x = remaining % x_size;
        debug_assert!(z < z_size);
        points.extend_from_slice(&[
            input.x_m[x],
            input.y_m[y],
            input.z_m[z],
            input.doppler_velocity_mps[doppler],
            candidate.snr_db,
        ]);
    }
    points
}

#[cfg(test)]
mod tests {
    use super::{
        CartesianSparsificationConfig, CartesianSparsificationInput, sparsify_cartesian_volume,
        sparsify_cartesian_volume_with_workers,
    };

    fn config() -> CartesianSparsificationConfig {
        CartesianSparsificationConfig {
            min_snr_db: -100.0,
            max_points: 256,
            spatial_peak_radius: 0,
            doppler_peak_radius: 0,
            max_doppler_peaks_per_spatial: None,
            boundary_margin_voxels: 0,
            noise_floor_scale: 1.0,
            static_point_capacity_fraction: 1.0,
            static_velocity_threshold_mps: 0.0,
            strongest_point_fallback: true,
        }
    }

    fn input(volume: &[f32], shape_dzyx: [usize; 4]) -> CartesianSparsificationInput<'_> {
        CartesianSparsificationInput {
            magnitude_dzyx: volume,
            shape_dzyx,
            doppler_velocity_mps: &[-1.0, 1.0],
            z_m: &[-0.5, 0.5],
            y_m: &[-1.0, 0.0, 1.0],
            x_m: &[1.0, 2.0, 3.0],
            spatial_mask_zyx: None,
            suppressed_doppler_index: None,
        }
    }

    #[test]
    fn materializes_metric_points_with_signed_velocity() {
        let mut volume = vec![0.0; 2 * 2 * 3 * 3];
        volume[0] = 2.0;
        volume[2 * 3 * 3 + (3 + 2) * 3 + 2] = 8.0;
        let result = sparsify_cartesian_volume(input(&volume, [2, 2, 3, 3]), config()).unwrap();

        assert_eq!(result.point_count, 2);
        assert_eq!(
            result.points,
            vec![1.0, -1.0, -0.5, -1.0, 0.0, 3.0, 1.0, 0.5, 1.0, 0.0]
        );
    }

    #[test]
    fn keeps_first_equal_spatial_plateau_index() {
        let mut volume = vec![0.0; 3 * 3 * 4];
        volume[(3 + 1) * 4 + 1] = 5.0;
        volume[(3 + 1) * 4 + 2] = 5.0;
        let mut policy = config();
        policy.spatial_peak_radius = 1;
        let result = sparsify_cartesian_volume(
            CartesianSparsificationInput {
                magnitude_dzyx: &volume,
                shape_dzyx: [1, 3, 3, 4],
                doppler_velocity_mps: &[0.0],
                z_m: &[0.0, 1.0, 2.0],
                y_m: &[0.0, 1.0, 2.0],
                x_m: &[0.0, 1.0, 2.0, 3.0],
                spatial_mask_zyx: None,
                suppressed_doppler_index: None,
            },
            policy,
        )
        .unwrap();

        assert_eq!(result.point_count, 1);
        assert_eq!(result.points[0], 1.0);
    }

    #[test]
    fn parallel_sparsification_matches_single_worker() {
        let mut volume = (0..2 * 2 * 3 * 3)
            .map(|index| (index % 11 + 1) as f32)
            .collect::<Vec<_>>();
        volume[7] = volume[6];
        let mut policy = config();
        policy.max_points = 7;
        policy.spatial_peak_radius = 1;
        policy.doppler_peak_radius = 1;
        policy.max_doppler_peaks_per_spatial = Some(1);
        let input = input(&volume, [2, 2, 3, 3]);

        let sequential = sparsify_cartesian_volume_with_workers(input, policy, 1).unwrap();
        let parallel = sparsify_cartesian_volume_with_workers(input, policy, 2).unwrap();

        assert_eq!(parallel, sequential);
    }

    #[test]
    fn keeps_first_equal_doppler_plateau_index() {
        let volume = [5.0, 5.0];
        let mut policy = config();
        policy.doppler_peak_radius = 1;
        let result = sparsify_cartesian_volume(
            CartesianSparsificationInput {
                magnitude_dzyx: &volume,
                shape_dzyx: [2, 1, 1, 1],
                doppler_velocity_mps: &[-1.0, 1.0],
                z_m: &[0.0],
                y_m: &[0.0],
                x_m: &[0.0],
                spatial_mask_zyx: None,
                suppressed_doppler_index: None,
            },
            policy,
        )
        .unwrap();

        assert_eq!(result.point_count, 1);
        assert_eq!(result.points[3], -1.0);
    }

    #[test]
    fn excludes_suppressed_doppler_slice_from_points_and_counts() {
        let volume = [2.0, 8.0];
        let result = sparsify_cartesian_volume(
            CartesianSparsificationInput {
                magnitude_dzyx: &volume,
                shape_dzyx: [2, 1, 1, 1],
                doppler_velocity_mps: &[-1.0, 0.0],
                z_m: &[0.0],
                y_m: &[0.0],
                x_m: &[1.0],
                spatial_mask_zyx: None,
                suppressed_doppler_index: Some(1),
            },
            config(),
        )
        .unwrap();

        assert_eq!(result.point_count, 1);
        assert_eq!(result.points[3], -1.0);
        assert_eq!(result.positive_volume_voxels, 1);
        assert_eq!(result.noise_floor_max, 2.0);
    }
}
