//! Stateful two-dimensional radar target tracking.

mod cluster;
mod kalman;
mod measurement;
mod metrics;
mod state;

use std::fmt;

use crate::assignment::AssignmentError;
use crate::clustering::ClusterError;

pub use cluster::ClusterTracker2D;
pub use measurement::MeasurementTracker2D;
pub use metrics::{
    TrackObservationMetrics, TrackingMetricsError, TrackingMetricsInput, TrackingSequenceMetrics,
    summarize_tracking_metrics,
};

/// Cartesian association gates for one two-dimensional tracker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackGatingConfig {
    pub(crate) max_distance_m: f64,
    pub(crate) max_radial_velocity_difference_mps: Option<f64>,
    pub(crate) max_mahalanobis_distance: Option<f64>,
}

impl TrackGatingConfig {
    /// Construct validated Cartesian association gates.
    pub fn new(
        max_distance_m: f64,
        max_radial_velocity_difference_mps: Option<f64>,
        max_mahalanobis_distance: Option<f64>,
    ) -> Result<Self, TrackingError> {
        if !is_positive_finite(max_distance_m) {
            return Err(TrackingError::InvalidConfiguration("max_distance_m"));
        }
        if !optional_positive_finite(max_radial_velocity_difference_mps) {
            return Err(TrackingError::InvalidConfiguration(
                "max_radial_velocity_difference_mps",
            ));
        }
        if !optional_positive_finite(max_mahalanobis_distance) {
            return Err(TrackingError::InvalidConfiguration(
                "max_mahalanobis_distance",
            ));
        }
        Ok(Self {
            max_distance_m,
            max_radial_velocity_difference_mps,
            max_mahalanobis_distance,
        })
    }
}

/// Evidence required to allocate a new track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackAllocationConfig {
    pub(crate) min_points: usize,
    pub(crate) min_abs_radial_velocity_mps: f64,
    pub(crate) min_total_snr: Option<f64>,
    pub(crate) max_new_tracks_per_frame: Option<usize>,
}

impl TrackAllocationConfig {
    /// Construct validated track-allocation requirements.
    pub fn new(
        min_points: usize,
        min_abs_radial_velocity_mps: f64,
        min_total_snr: Option<f64>,
        max_new_tracks_per_frame: Option<usize>,
    ) -> Result<Self, TrackingError> {
        if min_points == 0 {
            return Err(TrackingError::InvalidConfiguration("min_points"));
        }
        if !min_abs_radial_velocity_mps.is_finite() || min_abs_radial_velocity_mps < 0.0 {
            return Err(TrackingError::InvalidConfiguration(
                "min_abs_radial_velocity_mps",
            ));
        }
        if !optional_positive_finite(min_total_snr) {
            return Err(TrackingError::InvalidConfiguration("min_total_snr"));
        }
        if max_new_tracks_per_frame == Some(0) {
            return Err(TrackingError::InvalidConfiguration(
                "max_new_tracks_per_frame",
            ));
        }
        Ok(Self {
            min_points,
            min_abs_radial_velocity_mps,
            min_total_snr,
            max_new_tracks_per_frame,
        })
    }
}

/// Hit and miss limits that control a track lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrackLifecycleConfig {
    pub(crate) confirmation_hits: usize,
    pub(crate) tentative_max_misses: usize,
    pub(crate) confirmed_max_misses: usize,
}

impl TrackLifecycleConfig {
    /// Construct validated lifecycle limits.
    pub fn new(
        confirmation_hits: usize,
        tentative_max_misses: usize,
        confirmed_max_misses: usize,
    ) -> Result<Self, TrackingError> {
        if confirmation_hits == 0 {
            return Err(TrackingError::InvalidConfiguration("confirmation_hits"));
        }
        if tentative_max_misses == 0 {
            return Err(TrackingError::InvalidConfiguration("tentative_max_misses"));
        }
        if confirmed_max_misses == 0 {
            return Err(TrackingError::InvalidConfiguration("confirmed_max_misses"));
        }
        Ok(Self {
            confirmation_hits,
            tentative_max_misses,
            confirmed_max_misses,
        })
    }
}

/// Inclusive Cartesian tracking region in radar x/y coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingBox2D {
    pub(crate) x_min_m: f64,
    pub(crate) x_max_m: f64,
    pub(crate) y_min_m: f64,
    pub(crate) y_max_m: f64,
}

impl TrackingBox2D {
    /// Construct one finite, non-empty tracking boundary.
    pub fn new(
        x_min_m: f64,
        x_max_m: f64,
        y_min_m: f64,
        y_max_m: f64,
    ) -> Result<Self, TrackingError> {
        if !x_min_m.is_finite()
            || !x_max_m.is_finite()
            || !y_min_m.is_finite()
            || !y_max_m.is_finite()
            || x_min_m >= x_max_m
            || y_min_m >= y_max_m
        {
            return Err(TrackingError::InvalidConfiguration("boundary_boxes"));
        }
        Ok(Self {
            x_min_m,
            x_max_m,
            y_min_m,
            y_max_m,
        })
    }

    pub(crate) fn contains(self, x_m: f64, y_m: f64) -> bool {
        self.x_min_m <= x_m && x_m <= self.x_max_m && self.y_min_m <= y_m && y_m <= self.y_max_m
    }
}

/// Scene regions that constrain allocation and track lifetime.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackSceneryConfig {
    pub(crate) boundary_boxes: Vec<TrackingBox2D>,
    pub(crate) outside_max_frames: usize,
}

impl TrackSceneryConfig {
    /// Construct validated tracking scenery.
    pub fn new(
        boundary_boxes: Vec<TrackingBox2D>,
        outside_max_frames: usize,
    ) -> Result<Self, TrackingError> {
        if outside_max_frames == 0 {
            return Err(TrackingError::InvalidConfiguration("outside_max_frames"));
        }
        Ok(Self {
            boundary_boxes,
            outside_max_frames,
        })
    }

    pub(crate) fn contains(&self, x_m: f64, y_m: f64) -> bool {
        self.boundary_boxes.is_empty()
            || self
                .boundary_boxes
                .iter()
                .copied()
                .any(|boundary| boundary.contains(x_m, y_m))
    }
}

/// Constant-velocity state dynamics for a two-dimensional tracker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerDynamicsConfig {
    pub(crate) frame_period_s: f64,
    pub(crate) max_acceleration_mps2: [f64; 2],
    pub(crate) measurement_noise_m: f64,
    pub(crate) initial_velocity_std_mps: f64,
    pub(crate) extent_covariance_smoothing: f64,
}

impl TrackerDynamicsConfig {
    /// Construct validated constant-velocity dynamics.
    pub fn new(
        frame_period_s: f64,
        max_acceleration_mps2: [f64; 2],
        measurement_noise_m: f64,
        initial_velocity_std_mps: f64,
        extent_covariance_smoothing: f64,
    ) -> Result<Self, TrackingError> {
        if !is_positive_finite(frame_period_s) {
            return Err(TrackingError::InvalidConfiguration("frame_period_s"));
        }
        if max_acceleration_mps2
            .iter()
            .any(|&value| !is_positive_finite(value))
        {
            return Err(TrackingError::InvalidConfiguration("max_acceleration_mps2"));
        }
        if !is_positive_finite(measurement_noise_m) {
            return Err(TrackingError::InvalidConfiguration("measurement_noise_m"));
        }
        if !is_positive_finite(initial_velocity_std_mps) {
            return Err(TrackingError::InvalidConfiguration(
                "initial_velocity_std_mps",
            ));
        }
        if !(extent_covariance_smoothing.is_finite()
            && 0.0 < extent_covariance_smoothing
            && extent_covariance_smoothing <= 1.0)
        {
            return Err(TrackingError::InvalidConfiguration(
                "extent_covariance_smoothing",
            ));
        }
        Ok(Self {
            frame_period_s,
            max_acceleration_mps2,
            measurement_noise_m,
            initial_velocity_std_mps,
            extent_covariance_smoothing,
        })
    }
}

/// Complete native configuration for a two-dimensional tracker.
#[derive(Clone, Debug, PartialEq)]
pub struct Tracker2DConfig {
    pub(crate) dynamics: TrackerDynamicsConfig,
    pub(crate) gating: TrackGatingConfig,
    pub(crate) allocation: TrackAllocationConfig,
    pub(crate) lifecycle: TrackLifecycleConfig,
    pub(crate) scenery: TrackSceneryConfig,
    pub(crate) max_tracks: usize,
}

impl Tracker2DConfig {
    /// Construct one validated native tracker configuration.
    pub fn new(
        dynamics: TrackerDynamicsConfig,
        gating: TrackGatingConfig,
        allocation: TrackAllocationConfig,
        lifecycle: TrackLifecycleConfig,
        scenery: TrackSceneryConfig,
        max_tracks: usize,
    ) -> Result<Self, TrackingError> {
        if max_tracks == 0 {
            return Err(TrackingError::InvalidConfiguration("max_tracks"));
        }
        Ok(Self {
            dynamics,
            gating,
            allocation,
            lifecycle,
            scenery,
            max_tracks,
        })
    }
}

/// Contiguous cluster summaries for one tracker step.
#[derive(Clone, Copy, Debug)]
pub struct ClusterMeasurements<'a> {
    pub centers: &'a [f32],
    pub extents: &'a [f32],
    pub mean_velocities: &'a [f32],
    pub point_counts: &'a [i64],
}

/// Packed Cartesian point measurements for one measurement-tracker step.
#[derive(Clone, Copy, Debug)]
pub struct PointMeasurements<'a> {
    pub coordinates: &'a [f32],
    pub velocities: &'a [f32],
    pub snrs: &'a [f32],
}

/// Lifecycle code for one native tracker state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum NativeTrackStatus {
    Tentative = 0,
    Confirmed = 1,
    Coasting = 2,
}

impl NativeTrackStatus {
    pub(crate) fn code(self) -> u8 {
        self as u8
    }
}

/// Rust-owned arrays describing one tracker update.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackStepResult {
    pub track_ids: Vec<i64>,
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub position_covariances: Vec<f32>,
    pub extent_covariances: Vec<f32>,
    pub statuses: Vec<u8>,
    pub ages: Vec<i64>,
    pub missed_counts: Vec<i64>,
    pub observation_track_ids: Vec<i64>,
}

/// Native tracker validation and numerical errors.
#[derive(Clone, Debug, PartialEq)]
pub enum TrackingError {
    InvalidConfiguration(&'static str),
    ClusterMatrixLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    ClusterVectorLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteClusterValues {
        name: &'static str,
    },
    NegativeClusterExtent {
        index: usize,
        value: f32,
    },
    NonPositiveClusterPointCount {
        index: usize,
        value: i64,
    },
    ClusterTrackingDoesNotSupportSnrAllocation,
    MissingAllocationSnr,
    PointMatrixLength {
        expected: usize,
        actual: usize,
    },
    PointVectorLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinitePointValues {
        name: &'static str,
    },
    SingularInnovationCovariance,
    StateLockPoisoned,
    TrackIdOverflow,
    Assignment(AssignmentError),
    Clustering(ClusterError),
}

impl fmt::Display for TrackingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(name) => {
                write!(formatter, "Tracking configuration {name} is invalid.")
            }
            Self::ClusterMatrixLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "Cluster {name} matrix has {actual} values; expected {expected}."
            ),
            Self::ClusterVectorLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "Cluster {name} vector has {actual} values; expected {expected}."
            ),
            Self::NonFiniteClusterValues { name } => {
                write!(formatter, "Cluster {name} contains NaN or Inf values.")
            }
            Self::NegativeClusterExtent { index, value } => write!(
                formatter,
                "Cluster extent at index {index} must be non-negative; got {value}."
            ),
            Self::NonPositiveClusterPointCount { index, value } => write!(
                formatter,
                "Cluster point count at index {index} must be positive; got {value}."
            ),
            Self::ClusterTrackingDoesNotSupportSnrAllocation => write!(
                formatter,
                "Cluster tracking does not support minimum total SNR allocation."
            ),
            Self::MissingAllocationSnr => write!(
                formatter,
                "Configured minimum total SNR allocation requires point SNR values."
            ),
            Self::PointMatrixLength { expected, actual } => write!(
                formatter,
                "Point coordinate matrix has {actual} values; expected {expected}."
            ),
            Self::PointVectorLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "Point {name} vector has {actual} values; expected {expected}."
            ),
            Self::NonFinitePointValues { name } => {
                write!(formatter, "Point {name} contains NaN or Inf values.")
            }
            Self::SingularInnovationCovariance => {
                write!(formatter, "Tracker innovation covariance is singular.")
            }
            Self::StateLockPoisoned => write!(formatter, "Native tracker state lock is poisoned."),
            Self::TrackIdOverflow => write!(formatter, "Native track ID exceeds int64."),
            Self::Assignment(error) => error.fmt(formatter),
            Self::Clustering(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TrackingError {}

impl From<AssignmentError> for TrackingError {
    fn from(error: AssignmentError) -> Self {
        Self::Assignment(error)
    }
}

impl From<ClusterError> for TrackingError {
    fn from(error: ClusterError) -> Self {
        Self::Clustering(error)
    }
}

fn is_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn optional_positive_finite(value: Option<f64>) -> bool {
    value.is_none_or(is_positive_finite)
}
