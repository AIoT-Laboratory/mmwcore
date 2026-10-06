//! Calibrated planar-array projection onto a Cartesian Doppler volume.

mod plan;

use std::fmt;

use crate::cube::CubeTransformError;

pub use plan::PlanarCartesianProjectionPlan;

/// Physical and sampling contract for one planar Cartesian projection.
#[derive(Clone, Copy, Debug)]
pub struct PlanarCartesianProjectionConfig {
    pub range_resolution_m: f32,
    pub source_doppler_bins: usize,
    pub source_velocity_start_mps: f32,
    pub source_velocity_step_mps: f32,
    pub target_doppler_bins: usize,
    pub target_velocity_start_mps: f32,
    pub target_velocity_step_mps: f32,
    pub grid_shape_zyx: [usize; 3],
    pub grid_origin_xyz_m: [f32; 3],
    pub grid_voxel_size_xyz_m: [f32; 3],
    pub mount_height_m: f32,
    pub mount_pitch_deg: f32,
    pub azimuth_n_fft: usize,
    pub elevation_n_fft: usize,
    pub aperture_spacing_wavelengths: f32,
}

impl PlanarCartesianProjectionConfig {
    pub fn validate(self) -> Result<(), CartesianProjectionError> {
        for value in [
            self.range_resolution_m,
            self.source_velocity_step_mps,
            self.target_velocity_step_mps,
            self.aperture_spacing_wavelengths,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(CartesianProjectionError::InvalidPositiveResolution);
            }
        }
        if self.source_doppler_bins <= 1
            || self.target_doppler_bins == 0
            || self.azimuth_n_fft <= 1
            || self.elevation_n_fft <= 1
        {
            return Err(CartesianProjectionError::InvalidAxisSize);
        }
        if !self.source_velocity_start_mps.is_finite()
            || !self.target_velocity_start_mps.is_finite()
        {
            return Err(CartesianProjectionError::InvalidVelocityOrigin);
        }
        if self.grid_shape_zyx.into_iter().any(|size| size == 0) {
            return Err(CartesianProjectionError::InvalidGridShape);
        }
        if self
            .grid_origin_xyz_m
            .into_iter()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(CartesianProjectionError::InvalidGridOrigin);
        }
        if self
            .grid_voxel_size_xyz_m
            .into_iter()
            .any(|size| !size.is_finite() || size <= 0.0)
        {
            return Err(CartesianProjectionError::InvalidGridVoxelSize);
        }
        if !self.mount_height_m.is_finite() || self.mount_height_m <= 0.0 {
            return Err(CartesianProjectionError::InvalidMountHeight);
        }
        if !matches!(self.mount_pitch_deg, 0.0 | 30.0 | 90.0) {
            return Err(CartesianProjectionError::InvalidMountPitch);
        }
        Ok(())
    }
}

/// Native Cartesian magnitude and source-selection diagnostics.
#[derive(Debug)]
pub struct PlanarCartesianProjection {
    pub magnitude_dzyx: Vec<f32>,
    pub shape_dzyx: [usize; 4],
    pub doppler_start: usize,
    pub doppler_stop: usize,
    pub range_start: usize,
    pub range_stop: usize,
    pub spatial_valid_count: usize,
    pub doppler_valid_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CartesianProjectionError {
    Cube(CubeTransformError),
    SourceRank { actual: usize },
    SourceRequiresSingleFrame { actual: usize },
    SourceDopplerMismatch { expected: usize, actual: usize },
    SourceApertureMismatch { expected: usize, actual: usize },
    SourceRangeMismatch { expected: usize, actual: usize },
    InvalidPositiveResolution,
    InvalidAxisSize,
    InvalidVelocityOrigin,
    InvalidGridShape,
    InvalidGridOrigin,
    InvalidGridVoxelSize,
    InvalidMountHeight,
    InvalidMountPitch,
    NoSpatialSupport,
    NoDopplerSupport,
    EmptySourceSelection,
    NonFiniteGridCoordinate,
    NonFiniteDopplerCoordinate,
    WorkerPanicked,
}

impl fmt::Display for CartesianProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cube(error) => error.fmt(formatter),
            Self::SourceRank { actual } => {
                write!(
                    formatter,
                    "Planar Cartesian source rank must be 4; got {actual}."
                )
            }
            Self::SourceRequiresSingleFrame { actual } => write!(
                formatter,
                "Planar Cartesian projection requires exactly one radar frame; got {actual}."
            ),
            Self::SourceDopplerMismatch { expected, actual } => write!(
                formatter,
                "Planar Cartesian source Doppler size must be {expected}; got {actual}."
            ),
            Self::SourceApertureMismatch { expected, actual } => write!(
                formatter,
                "Planar Cartesian source virtual-array size must be {expected}; got {actual}."
            ),
            Self::SourceRangeMismatch { expected, actual } => write!(
                formatter,
                "Planar Cartesian source range size must be {expected}; got {actual}."
            ),
            Self::InvalidPositiveResolution => write!(
                formatter,
                "Cartesian projection resolutions must be finite and positive."
            ),
            Self::InvalidAxisSize => write!(
                formatter,
                "Cartesian projection FFT and Doppler sizes are invalid."
            ),
            Self::InvalidVelocityOrigin => write!(
                formatter,
                "Cartesian projection velocity origins must be finite."
            ),
            Self::InvalidGridShape => write!(
                formatter,
                "Cartesian projection grid_shape_zyx must be positive."
            ),
            Self::InvalidGridOrigin => write!(
                formatter,
                "Cartesian projection grid origin must be finite."
            ),
            Self::InvalidGridVoxelSize => write!(
                formatter,
                "Cartesian projection voxel sizes must be positive."
            ),
            Self::InvalidMountHeight => {
                write!(
                    formatter,
                    "Cartesian mount height must be finite and positive."
                )
            }
            Self::InvalidMountPitch => {
                write!(
                    formatter,
                    "Cartesian mount pitch must be 0, 30, or 90 degrees."
                )
            }
            Self::NoSpatialSupport => write!(
                formatter,
                "Cartesian target grid has no support in the source radar field of view."
            ),
            Self::NoDopplerSupport => write!(
                formatter,
                "Cartesian target Doppler axis has no support in the source radar axis."
            ),
            Self::EmptySourceSelection => write!(
                formatter,
                "Cartesian interpolation source selection is empty."
            ),
            Self::NonFiniteGridCoordinate => write!(
                formatter,
                "Cartesian projection grid coordinates must remain finite."
            ),
            Self::NonFiniteDopplerCoordinate => write!(
                formatter,
                "Cartesian projection Doppler coordinates must remain finite."
            ),
            Self::WorkerPanicked => write!(formatter, "Cartesian projection worker panicked."),
        }
    }
}

impl std::error::Error for CartesianProjectionError {}

impl From<CubeTransformError> for CartesianProjectionError {
    fn from(error: CubeTransformError) -> Self {
        Self::Cube(error)
    }
}
