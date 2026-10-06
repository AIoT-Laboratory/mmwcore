//! Candidate-level azimuth and elevation recovery from calibrated array data.

mod azimuth;
mod common;
mod elevation;

pub use azimuth::{
    CandidateAzimuthConfig, CandidateAzimuthInput, CandidateAzimuthPeaks,
    estimate_candidate_azimuths,
};
pub use common::{
    CandidateAoaError, CandidateCubeAxes, CandidateCubeInput, CandidateIndexColumns,
    CandidateMatrixInput,
};
pub use elevation::{
    CandidateElevationColumns, CandidateElevationConfig, CandidateElevationInput,
    CandidateElevations, estimate_candidate_elevations,
};
