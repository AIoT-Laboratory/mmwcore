use crate::angle::{
    AngleAxis, AngleBinCalibrationConfig, AngleBinCalibrationInput, calibrate_angle_bins,
};
use crate::fft::FftWindow;

use super::CandidateAoaError;
use super::common::{
    CandidateCubeInput, CandidateCubeLayout, CandidateIndexColumns, CandidateMatrixInput,
    extract_candidate_vectors, first_peak, row_spectra, validate_candidates,
    validate_index_columns,
};

/// Candidate-level azimuth input and ULA geometry.
#[derive(Clone, Copy, Debug)]
pub struct CandidateAzimuthInput<'a> {
    pub cube: CandidateCubeInput<'a>,
    pub candidates: CandidateMatrixInput<'a>,
    pub columns: CandidateIndexColumns,
    pub positions_wavelengths: &'a [f32],
    pub position_count: usize,
}

/// Angle FFT configuration for candidate-level azimuth recovery.
#[derive(Clone, Copy, Debug)]
pub struct CandidateAzimuthConfig {
    pub n_fft: usize,
    pub window: FftWindow,
    pub fftshift: bool,
    pub angle_axis: AngleAxis,
}

/// One strongest azimuth peak for every input candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateAzimuthPeaks {
    pub peak_bins: Vec<usize>,
    pub angles_rad: Vec<f32>,
    pub magnitudes: Vec<f32>,
}

/// Recover one calibrated azimuth peak for each range-Doppler candidate.
pub fn estimate_candidate_azimuths(
    input: CandidateAzimuthInput<'_>,
    config: CandidateAzimuthConfig,
) -> Result<CandidateAzimuthPeaks, CandidateAoaError> {
    let cube = CandidateCubeLayout::new(input.cube)?;
    validate_candidates(input.candidates)?;
    validate_index_columns(input.candidates.shape[1], input.columns)?;
    let antenna_count = cube.antenna_count();
    if input.position_count != antenna_count {
        return Err(CandidateAoaError::LayoutAntennaMismatch {
            expected: antenna_count,
            actual: input.position_count,
        });
    }
    let antenna_indices = (0..antenna_count).collect::<Vec<_>>();
    let vectors =
        extract_candidate_vectors(&cube, input.candidates, input.columns, &antenna_indices)?;
    let spectra = row_spectra(
        &vectors,
        input.candidates.shape[0],
        antenna_count,
        config.n_fft,
        config.window,
        config.fftshift,
    )?;
    let angles = calibrate_angle_bins(
        AngleBinCalibrationInput {
            positions_wavelengths: input.positions_wavelengths,
            position_count: input.position_count,
        },
        AngleBinCalibrationConfig {
            num_bins: config.n_fft,
            axis: config.angle_axis,
            fftshift: config.fftshift,
        },
    )?;

    let candidate_count = input.candidates.shape[0];
    let mut peak_bins = Vec::with_capacity(candidate_count);
    let mut angles_rad = Vec::with_capacity(candidate_count);
    let mut magnitudes = Vec::with_capacity(candidate_count);
    for row in spectra.chunks_exact(config.n_fft) {
        let (peak_bin, magnitude) = first_peak(row);
        peak_bins.push(peak_bin);
        angles_rad.push(angles[peak_bin]);
        magnitudes.push(magnitude);
    }
    Ok(CandidateAzimuthPeaks {
        peak_bins,
        angles_rad,
        magnitudes,
    })
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::{CandidateAzimuthConfig, CandidateAzimuthInput, estimate_candidate_azimuths};
    use crate::angle::AngleAxis;
    use crate::candidate_aoa::common::{
        CandidateCubeAxes, CandidateCubeInput, CandidateIndexColumns, CandidateMatrixInput,
    };
    use crate::fft::FftWindow;

    #[test]
    fn recovers_shifted_candidate_azimuth_peak() {
        let cube = vec![Complex32::new(1.0, 0.0); 8];
        let candidates = [0.0, 1.0, 0.0];
        let positions = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let result = estimate_candidate_azimuths(
            CandidateAzimuthInput {
                cube: CandidateCubeInput {
                    data: &cube,
                    shape: &[1, 1, 4, 2],
                    axes: CandidateCubeAxes {
                        frame: 0,
                        doppler: 1,
                        antenna: 2,
                        range: 3,
                    },
                },
                candidates: CandidateMatrixInput {
                    values: &candidates,
                    shape: [1, 3],
                },
                columns: CandidateIndexColumns {
                    frame: 0,
                    range: 1,
                    doppler: 2,
                },
                positions_wavelengths: &positions,
                position_count: 4,
            },
            CandidateAzimuthConfig {
                n_fft: 4,
                window: FftWindow::None,
                fftshift: true,
                angle_axis: AngleAxis::Azimuth,
            },
        )
        .unwrap();

        assert_eq!(result.peak_bins, [2]);
        assert_eq!(result.angles_rad, [0.0]);
        assert_eq!(result.magnitudes, [4.0]);
    }
}
