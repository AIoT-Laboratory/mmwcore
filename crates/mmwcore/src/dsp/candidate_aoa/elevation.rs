use crate::exact_candidate_index;
use crate::fft::FftWindow;

use super::CandidateAoaError;
use super::common::{
    CandidateCubeInput, CandidateCubeLayout, CandidateIndexColumns, CandidateMatrixInput,
    candidate_row, extract_candidate_vectors, row_spectra, validate_candidates, validate_column,
    validate_index_columns,
};

/// Additional candidate columns required for elevation recovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateElevationColumns {
    pub indices: CandidateIndexColumns,
    pub azimuth_bin: usize,
    pub azimuth_rad: usize,
}

/// Candidate-level elevation input from a pair of horizontal virtual rows.
#[derive(Clone, Copy, Debug)]
pub struct CandidateElevationInput<'a> {
    pub cube: CandidateCubeInput<'a>,
    pub candidates: CandidateMatrixInput<'a>,
    pub columns: CandidateElevationColumns,
    pub azimuth_antenna_indices: &'a [usize],
    pub elevation_antenna_indices: &'a [usize],
    pub azimuth_positions_wavelengths: &'a [f64],
    pub azimuth_position_count: usize,
    pub elevation_positions_wavelengths: &'a [f64],
    pub elevation_position_count: usize,
}

/// Shared angle FFT configuration for paired-row elevation recovery.
#[derive(Clone, Copy, Debug)]
pub struct CandidateElevationConfig {
    pub n_fft: usize,
    pub window: FftWindow,
    pub fftshift: bool,
}

/// Valid elevation estimates retained in input candidate order.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateElevations {
    pub valid_candidate_indices: Vec<usize>,
    pub angles_rad: Vec<f32>,
    pub magnitudes: Vec<f32>,
    pub row_offsets_wavelengths: [f64; 2],
}

/// Recover physical elevations from paired horizontal virtual rows.
pub fn estimate_candidate_elevations(
    input: CandidateElevationInput<'_>,
    config: CandidateElevationConfig,
) -> Result<CandidateElevations, CandidateAoaError> {
    let cube = CandidateCubeLayout::new(input.cube)?;
    validate_candidates(input.candidates)?;
    validate_index_columns(input.candidates.shape[1], input.columns.indices)?;
    validate_column(
        input.candidates.shape[1],
        "azimuth_bin",
        input.columns.azimuth_bin,
    )?;
    validate_column(
        input.candidates.shape[1],
        "azimuth_rad",
        input.columns.azimuth_rad,
    )?;
    validate_subarray_layout(
        "azimuth",
        input.azimuth_antenna_indices,
        input.azimuth_positions_wavelengths,
        input.azimuth_position_count,
    )?;
    validate_subarray_layout(
        "elevation",
        input.elevation_antenna_indices,
        input.elevation_positions_wavelengths,
        input.elevation_position_count,
    )?;
    validate_subarray_indices(input.azimuth_antenna_indices, cube.antenna_count())?;
    validate_subarray_indices(input.elevation_antenna_indices, cube.antenna_count())?;

    let azimuth_vectors = extract_candidate_vectors(
        &cube,
        input.candidates,
        input.columns.indices,
        input.azimuth_antenna_indices,
    )?;
    let elevation_vectors = extract_candidate_vectors(
        &cube,
        input.candidates,
        input.columns.indices,
        input.elevation_antenna_indices,
    )?;
    let azimuth_spectra = row_spectra(
        &azimuth_vectors,
        input.candidates.shape[0],
        input.azimuth_antenna_indices.len(),
        config.n_fft,
        config.window,
        config.fftshift,
    )?;
    let elevation_spectra = row_spectra(
        &elevation_vectors,
        input.candidates.shape[0],
        input.elevation_antenna_indices.len(),
        config.n_fft,
        config.window,
        config.fftshift,
    )?;
    let row_offsets_wavelengths = paired_row_offsets(
        input.azimuth_positions_wavelengths,
        input.azimuth_position_count,
        input.elevation_positions_wavelengths,
        input.elevation_position_count,
    )?;
    let x_offset = row_offsets_wavelengths[0] as f32;
    let z_offset = row_offsets_wavelengths[1] as f32;
    let ambiguity_period = 1.0 / z_offset.abs();

    let candidate_count = input.candidates.shape[0];
    let mut valid_candidate_indices = Vec::with_capacity(candidate_count);
    let mut angles_rad = Vec::with_capacity(candidate_count);
    let mut magnitudes = Vec::with_capacity(candidate_count);
    for candidate_index in 0..candidate_count {
        let row = candidate_row(input.candidates, candidate_index);
        let peak_bin = exact_candidate_index(row[input.columns.azimuth_bin], config.n_fft)
            .ok_or(CandidateAoaError::AzimuthBinOutOfBounds)?;
        let azimuth_peak = azimuth_spectra[candidate_index * config.n_fft + peak_bin];
        let elevation_peak = elevation_spectra[candidate_index * config.n_fft + peak_bin];
        let lateral_direction = row[input.columns.azimuth_rad].sin();
        let phase_cycles = (elevation_peak * azimuth_peak.conj()).arg() / core::f32::consts::TAU;
        let vertical_direction = ((phase_cycles - x_offset * lateral_direction) / z_offset
            + ambiguity_period / 2.0)
            .rem_euclid(ambiguity_period)
            - ambiguity_period / 2.0;
        let physically_valid = vertical_direction.abs() <= 1.0 + 1e-6
            && lateral_direction.powi(2) + vertical_direction.powi(2) <= 1.0 + 1e-6;
        if physically_valid {
            valid_candidate_indices.push(candidate_index);
            angles_rad.push(vertical_direction.clamp(-1.0, 1.0).asin());
            magnitudes.push(elevation_peak.norm());
        }
    }
    Ok(CandidateElevations {
        valid_candidate_indices,
        angles_rad,
        magnitudes,
        row_offsets_wavelengths,
    })
}

fn validate_subarray_layout(
    name: &'static str,
    indices: &[usize],
    positions: &[f64],
    position_count: usize,
) -> Result<(), CandidateAoaError> {
    let expected = position_count
        .checked_mul(3)
        .ok_or(CandidateAoaError::CandidateShapeOverflow)?;
    if positions.len() != expected {
        return Err(CandidateAoaError::SubarrayLayoutMismatch {
            name,
            expected,
            actual: positions.len(),
        });
    }
    if indices.len() != position_count {
        return Err(CandidateAoaError::SubarrayLayoutMismatch {
            name,
            expected: indices.len(),
            actual: position_count,
        });
    }
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(CandidateAoaError::NonFiniteCandidates);
    }
    Ok(())
}

fn validate_subarray_indices(
    indices: &[usize],
    antenna_count: usize,
) -> Result<(), CandidateAoaError> {
    if indices.iter().any(|&index| index >= antenna_count) {
        return Err(CandidateAoaError::SubarrayIndexOutOfBounds);
    }
    Ok(())
}

fn paired_row_offsets(
    azimuth_positions: &[f64],
    azimuth_count: usize,
    elevation_positions: &[f64],
    elevation_count: usize,
) -> Result<[f64; 2], CandidateAoaError> {
    validate_horizontal_row("azimuth", azimuth_positions, azimuth_count)?;
    validate_horizontal_row("elevation", elevation_positions, elevation_count)?;
    let azimuth_spacing = position(azimuth_positions, 1, 0) - position(azimuth_positions, 0, 0);
    let elevation_spacing =
        position(elevation_positions, 1, 0) - position(elevation_positions, 0, 0);
    if !all_close(azimuth_spacing, elevation_spacing) {
        return Err(CandidateAoaError::UnequalRowSpacing);
    }
    if !all_close(
        position(azimuth_positions, 0, 1),
        position(elevation_positions, 0, 1),
    ) {
        return Err(CandidateAoaError::DifferentForwardCoordinates);
    }
    let x_offset = position(elevation_positions, 0, 0) - position(azimuth_positions, 0, 0);
    let z_offset = position(elevation_positions, 0, 2) - position(azimuth_positions, 0, 2);
    if all_close(z_offset, 0.0) || z_offset.abs() > 0.5 + 1e-6 {
        return Err(CandidateAoaError::InvalidVerticalSeparation);
    }
    Ok([x_offset, z_offset])
}

fn validate_horizontal_row(
    name: &'static str,
    positions: &[f64],
    position_count: usize,
) -> Result<(), CandidateAoaError> {
    if position_count < 2 {
        return Err(CandidateAoaError::RowRequiresTwoAntennas { name });
    }
    let y0 = position(positions, 0, 1);
    let z0 = position(positions, 0, 2);
    if (1..position_count).any(|index| {
        !all_close(position(positions, index, 1), y0)
            || !all_close(position(positions, index, 2), z0)
    }) {
        return Err(CandidateAoaError::NonHorizontalRow { name });
    }
    let spacing = position(positions, 1, 0) - position(positions, 0, 0);
    if spacing <= 0.0
        || (2..position_count).any(|index| {
            let current = position(positions, index, 0) - position(positions, index - 1, 0);
            current <= 0.0 || !all_close(current, spacing)
        })
    {
        return Err(CandidateAoaError::UnequalRowSpacing);
    }
    Ok(())
}

fn position(positions: &[f64], index: usize, axis: usize) -> f64 {
    positions[index * 3 + axis]
}

fn all_close(value: f64, reference: f64) -> bool {
    (value - reference).abs() <= 1e-8 + 1e-5 * reference.abs()
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::{
        CandidateElevationColumns, CandidateElevationConfig, CandidateElevationInput,
        estimate_candidate_elevations,
    };
    use crate::candidate_aoa::common::{
        CandidateCubeAxes, CandidateCubeInput, CandidateIndexColumns, CandidateMatrixInput,
    };
    use crate::fft::FftWindow;

    #[test]
    fn recovers_paired_row_elevation_direction() {
        let lateral_direction = 0.25_f32;
        let vertical_direction = 0.25_f32;
        let azimuth_positions = [0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 1.0, 0.0, 0.5, 1.5, 0.0, 0.5];
        let elevation_positions = [1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 2.5, 0.0, 0.0];
        let mut cube = Vec::new();
        for positions in [&azimuth_positions[..], &elevation_positions[..]] {
            for antenna in 0..4 {
                let phase = core::f32::consts::TAU
                    * (positions[antenna * 3] as f32 * lateral_direction
                        + positions[antenna * 3 + 2] as f32 * vertical_direction);
                cube.push(Complex32::from_polar(1.0, phase));
            }
        }
        let candidates = [0.0, 0.0, 0.0, 5.0, lateral_direction.asin()];
        let result = estimate_candidate_elevations(
            CandidateElevationInput {
                cube: CandidateCubeInput {
                    data: &cube,
                    shape: &[1, 1, 8, 1],
                    axes: CandidateCubeAxes {
                        frame: 0,
                        doppler: 1,
                        antenna: 2,
                        range: 3,
                    },
                },
                candidates: CandidateMatrixInput {
                    values: &candidates,
                    shape: [1, 5],
                },
                columns: CandidateElevationColumns {
                    indices: CandidateIndexColumns {
                        frame: 0,
                        range: 1,
                        doppler: 2,
                    },
                    azimuth_bin: 3,
                    azimuth_rad: 4,
                },
                azimuth_antenna_indices: &[0, 1, 2, 3],
                elevation_antenna_indices: &[4, 5, 6, 7],
                azimuth_positions_wavelengths: &azimuth_positions,
                azimuth_position_count: 4,
                elevation_positions_wavelengths: &elevation_positions,
                elevation_position_count: 4,
            },
            CandidateElevationConfig {
                n_fft: 8,
                window: FftWindow::None,
                fftshift: true,
            },
        )
        .unwrap();

        assert_eq!(result.valid_candidate_indices, [0]);
        assert!((result.angles_rad[0] - vertical_direction.asin()).abs() < 1e-5);
        assert_eq!(result.row_offsets_wavelengths, [1.0, -0.5]);
    }
}
