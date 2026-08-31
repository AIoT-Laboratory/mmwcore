//! Fixed-geometry Cartesian projection plan.

use std::collections::HashSet;
use std::sync::Arc;

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use super::{CartesianProjectionError, PlanarCartesianProjection, PlanarCartesianProjectionConfig};
use crate::cube::{CubeTransformError, checked_product, validate_shape};

/// Reusable fixed-geometry plan for one planar Cartesian projection route.
pub struct PlanarCartesianProjectionPlan {
    config: PlanarCartesianProjectionConfig,
    source_range_bins: usize,
    source_virtual_channels: usize,
    aperture_channels: Vec<ApertureChannel>,
    aperture_elevation_bins: usize,
    spatial_samples: Vec<ValidSpatialSample>,
    doppler_samples: Vec<ValidDopplerSample>,
    shape_dzyx: [usize; 4],
    spatial_count: usize,
    doppler_start: usize,
    doppler_stop: usize,
    range_start: usize,
    range_stop: usize,
    azimuth_fft: Arc<dyn Fft<f32>>,
    elevation_fft: Arc<dyn Fft<f32>>,
}

impl PlanarCartesianProjectionPlan {
    pub fn new(
        source_range_bins: usize,
        grid_indices: &[(usize, usize)],
        config: PlanarCartesianProjectionConfig,
    ) -> Result<Self, CartesianProjectionError> {
        config.validate()?;
        if source_range_bins == 0 {
            return Err(CartesianProjectionError::InvalidAxisSize);
        }
        let (spatial_samples, spatial_valid_count) = spatial_samples(config, source_range_bins)?;
        let (doppler_samples, doppler_valid_count) = doppler_samples(config)?;
        if spatial_valid_count == 0 {
            return Err(CartesianProjectionError::NoSpatialSupport);
        }
        if doppler_valid_count == 0 {
            return Err(CartesianProjectionError::NoDopplerSupport);
        }
        let (doppler_start, doppler_stop) = selection_bounds(
            doppler_samples
                .iter()
                .filter(|sample| sample.valid)
                .map(|sample| sample.source_index),
            config.source_doppler_bins,
        )?;
        let (range_start, range_stop) = selection_bounds(
            spatial_samples
                .iter()
                .filter(|sample| sample.valid)
                .map(|sample| sample.range_index),
            source_range_bins,
        )?;
        let (aperture_channels, aperture_elevation_bins) = aperture_channels(grid_indices, config)?;
        let valid_spatial_samples =
            build_spatial_interpolation_plan(&spatial_samples, range_start, range_stop, config);
        let valid_doppler_samples =
            build_doppler_interpolation_plan(&doppler_samples, doppler_start, doppler_stop);
        let shape_dzyx = [
            config.target_doppler_bins,
            config.grid_shape_zyx[0],
            config.grid_shape_zyx[1],
            config.grid_shape_zyx[2],
        ];
        let spatial_count = checked_product(&shape_dzyx[1..])?;
        let mut planner = FftPlanner::<f32>::new();
        let azimuth_fft = planner.plan_fft_forward(config.azimuth_n_fft);
        let elevation_fft = planner.plan_fft_forward(config.elevation_n_fft);
        Ok(Self {
            config,
            source_range_bins,
            source_virtual_channels: grid_indices.len(),
            aperture_channels,
            aperture_elevation_bins,
            spatial_samples: valid_spatial_samples,
            doppler_samples: valid_doppler_samples,
            shape_dzyx,
            spatial_count,
            doppler_start,
            doppler_stop,
            range_start,
            range_stop,
            azimuth_fft,
            elevation_fft,
        })
    }

    /// Project one `(frame, doppler, virtual_rx, range)` complex cube to DZYX magnitude.
    pub fn project(
        &self,
        data: &[Complex32],
        shape: &[usize],
    ) -> Result<PlanarCartesianProjection, CartesianProjectionError> {
        self.validate_source(data, shape)?;
        let source_magnitudes = self.source_spatial_magnitudes(data)?;
        let magnitude_dzyx = self.interpolate_doppler_magnitude(&source_magnitudes)?;
        Ok(PlanarCartesianProjection {
            magnitude_dzyx,
            shape_dzyx: self.shape_dzyx,
            doppler_start: self.doppler_start,
            doppler_stop: self.doppler_stop,
            range_start: self.range_start,
            range_stop: self.range_stop,
            spatial_valid_count: self.spatial_samples.len(),
            doppler_valid_count: self.doppler_samples.len(),
        })
    }

    fn validate_source(
        &self,
        data: &[Complex32],
        shape: &[usize],
    ) -> Result<(), CartesianProjectionError> {
        validate_shape(data, shape)?;
        if shape.len() != 4 {
            return Err(CartesianProjectionError::SourceRank {
                actual: shape.len(),
            });
        }
        if shape[0] != 1 {
            return Err(CartesianProjectionError::SourceRequiresSingleFrame { actual: shape[0] });
        }
        if shape[1] != self.config.source_doppler_bins {
            return Err(CartesianProjectionError::SourceDopplerMismatch {
                expected: self.config.source_doppler_bins,
                actual: shape[1],
            });
        }
        if shape[2] != self.source_virtual_channels {
            return Err(CartesianProjectionError::SourceApertureMismatch {
                expected: self.source_virtual_channels,
                actual: shape[2],
            });
        }
        if shape[3] != self.source_range_bins {
            return Err(CartesianProjectionError::SourceRangeMismatch {
                expected: self.source_range_bins,
                actual: shape[3],
            });
        }
        Ok(())
    }

    fn source_spatial_magnitudes(
        &self,
        source: &[Complex32],
    ) -> Result<Vec<f32>, CartesianProjectionError> {
        let selected_doppler_bins = self.doppler_stop - self.doppler_start;
        let worker_count = std::thread::available_parallelism()
            .map_or(1, |parallelism| parallelism.get())
            .min(4)
            .min(selected_doppler_bins);
        self.source_spatial_magnitudes_with_workers(source, worker_count)
    }

    fn source_spatial_magnitudes_with_workers(
        &self,
        source: &[Complex32],
        worker_count: usize,
    ) -> Result<Vec<f32>, CartesianProjectionError> {
        let selected_doppler_bins = self.doppler_stop - self.doppler_start;
        let worker_count = worker_count.clamp(1, selected_doppler_bins);
        let chunk_size = selected_doppler_bins.div_ceil(worker_count);
        let output_size = checked_product(&[selected_doppler_bins, self.spatial_count])?;
        let mut magnitudes = vec![0.0; output_size];
        if worker_count == 1 {
            self.fill_source_spatial_magnitude_chunk(
                source,
                0,
                selected_doppler_bins,
                &mut magnitudes,
            )?;
            return Ok(magnitudes);
        }

        let chunk_length = chunk_size * self.spatial_count;
        std::thread::scope(|scope| -> Result<(), CartesianProjectionError> {
            let mut handles = Vec::with_capacity(worker_count);
            for (chunk_index, output) in magnitudes.chunks_mut(chunk_length).enumerate() {
                let start = chunk_index * chunk_size;
                let stop = (start + chunk_size).min(selected_doppler_bins);
                handles.push(scope.spawn(move || {
                    self.fill_source_spatial_magnitude_chunk(source, start, stop, output)
                }));
            }

            for handle in handles {
                handle
                    .join()
                    .map_err(|_| CartesianProjectionError::WorkerPanicked)??;
            }
            Ok(())
        })?;
        Ok(magnitudes)
    }

    fn fill_source_spatial_magnitude_chunk(
        &self,
        source: &[Complex32],
        selected_doppler_start: usize,
        selected_doppler_stop: usize,
        magnitudes: &mut [f32],
    ) -> Result<(), CartesianProjectionError> {
        let selected_range_bins = self.range_stop - self.range_start;
        let azimuth_shape = [
            selected_range_bins,
            self.aperture_elevation_bins,
            self.config.azimuth_n_fft,
        ];
        let angle_shape = [
            selected_range_bins,
            self.config.azimuth_n_fft,
            self.config.elevation_n_fft,
        ];
        let mut azimuth = vec![Complex32::new(0.0, 0.0); checked_product(&azimuth_shape)?];
        let mut angle = vec![Complex32::new(0.0, 0.0); checked_product(&angle_shape)?];
        let chunk_doppler_bins = selected_doppler_stop - selected_doppler_start;
        debug_assert_eq!(
            magnitudes.len(),
            checked_product(&[chunk_doppler_bins, self.spatial_count])?
        );

        for selected_doppler in selected_doppler_start..selected_doppler_stop {
            azimuth.fill(Complex32::new(0.0, 0.0));
            angle.fill(Complex32::new(0.0, 0.0));
            let source_doppler = self.doppler_start + selected_doppler;
            for channel in &self.aperture_channels {
                let source_start = (source_doppler * self.source_virtual_channels
                    + channel.source_channel)
                    * self.source_range_bins
                    + self.range_start;
                for selected_range in 0..selected_range_bins {
                    let destination = ((selected_range * self.aperture_elevation_bins
                        + channel.elevation)
                        * self.config.azimuth_n_fft)
                        + channel.azimuth;
                    azimuth[destination] = source[source_start + selected_range];
                }
            }
            self.azimuth_fft.process(&mut azimuth);

            for range in 0..selected_range_bins {
                for elevation in 0..self.aperture_elevation_bins {
                    let source_start = (range * self.aperture_elevation_bins + elevation)
                        * self.config.azimuth_n_fft;
                    for azimuth_index in 0..self.config.azimuth_n_fft {
                        let destination = ((range * self.config.azimuth_n_fft + azimuth_index)
                            * self.config.elevation_n_fft)
                            + elevation;
                        angle[destination] = azimuth[source_start + azimuth_index];
                    }
                }
            }
            self.elevation_fft.process(&mut angle);
            let magnitude_base = (selected_doppler - selected_doppler_start) * self.spatial_count;
            for spatial in &self.spatial_samples {
                magnitudes[magnitude_base + spatial.output_index] =
                    interpolate_spatial(&angle, 0, spatial).norm();
            }
        }
        Ok(())
    }

    fn interpolate_doppler_magnitude(
        &self,
        source_magnitudes: &[f32],
    ) -> Result<Vec<f32>, CartesianProjectionError> {
        let mut output = vec![0.0; checked_product(&self.shape_dzyx)?];
        for doppler in &self.doppler_samples {
            let lower_base = doppler.lower_index * self.spatial_count;
            let upper_base = doppler.upper_index * self.spatial_count;
            let output_base = doppler.output_index * self.spatial_count;
            for spatial in &self.spatial_samples {
                let lower_magnitude = source_magnitudes[lower_base + spatial.output_index];
                let upper_magnitude = source_magnitudes[upper_base + spatial.output_index];
                let magnitude =
                    lower_magnitude * (1.0 - doppler.fraction) + upper_magnitude * doppler.fraction;
                output[output_base + spatial.output_index] = magnitude;
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug)]
struct SpatialSample {
    range_index: f32,
    azimuth_index: f32,
    elevation_index: f32,
    valid: bool,
}

#[derive(Clone, Copy, Debug)]
struct DopplerSample {
    source_index: f32,
    valid: bool,
}

#[derive(Clone, Copy, Debug)]
struct ApertureChannel {
    source_channel: usize,
    azimuth: usize,
    elevation: usize,
}

#[derive(Clone, Copy, Debug)]
struct SpatialCorner {
    offset: usize,
    weight: f32,
}

#[derive(Clone, Copy, Debug)]
struct ValidSpatialSample {
    output_index: usize,
    corners: [SpatialCorner; 8],
}

#[derive(Clone, Copy, Debug)]
struct ValidDopplerSample {
    output_index: usize,
    lower_index: usize,
    upper_index: usize,
    fraction: f32,
}

fn spatial_samples(
    config: PlanarCartesianProjectionConfig,
    source_range_bins: usize,
) -> Result<(Vec<SpatialSample>, usize), CartesianProjectionError> {
    let pitch_rad = config.mount_pitch_deg.to_radians();
    let (pitch_sin, pitch_cos) = pitch_rad.sin_cos();
    let [z_size, y_size, x_size] = config.grid_shape_zyx;
    let sample_count = checked_product(&config.grid_shape_zyx)?;
    let mut samples = Vec::with_capacity(sample_count);
    let mut valid_count = 0;
    for z_index in 0..z_size {
        let z = config.grid_origin_xyz_m[2] + z_index as f32 * config.grid_voxel_size_xyz_m[2];
        for y_index in 0..y_size {
            let y = config.grid_origin_xyz_m[1] + y_index as f32 * config.grid_voxel_size_xyz_m[1];
            for x_index in 0..x_size {
                let x =
                    config.grid_origin_xyz_m[0] + x_index as f32 * config.grid_voxel_size_xyz_m[0];
                if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                    return Err(CartesianProjectionError::NonFiniteGridCoordinate);
                }
                // The declared grid is level forward/lateral/up. Rotate each fixed output
                // coordinate back into sensor forward/lateral/up before radar sampling.
                let sensor_x = pitch_cos * x - pitch_sin * (z - config.mount_height_m);
                let sensor_y = y;
                let sensor_z = pitch_sin * x + pitch_cos * (z - config.mount_height_m);
                let radial_range =
                    (sensor_x * sensor_x + sensor_y * sensor_y + sensor_z * sensor_z).sqrt();
                let nonzero = radial_range > f32::EPSILON;
                let azimuth_direction = if nonzero {
                    sensor_y / radial_range
                } else {
                    0.0
                };
                let elevation_direction = if nonzero {
                    sensor_z / radial_range
                } else {
                    0.0
                };
                let range_index = radial_range / config.range_resolution_m;
                let azimuth_index = azimuth_direction
                    * config.aperture_spacing_wavelengths
                    * config.azimuth_n_fft as f32
                    + config.azimuth_n_fft as f32 / 2.0;
                let elevation_index = elevation_direction
                    * config.aperture_spacing_wavelengths
                    * config.elevation_n_fft as f32
                    + config.elevation_n_fft as f32 / 2.0;
                let visible = azimuth_direction * azimuth_direction
                    + elevation_direction * elevation_direction
                    <= 1.0 + 1.0e-6;
                let valid = nonzero
                    && sensor_x >= 0.0
                    && visible
                    && range_index >= 0.0
                    && range_index <= source_range_bins.saturating_sub(1) as f32
                    && azimuth_index >= 0.0
                    && azimuth_index <= config.azimuth_n_fft.saturating_sub(1) as f32
                    && elevation_index >= 0.0
                    && elevation_index <= config.elevation_n_fft.saturating_sub(1) as f32;
                valid_count += if valid { 1 } else { 0 };
                samples.push(SpatialSample {
                    range_index,
                    azimuth_index,
                    elevation_index,
                    valid,
                });
            }
        }
    }
    Ok((samples, valid_count))
}

fn doppler_samples(
    config: PlanarCartesianProjectionConfig,
) -> Result<(Vec<DopplerSample>, usize), CartesianProjectionError> {
    let mut samples = Vec::with_capacity(config.target_doppler_bins);
    let mut valid_count = 0;
    for target_index in 0..config.target_doppler_bins {
        let target_velocity = config.target_velocity_start_mps
            + config.target_velocity_step_mps * target_index as f32;
        if !target_velocity.is_finite() {
            return Err(CartesianProjectionError::NonFiniteDopplerCoordinate);
        }
        let source_index =
            (target_velocity - config.source_velocity_start_mps) / config.source_velocity_step_mps;
        let valid = source_index >= 0.0 && source_index <= (config.source_doppler_bins - 1) as f32;
        valid_count += if valid { 1 } else { 0 };
        samples.push(DopplerSample {
            source_index,
            valid,
        });
    }
    Ok((samples, valid_count))
}

fn selection_bounds(
    indices: impl Iterator<Item = f32>,
    size: usize,
) -> Result<(usize, usize), CartesianProjectionError> {
    let mut min_index = f32::INFINITY;
    let mut max_index = f32::NEG_INFINITY;
    for index in indices {
        min_index = min_index.min(index);
        max_index = max_index.max(index);
    }
    if !min_index.is_finite() || !max_index.is_finite() {
        return Err(CartesianProjectionError::EmptySourceSelection);
    }
    let start = (min_index.floor() as usize).min(size);
    let stop = (max_index.ceil() as usize).saturating_add(1).min(size);
    if start >= stop {
        return Err(CartesianProjectionError::EmptySourceSelection);
    }
    Ok((start, stop))
}

fn aperture_channels(
    grid_indices: &[(usize, usize)],
    config: PlanarCartesianProjectionConfig,
) -> Result<(Vec<ApertureChannel>, usize), CartesianProjectionError> {
    let maximum_elevation = grid_indices
        .iter()
        .map(|&(_, elevation)| elevation)
        .max()
        .ok_or(CubeTransformError::PlanarPositionMismatch {
            expected: 1,
            actual: 0,
        })?;
    let aperture_elevation_bins = maximum_elevation
        .checked_add(1)
        .ok_or(CubeTransformError::ShapeOverflow)?
        .min(config.elevation_n_fft);
    let mut seen = HashSet::new();
    let channels = grid_indices
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(source_channel, (azimuth, elevation))| {
            let first = seen.insert((azimuth, elevation));
            (first && azimuth < config.azimuth_n_fft && elevation < config.elevation_n_fft)
                .then_some(ApertureChannel {
                    source_channel,
                    azimuth,
                    elevation,
                })
        })
        .collect();
    Ok((channels, aperture_elevation_bins))
}

fn build_spatial_interpolation_plan(
    spatial_samples: &[SpatialSample],
    range_start: usize,
    range_stop: usize,
    config: PlanarCartesianProjectionConfig,
) -> Vec<ValidSpatialSample> {
    let selected_range_bins = range_stop - range_start;
    spatial_samples
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, sample)| sample.valid)
        .map(|(output_index, sample)| {
            spatial_interpolation_sample(
                output_index,
                sample,
                range_start,
                selected_range_bins,
                config,
            )
        })
        .collect()
}

fn spatial_interpolation_sample(
    output_index: usize,
    sample: SpatialSample,
    range_start: usize,
    selected_range_bins: usize,
    config: PlanarCartesianProjectionConfig,
) -> ValidSpatialSample {
    let range_index = sample.range_index - range_start as f32;
    let range_lower = range_index.floor() as usize;
    let azimuth_lower = sample.azimuth_index.floor() as usize;
    let elevation_lower = sample.elevation_index.floor() as usize;
    let range_upper = (range_lower + 1).min(selected_range_bins - 1);
    let azimuth_upper = (azimuth_lower + 1).min(config.azimuth_n_fft - 1);
    let elevation_upper = (elevation_lower + 1).min(config.elevation_n_fft - 1);
    let range_fraction = range_index - range_lower as f32;
    let azimuth_fraction = sample.azimuth_index - azimuth_lower as f32;
    let elevation_fraction = sample.elevation_index - elevation_lower as f32;
    let mut corners = [SpatialCorner {
        offset: 0,
        weight: 0.0,
    }; 8];
    let mut corner_index = 0;
    for range_side in 0..2 {
        let range = if range_side == 0 {
            range_lower
        } else {
            range_upper
        };
        for azimuth_side in 0..2 {
            let azimuth = if azimuth_side == 0 {
                azimuth_lower
            } else {
                azimuth_upper
            };
            for elevation_side in 0..2 {
                let elevation = if elevation_side == 0 {
                    elevation_lower
                } else {
                    elevation_upper
                };
                let mut weight = 1.0_f32;
                weight *= if range_side == 0 {
                    1.0 - range_fraction
                } else {
                    range_fraction
                };
                weight *= if azimuth_side == 0 {
                    1.0 - azimuth_fraction
                } else {
                    azimuth_fraction
                };
                weight *= if elevation_side == 0 {
                    1.0 - elevation_fraction
                } else {
                    elevation_fraction
                };
                corners[corner_index] = SpatialCorner {
                    offset: angle_offset(
                        range,
                        unshifted_index(azimuth, config.azimuth_n_fft),
                        unshifted_index(elevation, config.elevation_n_fft),
                        config.azimuth_n_fft,
                        config.elevation_n_fft,
                    ),
                    weight,
                };
                corner_index += 1;
            }
        }
    }
    ValidSpatialSample {
        output_index,
        corners,
    }
}

fn angle_offset(
    range: usize,
    azimuth: usize,
    elevation: usize,
    azimuth_bins: usize,
    elevation_bins: usize,
) -> usize {
    (range * azimuth_bins + azimuth) * elevation_bins + elevation
}

fn unshifted_index(shifted_index: usize, length: usize) -> usize {
    (shifted_index + length.div_ceil(2)) % length
}

fn build_doppler_interpolation_plan(
    samples: &[DopplerSample],
    doppler_start: usize,
    doppler_stop: usize,
) -> Vec<ValidDopplerSample> {
    let selected_doppler_bins = doppler_stop - doppler_start;
    samples
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, sample)| sample.valid)
        .map(|(output_index, sample)| {
            let source_index = sample.source_index - doppler_start as f32;
            let lower_index = (source_index.floor() as usize).min(selected_doppler_bins - 1);
            ValidDopplerSample {
                output_index,
                lower_index,
                upper_index: (lower_index + 1).min(selected_doppler_bins - 1),
                fraction: source_index - lower_index as f32,
            }
        })
        .collect()
}

fn interpolate_spatial(
    angle: &[Complex32],
    doppler_base: usize,
    sample: &ValidSpatialSample,
) -> Complex32 {
    sample
        .corners
        .iter()
        .fold(Complex32::new(0.0, 0.0), |value, corner| {
            value + angle[doppler_base + corner.offset] * corner.weight
        })
}

#[cfg(test)]
mod tests {
    use super::{PlanarCartesianProjectionConfig, PlanarCartesianProjectionPlan};
    use num_complex::Complex32;

    const APERTURE: &[(usize, usize)] = &[(0, 0), (1, 0), (0, 1), (1, 1)];

    fn config(target_velocity_start_mps: f32) -> PlanarCartesianProjectionConfig {
        PlanarCartesianProjectionConfig {
            range_resolution_m: 0.5,
            source_doppler_bins: 3,
            source_velocity_start_mps: -1.0,
            source_velocity_step_mps: 1.0,
            target_doppler_bins: 1,
            target_velocity_start_mps,
            target_velocity_step_mps: 1.0,
            grid_shape_zyx: [1, 1, 1],
            grid_origin_xyz_m: [1.0, 0.0, 1.0],
            grid_voxel_size_xyz_m: [0.5, 0.5, 0.5],
            mount_height_m: 1.0,
            mount_pitch_deg: 0.0,
            azimuth_n_fft: 4,
            elevation_n_fft: 4,
            aperture_spacing_wavelengths: 0.5,
        }
    }

    fn broadside_source() -> Vec<Complex32> {
        let mut source = vec![Complex32::new(0.0, 0.0); 3 * 4 * 4];
        for channel in 0..4 {
            source[(4 + channel) * 4 + 2] = Complex32::new(1.0, 0.0);
        }
        source
    }

    #[test]
    fn projects_broadside_target_to_metric_voxel() {
        let plan = PlanarCartesianProjectionPlan::new(4, APERTURE, config(0.0)).unwrap();
        let result = plan.project(&broadside_source(), &[1, 3, 4, 4]).unwrap();

        assert_eq!(result.shape_dzyx, [1, 1, 1, 1]);
        assert_eq!(result.doppler_start, 1);
        assert_eq!(result.doppler_stop, 2);
        assert_eq!(result.range_start, 2);
        assert_eq!(result.range_stop, 3);
        assert_eq!(result.spatial_valid_count, 1);
        assert_eq!(result.doppler_valid_count, 1);
        assert!((result.magnitude_dzyx[0] - 4.0).abs() < 1.0e-5);
    }

    #[test]
    fn interpolates_physical_doppler_magnitude() {
        let mut source = broadside_source();
        for channel in 0..4 {
            source[(8 + channel) * 4 + 2] = Complex32::new(3.0, 0.0);
        }
        let plan = PlanarCartesianProjectionPlan::new(4, APERTURE, config(0.5)).unwrap();
        let result = plan.project(&source, &[1, 3, 4, 4]).unwrap();

        assert_eq!(result.doppler_start, 1);
        assert_eq!(result.doppler_stop, 3);
        assert!((result.magnitude_dzyx[0] - 8.0).abs() < 1.0e-5);
    }

    #[test]
    fn downward_mount_samples_level_grid_at_supported_angles() {
        let cases = [
            (0.0_f32, 1.0_f32, 1.0_f32),
            (30.0, 3.0_f32.sqrt() / 2.0, 0.5),
            (90.0, 0.0, 0.0),
        ];
        for (pitch_deg, level_forward, level_up) in cases {
            let mut mounted = config(0.0);
            mounted.mount_pitch_deg = pitch_deg;
            mounted.grid_origin_xyz_m = [level_forward, 0.0, level_up];
            let plan = PlanarCartesianProjectionPlan::new(4, APERTURE, mounted).unwrap();
            let result = plan.project(&broadside_source(), &[1, 3, 4, 4]).unwrap();

            assert!((result.magnitude_dzyx[0] - 4.0).abs() < 1.0e-5);
            assert_eq!((result.range_start, result.range_stop), (2, 3));
        }
    }

    #[test]
    fn parallel_doppler_projection_matches_single_worker() {
        let mut parallel_config = config(-1.0);
        parallel_config.target_doppler_bins = 3;
        let plan = PlanarCartesianProjectionPlan::new(4, APERTURE, parallel_config).unwrap();
        let mut source = broadside_source();
        for doppler in 0..3 {
            for channel in 0..4 {
                source[(doppler * 4 + channel) * 4 + 2] = Complex32::new((doppler + 1) as f32, 0.0);
            }
        }

        let sequential = plan
            .source_spatial_magnitudes_with_workers(&source, 1)
            .unwrap();
        let parallel = plan
            .source_spatial_magnitudes_with_workers(&source, 3)
            .unwrap();

        assert_eq!(parallel, sequential);
    }
}
