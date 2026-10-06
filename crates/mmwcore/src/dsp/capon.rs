//! Floating-point ISK dynamic Capon frontend (Toolbox 4.00.00.05 method 1).
//!
//! This deliberately supports the default single-elevation, RA-CASO chain only.
//! It retains the detector's edge/order quirks, but is not DSP bit emulation.
//! Input: uncompensated, unwindowed slow-time samples in (loop, antenna, range)
//! order; TX0/TX1/TX2, four RX each. Output coordinates: forward/right/up.

mod cfar;
mod matrix;

use num_complex::{Complex32, Complex64};
use rustfft::FftPlanner;
use serde::Serialize;

const ANTENNAS: usize = 12;
const ROW: [usize; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
const M: [f64; 12] = [0., -1., -2., -3., -2., -3., -4., -5., -4., -5., -6., -7.];
const N: [f64; 12] = [-1., -1., -1., -1., 0., 0., 0., 0., -1., -1., -1., -1.];
pub const AZIMUTH_BINS: usize = 187;
pub const ELEVATION_BINS: usize = 27;

/// Physical bin spacing; velocity spacing corresponds to `doppler_bins`, not
/// necessarily the number of measured loops when zero-padding is requested.
#[derive(Clone, Copy, Debug)]
pub struct IskCaponConfig {
    pub range_resolution_m: f64,
    pub velocity_resolution_mps: f64,
    pub doppler_bins: usize,
}

impl IskCaponConfig {
    fn validate(self, loops: usize, ranges: usize) -> Result<(), String> {
        if !self.range_resolution_m.is_finite()
            || self.range_resolution_m <= 0.
            || !self.velocity_resolution_mps.is_finite()
            || self.velocity_resolution_mps <= 0.
        {
            return Err("Capon range/velocity bin spacing must be finite and positive".into());
        }
        if loops < 2
            || ranges < 37
            || !self.doppler_bins.is_power_of_two()
            || self.doppler_bins < loops
            || self.doppler_bins > 65536
        {
            return Err("Capon requires loops >= 2, ranges >= 37 and power-of-two Doppler FFT >= loops (<= 65536)".into());
        }
        if self.range_resolution_m * ranges as f64 > f32::MAX as f64
            || self.velocity_resolution_mps * (self.doppler_bins / 2) as f64 > f32::MAX as f64
        {
            return Err("Capon physical coordinates must fit finite float32".into());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct CaponDetection {
    pub range_bin: usize,
    pub azimuth_bin: usize,
    pub elevation_bin: usize,
    pub elevation_interpolated_bin: f64,
    pub doppler_bin: i32,
    pub xyz_m: [f64; 3],
    pub velocity_mps: f64,
    /// Linear range-CFAR ratio; no DSP int16 packing or dB conversion.
    pub snr: f64,
    pub range_noise: f32,
    pub ra_power: f32,
    pub elevation_power: f64,
}

#[derive(Debug, Default, Serialize)]
pub struct CaponDiagnostics {
    pub zero_covariance_ranges: usize,
    pub range_candidates: usize,
    pub accepted_before_capacity: usize,
    pub capacity_dropped: usize,
    pub nonvisible_dropped: usize,
    pub zero_noise_dropped: usize,
}

#[derive(Debug)]
pub struct IskCaponFrame {
    /// Row-major (TI nu, range) linear Capon power; no logarithm.
    pub ra_power: Vec<f32>,
    pub detections: Vec<CaponDetection>,
    pub diagnostics: CaponDiagnostics,
}

pub fn nu_grid() -> Vec<f64> {
    grid(70., 0.75, AZIMUTH_BINS)
}

pub fn mu_grid() -> Vec<f64> {
    grid(20., 1.5, ELEVATION_BINS)
}

fn grid(fov: f64, step: f64, count: usize) -> Vec<f64> {
    let start = -fov.to_radians().sin();
    (0..count)
        .map(|i| start - start * step / fov * i as f64)
        .collect()
}

fn steering(nu: f64, mu: f64) -> Vec<Complex64> {
    (0..ANTENNAS)
        .map(|j| {
            Complex64::from_polar(1., -std::f64::consts::PI * (0.9813 * M[j] * nu + N[j] * mu))
        })
        .collect()
}

fn response(inverse: &[Complex64], a: &[Complex64]) -> (f64, Vec<Complex64>) {
    let weights = matrix::multiply(inverse, a);
    let denominator: f64 = a.iter().zip(&weights).map(|(a, w)| (a.conj() * w).re).sum();
    (denominator.recip(), weights)
}

/// Run all dynamic stages after the range FFT. Static means are removed once.
pub fn isk_capon(
    data: &[Complex32],
    loops: usize,
    ranges: usize,
    config: IskCaponConfig,
) -> Result<IskCaponFrame, String> {
    config.validate(loops, ranges)?;
    if loops
        .checked_mul(ANTENNAS)
        .and_then(|n| n.checked_mul(ranges))
        != Some(data.len())
    {
        return Err("Capon input must have shape (loop, 12, range)".into());
    }
    if data.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err("Capon input must be finite".into());
    }
    let nu = nu_grid();
    let mu = mu_grid();
    let row_steering: Vec<Vec<_>> = nu
        .iter()
        .map(|&u| {
            let all = steering(u, 0.);
            ROW.iter().map(|&j| all[j]).collect()
        })
        .collect();
    let mut samples = Vec::with_capacity(ranges);
    let mut covariances = Vec::with_capacity(ranges);
    let mut output = IskCaponFrame {
        ra_power: vec![0.; AZIMUTH_BINS * ranges],
        detections: Vec::new(),
        diagnostics: CaponDiagnostics::default(),
    };
    for r in 0..ranges {
        let mut x: Vec<Complex64> = (0..loops)
            .flat_map(|t| {
                (0..ANTENNAS).map(move |j| {
                    let z = data[(t * ANTENNAS + j) * ranges + r];
                    Complex64::new(f64::from(z.re), f64::from(z.im))
                })
            })
            .collect();
        for j in 0..ANTENNAS {
            let mean: Complex64 =
                (0..loops).map(|t| x[t * ANTENNAS + j]).sum::<Complex64>() / loops as f64;
            for t in 0..loops {
                x[t * ANTENNAS + j] -= mean;
            }
        }
        let covariance = matrix::covariance(&x, ANTENNAS);
        let row_cov: Vec<_> = ROW
            .iter()
            .flat_map(|&i| {
                ROW.iter()
                    .map(|&j| covariance[i * ANTENNAS + j])
                    .collect::<Vec<_>>()
            })
            .collect();
        if let Some(inverse) = matrix::loaded_inverse(&row_cov, ROW.len(), 0.001)? {
            for (a, s) in row_steering.iter().enumerate() {
                let power = response(&inverse, s).0;
                if !power.is_finite() || power < 0. || power > f32::MAX as f64 {
                    return Err("Capon RA power exceeds finite float32 range".into());
                }
                output.ra_power[a * ranges + r] = power as f32;
            }
        } else {
            output.diagnostics.zero_covariance_ranges += 1;
        }
        samples.push(x);
        covariances.push(covariance);
    }
    let candidates = cfar::detect(&output.ra_power, ranges, &mut output.diagnostics);
    let mut inverses: Vec<Option<Vec<Complex64>>> = vec![None; ranges];
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(config.doppler_bins);
    let mut spectrum = vec![Complex64::default(); config.doppler_bins];
    let mut scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];
    for candidate in candidates {
        let r = candidate.range;
        if candidate.noise <= 0. {
            // Zero-background synthetic data have no finite SNR. Never fabricate
            // a noise floor or pass infinity into GTRACK.
            output.diagnostics.zero_noise_dropped += 1;
            continue;
        }
        if inverses[r].is_none() {
            inverses[r] = matrix::loaded_inverse(&covariances[r], ANTENNAS, 0.03)?;
        }
        let Some(inverse) = &inverses[r] else {
            continue;
        };
        let responses: Vec<_> = mu
            .iter()
            .map(|&v| response(inverse, &steering(nu[candidate.angle], v)))
            .collect();
        let peak = first_peak(responses.iter().map(|p| p.0));
        let left = peak.saturating_sub(1);
        let right = (peak + 1).min(ELEVATION_BINS - 1);
        let indices = [left, peak, right];
        let total: f64 = indices.iter().map(|&i| responses[i].0).sum();
        let interpolated: f64 = indices
            .iter()
            .map(|&i| i as f64 * responses[i].0)
            .sum::<f64>()
            / total;
        let up = mu[0] + (mu[1] - mu[0]) * interpolated;
        // TI azimuth has the opposite lateral sign to the workspace convention.
        // Keep the source's reported direction (do not fold 0.9813 into XYZ).
        let lateral = -nu[candidate.angle];
        let forward_squared = 1. - lateral * lateral - up * up;
        if forward_squared <= 0. {
            output.diagnostics.nonvisible_dropped += 1;
            continue;
        }
        let weights = &responses[peak].1; // Discrete-peak, unnormalized R^-1 a.
        // Each candidate reuses the same buffers. Clear the padded tail as well
        // as measured loops so the previous candidate cannot leak into this FFT.
        spectrum.fill(Complex64::default());
        for (t, value) in spectrum.iter_mut().take(loops).enumerate() {
            *value = weights
                .iter()
                .enumerate()
                .map(|(j, w)| w.conj() * samples[r][t * ANTENNAS + j])
                .sum();
        }
        fft.process_with_scratch(&mut spectrum, &mut scratch);
        let bin = first_peak(spectrum.iter().map(|z| z.norm_sqr()));
        // TI uses > (not >=): the aliased Nyquist bin is reported positive.
        let signed = if bin > config.doppler_bins / 2 {
            bin as i32 - config.doppler_bins as i32
        } else {
            bin as i32
        };
        let radius = r as f64 * config.range_resolution_m;
        let power = output.ra_power[candidate.angle * ranges + r];
        output.detections.push(CaponDetection {
            range_bin: r,
            azimuth_bin: candidate.angle,
            elevation_bin: peak,
            elevation_interpolated_bin: interpolated,
            doppler_bin: signed,
            xyz_m: [
                radius * forward_squared.sqrt(),
                radius * lateral,
                radius * up,
            ],
            velocity_mps: signed as f64 * config.velocity_resolution_mps,
            snr: f64::from(power) / f64::from(candidate.noise),
            range_noise: candidate.noise,
            ra_power: power,
            elevation_power: responses[peak].0,
        });
    }
    Ok(output)
}

fn first_peak(values: impl Iterator<Item = f64>) -> usize {
    let mut best = f64::NEG_INFINITY;
    let mut index = 0;
    for (i, value) in values.enumerate() {
        if value > best {
            best = value;
            index = i;
        }
    }
    index
}
