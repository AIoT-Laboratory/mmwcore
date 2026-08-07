//! Cell-averaging CFAR detection over canonical range-Doppler tensors.

use std::fmt;

use num_complex::Complex32;

use crate::detection::{
    DetectionError, RangeDopplerAxes, ReceiverAggregation, range_doppler_magnitude_complex,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CfarMode {
    Ca = 0,
    Go = 1,
    So = 2,
    Cacc = 3,
}

impl TryFrom<u8> for CfarMode {
    type Error = CfarError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Ca),
            1 => Ok(Self::Go),
            2 => Ok(Self::So),
            3 => Ok(Self::Cacc),
            _ => Err(CfarError::UnsupportedMode { mode: value }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CfarInputScale {
    Magnitude = 0,
    Power = 1,
}

impl TryFrom<u8> for CfarInputScale {
    type Error = CfarError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Magnitude),
            1 => Ok(Self::Power),
            _ => Err(CfarError::UnsupportedInputScale { scale: value }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfar1DConfig {
    training_cells: usize,
    guard_cells: usize,
    threshold_scale: f32,
    mode: CfarMode,
    cyclic: bool,
    left_skip: usize,
    right_skip: usize,
}

impl Cfar1DConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        training_cells: usize,
        guard_cells: usize,
        threshold_scale: f32,
        mode: CfarMode,
        cyclic: bool,
        left_skip: usize,
        right_skip: usize,
    ) -> Result<Self, CfarError> {
        if training_cells == 0 {
            return Err(CfarError::ZeroTrainingCells);
        }
        if !threshold_scale.is_finite() || threshold_scale < 0.0 {
            return Err(CfarError::InvalidThresholdScale);
        }
        Ok(Self {
            training_cells,
            guard_cells,
            threshold_scale,
            mode,
            cyclic,
            left_skip,
            right_skip,
        })
    }

    pub const fn training_cells(self) -> usize {
        self.training_cells
    }

    pub const fn guard_cells(self) -> usize {
        self.guard_cells
    }

    pub const fn threshold_scale(self) -> f32 {
        self.threshold_scale
    }

    pub const fn mode(self) -> CfarMode {
        self.mode
    }

    pub const fn cyclic(self) -> bool {
        self.cyclic
    }

    pub const fn left_skip(self) -> usize {
        self.left_skip
    }

    pub const fn right_skip(self) -> usize {
        self.right_skip
    }

    fn radius(self) -> Result<usize, CfarError> {
        self.training_cells
            .checked_add(self.guard_cells)
            .ok_or(CfarError::ConfigurationOverflow)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfar2DConfig {
    training_cells: usize,
    guard_cells: usize,
    threshold_scale: f32,
}

impl Cfar2DConfig {
    pub fn new(
        training_cells: usize,
        guard_cells: usize,
        threshold_scale: f32,
    ) -> Result<Self, CfarError> {
        if training_cells == 0 {
            return Err(CfarError::ZeroTrainingCells);
        }
        if !threshold_scale.is_finite() || threshold_scale < 0.0 {
            return Err(CfarError::InvalidThresholdScale);
        }
        Ok(Self {
            training_cells,
            guard_cells,
            threshold_scale,
        })
    }

    pub const fn training_cells(self) -> usize {
        self.training_cells
    }

    pub const fn guard_cells(self) -> usize {
        self.guard_cells
    }

    pub const fn threshold_scale(self) -> f32 {
        self.threshold_scale
    }

    fn radius(self) -> Result<usize, CfarError> {
        self.training_cells
            .checked_add(self.guard_cells)
            .ok_or(CfarError::ConfigurationOverflow)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cfar1DResult {
    pub indices: Vec<usize>,
    pub noise: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CfarDetections {
    pub indices: Vec<usize>,
    pub magnitudes: Vec<f32>,
    pub noise: Vec<f32>,
    pub snr: Vec<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CfarError {
    Detection(DetectionError),
    ZeroTrainingCells,
    InvalidThresholdScale,
    ConfigurationOverflow,
    InvalidPowerInput,
    UnsupportedMode { mode: u8 },
    UnsupportedInputScale { scale: u8 },
}

impl fmt::Display for CfarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Detection(error) => error.fmt(formatter),
            Self::ZeroTrainingCells => write!(formatter, "CFAR training cells must be positive."),
            Self::InvalidThresholdScale => {
                write!(
                    formatter,
                    "CFAR threshold scale must be finite and non-negative."
                )
            }
            Self::ConfigurationOverflow => write!(formatter, "CFAR configuration size overflows."),
            Self::InvalidPowerInput => write!(
                formatter,
                "CFAR input power must contain finite non-negative values."
            ),
            Self::UnsupportedMode { mode } => {
                write!(formatter, "Unsupported native CFAR mode code {mode}.")
            }
            Self::UnsupportedInputScale { scale } => {
                write!(
                    formatter,
                    "Unsupported native CFAR input scale code {scale}."
                )
            }
        }
    }
}

impl std::error::Error for CfarError {}

impl From<DetectionError> for CfarError {
    fn from(error: DetectionError) -> Self {
        Self::Detection(error)
    }
}

pub fn detect_cfar_1d(power: &[f32], config: Cfar1DConfig) -> Result<Cfar1DResult, CfarError> {
    if power.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(CfarError::InvalidPowerInput);
    }

    let radius = config.radius()?;
    if radius >= half_ceil(power.len()) {
        return Ok(Cfar1DResult {
            indices: Vec::new(),
            noise: Vec::new(),
        });
    }

    let (start, stop) = if config.cyclic {
        (
            config.left_skip,
            power.len().saturating_sub(config.right_skip),
        )
    } else {
        (
            config.left_skip.saturating_add(radius),
            power
                .len()
                .saturating_sub(config.right_skip)
                .saturating_sub(radius),
        )
    };
    let mut indices = Vec::new();
    let mut noise_values = Vec::new();

    for cut in start..stop {
        let (left_sum, right_sum) = cfar_window_sums(power, cut, radius, config)?;
        let noise = cfar_noise(left_sum, right_sum, config);
        if power[cut] > noise * config.threshold_scale {
            indices.push(cut);
            noise_values.push(noise);
        }
    }

    Ok(Cfar1DResult {
        indices,
        noise: noise_values,
    })
}

pub fn detect_range_doppler_cfar_complex(
    data: &[Complex32],
    shape: &[usize],
    axes: RangeDopplerAxes,
    aggregation: ReceiverAggregation,
    range_config: Cfar1DConfig,
    doppler_config: Option<Cfar1DConfig>,
    input_scale: CfarInputScale,
) -> Result<CfarDetections, CfarError> {
    let (magnitude, [frames, doppler_bins, range_bins]) =
        range_doppler_magnitude_complex(data, shape, axes, aggregation)?;
    let signal = signal_from_magnitude(&magnitude, input_scale);
    let mut range_noise = vec![None; signal.len()];
    let mut doppler_noise = doppler_config.map(|_| vec![None; signal.len()]);

    for frame in 0..frames {
        for doppler in 0..doppler_bins {
            let start = range_doppler_index(frame, doppler, 0, doppler_bins, range_bins);
            let result = detect_cfar_1d(&signal[start..start + range_bins], range_config)?;
            for (range, noise) in result.indices.into_iter().zip(result.noise) {
                range_noise[range_doppler_index(frame, doppler, range, doppler_bins, range_bins)] =
                    Some(noise);
            }
        }

        if let (Some(config), Some(noise_map)) = (doppler_config, doppler_noise.as_mut()) {
            let mut line = Vec::with_capacity(doppler_bins);
            for range in 0..range_bins {
                line.clear();
                for doppler in 0..doppler_bins {
                    line.push(
                        signal
                            [range_doppler_index(frame, doppler, range, doppler_bins, range_bins)],
                    );
                }
                let result = detect_cfar_1d(&line, config)?;
                for (doppler, noise) in result.indices.into_iter().zip(result.noise) {
                    noise_map
                        [range_doppler_index(frame, doppler, range, doppler_bins, range_bins)] =
                        Some(noise);
                }
            }
        }
    }

    let mut detections = CfarDetections {
        indices: Vec::new(),
        magnitudes: Vec::new(),
        noise: Vec::new(),
        snr: Vec::new(),
    };
    for frame in 0..frames {
        for doppler in 0..doppler_bins {
            for range in 0..range_bins {
                let index = range_doppler_index(frame, doppler, range, doppler_bins, range_bins);
                let Some(range_noise_value) = range_noise[index] else {
                    continue;
                };
                let effective_noise = match doppler_noise.as_ref() {
                    Some(noise_map) => {
                        let Some(doppler_noise_value) = noise_map[index] else {
                            continue;
                        };
                        range_noise_value.max(doppler_noise_value)
                    }
                    None => range_noise_value,
                };
                detections.indices.extend([frame, doppler, range]);
                detections.magnitudes.push(magnitude[index]);
                detections.noise.push(effective_noise);
                detections
                    .snr
                    .push(linear_snr(signal[index], effective_noise));
            }
        }
    }
    Ok(detections)
}

pub fn detect_cfar_2d_complex(
    data: &[Complex32],
    shape: &[usize],
    axes: RangeDopplerAxes,
    aggregation: ReceiverAggregation,
    config: Cfar2DConfig,
) -> Result<CfarDetections, CfarError> {
    let (magnitude, [frames, doppler_bins, range_bins]) =
        range_doppler_magnitude_complex(data, shape, axes, aggregation)?;
    let radius = config.radius()?;
    if radius >= half_ceil(doppler_bins) || radius >= half_ceil(range_bins) {
        return Ok(CfarDetections {
            indices: Vec::new(),
            magnitudes: Vec::new(),
            noise: Vec::new(),
            snr: Vec::new(),
        });
    }

    let mut detections = CfarDetections {
        indices: Vec::new(),
        magnitudes: Vec::new(),
        noise: Vec::new(),
        snr: Vec::new(),
    };
    for frame in 0..frames {
        for doppler in radius..doppler_bins - radius {
            for range in radius..range_bins - radius {
                let noise = cfar_2d_noise(
                    &magnitude,
                    [frame, doppler, range],
                    [doppler_bins, range_bins],
                    radius,
                    config.guard_cells,
                );
                let index = range_doppler_index(frame, doppler, range, doppler_bins, range_bins);
                let cut = magnitude[index];
                if cut > noise * config.threshold_scale {
                    detections.indices.extend([frame, doppler, range]);
                    detections.magnitudes.push(cut);
                    detections.noise.push(noise);
                    detections.snr.push(linear_snr(cut, noise));
                }
            }
        }
    }
    Ok(detections)
}

fn cfar_window_sums(
    power: &[f32],
    cut: usize,
    radius: usize,
    config: Cfar1DConfig,
) -> Result<(f32, f32), CfarError> {
    let mut left_sum = 0.0_f32;
    let mut right_sum = 0.0_f32;
    for offset in config.guard_cells + 1..=radius {
        let left_index = if config.cyclic {
            subtract_modulo(cut, offset, power.len())
        } else {
            cut.checked_sub(offset)
                .ok_or(CfarError::ConfigurationOverflow)?
        };
        let right_index = if config.cyclic {
            add_modulo(cut, offset, power.len())
        } else {
            cut.checked_add(offset)
                .ok_or(CfarError::ConfigurationOverflow)?
        };
        left_sum += power[left_index];
        right_sum += power[right_index];
    }
    Ok((left_sum, right_sum))
}

fn cfar_noise(left_sum: f32, right_sum: f32, config: Cfar1DConfig) -> f32 {
    match config.mode {
        CfarMode::Ca => (left_sum + right_sum) / (2 * config.training_cells) as f32,
        CfarMode::Go => {
            (left_sum / config.training_cells as f32).max(right_sum / config.training_cells as f32)
        }
        CfarMode::So => {
            (left_sum / config.training_cells as f32).min(right_sum / config.training_cells as f32)
        }
        CfarMode::Cacc => left_sum + right_sum,
    }
}

fn cfar_2d_noise(
    magnitude: &[f32],
    location: [usize; 3],
    dimensions: [usize; 2],
    radius: usize,
    guard_cells: usize,
) -> f32 {
    let [frame, doppler, range] = location;
    let [doppler_bins, range_bins] = dimensions;
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for neighbor_doppler in doppler - radius..=doppler + radius {
        for neighbor_range in range - radius..=range + radius {
            if doppler.abs_diff(neighbor_doppler) <= guard_cells
                && range.abs_diff(neighbor_range) <= guard_cells
            {
                continue;
            }
            sum += magnitude[range_doppler_index(
                frame,
                neighbor_doppler,
                neighbor_range,
                doppler_bins,
                range_bins,
            )];
            count += 1;
        }
    }
    sum / count as f32
}

fn signal_from_magnitude(magnitude: &[f32], input_scale: CfarInputScale) -> Vec<f32> {
    match input_scale {
        CfarInputScale::Magnitude => magnitude.to_vec(),
        CfarInputScale::Power => magnitude.iter().map(|value| value * value).collect(),
    }
}

fn linear_snr(signal: f32, noise: f32) -> f32 {
    if noise <= 0.0 {
        f32::MAX
    } else {
        signal / noise
    }
}

fn subtract_modulo(index: usize, offset: usize, length: usize) -> usize {
    if index >= offset {
        index - offset
    } else {
        length - (offset - index)
    }
}

fn add_modulo(index: usize, offset: usize, length: usize) -> usize {
    let boundary = length - offset;
    if index >= boundary {
        index - boundary
    } else {
        index + offset
    }
}

fn range_doppler_index(
    frame: usize,
    doppler: usize,
    range: usize,
    doppler_bins: usize,
    range_bins: usize,
) -> usize {
    (frame * doppler_bins + doppler) * range_bins + range
}

fn half_ceil(value: usize) -> usize {
    value / 2 + value % 2
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::{
        Cfar1DConfig, Cfar2DConfig, CfarInputScale, CfarMode, detect_cfar_1d,
        detect_cfar_2d_complex, detect_range_doppler_cfar_complex,
    };
    use crate::detection::{RangeDopplerAxes, ReceiverAggregation};

    #[test]
    fn validates_and_exposes_cfar_configs() {
        for threshold_scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            assert_eq!(
                Cfar1DConfig::new(1, 2, threshold_scale, CfarMode::Go, true, 3, 4),
                Err(super::CfarError::InvalidThresholdScale)
            );
            assert_eq!(
                Cfar2DConfig::new(1, 2, threshold_scale),
                Err(super::CfarError::InvalidThresholdScale)
            );
        }

        let one = Cfar1DConfig::new(1, 2, 3.0, CfarMode::Go, true, 4, 5).unwrap();
        assert_eq!(one.training_cells(), 1);
        assert_eq!(one.guard_cells(), 2);
        assert_eq!(one.threshold_scale(), 3.0);
        assert_eq!(one.mode(), CfarMode::Go);
        assert!(one.cyclic());
        assert_eq!(one.left_skip(), 4);
        assert_eq!(one.right_skip(), 5);

        let two = Cfar2DConfig::new(6, 7, 8.0).unwrap();
        assert_eq!(two.training_cells(), 6);
        assert_eq!(two.guard_cells(), 7);
        assert_eq!(two.threshold_scale(), 8.0);
    }

    #[test]
    fn one_dimensional_cfar_matches_window_reduction_modes() {
        let power = [1.0, 1.0, 0.0, 20.0, 0.0, 5.0, 5.0];
        let expected_noise = [1.5, 2.5, 0.5, 6.0];
        for (mode, expected) in [CfarMode::Ca, CfarMode::Go, CfarMode::So, CfarMode::Cacc]
            .into_iter()
            .zip(expected_noise)
        {
            let result = detect_cfar_1d(
                &power,
                Cfar1DConfig::new(2, 0, 1.1, mode, false, 0, 0).unwrap(),
            )
            .unwrap();
            assert_eq!(result.indices, [3]);
            assert_eq!(result.noise, [expected]);
        }
    }

    #[test]
    fn one_dimensional_cfar_wraps_cyclic_windows() {
        let result = detect_cfar_1d(
            &[10.0, 1.0, 1.0, 1.0, 1.0],
            Cfar1DConfig::new(1, 0, 2.0, CfarMode::Ca, true, 0, 0).unwrap(),
        )
        .unwrap();

        assert_eq!(result.indices, [0]);
        assert_eq!(result.noise, [1.0]);
    }

    #[test]
    fn composed_range_doppler_cfar_keeps_canonical_order_and_power_domain() {
        let mut data = vec![Complex32::new(1.0, 0.0); 49];
        data[(3 * 7) + 3] = Complex32::new(10.0, 0.0);
        let detections = detect_range_doppler_cfar_complex(
            &data,
            &[1, 7, 1, 7],
            RangeDopplerAxes {
                frame: 0,
                doppler: 1,
                receiver: 2,
                range: 3,
            },
            ReceiverAggregation::Max,
            Cfar1DConfig::new(1, 1, 20.0, CfarMode::Ca, false, 0, 0).unwrap(),
            None,
            CfarInputScale::Power,
        )
        .unwrap();

        assert_eq!(detections.indices, [0, 3, 3]);
        assert_eq!(detections.magnitudes, [10.0]);
        assert_eq!(detections.noise, [1.0]);
        assert_eq!(detections.snr, [100.0]);
    }

    #[test]
    fn two_dimensional_cfar_skips_edges_and_uses_guard_region() {
        let mut data = vec![Complex32::new(1.0, 0.0); 25];
        data[(2 * 5) + 2] = Complex32::new(8.0, 0.0);
        data[0] = Complex32::new(100.0, 0.0);
        let detections = detect_cfar_2d_complex(
            &data,
            &[1, 5, 1, 5],
            RangeDopplerAxes {
                frame: 0,
                doppler: 1,
                receiver: 2,
                range: 3,
            },
            ReceiverAggregation::Max,
            Cfar2DConfig::new(1, 0, 4.0).unwrap(),
        )
        .unwrap();

        assert_eq!(detections.indices, [0, 2, 2]);
        assert_eq!(detections.magnitudes, [8.0]);
        assert_eq!(detections.noise, [1.0]);
        assert_eq!(detections.snr, [8.0]);
    }
}
