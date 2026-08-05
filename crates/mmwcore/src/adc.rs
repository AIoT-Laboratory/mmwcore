//! Raw ADC layout decoding.

use std::fmt;

use num_complex::Complex32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AdcComplexLayout {
    IqInterleaved = 0,
    SampleIThenQ = 1,
    Group2IThenQ = 2,
}

impl TryFrom<u8> for AdcComplexLayout {
    type Error = AdcDecodeError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::IqInterleaved),
            1 => Ok(Self::SampleIThenQ),
            2 => Ok(Self::Group2IThenQ),
            _ => Err(AdcDecodeError::UnsupportedLayout(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdcFrameSpec {
    pub num_chirps: usize,
    pub num_rx: usize,
    pub num_samples: usize,
    pub layout: AdcComplexLayout,
}

impl AdcFrameSpec {
    pub fn new(
        num_chirps: usize,
        num_rx: usize,
        num_samples: usize,
        layout: AdcComplexLayout,
    ) -> Result<Self, AdcDecodeError> {
        for (name, value) in [
            ("num_chirps", num_chirps),
            ("num_rx", num_rx),
            ("num_samples", num_samples),
        ] {
            if value == 0 {
                return Err(AdcDecodeError::ZeroDimension(name));
            }
        }
        Ok(Self {
            num_chirps,
            num_rx,
            num_samples,
            layout,
        })
    }

    pub fn raw_values_per_frame(self) -> Result<usize, AdcDecodeError> {
        self.num_chirps
            .checked_mul(self.num_rx)
            .and_then(|value| value.checked_mul(self.num_samples))
            .and_then(|value| value.checked_mul(2))
            .ok_or(AdcDecodeError::FrameSizeOverflow)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum AdcDecodeError {
    ZeroDimension(&'static str),
    FrameSizeOverflow,
    IncompleteFrames { samples: usize, frame_size: usize },
    NoCompleteFrames,
    Group2RequiresEvenSamples { num_samples: usize },
    UnsupportedLayout(u8),
}

impl fmt::Display for AdcDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(formatter, "ADC {name} must be positive."),
            Self::FrameSizeOverflow => write!(formatter, "ADC frame size overflows usize."),
            Self::IncompleteFrames {
                samples,
                frame_size,
            } => write!(
                formatter,
                "Raw ADC sample count is not a whole number of frames; got {samples} values for frame size {frame_size}."
            ),
            Self::NoCompleteFrames => write!(formatter, "No complete ADC frames are available."),
            Self::Group2RequiresEvenSamples { num_samples } => write!(
                formatter,
                "GROUP2_I_THEN_Q requires even num_samples; got {num_samples}."
            ),
            Self::UnsupportedLayout(value) => {
                write!(formatter, "Unsupported ADC layout code: {value}.")
            }
        }
    }
}

impl std::error::Error for AdcDecodeError {}

#[derive(Clone, Debug, PartialEq)]
pub struct AdcCube {
    data: Vec<Complex32>,
    shape: [usize; 4],
}

impl AdcCube {
    pub fn data(&self) -> &[Complex32] {
        &self.data
    }

    pub fn into_data(self) -> Vec<Complex32> {
        self.data
    }

    pub const fn shape(&self) -> [usize; 4] {
        self.shape
    }
}

pub fn decode_adc_i16(
    samples: &[i16],
    spec: AdcFrameSpec,
    drop_incomplete: bool,
) -> Result<AdcCube, AdcDecodeError> {
    let frame_size = spec.raw_values_per_frame()?;
    let remainder = samples.len() % frame_size;
    if remainder != 0 && !drop_incomplete {
        return Err(AdcDecodeError::IncompleteFrames {
            samples: samples.len(),
            frame_size,
        });
    }

    let complete_values = samples.len() - remainder;
    if complete_values == 0 {
        return Err(AdcDecodeError::NoCompleteFrames);
    }
    if spec.layout == AdcComplexLayout::Group2IThenQ && spec.num_samples % 2 != 0 {
        return Err(AdcDecodeError::Group2RequiresEvenSamples {
            num_samples: spec.num_samples,
        });
    }

    let num_frames = complete_values / frame_size;
    let complex_values = num_frames
        .checked_mul(spec.num_chirps)
        .and_then(|value| value.checked_mul(spec.num_rx))
        .and_then(|value| value.checked_mul(spec.num_samples))
        .ok_or(AdcDecodeError::FrameSizeOverflow)?;
    let mut data = vec![Complex32::new(0.0, 0.0); complex_values];
    let complete_samples = &samples[..complete_values];

    match spec.layout {
        AdcComplexLayout::IqInterleaved => {
            for (output_index, pair) in complete_samples.chunks_exact(2).enumerate() {
                data[output_index] = Complex32::new(f32::from(pair[0]), f32::from(pair[1]));
            }
        }
        AdcComplexLayout::SampleIThenQ => {
            decode_sample_i_then_q(complete_samples, &mut data, num_frames, spec)
        }
        AdcComplexLayout::Group2IThenQ => {
            decode_group2_i_then_q(complete_samples, &mut data, num_frames, spec)
        }
    }

    Ok(AdcCube {
        data,
        shape: [num_frames, spec.num_chirps, spec.num_rx, spec.num_samples],
    })
}

fn decode_sample_i_then_q(
    samples: &[i16],
    data: &mut [Complex32],
    num_frames: usize,
    spec: AdcFrameSpec,
) {
    for frame in 0..num_frames {
        for chirp in 0..spec.num_chirps {
            for sample in 0..spec.num_samples {
                let group_start = (((frame * spec.num_chirps + chirp) * spec.num_samples + sample)
                    * 2)
                    * spec.num_rx;
                for rx in 0..spec.num_rx {
                    let output_index = cube_index(frame, chirp, rx, sample, spec);
                    data[output_index] = Complex32::new(
                        f32::from(samples[group_start + rx]),
                        f32::from(samples[group_start + spec.num_rx + rx]),
                    );
                }
            }
        }
    }
}

fn decode_group2_i_then_q(
    samples: &[i16],
    data: &mut [Complex32],
    num_frames: usize,
    spec: AdcFrameSpec,
) {
    let groups_per_rx = spec.num_samples / 2;
    for frame in 0..num_frames {
        for chirp in 0..spec.num_chirps {
            for rx in 0..spec.num_rx {
                for group in 0..groups_per_rx {
                    let group_start = (((frame * spec.num_chirps + chirp) * spec.num_rx + rx)
                        * groups_per_rx
                        + group)
                        * 4;
                    let sample_start = group * 2;
                    data[cube_index(frame, chirp, rx, sample_start, spec)] = Complex32::new(
                        f32::from(samples[group_start]),
                        f32::from(samples[group_start + 2]),
                    );
                    data[cube_index(frame, chirp, rx, sample_start + 1, spec)] = Complex32::new(
                        f32::from(samples[group_start + 1]),
                        f32::from(samples[group_start + 3]),
                    );
                }
            }
        }
    }
}

fn cube_index(frame: usize, chirp: usize, rx: usize, sample: usize, spec: AdcFrameSpec) -> usize {
    (((frame * spec.num_chirps + chirp) * spec.num_rx + rx) * spec.num_samples) + sample
}

#[cfg(test)]
mod tests {
    use super::{AdcComplexLayout, AdcDecodeError, AdcFrameSpec, decode_adc_i16};
    use num_complex::Complex32;

    #[test]
    fn decodes_iq_interleaved_layout() {
        let spec = AdcFrameSpec::new(1, 2, 2, AdcComplexLayout::IqInterleaved).unwrap();
        let cube = decode_adc_i16(&[1, 10, 2, 20, 3, 30, 4, 40], spec, false).unwrap();

        assert_eq!(cube.shape(), [1, 1, 2, 2]);
        assert_eq!(
            cube.data(),
            [
                Complex32::new(1.0, 10.0),
                Complex32::new(2.0, 20.0),
                Complex32::new(3.0, 30.0),
                Complex32::new(4.0, 40.0),
            ]
        );
    }

    #[test]
    fn decodes_sample_i_then_q_layout() {
        let spec = AdcFrameSpec::new(1, 2, 2, AdcComplexLayout::SampleIThenQ).unwrap();
        let cube = decode_adc_i16(&[1, 3, 10, 30, 2, 4, 20, 40], spec, false).unwrap();

        assert_eq!(
            cube.data(),
            [
                Complex32::new(1.0, 10.0),
                Complex32::new(2.0, 20.0),
                Complex32::new(3.0, 30.0),
                Complex32::new(4.0, 40.0),
            ]
        );
    }

    #[test]
    fn decodes_group2_i_then_q_layout() {
        let spec = AdcFrameSpec::new(1, 2, 4, AdcComplexLayout::Group2IThenQ).unwrap();
        let cube = decode_adc_i16(
            &[1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60, 7, 8, 70, 80],
            spec,
            false,
        )
        .unwrap();

        assert_eq!(
            cube.data(),
            [
                Complex32::new(1.0, 10.0),
                Complex32::new(2.0, 20.0),
                Complex32::new(3.0, 30.0),
                Complex32::new(4.0, 40.0),
                Complex32::new(5.0, 50.0),
                Complex32::new(6.0, 60.0),
                Complex32::new(7.0, 70.0),
                Complex32::new(8.0, 80.0),
            ]
        );
    }

    #[test]
    fn rejects_incomplete_frames_without_drop() {
        let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::IqInterleaved).unwrap();

        assert_eq!(
            decode_adc_i16(&[1, 2, 3, 4, 5], spec, false),
            Err(AdcDecodeError::IncompleteFrames {
                samples: 5,
                frame_size: 4,
            })
        );
    }

    #[test]
    fn drops_incomplete_tail_when_requested() {
        let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::IqInterleaved).unwrap();
        let cube = decode_adc_i16(&[1, 10, 2, 20, 999], spec, true).unwrap();

        assert_eq!(
            cube.data(),
            [Complex32::new(1.0, 10.0), Complex32::new(2.0, 20.0)]
        );
    }

    #[test]
    fn rejects_odd_group2_sample_count_after_frame_validation() {
        let spec = AdcFrameSpec::new(1, 1, 3, AdcComplexLayout::Group2IThenQ).unwrap();

        assert_eq!(
            decode_adc_i16(&[0; 6], spec, false),
            Err(AdcDecodeError::Group2RequiresEvenSamples { num_samples: 3 })
        );
    }
}
