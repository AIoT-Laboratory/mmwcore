//! Raw ADC layout decoding.

use std::fmt;

use num_complex::Complex32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AdcComplexLayout {
    IqInterleaved = 0,
    SampleIThenQ = 1,
    Group2IThenQ = 2,
    Group4IThenQ = 3,
}

impl TryFrom<u8> for AdcComplexLayout {
    type Error = AdcDecodeError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::IqInterleaved),
            1 => Ok(Self::SampleIThenQ),
            2 => Ok(Self::Group2IThenQ),
            3 => Ok(Self::Group4IThenQ),
            _ => Err(AdcDecodeError::UnsupportedLayout(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdcFrameSpec {
    num_chirps: usize,
    num_rx: usize,
    num_samples: usize,
    layout: AdcComplexLayout,
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

    pub const fn num_chirps(self) -> usize {
        self.num_chirps
    }

    pub const fn num_rx(self) -> usize {
        self.num_rx
    }

    pub const fn num_samples(self) -> usize {
        self.num_samples
    }

    pub const fn layout(self) -> AdcComplexLayout {
        self.layout
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
    IncompleteFrames {
        samples: usize,
        frame_size: usize,
    },
    NoCompleteFrames,
    Group2RequiresEvenSamples {
        num_samples: usize,
    },
    Group4RequiresAlignedFrame {
        num_chirps: usize,
        num_rx: usize,
        num_samples: usize,
    },
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
            Self::Group4RequiresAlignedFrame {
                num_chirps,
                num_rx,
                num_samples,
            } => write!(
                formatter,
                "GROUP4_I_THEN_Q requires num_chirps * num_rx * num_samples to be divisible by 4; got {num_chirps} * {num_rx} * {num_samples}."
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
    let complex_values_per_chirp = spec
        .num_rx
        .checked_mul(spec.num_samples)
        .ok_or(AdcDecodeError::FrameSizeOverflow)?;
    let complex_values_per_frame = spec
        .num_chirps
        .checked_mul(complex_values_per_chirp)
        .ok_or(AdcDecodeError::FrameSizeOverflow)?;
    if spec.layout == AdcComplexLayout::Group2IThenQ && spec.num_samples % 2 != 0 {
        return Err(AdcDecodeError::Group2RequiresEvenSamples {
            num_samples: spec.num_samples,
        });
    }
    if spec.layout == AdcComplexLayout::Group4IThenQ && complex_values_per_frame % 4 != 0 {
        return Err(AdcDecodeError::Group4RequiresAlignedFrame {
            num_chirps: spec.num_chirps,
            num_rx: spec.num_rx,
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
        AdcComplexLayout::Group4IThenQ => decode_group4_i_then_q(
            complete_samples,
            &mut data,
            num_frames,
            complex_values_per_frame,
            complex_values_per_chirp,
            spec,
        ),
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

fn decode_group4_i_then_q(
    samples: &[i16],
    data: &mut [Complex32],
    num_frames: usize,
    complex_values_per_frame: usize,
    complex_values_per_chirp: usize,
    spec: AdcFrameSpec,
) {
    let groups_per_frame = complex_values_per_frame / 4;
    for frame in 0..num_frames {
        for group in 0..groups_per_frame {
            let group_start = (frame * groups_per_frame + group) * 8;
            for lane in 0..4 {
                let logical_index = group * 4 + lane;
                let chirp = logical_index / complex_values_per_chirp;
                let chirp_index = logical_index % complex_values_per_chirp;
                let sample = chirp_index / spec.num_rx;
                let rx = chirp_index % spec.num_rx;
                data[cube_index(frame, chirp, rx, sample, spec)] = Complex32::new(
                    f32::from(samples[group_start + lane]),
                    f32::from(samples[group_start + 4 + lane]),
                );
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
    fn exposes_validated_frame_spec_fields() {
        let spec = AdcFrameSpec::new(2, 3, 4, AdcComplexLayout::Group2IThenQ).unwrap();

        assert_eq!(spec.num_chirps(), 2);
        assert_eq!(spec.num_rx(), 3);
        assert_eq!(spec.num_samples(), 4);
        assert_eq!(spec.layout(), AdcComplexLayout::Group2IThenQ);
    }

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
    fn decodes_group4_i_then_q_channel_interleaved_layout() {
        let spec = AdcFrameSpec::new(2, 1, 2, AdcComplexLayout::Group4IThenQ).unwrap();
        let cube = decode_adc_i16(&[1, 2, 3, 4, 10, 20, 30, 40], spec, false).unwrap();

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

    #[test]
    fn rejects_group4_frames_without_four_complex_value_alignment() {
        let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::Group4IThenQ).unwrap();

        assert_eq!(
            decode_adc_i16(&[0; 8], spec, false),
            Err(AdcDecodeError::Group4RequiresAlignedFrame {
                num_chirps: 1,
                num_rx: 1,
                num_samples: 2,
            })
        );
    }

    #[test]
    fn rejects_incomplete_group4_frames_before_decoding() {
        let spec = AdcFrameSpec::new(1, 2, 2, AdcComplexLayout::Group4IThenQ).unwrap();

        assert_eq!(
            decode_adc_i16(&[0; 9], spec, false),
            Err(AdcDecodeError::IncompleteFrames {
                samples: 9,
                frame_size: 8,
            })
        );
    }

    #[test]
    fn rejects_group4_frame_size_overflow() {
        let spec = AdcFrameSpec::new(usize::MAX, 4, 1, AdcComplexLayout::Group4IThenQ).unwrap();

        assert_eq!(
            decode_adc_i16(&[], spec, false),
            Err(AdcDecodeError::FrameSizeOverflow)
        );
    }
}
