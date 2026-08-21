//! Lossless ADC coding for independently decodable frame groups.

use std::fmt;

/// Default number of `int16` samples in one adaptive Rice block.
pub const ADC_RICE_BLOCK_SAMPLES: usize = 512;
/// Default maximum number of temporally dependent frames in one archive group.
pub const ADC_RICE_RESTART_FRAMES: usize = 4;

const RAW_BLOCK_TAG: u8 = u8::MAX;
const MAX_RICE_PARAMETER: u8 = 16;
const MIN_BLOCK_SAMPLES: usize = 256;
const MAX_BLOCK_SAMPLES: usize = 1024;
const MAX_ZIGZAG_DELTA: u32 = 131_070;

/// Encode complete little-endian `int16` ADC frames as one independent Rice group.
///
/// Frame zero is coded as absolute samples. Every later frame is predicted from the sample at
/// the same flattened capture coordinate in the previous frame. Blocks select the shorter of
/// Rice-coded ZigZag residuals and exact raw `int16` bytes.
pub fn encode_adc_archive_chunk(
    raw: &[u8],
    frame_bytes: usize,
    block_samples: usize,
) -> Result<Vec<u8>, AdcArchiveCodecError> {
    let frame_count = validate_chunk(raw, frame_bytes, block_samples)?;
    let frame_samples = frame_bytes / 2;
    let mut previous = vec![0_i16; frame_samples];
    let maximum_encoded = maximum_adc_archive_chunk_bytes(frame_bytes, frame_count, block_samples)?;
    let mut encoded = Vec::new();
    encoded.try_reserve_exact(maximum_encoded).map_err(|_| {
        AdcArchiveCodecError::CannotAllocateOutput {
            expected_bytes: maximum_encoded,
        }
    })?;
    let mut residuals = Vec::with_capacity(block_samples);
    let mut rice = BitWriter::default();

    for (frame_index, frame) in raw.chunks_exact(frame_bytes).enumerate() {
        for block_start in (0..frame_samples).step_by(block_samples) {
            let block_stop = (block_start + block_samples).min(frame_samples);
            let raw_block = &frame[block_start * 2..block_stop * 2];
            populate_residuals(
                raw_block,
                frame_index == 0,
                &mut previous[block_start..block_stop],
                &mut residuals,
            );
            let (parameter, bit_count) = best_rice_parameter_and_bit_count(&residuals);
            let expected_rice_bytes = usize::try_from(bit_count.div_ceil(8))
                .map_err(|_| AdcArchiveCodecError::OutputSizeOverflow)?;
            if expected_rice_bytes < raw_block.len() {
                rice.reset();
                rice.try_reserve(expected_rice_bytes)?;
                for &value in &residuals {
                    rice.write_rice(value, parameter);
                }
                let rice_bytes = rice.finish();
                debug_assert_eq!(rice_bytes.len(), expected_rice_bytes);
                encoded.push(parameter);
                encoded.extend_from_slice(rice_bytes);
            } else {
                encoded.push(RAW_BLOCK_TAG);
                encoded.extend_from_slice(raw_block);
            }
        }
    }
    Ok(encoded)
}

/// Decode one independently encoded ADC frame group exactly.
pub fn decode_adc_archive_chunk(
    encoded: &[u8],
    frame_bytes: usize,
    frame_count: usize,
    block_samples: usize,
) -> Result<Vec<u8>, AdcArchiveCodecError> {
    validate_dimensions(frame_bytes, frame_count, block_samples)?;
    if encoded.is_empty() {
        return Err(AdcArchiveCodecError::EmptyChunk {
            name: "encoded chunk",
        });
    }
    let expected_bytes = frame_bytes
        .checked_mul(frame_count)
        .ok_or(AdcArchiveCodecError::OutputSizeOverflow)?;
    let frame_samples = frame_bytes / 2;
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(expected_bytes)
        .map_err(|_| AdcArchiveCodecError::CannotAllocateOutput { expected_bytes })?;
    let mut previous = vec![0_i16; frame_samples];
    let mut cursor = 0_usize;

    for frame_index in 0..frame_count {
        for block_start in (0..frame_samples).step_by(block_samples) {
            let block_stop = (block_start + block_samples).min(frame_samples);
            let sample_count = block_stop - block_start;
            let tag = *encoded
                .get(cursor)
                .ok_or(AdcArchiveCodecError::TruncatedBlock)?;
            cursor += 1;
            if tag == RAW_BLOCK_TAG {
                let raw_bytes = sample_count * 2;
                let stop = cursor
                    .checked_add(raw_bytes)
                    .ok_or(AdcArchiveCodecError::OutputSizeOverflow)?;
                let block = encoded
                    .get(cursor..stop)
                    .ok_or(AdcArchiveCodecError::TruncatedBlock)?;
                decoded.extend_from_slice(block);
                for (offset, sample) in block.chunks_exact(2).enumerate() {
                    previous[block_start + offset] =
                        i16::from_le_bytes(sample.try_into().expect("validated raw block sample"));
                }
                cursor = stop;
                continue;
            }
            if tag > MAX_RICE_PARAMETER {
                return Err(AdcArchiveCodecError::InvalidBlockTag { tag });
            }

            let mut reader = BitReader::new(&encoded[cursor..]);
            for previous_sample in &mut previous[block_start..block_stop] {
                let mapped = reader.read_rice(tag)?;
                if mapped > MAX_ZIGZAG_DELTA {
                    return Err(AdcArchiveCodecError::ResidualOutOfRange { value: mapped });
                }
                let residual = unzigzag_i32(mapped);
                let reconstructed = if frame_index == 0 {
                    residual
                } else {
                    i32::from(*previous_sample) + residual
                };
                let sample = i16::try_from(reconstructed).map_err(|_| {
                    AdcArchiveCodecError::ReconstructedSampleOutOfRange {
                        value: reconstructed,
                    }
                })?;
                decoded.extend_from_slice(&sample.to_le_bytes());
                *previous_sample = sample;
            }
            cursor = cursor
                .checked_add(reader.finish_block()?)
                .ok_or(AdcArchiveCodecError::OutputSizeOverflow)?;
        }
    }
    if cursor != encoded.len() {
        return Err(AdcArchiveCodecError::TrailingEncodedBytes {
            trailing_bytes: encoded.len() - cursor,
        });
    }
    debug_assert_eq!(decoded.len(), expected_bytes);
    Ok(decoded)
}

/// Strict upper bound for one encoded chunk, including adaptive block tags.
pub fn maximum_adc_archive_chunk_bytes(
    frame_bytes: usize,
    frame_count: usize,
    block_samples: usize,
) -> Result<usize, AdcArchiveCodecError> {
    validate_dimensions(frame_bytes, frame_count, block_samples)?;
    let blocks_per_frame = (frame_bytes / 2).div_ceil(block_samples);
    frame_bytes
        .checked_mul(frame_count)
        .and_then(|raw| raw.checked_add(blocks_per_frame.checked_mul(frame_count)?))
        .ok_or(AdcArchiveCodecError::OutputSizeOverflow)
}

fn validate_chunk(
    raw: &[u8],
    frame_bytes: usize,
    block_samples: usize,
) -> Result<usize, AdcArchiveCodecError> {
    if raw.is_empty() {
        return Err(AdcArchiveCodecError::EmptyChunk { name: "raw chunk" });
    }
    if frame_bytes == 0 || !frame_bytes.is_multiple_of(2) {
        return Err(AdcArchiveCodecError::InvalidFrameBytes { frame_bytes });
    }
    if !raw.len().is_multiple_of(frame_bytes) {
        return Err(AdcArchiveCodecError::IncompleteFrameChunk {
            chunk_bytes: raw.len(),
            frame_bytes,
        });
    }
    let frame_count = raw.len() / frame_bytes;
    validate_dimensions(frame_bytes, frame_count, block_samples)?;
    Ok(frame_count)
}

fn validate_dimensions(
    frame_bytes: usize,
    frame_count: usize,
    block_samples: usize,
) -> Result<(), AdcArchiveCodecError> {
    if frame_bytes == 0 || !frame_bytes.is_multiple_of(2) {
        return Err(AdcArchiveCodecError::InvalidFrameBytes { frame_bytes });
    }
    if frame_count == 0 {
        return Err(AdcArchiveCodecError::EmptyChunk {
            name: "frame group",
        });
    }
    if !(MIN_BLOCK_SAMPLES..=MAX_BLOCK_SAMPLES).contains(&block_samples)
        || !block_samples.is_power_of_two()
    {
        return Err(AdcArchiveCodecError::InvalidBlockSamples { block_samples });
    }
    Ok(())
}

fn best_rice_parameter_and_bit_count(values: &[u32]) -> (u8, u64) {
    let mut costs = [0_u64; MAX_RICE_PARAMETER as usize + 1];
    for &value in values {
        for (parameter, cost) in costs.iter_mut().enumerate() {
            *cost += u64::from(value >> parameter) + 1 + parameter as u64;
        }
    }
    costs
        .iter()
        .enumerate()
        .min_by_key(|(_, cost)| *cost)
        .map(|(parameter, &cost)| (parameter as u8, cost))
        .expect("Rice parameter range is non-empty")
}

fn populate_residuals(
    raw_block: &[u8],
    absolute_frame: bool,
    previous: &mut [i16],
    residuals: &mut Vec<u32>,
) {
    debug_assert_eq!(raw_block.len(), previous.len() * 2);
    residuals.clear();
    for (sample_bytes, previous_sample) in raw_block.chunks_exact(2).zip(previous) {
        let current = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
        let residual = if absolute_frame {
            i32::from(current)
        } else {
            i32::from(current) - i32::from(*previous_sample)
        };
        residuals.push(zigzag_i32(residual));
        *previous_sample = current;
    }
}

fn zigzag_i32(value: i32) -> u32 {
    ((value << 1) ^ (value >> 31)) as u32
}

fn unzigzag_i32(value: u32) -> i32 {
    ((value >> 1) as i32) ^ -((value & 1) as i32)
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    fn reset(&mut self) {
        self.bytes.clear();
        self.current = 0;
        self.used = 0;
    }

    fn try_reserve(&mut self, encoded_bytes: usize) -> Result<(), AdcArchiveCodecError> {
        self.bytes.try_reserve_exact(encoded_bytes).map_err(|_| {
            AdcArchiveCodecError::CannotAllocateOutput {
                expected_bytes: encoded_bytes,
            }
        })
    }

    fn write_rice(&mut self, value: u32, parameter: u8) {
        let quotient = value >> parameter;
        self.write_zeroes(quotient as usize);
        self.write_bits(1, 1);
        if parameter > 0 {
            let mask = (1_u32 << parameter) - 1;
            self.write_bits(value & mask, parameter);
        }
    }

    fn write_zeroes(&mut self, mut count: usize) {
        if self.used != 0 {
            let available = usize::from(8 - self.used);
            if count < available {
                self.used += count as u8;
                return;
            }
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
            count -= available;
        }

        let whole_bytes = count / 8;
        if whole_bytes != 0 {
            self.bytes.resize(self.bytes.len() + whole_bytes, 0);
        }
        self.used = (count % 8) as u8;
    }

    fn write_bits(&mut self, value: u32, mut count: u8) {
        while count != 0 {
            if self.used == 0 && count >= 8 {
                let shift = count - 8;
                self.bytes.push((value >> shift) as u8);
                count -= 8;
                continue;
            }

            let available = 8 - self.used;
            let take = count.min(available);
            let shift = count - take;
            let mask = (1_u32 << take) - 1;
            let bits = ((value >> shift) & mask) as u8;
            self.current |= bits << (available - take);
            self.used += take;
            count -= take;
            if self.used == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.used = 0;
            }
        }
    }

    fn finish(&mut self) -> &[u8] {
        if self.used != 0 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
        &self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit_index: usize,
}

impl<'a> BitReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_index: 0,
        }
    }

    fn read_rice(&mut self, parameter: u8) -> Result<u32, AdcArchiveCodecError> {
        let maximum_quotient = MAX_ZIGZAG_DELTA >> parameter;
        let quotient = self.read_unary(maximum_quotient)?;
        let remainder = self.read_bits(parameter)?;
        quotient
            .checked_shl(u32::from(parameter))
            .and_then(|value| value.checked_add(remainder))
            .ok_or(AdcArchiveCodecError::RiceQuotientOutOfRange)
    }

    fn read_bits(&mut self, count: u8) -> Result<u32, AdcArchiveCodecError> {
        let mut count = count;
        let mut value = 0_u32;
        while count != 0 {
            let byte = *self
                .bytes
                .get(self.bit_index / 8)
                .ok_or(AdcArchiveCodecError::TruncatedBlock)?;
            let offset = (self.bit_index % 8) as u8;
            let available = 8 - offset;
            let take = count.min(available);
            let mask = (1_u16 << take) - 1;
            let bits = u32::from((u16::from(byte) >> (available - take)) & mask);
            value = (value << take) | bits;
            self.bit_index += usize::from(take);
            count -= take;
        }
        Ok(value)
    }

    fn read_unary(&mut self, maximum_quotient: u32) -> Result<u32, AdcArchiveCodecError> {
        let mut quotient = 0_u32;
        loop {
            let byte = *self
                .bytes
                .get(self.bit_index / 8)
                .ok_or(AdcArchiveCodecError::TruncatedBlock)?;
            let offset = (self.bit_index % 8) as u8;
            let available = 8 - offset;
            let remaining_mask = (1_u16 << available) - 1;
            let remaining = (u16::from(byte) & remaining_mask) as u8;
            if remaining == 0 {
                quotient = quotient
                    .checked_add(u32::from(available))
                    .ok_or(AdcArchiveCodecError::RiceQuotientOutOfRange)?;
                if quotient > maximum_quotient {
                    return Err(AdcArchiveCodecError::RiceQuotientOutOfRange);
                }
                self.bit_index += usize::from(available);
                continue;
            }

            let zeroes = remaining.leading_zeros() as u8 - offset;
            quotient = quotient
                .checked_add(u32::from(zeroes))
                .ok_or(AdcArchiveCodecError::RiceQuotientOutOfRange)?;
            if quotient > maximum_quotient {
                return Err(AdcArchiveCodecError::RiceQuotientOutOfRange);
            }
            self.bit_index += usize::from(zeroes) + 1;
            return Ok(quotient);
        }
    }

    fn finish_block(&mut self) -> Result<usize, AdcArchiveCodecError> {
        let remainder = self.bit_index % 8;
        if remainder == 0 {
            return Ok(self.bit_index / 8);
        }
        let byte = *self
            .bytes
            .get(self.bit_index / 8)
            .ok_or(AdcArchiveCodecError::TruncatedBlock)?;
        let padding_mask = (1_u16 << (8 - remainder)) - 1;
        if u16::from(byte) & padding_mask != 0 {
            return Err(AdcArchiveCodecError::NonZeroPadding);
        }
        self.bit_index += 8 - remainder;
        Ok(self.bit_index / 8)
    }
}

/// ADC archive Rice codec validation and decoding errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdcArchiveCodecError {
    EmptyChunk {
        name: &'static str,
    },
    InvalidFrameBytes {
        frame_bytes: usize,
    },
    IncompleteFrameChunk {
        chunk_bytes: usize,
        frame_bytes: usize,
    },
    InvalidBlockSamples {
        block_samples: usize,
    },
    CannotAllocateOutput {
        expected_bytes: usize,
    },
    OutputSizeOverflow,
    TruncatedBlock,
    InvalidBlockTag {
        tag: u8,
    },
    RiceQuotientOutOfRange,
    ResidualOutOfRange {
        value: u32,
    },
    ReconstructedSampleOutOfRange {
        value: i32,
    },
    NonZeroPadding,
    TrailingEncodedBytes {
        trailing_bytes: usize,
    },
}

impl fmt::Display for AdcArchiveCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyChunk { name } => write!(formatter, "ADC archive {name} must be non-empty."),
            Self::InvalidFrameBytes { frame_bytes } => write!(
                formatter,
                "ADC archive frame_bytes must be a positive multiple of two; got {frame_bytes}."
            ),
            Self::IncompleteFrameChunk {
                chunk_bytes,
                frame_bytes,
            } => write!(
                formatter,
                "ADC archive chunk has {chunk_bytes} bytes, which is not a multiple of frame_bytes {frame_bytes}."
            ),
            Self::InvalidBlockSamples { block_samples } => write!(
                formatter,
                "ADC archive Rice block_samples must be a power of two in [256, 1024]; got {block_samples}."
            ),
            Self::CannotAllocateOutput { expected_bytes } => write!(
                formatter,
                "Cannot allocate bounded ADC archive output for {expected_bytes} bytes."
            ),
            Self::OutputSizeOverflow => {
                write!(formatter, "ADC archive output size overflows usize.")
            }
            Self::TruncatedBlock => write!(formatter, "ADC archive Rice block is truncated."),
            Self::InvalidBlockTag { tag } => {
                write!(
                    formatter,
                    "ADC archive Rice block tag {tag} is unsupported."
                )
            }
            Self::RiceQuotientOutOfRange => {
                write!(
                    formatter,
                    "ADC archive Rice quotient exceeds the int16 delta domain."
                )
            }
            Self::ResidualOutOfRange { value } => write!(
                formatter,
                "ADC archive ZigZag residual {value} exceeds the int16 frame-delta domain."
            ),
            Self::ReconstructedSampleOutOfRange { value } => write!(
                formatter,
                "ADC archive reconstructed sample {value} exceeds int16."
            ),
            Self::NonZeroPadding => {
                write!(formatter, "ADC archive Rice block padding must be zero.")
            }
            Self::TrailingEncodedBytes { trailing_bytes } => write!(
                formatter,
                "ADC archive chunk contains {trailing_bytes} trailing encoded bytes."
            ),
        }
    }
}

impl std::error::Error for AdcArchiveCodecError {}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::{
        ADC_RICE_BLOCK_SAMPLES, AdcArchiveCodecError, RAW_BLOCK_TAG, decode_adc_archive_chunk,
        encode_adc_archive_chunk, maximum_adc_archive_chunk_bytes,
    };

    fn samples_bytes(frames: &[&[i16]]) -> Vec<u8> {
        frames
            .iter()
            .flat_map(|frame| frame.iter())
            .flat_map(|sample| sample.to_le_bytes())
            .collect()
    }

    #[test]
    fn round_trip_preserves_absolute_and_homologous_delta_frames() {
        let first = vec![-32_768_i16; ADC_RICE_BLOCK_SAMPLES];
        let second = vec![32_767_i16; ADC_RICE_BLOCK_SAMPLES];
        let third: Vec<i16> = (0..ADC_RICE_BLOCK_SAMPLES)
            .map(|index| index as i16 - 256)
            .collect();
        let raw = samples_bytes(&[&first, &second, &third]);
        let frame_bytes = first.len() * 2;

        let encoded = encode_adc_archive_chunk(&raw, frame_bytes, ADC_RICE_BLOCK_SAMPLES).unwrap();
        let decoded =
            decode_adc_archive_chunk(&encoded, frame_bytes, 3, ADC_RICE_BLOCK_SAMPLES).unwrap();

        assert_eq!(decoded, raw);
        assert!(
            encoded.len()
                <= maximum_adc_archive_chunk_bytes(frame_bytes, 3, ADC_RICE_BLOCK_SAMPLES).unwrap()
        );
    }

    #[test]
    fn static_frames_use_temporal_rice_residuals() {
        let frame = vec![1234_i16; ADC_RICE_BLOCK_SAMPLES];
        let raw = samples_bytes(&[&frame, &frame]);
        let encoded =
            encode_adc_archive_chunk(&raw, frame.len() * 2, ADC_RICE_BLOCK_SAMPLES).unwrap();

        assert_ne!(encoded[0], RAW_BLOCK_TAG);
        assert!(encoded.len() < raw.len() / 2);
    }

    #[test]
    fn high_entropy_blocks_fall_back_to_exact_raw_samples() {
        let frame: Vec<i16> = (0..ADC_RICE_BLOCK_SAMPLES)
            .map(|index| (index as u16).wrapping_mul(32_749) as i16)
            .collect();
        let raw = samples_bytes(&[&frame]);
        let encoded = encode_adc_archive_chunk(&raw, raw.len(), ADC_RICE_BLOCK_SAMPLES).unwrap();

        assert_eq!(encoded[0], RAW_BLOCK_TAG);
        assert_eq!(encoded.len(), raw.len() + 1);
        assert_eq!(
            decode_adc_archive_chunk(&encoded, raw.len(), 1, ADC_RICE_BLOCK_SAMPLES).unwrap(),
            raw
        );
    }

    #[test]
    fn rejects_invalid_dimensions_and_ambiguous_payloads() {
        assert_eq!(
            encode_adc_archive_chunk(&[], 1024, ADC_RICE_BLOCK_SAMPLES),
            Err(AdcArchiveCodecError::EmptyChunk { name: "raw chunk" })
        );
        assert_eq!(
            encode_adc_archive_chunk(&[0; 4], 3, ADC_RICE_BLOCK_SAMPLES),
            Err(AdcArchiveCodecError::InvalidFrameBytes { frame_bytes: 3 })
        );
        assert_eq!(
            encode_adc_archive_chunk(&[0; 6], 4, ADC_RICE_BLOCK_SAMPLES),
            Err(AdcArchiveCodecError::IncompleteFrameChunk {
                chunk_bytes: 6,
                frame_bytes: 4,
            })
        );
        assert!(matches!(
            decode_adc_archive_chunk(&[17], 1024, 1, ADC_RICE_BLOCK_SAMPLES),
            Err(AdcArchiveCodecError::InvalidBlockTag { tag: 17 })
        ));
        assert!(matches!(
            decode_adc_archive_chunk(&[RAW_BLOCK_TAG, 0], 1024, 1, ADC_RICE_BLOCK_SAMPLES),
            Err(AdcArchiveCodecError::TruncatedBlock)
        ));
    }

    #[test]
    fn all_supported_block_sizes_round_trip_partial_blocks_and_restart_groups() {
        let frame_samples = 777;
        let frames: Vec<Vec<i16>> = (0..9)
            .map(|frame| {
                (0..frame_samples)
                    .map(|sample| {
                        (sample as u16)
                            .wrapping_mul(32_749)
                            .wrapping_add(frame * 113) as i16
                    })
                    .collect()
            })
            .collect();
        let frame_refs: Vec<&[i16]> = frames.iter().map(Vec::as_slice).collect();
        let raw = samples_bytes(&frame_refs);
        let frame_bytes = frame_samples * 2;

        for block_samples in [256, 512, 1024] {
            let first_group_bytes = frame_bytes * 4;
            let first =
                encode_adc_archive_chunk(&raw[..first_group_bytes], frame_bytes, block_samples)
                    .unwrap();
            let second = encode_adc_archive_chunk(
                &raw[first_group_bytes..frame_bytes * 8],
                frame_bytes,
                block_samples,
            )
            .unwrap();
            let third =
                encode_adc_archive_chunk(&raw[frame_bytes * 8..], frame_bytes, block_samples)
                    .unwrap();

            let mut decoded =
                decode_adc_archive_chunk(&first, frame_bytes, 4, block_samples).unwrap();
            decoded
                .extend(decode_adc_archive_chunk(&second, frame_bytes, 4, block_samples).unwrap());
            decoded
                .extend(decode_adc_archive_chunk(&third, frame_bytes, 1, block_samples).unwrap());
            assert_eq!(decoded, raw);
        }
    }

    #[test]
    fn optimized_encoder_preserves_v3_golden_bitstreams() {
        let frame_samples = 777;
        let frames: Vec<Vec<i16>> = (0..4)
            .map(|frame| {
                (0..frame_samples)
                    .map(|sample| {
                        (sample as u16)
                            .wrapping_mul(32_749)
                            .wrapping_add((frame * 113) as u16) as i16
                    })
                    .collect()
            })
            .collect();
        let frame_refs: Vec<&[i16]> = frames.iter().map(Vec::as_slice).collect();
        let raw = samples_bytes(&frame_refs);
        let expected = [
            (
                256,
                4_531,
                "17aaa189ebb8e187133339a19cb035f399098c964a15bf2230ec37aff2be8601",
            ),
            (
                512,
                4_667,
                "838e9d067abbae4fb76c3a75991470644cd6a033634602d6e14b3e307e1ef4a8",
            ),
            (
                1024,
                4_756,
                "6ff95c742e489be39d62c87eba7095010ffa583061d668d0b82240cabeae3185",
            ),
        ];

        for (block_samples, encoded_bytes, sha256) in expected {
            let encoded = encode_adc_archive_chunk(&raw, frame_samples * 2, block_samples).unwrap();

            assert_eq!(encoded.len(), encoded_bytes);
            assert_eq!(format!("{:x}", Sha256::digest(&encoded)), sha256);
        }
    }
}
