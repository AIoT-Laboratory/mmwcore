//! Exact single-frame ADC archive codec primitives.

use std::fmt;
use std::io::Write;

use flate2::{Compression, Decompress, FlushDecompress, Status, write::ZlibEncoder};

const ZLIB_LEVEL: u32 = 1;

/// Encode one non-empty little-endian int16 ADC frame with byte-shuffle zlib level 1.
pub fn encode_adc_archive_frame(raw: &[u8]) -> Result<Vec<u8>, AdcArchiveCodecError> {
    validate_frame_bytes(raw.len(), "raw frame")?;
    let shuffled = shuffle_i16_bytes(raw);
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(ZLIB_LEVEL));
    encoder
        .write_all(&shuffled)
        .map_err(|_| AdcArchiveCodecError::CompressionFailed)?;
    encoder
        .finish()
        .map_err(|_| AdcArchiveCodecError::CompressionFailed)
}

/// Decode one zlib-compressed byte-shuffled little-endian int16 ADC frame exactly.
pub fn decode_adc_archive_frame(
    compressed: &[u8],
    expected_raw_bytes: usize,
) -> Result<Vec<u8>, AdcArchiveCodecError> {
    validate_frame_bytes(expected_raw_bytes, "expected raw byte count")?;

    let mut shuffled = Vec::new();
    shuffled
        .try_reserve_exact(expected_raw_bytes)
        .map_err(|_| AdcArchiveCodecError::CannotAllocateOutput { expected_raw_bytes })?;
    shuffled.resize(expected_raw_bytes, 0);

    let mut decompressor = Decompress::new(true);
    let status = match decompressor.decompress(compressed, &mut shuffled, FlushDecompress::Finish) {
        Ok(status) => status,
        Err(_) if decompressor.total_out() as usize == expected_raw_bytes => {
            return Err(AdcArchiveCodecError::OutputLengthExceedsExpected { expected_raw_bytes });
        }
        Err(_) => return Err(AdcArchiveCodecError::InvalidCompressedFrame),
    };
    let produced = usize::try_from(decompressor.total_out())
        .map_err(|_| AdcArchiveCodecError::OutputLengthExceedsExpected { expected_raw_bytes })?;
    if produced > expected_raw_bytes
        || (status != Status::StreamEnd && produced == expected_raw_bytes)
    {
        return Err(AdcArchiveCodecError::OutputLengthExceedsExpected { expected_raw_bytes });
    }
    if status != Status::StreamEnd {
        return Err(AdcArchiveCodecError::InvalidCompressedFrame);
    }

    let consumed = usize::try_from(decompressor.total_in())
        .map_err(|_| AdcArchiveCodecError::InvalidCompressedFrame)?;
    if consumed != compressed.len() {
        return Err(AdcArchiveCodecError::TrailingCompressedBytes {
            trailing_bytes: compressed.len() - consumed,
        });
    }
    if produced != expected_raw_bytes {
        return Err(AdcArchiveCodecError::OutputLengthMismatch {
            expected_raw_bytes,
            actual_raw_bytes: produced,
        });
    }

    shuffled.truncate(produced);
    Ok(unshuffle_i16_bytes(&shuffled))
}

fn validate_frame_bytes(bytes: usize, name: &'static str) -> Result<(), AdcArchiveCodecError> {
    if bytes == 0 {
        return Err(AdcArchiveCodecError::EmptyFrame { name });
    }
    if !bytes.is_multiple_of(2) {
        return Err(AdcArchiveCodecError::OddByteLength { name, bytes });
    }
    Ok(())
}

fn shuffle_i16_bytes(raw: &[u8]) -> Vec<u8> {
    let sample_count = raw.len() / 2;
    let mut shuffled = vec![0_u8; raw.len()];
    for sample in 0..sample_count {
        shuffled[sample] = raw[sample * 2];
        shuffled[sample_count + sample] = raw[sample * 2 + 1];
    }
    shuffled
}

fn unshuffle_i16_bytes(shuffled: &[u8]) -> Vec<u8> {
    let sample_count = shuffled.len() / 2;
    let mut raw = vec![0_u8; shuffled.len()];
    for sample in 0..sample_count {
        raw[sample * 2] = shuffled[sample];
        raw[sample * 2 + 1] = shuffled[sample_count + sample];
    }
    raw
}

/// ADC archive frame codec validation and decoding errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdcArchiveCodecError {
    EmptyFrame {
        name: &'static str,
    },
    OddByteLength {
        name: &'static str,
        bytes: usize,
    },
    CannotAllocateOutput {
        expected_raw_bytes: usize,
    },
    CompressionFailed,
    InvalidCompressedFrame,
    OutputLengthExceedsExpected {
        expected_raw_bytes: usize,
    },
    OutputLengthMismatch {
        expected_raw_bytes: usize,
        actual_raw_bytes: usize,
    },
    TrailingCompressedBytes {
        trailing_bytes: usize,
    },
}

impl fmt::Display for AdcArchiveCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrame { name } => write!(formatter, "ADC archive {name} must be non-empty."),
            Self::OddByteLength { name, bytes } => write!(
                formatter,
                "ADC archive {name} must contain an even number of bytes; got {bytes}."
            ),
            Self::CannotAllocateOutput { expected_raw_bytes } => write!(
                formatter,
                "Cannot allocate bounded ADC archive decode output for {expected_raw_bytes} bytes."
            ),
            Self::CompressionFailed => {
                write!(formatter, "ADC archive frame zlib compression failed.")
            }
            Self::InvalidCompressedFrame => {
                write!(
                    formatter,
                    "ADC archive frame is not one complete valid zlib stream."
                )
            }
            Self::OutputLengthExceedsExpected { expected_raw_bytes } => write!(
                formatter,
                "ADC archive frame decompressed beyond expected raw byte count {expected_raw_bytes}."
            ),
            Self::OutputLengthMismatch {
                expected_raw_bytes,
                actual_raw_bytes,
            } => write!(
                formatter,
                "ADC archive frame decompressed to {actual_raw_bytes} bytes, expected {expected_raw_bytes}."
            ),
            Self::TrailingCompressedBytes { trailing_bytes } => write!(
                formatter,
                "ADC archive frame contains {trailing_bytes} trailing compressed bytes."
            ),
        }
    }
}

impl std::error::Error for AdcArchiveCodecError {}

#[cfg(test)]
mod tests {
    use super::{
        AdcArchiveCodecError, decode_adc_archive_frame, encode_adc_archive_frame, shuffle_i16_bytes,
    };

    #[test]
    fn round_trip_preserves_little_endian_int16_bytes() {
        let raw = [0x34, 0x12, 0xfe, 0xff, 0x00, 0x80, 0xff, 0x7f];

        let encoded = encode_adc_archive_frame(&raw).unwrap();
        let decoded = decode_adc_archive_frame(&encoded, raw.len()).unwrap();

        assert_eq!(decoded, raw);
    }

    #[test]
    fn rejects_invalid_frame_sizes_and_decode_payloads() {
        assert_eq!(
            encode_adc_archive_frame(&[]),
            Err(AdcArchiveCodecError::EmptyFrame { name: "raw frame" })
        );
        assert_eq!(
            encode_adc_archive_frame(&[1]),
            Err(AdcArchiveCodecError::OddByteLength {
                name: "raw frame",
                bytes: 1,
            })
        );

        let encoded = encode_adc_archive_frame(&[1, 2, 3, 4]).unwrap();
        assert_eq!(
            decode_adc_archive_frame(&encoded, 0),
            Err(AdcArchiveCodecError::EmptyFrame {
                name: "expected raw byte count",
            })
        );
        assert_eq!(
            decode_adc_archive_frame(&encoded, 3),
            Err(AdcArchiveCodecError::OddByteLength {
                name: "expected raw byte count",
                bytes: 3,
            })
        );
        assert_eq!(
            decode_adc_archive_frame(&encoded, 2),
            Err(AdcArchiveCodecError::OutputLengthExceedsExpected {
                expected_raw_bytes: 2,
            })
        );
        assert_eq!(
            decode_adc_archive_frame(&encoded, 6),
            Err(AdcArchiveCodecError::OutputLengthMismatch {
                expected_raw_bytes: 6,
                actual_raw_bytes: 4,
            })
        );
    }

    #[test]
    fn rejects_corruption_and_trailing_compressed_bytes() {
        let raw = [1, 2, 3, 4];
        let mut encoded = encode_adc_archive_frame(&raw).unwrap();
        encoded.extend_from_slice(&[0xde, 0xad]);
        assert_eq!(
            decode_adc_archive_frame(&encoded, raw.len()),
            Err(AdcArchiveCodecError::TrailingCompressedBytes { trailing_bytes: 2 })
        );

        assert_eq!(
            decode_adc_archive_frame(&[0x78, 0x01, 0x00], raw.len()),
            Err(AdcArchiveCodecError::InvalidCompressedFrame)
        );
    }

    #[test]
    fn shuffle_separates_little_endian_byte_planes() {
        assert_eq!(
            shuffle_i16_bytes(&[0x34, 0x12, 0xfe, 0xff]),
            [0x34, 0xfe, 0x12, 0xff]
        );
    }
}
