//! Exact single-frame ADC evidence codec primitives.

use std::fmt;
use std::io::Write;

use flate2::{Compression, Decompress, FlushDecompress, Status, write::ZlibEncoder};

const ZLIB_LEVEL: u32 = 1;

/// Encode one non-empty little-endian int16 ADC frame with byte-shuffle zlib level 1.
pub fn encode_evidence_frame(raw: &[u8]) -> Result<Vec<u8>, EvidenceCodecError> {
    validate_frame_bytes(raw.len(), "raw frame")?;
    let shuffled = shuffle_i16_bytes(raw);
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(ZLIB_LEVEL));
    encoder
        .write_all(&shuffled)
        .map_err(|_| EvidenceCodecError::CompressionFailed)?;
    encoder
        .finish()
        .map_err(|_| EvidenceCodecError::CompressionFailed)
}

/// Decode one zlib-compressed byte-shuffled little-endian int16 ADC frame exactly.
pub fn decode_evidence_frame(
    compressed: &[u8],
    expected_raw_bytes: usize,
) -> Result<Vec<u8>, EvidenceCodecError> {
    validate_frame_bytes(expected_raw_bytes, "expected raw byte count")?;

    let mut shuffled = Vec::new();
    shuffled
        .try_reserve_exact(expected_raw_bytes)
        .map_err(|_| EvidenceCodecError::CannotAllocateOutput { expected_raw_bytes })?;
    shuffled.resize(expected_raw_bytes, 0);

    let mut decompressor = Decompress::new(true);
    let status = match decompressor.decompress(compressed, &mut shuffled, FlushDecompress::Finish) {
        Ok(status) => status,
        Err(_) if decompressor.total_out() as usize == expected_raw_bytes => {
            return Err(EvidenceCodecError::OutputLengthExceedsExpected { expected_raw_bytes });
        }
        Err(_) => return Err(EvidenceCodecError::InvalidCompressedFrame),
    };
    let produced = usize::try_from(decompressor.total_out())
        .map_err(|_| EvidenceCodecError::OutputLengthExceedsExpected { expected_raw_bytes })?;
    if produced > expected_raw_bytes
        || (status != Status::StreamEnd && produced == expected_raw_bytes)
    {
        return Err(EvidenceCodecError::OutputLengthExceedsExpected { expected_raw_bytes });
    }
    if status != Status::StreamEnd {
        return Err(EvidenceCodecError::InvalidCompressedFrame);
    }

    let consumed = usize::try_from(decompressor.total_in())
        .map_err(|_| EvidenceCodecError::InvalidCompressedFrame)?;
    if consumed != compressed.len() {
        return Err(EvidenceCodecError::TrailingCompressedBytes {
            trailing_bytes: compressed.len() - consumed,
        });
    }
    if produced != expected_raw_bytes {
        return Err(EvidenceCodecError::OutputLengthMismatch {
            expected_raw_bytes,
            actual_raw_bytes: produced,
        });
    }

    shuffled.truncate(produced);
    Ok(unshuffle_i16_bytes(&shuffled))
}

fn validate_frame_bytes(bytes: usize, name: &'static str) -> Result<(), EvidenceCodecError> {
    if bytes == 0 {
        return Err(EvidenceCodecError::EmptyFrame { name });
    }
    if !bytes.is_multiple_of(2) {
        return Err(EvidenceCodecError::OddByteLength { name, bytes });
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

/// Evidence frame codec validation and decoding errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceCodecError {
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

impl fmt::Display for EvidenceCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrame { name } => write!(formatter, "Evidence {name} must be non-empty."),
            Self::OddByteLength { name, bytes } => write!(
                formatter,
                "Evidence {name} must contain an even number of bytes; got {bytes}."
            ),
            Self::CannotAllocateOutput { expected_raw_bytes } => write!(
                formatter,
                "Cannot allocate bounded evidence decode output for {expected_raw_bytes} bytes."
            ),
            Self::CompressionFailed => write!(formatter, "Evidence frame zlib compression failed."),
            Self::InvalidCompressedFrame => {
                write!(
                    formatter,
                    "Evidence frame is not one complete valid zlib stream."
                )
            }
            Self::OutputLengthExceedsExpected { expected_raw_bytes } => write!(
                formatter,
                "Evidence frame decompressed beyond expected raw byte count {expected_raw_bytes}."
            ),
            Self::OutputLengthMismatch {
                expected_raw_bytes,
                actual_raw_bytes,
            } => write!(
                formatter,
                "Evidence frame decompressed to {actual_raw_bytes} bytes, expected {expected_raw_bytes}."
            ),
            Self::TrailingCompressedBytes { trailing_bytes } => write!(
                formatter,
                "Evidence frame contains {trailing_bytes} trailing compressed bytes."
            ),
        }
    }
}

impl std::error::Error for EvidenceCodecError {}

#[cfg(test)]
mod tests {
    use super::{
        EvidenceCodecError, decode_evidence_frame, encode_evidence_frame, shuffle_i16_bytes,
    };

    #[test]
    fn round_trip_preserves_little_endian_int16_bytes() {
        let raw = [0x34, 0x12, 0xfe, 0xff, 0x00, 0x80, 0xff, 0x7f];

        let encoded = encode_evidence_frame(&raw).unwrap();
        let decoded = decode_evidence_frame(&encoded, raw.len()).unwrap();

        assert_eq!(decoded, raw);
    }

    #[test]
    fn rejects_invalid_frame_sizes_and_decode_payloads() {
        assert_eq!(
            encode_evidence_frame(&[]),
            Err(EvidenceCodecError::EmptyFrame { name: "raw frame" })
        );
        assert_eq!(
            encode_evidence_frame(&[1]),
            Err(EvidenceCodecError::OddByteLength {
                name: "raw frame",
                bytes: 1,
            })
        );

        let encoded = encode_evidence_frame(&[1, 2, 3, 4]).unwrap();
        assert_eq!(
            decode_evidence_frame(&encoded, 0),
            Err(EvidenceCodecError::EmptyFrame {
                name: "expected raw byte count",
            })
        );
        assert_eq!(
            decode_evidence_frame(&encoded, 3),
            Err(EvidenceCodecError::OddByteLength {
                name: "expected raw byte count",
                bytes: 3,
            })
        );
        assert_eq!(
            decode_evidence_frame(&encoded, 2),
            Err(EvidenceCodecError::OutputLengthExceedsExpected {
                expected_raw_bytes: 2,
            })
        );
        assert_eq!(
            decode_evidence_frame(&encoded, 6),
            Err(EvidenceCodecError::OutputLengthMismatch {
                expected_raw_bytes: 6,
                actual_raw_bytes: 4,
            })
        );
    }

    #[test]
    fn rejects_corruption_and_trailing_compressed_bytes() {
        let raw = [1, 2, 3, 4];
        let mut encoded = encode_evidence_frame(&raw).unwrap();
        encoded.extend_from_slice(&[0xde, 0xad]);
        assert_eq!(
            decode_evidence_frame(&encoded, raw.len()),
            Err(EvidenceCodecError::TrailingCompressedBytes { trailing_bytes: 2 })
        );

        assert_eq!(
            decode_evidence_frame(&[0x78, 0x01, 0x00], raw.len()),
            Err(EvidenceCodecError::InvalidCompressedFrame)
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
