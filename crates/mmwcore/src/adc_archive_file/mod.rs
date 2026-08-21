//! Self-describing, verified storage for finite raw ADC captures.

mod contract;
mod reader;
mod wire;
mod writer;

use std::fmt;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::time::UNIX_EPOCH;

use sha2::{Digest, Sha256};

pub use reader::{AdcArchiveFile, open_adc_archive_file};
pub use writer::write_adc_archive_file;

const HEADER_MAGIC: &[u8; 8] = b"MMWADCA3";
const FOOTER_MAGIC: &[u8; 8] = b"MMWACMT3";
const VERSION: u32 = 3;
const FIXED_HEADER_BYTES: usize = 112;
const INDEX_RECORD_BYTES: usize = 56;
const FOOTER_BYTES: usize = 160;
const CODEC_I16_FRAME_DELTA_RICE: u32 = 2;
const METADATA_RADAR_CAPTURE_JSON: u32 = 1;
const DEFAULT_RESTART_FRAMES: usize = 4;
const MAX_FRAME_BYTES: u64 = 64 * 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 1024 * 1024;
const MAX_RESTART_FRAMES: u32 = 64;

#[derive(Clone, Debug)]
struct ChunkRecord {
    offset: u64,
    stored_bytes: u64,
    frame_count: u32,
    raw_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    size: u64,
    modified_ns: u128,
}

fn file_identity(path: &Path) -> Result<FileIdentity, AdcArchiveFileError> {
    let metadata = fs::metadata(path).map_err(|value| io_error("stat file", value))?;
    if !metadata.is_file() {
        return Err(error(format!(
            "Path is not a regular file: {}",
            path.display()
        )));
    }
    let modified_ns = metadata
        .modified()
        .map_err(|value| io_error("read file modification time", value))?
        .duration_since(UNIX_EPOCH)
        .map_err(|_| error("File modification time precedes the Unix epoch."))?
        .as_nanos();
    Ok(FileIdentity {
        size: metadata.len(),
        modified_ns,
    })
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn read_exact_vec(
    reader: &mut impl Read,
    length: usize,
    label: &str,
) -> Result<Vec<u8>, AdcArchiveFileError> {
    let mut bytes = vec![0_u8; length];
    reader
        .read_exact(&mut bytes)
        .map_err(|value| io_error(&format!("read {label}"), value))?;
    Ok(bytes)
}

fn read_exact_array<const N: usize>(
    reader: &mut impl Read,
    label: &str,
) -> Result<[u8; N], AdcArchiveFileError> {
    let mut bytes = [0_u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|value| io_error(&format!("read {label}"), value))?;
    Ok(bytes)
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, AdcArchiveFileError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| error("ADC archive integer field is truncated."))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("u32 field length"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, AdcArchiveFileError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| error("ADC archive integer field is truncated."))?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("u64 field length"),
    ))
}

/// Parse one lowercase hexadecimal SHA-256 value for the archive writer.
pub fn sha256_from_hex(value: &str) -> Result<[u8; 32], AdcArchiveFileError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(error(
            "SHA-256 must contain 64 lowercase hexadecimal characters.",
        ));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        digest[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(digest)
}

fn hex_nibble(value: u8) -> Result<u8, AdcArchiveFileError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(error("SHA-256 contains a non-hexadecimal character.")),
    }
}

pub fn sha256_to_hex(value: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

/// ADC archive container validation or file-system failure.
#[derive(Debug)]
pub struct AdcArchiveFileError(AdcArchiveFileErrorKind);

#[derive(Debug)]
enum AdcArchiveFileErrorKind {
    Domain(String),
    Io {
        context: String,
        source: std::io::Error,
    },
}

impl AdcArchiveFileError {
    /// Return the underlying file-system category, or `None` for archive-domain failures.
    pub fn io_kind(&self) -> Option<std::io::ErrorKind> {
        match &self.0 {
            AdcArchiveFileErrorKind::Domain(_) => None,
            AdcArchiveFileErrorKind::Io { source, .. } => Some(source.kind()),
        }
    }
}

impl fmt::Display for AdcArchiveFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            AdcArchiveFileErrorKind::Domain(message) => formatter.write_str(message),
            AdcArchiveFileErrorKind::Io { context, source } => {
                write!(formatter, "{context}: {source}")
            }
        }
    }
}

impl std::error::Error for AdcArchiveFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            AdcArchiveFileErrorKind::Domain(_) => None,
            AdcArchiveFileErrorKind::Io { source, .. } => Some(source),
        }
    }
}

fn error(message: impl Into<String>) -> AdcArchiveFileError {
    AdcArchiveFileError(AdcArchiveFileErrorKind::Domain(message.into()))
}

fn io_error(context: &str, source: std::io::Error) -> AdcArchiveFileError {
    AdcArchiveFileError(AdcArchiveFileErrorKind::Io {
        context: context.to_owned(),
        source,
    })
}
