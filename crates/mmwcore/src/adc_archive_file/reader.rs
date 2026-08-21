use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::decode_adc_archive_chunk;

use super::contract::{canonical_capture_json, validate_capture_json};
use super::wire::{
    archive_chunk_count, decode_fixed_header, decode_footer, parse_index, validate_header_capture,
};
use super::{
    AdcArchiveFileError, ChunkRecord, FIXED_HEADER_BYTES, FOOTER_BYTES, FileIdentity,
    INDEX_RECORD_BYTES, error, file_identity, io_error, read_exact_array, read_exact_vec, sha256,
};

/// One opened, structurally verified ADC Archive v3 file.
#[derive(Debug)]
pub struct AdcArchiveFile {
    path: PathBuf,
    header: Vec<u8>,
    footer: [u8; FOOTER_BYTES],
    index: Vec<u8>,
    capture_json: String,
    capture_sha256: [u8; 32],
    adc_sha256: [u8; 32],
    frame_bytes: u64,
    frame_count: u64,
    block_samples: u32,
    restart_frames: u32,
    index_offset: u64,
    records: Vec<ChunkRecord>,
    identity: FileIdentity,
    verified_all: bool,
}

impl AdcArchiveFile {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn frame_bytes(&self) -> u64 {
        self.frame_bytes
    }

    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub const fn block_samples(&self) -> u32 {
        self.block_samples
    }

    pub const fn restart_frames(&self) -> u32 {
        self.restart_frames
    }

    pub const fn adc_sha256(&self) -> [u8; 32] {
        self.adc_sha256
    }

    pub const fn capture_sha256(&self) -> [u8; 32] {
        self.capture_sha256
    }

    pub fn capture_json(&self) -> &str {
        &self.capture_json
    }

    pub const fn archive_size(&self) -> u64 {
        self.identity.size
    }

    pub fn payload_bytes(&self) -> u64 {
        self.index_offset - self.header.len() as u64
    }

    pub fn index_bytes(&self) -> u64 {
        self.index.len() as u64
    }

    pub fn header_bytes(&self) -> u64 {
        self.header.len() as u64
    }

    pub fn capture_metadata_bytes(&self) -> u64 {
        (self.header.len() - FIXED_HEADER_BYTES) as u64
    }

    pub fn container_overhead_bytes(&self) -> u64 {
        self.header_bytes() + self.index_bytes() + FOOTER_BYTES as u64
    }

    pub fn read_frames(
        &mut self,
        start: u64,
        stop: u64,
        verify: bool,
    ) -> Result<Vec<u8>, AdcArchiveFileError> {
        if start > stop || stop > self.frame_count {
            return Err(error(format!(
                "Frame interval [{start}, {stop}) is outside [0, {}).",
                self.frame_count
            )));
        }
        if !verify && !self.verified_all {
            return Err(error(
                "Trusted reads require verify_all() on this archive object.",
            ));
        }
        let result = self.read_frame_range(start, stop, verify);
        if result.is_err() {
            self.verified_all = false;
        }
        result
    }

    pub fn verify_all(&mut self) -> Result<(), AdcArchiveFileError> {
        self.verified_all = false;
        if file_identity(&self.path)? != self.identity {
            return Err(error("ADC archive changed after it was opened."));
        }
        let mut logical = Sha256::new();
        let mut file = File::open(&self.path).map_err(|value| io_error("open archive", value))?;
        for record in &self.records {
            let raw = self.read_chunk(&mut file, record)?;
            if sha256(&raw) != record.raw_sha256 {
                return Err(error(
                    "Decoded chunk SHA-256 does not match the archive index.",
                ));
            }
            logical.update(&raw);
        }
        if file_identity(&self.path)? != self.identity {
            return Err(error("ADC archive changed during complete verification."));
        }
        let digest: [u8; 32] = logical.finalize().into();
        if digest != self.adc_sha256 {
            return Err(error(
                "Archive logical raw SHA-256 does not match the footer.",
            ));
        }
        self.verified_all = true;
        Ok(())
    }

    pub fn revalidate_input(&mut self) -> Result<(), AdcArchiveFileError> {
        if file_identity(&self.path)? != self.identity {
            self.verified_all = false;
            return Err(error("ADC archive changed after it was opened."));
        }
        let mut file = File::open(&self.path).map_err(|value| io_error("open archive", value))?;
        if read_exact_vec(&mut file, self.header.len(), "header")? != self.header {
            self.verified_all = false;
            return Err(error("ADC archive header changed after it was opened."));
        }
        file.seek(SeekFrom::Start(self.index_offset))
            .map_err(|value| io_error("seek archive index", value))?;
        if read_exact_vec(&mut file, self.index.len(), "index")? != self.index {
            self.verified_all = false;
            return Err(error("ADC archive index changed after it was opened."));
        }
        file.seek(SeekFrom::Start(self.identity.size - FOOTER_BYTES as u64))
            .map_err(|value| io_error("seek archive footer", value))?;
        if read_exact_array::<FOOTER_BYTES>(&mut file, "footer")? != self.footer {
            self.verified_all = false;
            return Err(error("ADC archive footer changed after it was opened."));
        }
        Ok(())
    }

    fn read_frame_range(
        &self,
        start: u64,
        stop: u64,
        verify: bool,
    ) -> Result<Vec<u8>, AdcArchiveFileError> {
        if file_identity(&self.path)? != self.identity {
            return Err(error("ADC archive changed after it was opened."));
        }
        let output_bytes = (stop - start)
            .checked_mul(self.frame_bytes)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| error("Requested ADC frame interval is too large."))?;
        if output_bytes == 0 {
            return Ok(Vec::new());
        }
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(output_bytes)
            .map_err(|_| error("Cannot allocate decoded ADC frame interval."))?;
        let restart = u64::from(self.restart_frames);
        let first_chunk = usize::try_from(start / restart)
            .map_err(|_| error("Chunk index does not fit memory."))?;
        let last_chunk = usize::try_from((stop - 1) / restart)
            .map_err(|_| error("Chunk index does not fit memory."))?;
        let frame_bytes = usize::try_from(self.frame_bytes)
            .map_err(|_| error("ADC frame length does not fit memory."))?;
        let mut file = File::open(&self.path).map_err(|value| io_error("open archive", value))?;
        for chunk_index in first_chunk..=last_chunk {
            let record = &self.records[chunk_index];
            let raw = self.read_chunk(&mut file, record)?;
            if verify && sha256(&raw) != record.raw_sha256 {
                return Err(error(
                    "Decoded chunk SHA-256 does not match the archive index.",
                ));
            }
            let chunk_first_frame = chunk_index as u64 * restart;
            let local_start = start.saturating_sub(chunk_first_frame) as usize;
            let local_stop = (stop - chunk_first_frame).min(u64::from(record.frame_count)) as usize;
            decoded.extend_from_slice(&raw[local_start * frame_bytes..local_stop * frame_bytes]);
        }
        if file_identity(&self.path)? != self.identity {
            return Err(error("ADC archive changed while frames were being read."));
        }
        debug_assert_eq!(decoded.len(), output_bytes);
        Ok(decoded)
    }

    fn read_chunk(
        &self,
        file: &mut File,
        record: &ChunkRecord,
    ) -> Result<Vec<u8>, AdcArchiveFileError> {
        file.seek(SeekFrom::Start(record.offset))
            .map_err(|value| io_error("seek encoded chunk", value))?;
        let encoded_length = usize::try_from(record.stored_bytes)
            .map_err(|_| error("Encoded chunk length does not fit memory."))?;
        let encoded = read_exact_vec(file, encoded_length, "encoded chunk")?;
        let frame_bytes = usize::try_from(self.frame_bytes)
            .map_err(|_| error("ADC frame length does not fit memory."))?;
        decode_adc_archive_chunk(
            &encoded,
            frame_bytes,
            record.frame_count as usize,
            self.block_samples as usize,
        )
        .map_err(|value| error(value.to_string()))
    }
}

/// Open a completely committed ADC Archive v3 file.
pub fn open_adc_archive_file(path: &Path) -> Result<AdcArchiveFile, AdcArchiveFileError> {
    let identity = file_identity(path)?;
    if identity.size < (FIXED_HEADER_BYTES + FOOTER_BYTES) as u64 {
        return Err(error(
            "ADC archive is too small for a v3 header and footer.",
        ));
    }
    let mut file = File::open(path).map_err(|value| io_error("open ADC archive", value))?;
    let fixed = read_exact_array::<FIXED_HEADER_BYTES>(&mut file, "fixed header")?;
    let decoded = decode_fixed_header(&fixed)?;
    let metadata_length = usize::try_from(decoded.metadata_bytes)
        .map_err(|_| error("ADC archive metadata length does not fit memory."))?;
    let metadata = read_exact_vec(&mut file, metadata_length, "capture metadata")?;
    if sha256(&metadata) != decoded.capture_sha256 {
        return Err(error(
            "ADC archive capture metadata SHA-256 does not match.",
        ));
    }
    let capture = validate_capture_json(&metadata)?;
    if canonical_capture_json(&capture)? != metadata {
        return Err(error(
            "ADC archive capture metadata is not canonical v3 JSON.",
        ));
    }
    validate_header_capture(&decoded, &capture)?;
    let mut header = Vec::with_capacity(FIXED_HEADER_BYTES + metadata.len());
    header.extend_from_slice(&fixed);
    header.extend_from_slice(&metadata);

    file.seek(SeekFrom::Start(identity.size - FOOTER_BYTES as u64))
        .map_err(|value| io_error("seek commit footer", value))?;
    let footer = read_exact_array::<FOOTER_BYTES>(&mut file, "commit footer")?;
    let decoded_footer = decode_footer(&footer, &header)?;
    let chunk_count = archive_chunk_count(decoded.frame_count, decoded.restart_frames)?;
    let expected_index_bytes = (chunk_count as u64)
        .checked_mul(INDEX_RECORD_BYTES as u64)
        .ok_or_else(|| error("ADC archive index length overflows u64."))?;
    if decoded_footer.index_bytes != expected_index_bytes
        || decoded_footer.index_offset + decoded_footer.index_bytes
            != identity.size - FOOTER_BYTES as u64
        || decoded_footer.index_offset < decoded.header_bytes
    {
        return Err(error(
            "ADC archive index bounds do not match the v3 layout.",
        ));
    }
    file.seek(SeekFrom::Start(decoded_footer.index_offset))
        .map_err(|value| io_error("seek chunk index", value))?;
    let index_length = usize::try_from(decoded_footer.index_bytes)
        .map_err(|_| error("ADC archive index length does not fit memory."))?;
    let index = read_exact_vec(&mut file, index_length, "chunk index")?;
    if sha256(&index) != decoded_footer.index_sha256 {
        return Err(error(
            "ADC archive index SHA-256 does not match the footer.",
        ));
    }
    let records = parse_index(&index, &decoded, decoded_footer.index_offset)?;
    let capture_json = String::from_utf8(metadata)
        .map_err(|_| error("ADC archive capture metadata is not UTF-8 JSON."))?;
    if file_identity(path)? != identity {
        return Err(error("ADC archive changed while it was being opened."));
    }
    Ok(AdcArchiveFile {
        path: path.to_path_buf(),
        header,
        footer,
        index,
        capture_json,
        capture_sha256: decoded.capture_sha256,
        adc_sha256: decoded_footer.adc_sha256,
        frame_bytes: decoded.frame_bytes,
        frame_count: decoded.frame_count,
        block_samples: decoded.block_samples,
        restart_frames: decoded.restart_frames,
        index_offset: decoded_footer.index_offset,
        records,
        identity,
        verified_all: false,
    })
}
