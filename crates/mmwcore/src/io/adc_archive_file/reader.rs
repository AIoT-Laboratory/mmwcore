use std::collections::BTreeMap;
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
    AdcArchiveFileError, ChunkRecord, FIXED_HEADER_BYTES, FOOTER_BYTES, INDEX_RECORD_BYTES, error,
    io_error, read_exact_array, read_exact_vec, regular_file_size, sha256,
};

/// One opened, structurally verified ADC Archive v3 file.
#[derive(Debug)]
pub struct AdcArchiveFile {
    path: PathBuf,
    header: Vec<u8>,
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
    archive_size: u64,
}

#[derive(Debug)]
struct WindowCopy {
    raw_start: usize,
    raw_stop: usize,
    output_start: usize,
}

fn frame_byte_offset(frame_count: u64, frame_bytes: usize) -> Option<usize> {
    usize::try_from(frame_count).ok()?.checked_mul(frame_bytes)
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
        self.archive_size
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
        self.read_window_batch(&[start], stop - start, verify)
    }

    /// Read fixed-length frame windows in caller order while decoding each touched chunk once.
    pub fn read_windows(
        &mut self,
        starts: &[u64],
        window_frames: u64,
        verify: bool,
    ) -> Result<Vec<u8>, AdcArchiveFileError> {
        if window_frames == 0 {
            return Err(error("ADC window length must be greater than zero."));
        }
        for (index, &start) in starts.iter().enumerate() {
            let stop = start
                .checked_add(window_frames)
                .ok_or_else(|| error(format!("ADC window at index {index} overflows u64.")))?;
            if stop > self.frame_count {
                return Err(error(format!(
                    "ADC window at index {index} [{start}, {stop}) is outside [0, {}).",
                    self.frame_count
                )));
            }
        }
        self.read_window_batch(starts, window_frames, verify)
    }

    pub fn verify_all(&self) -> Result<(), AdcArchiveFileError> {
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
        let digest: [u8; 32] = logical.finalize().into();
        if digest != self.adc_sha256 {
            return Err(error(
                "Archive logical raw SHA-256 does not match the footer.",
            ));
        }
        Ok(())
    }

    fn read_window_batch(
        &self,
        starts: &[u64],
        window_frames: u64,
        verify: bool,
    ) -> Result<Vec<u8>, AdcArchiveFileError> {
        let frame_bytes = usize::try_from(self.frame_bytes)
            .map_err(|_| error("ADC frame length does not fit memory."))?;
        let window_frames_usize = usize::try_from(window_frames)
            .map_err(|_| error("ADC window length does not fit memory."))?;
        let window_bytes = window_frames_usize
            .checked_mul(frame_bytes)
            .ok_or_else(|| error("Requested ADC window is too large."))?;
        let output_bytes = starts
            .len()
            .checked_mul(window_bytes)
            .ok_or_else(|| error("Requested ADC window batch is too large."))?;
        if output_bytes == 0 {
            return Ok(Vec::new());
        }
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(output_bytes)
            .map_err(|_| error("Cannot allocate decoded ADC window batch."))?;
        decoded.resize(output_bytes, 0);

        let restart = u64::from(self.restart_frames);
        let mut chunk_copies: BTreeMap<usize, Vec<WindowCopy>> = BTreeMap::new();
        for (window_index, &start) in starts.iter().enumerate() {
            let stop = start
                .checked_add(window_frames)
                .ok_or_else(|| error("ADC window end overflows u64."))?;
            let first_chunk = usize::try_from(start / restart)
                .map_err(|_| error("Chunk index does not fit memory."))?;
            let last_chunk = usize::try_from((stop - 1) / restart)
                .map_err(|_| error("Chunk index does not fit memory."))?;
            let output_base = window_index
                .checked_mul(window_bytes)
                .ok_or_else(|| error("ADC window output offset overflows memory."))?;
            for chunk_index in first_chunk..=last_chunk {
                let record = &self.records[chunk_index];
                let chunk_first_frame = chunk_index as u64 * restart;
                let copy_start_frame = start.max(chunk_first_frame);
                let copy_stop_frame = stop.min(chunk_first_frame + u64::from(record.frame_count));
                let raw_start =
                    frame_byte_offset(copy_start_frame - chunk_first_frame, frame_bytes)
                        .ok_or_else(|| error("ADC chunk byte offset overflows memory."))?;
                let raw_stop = frame_byte_offset(copy_stop_frame - chunk_first_frame, frame_bytes)
                    .ok_or_else(|| error("ADC chunk byte offset overflows memory."))?;
                let output_start = frame_byte_offset(copy_start_frame - start, frame_bytes)
                    .and_then(|value| output_base.checked_add(value))
                    .ok_or_else(|| error("ADC window output offset overflows memory."))?;
                chunk_copies
                    .entry(chunk_index)
                    .or_default()
                    .push(WindowCopy {
                        raw_start,
                        raw_stop,
                        output_start,
                    });
            }
        }

        let mut file = File::open(&self.path).map_err(|value| io_error("open archive", value))?;
        for (chunk_index, copies) in chunk_copies {
            let record = &self.records[chunk_index];
            let raw = self.read_chunk(&mut file, record)?;
            if verify && sha256(&raw) != record.raw_sha256 {
                return Err(error(
                    "Decoded chunk SHA-256 does not match the archive index.",
                ));
            }
            for copy in copies {
                let output_stop = copy
                    .output_start
                    .checked_add(copy.raw_stop - copy.raw_start)
                    .ok_or_else(|| error("ADC window output offset overflows memory."))?;
                decoded[copy.output_start..output_stop]
                    .copy_from_slice(&raw[copy.raw_start..copy.raw_stop]);
            }
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
    let archive_size = regular_file_size(path)?;
    if archive_size < (FIXED_HEADER_BYTES + FOOTER_BYTES) as u64 {
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

    file.seek(SeekFrom::Start(archive_size - FOOTER_BYTES as u64))
        .map_err(|value| io_error("seek commit footer", value))?;
    let footer = read_exact_array::<FOOTER_BYTES>(&mut file, "commit footer")?;
    let decoded_footer = decode_footer(&footer, &header)?;
    let chunk_count = archive_chunk_count(decoded.frame_count, decoded.restart_frames)?;
    let expected_index_bytes = (chunk_count as u64)
        .checked_mul(INDEX_RECORD_BYTES as u64)
        .ok_or_else(|| error("ADC archive index length overflows u64."))?;
    if decoded_footer.index_bytes != expected_index_bytes
        || decoded_footer.index_offset + decoded_footer.index_bytes
            != archive_size - FOOTER_BYTES as u64
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
    Ok(AdcArchiveFile {
        path: path.to_path_buf(),
        header,
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
        archive_size,
    })
}
