use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::{ADC_RICE_BLOCK_SAMPLES, ADC_RICE_RESTART_FRAMES, encode_adc_archive_chunk};

use super::contract::{canonical_capture_json, capture_frame_bytes, validate_capture_json};
use super::reader::{AdcArchiveFile, open_adc_archive_file};
use super::wire::{archive_chunk_count, encode_footer, encode_header, encode_index};
use super::{
    AdcArchiveFileError, ChunkRecord, FileIdentity, error, file_identity, io_error, sha256,
};

static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write one self-describing ADC Archive v3 file and publish it without overwrite.
pub fn write_adc_archive_file(
    source: &Path,
    destination: &Path,
    capture_json: &str,
    expected_adc_sha256: Option<[u8; 32]>,
) -> Result<AdcArchiveFile, AdcArchiveFileError> {
    let source_identity = file_identity(source)?;
    let capture = validate_capture_json(capture_json.as_bytes())?;
    let frame_bytes = capture_frame_bytes(&capture)?;
    let frame_count = capture
        .num_frames
        .ok_or_else(|| error("Embedded capture num_frames must be present."))?;
    let expected_size = frame_bytes
        .checked_mul(frame_count)
        .ok_or_else(|| error("Embedded capture size overflows u64."))?;
    if source_identity.size != expected_size || capture.expected_size_bytes != Some(expected_size) {
        return Err(error(
            "Source size does not match the embedded capture frame contract.",
        ));
    }
    require_new_destination(destination)?;

    let metadata = canonical_capture_json(&capture)?;
    let capture_sha256 = sha256(&metadata);
    let header = encode_header(
        &metadata,
        frame_bytes,
        frame_count,
        ADC_RICE_BLOCK_SAMPLES as u32,
        ADC_RICE_RESTART_FRAMES as u32,
        capture_sha256,
    )?;
    let temporary = temporary_path(destination)?;
    if let Err(failure) = write_temporary_archive(
        source,
        &temporary,
        &header,
        frame_bytes,
        frame_count,
        expected_adc_sha256,
        &source_identity,
    ) {
        let _ = fs::remove_file(&temporary);
        return Err(failure);
    }
    let prepared = match open_adc_archive_file(&temporary) {
        Ok(value) => value,
        Err(failure) => {
            let _ = fs::remove_file(&temporary);
            return Err(failure);
        }
    };
    if let Err(failure) = fs::hard_link(&temporary, destination) {
        let _ = fs::remove_file(&temporary);
        if failure.kind() == std::io::ErrorKind::AlreadyExists {
            return Err(error(format!(
                "ADC archive destination already exists: {}",
                destination.display()
            )));
        }
        return Err(io_error("publish ADC archive", failure));
    }
    let committed = match open_adc_archive_file(destination) {
        Ok(value) => value,
        Err(failure) => {
            let _ = fs::remove_file(destination);
            let _ = fs::remove_file(&temporary);
            return Err(failure);
        }
    };
    if prepared.adc_sha256() != committed.adc_sha256()
        || prepared.capture_sha256() != committed.capture_sha256()
    {
        let _ = fs::remove_file(destination);
        let _ = fs::remove_file(&temporary);
        return Err(error("Published ADC archive identity changed."));
    }
    if let Err(failure) = sync_parent(destination) {
        let _ = fs::remove_file(destination);
        let _ = fs::remove_file(&temporary);
        return Err(failure);
    }
    let _ = fs::remove_file(&temporary);
    Ok(committed)
}

fn write_temporary_archive(
    source: &Path,
    temporary: &Path,
    header: &[u8],
    frame_bytes: u64,
    frame_count: u64,
    expected_adc_sha256: Option<[u8; 32]>,
    source_identity: &FileIdentity,
) -> Result<(), AdcArchiveFileError> {
    let mut source_file = File::open(source).map_err(|value| io_error("open ADC source", value))?;
    let mut archive = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary)
        .map_err(|value| io_error("create temporary ADC archive", value))?;
    archive
        .write_all(header)
        .map_err(|value| io_error("write ADC archive header", value))?;
    let frame_length =
        usize::try_from(frame_bytes).map_err(|_| error("ADC frame size does not fit memory."))?;
    let restart_frames = ADC_RICE_RESTART_FRAMES;
    let maximum_chunk_bytes = frame_length
        .checked_mul(restart_frames)
        .ok_or_else(|| error("ADC chunk size does not fit memory."))?;
    let capacity = archive_chunk_count(frame_count, restart_frames as u32)?;
    let mut raw = vec![0_u8; maximum_chunk_bytes];
    let mut records = Vec::with_capacity(capacity);
    let mut logical = Sha256::new();
    let mut offset = header.len() as u64;
    let mut remaining_frames = frame_count;
    while remaining_frames > 0 {
        let chunk_frames = remaining_frames.min(restart_frames as u64) as usize;
        let raw_bytes = frame_length
            .checked_mul(chunk_frames)
            .ok_or_else(|| error("ADC chunk size does not fit memory."))?;
        let chunk = &mut raw[..raw_bytes];
        source_file
            .read_exact(chunk)
            .map_err(|value| io_error("read ADC source chunk", value))?;
        logical.update(&*chunk);
        let encoded = encode_adc_archive_chunk(chunk, frame_length, ADC_RICE_BLOCK_SAMPLES)
            .map_err(|value| error(value.to_string()))?;
        archive
            .write_all(&encoded)
            .map_err(|value| io_error("write encoded ADC chunk", value))?;
        records.push(ChunkRecord {
            offset,
            stored_bytes: encoded.len() as u64,
            frame_count: chunk_frames as u32,
            raw_sha256: sha256(chunk),
        });
        offset = offset
            .checked_add(encoded.len() as u64)
            .ok_or_else(|| error("ADC archive payload offset overflows u64."))?;
        remaining_frames -= chunk_frames as u64;
    }
    let adc_sha256: [u8; 32] = logical.finalize().into();
    if expected_adc_sha256.is_some_and(|expected| expected != adc_sha256) {
        return Err(error(
            "Source logical SHA-256 does not match expected_adc_sha256.",
        ));
    }
    if file_identity(source)? != *source_identity {
        return Err(error(
            "ADC source changed while the archive was being written.",
        ));
    }
    let index = encode_index(&records);
    let footer = encode_footer(
        offset,
        index.len() as u64,
        header,
        sha256(&index),
        adc_sha256,
    );
    archive
        .write_all(&index)
        .and_then(|_| archive.write_all(&footer))
        .and_then(|_| archive.flush())
        .and_then(|_| archive.sync_all())
        .map_err(|value| io_error("commit ADC archive", value))?;
    Ok(())
}

fn require_new_destination(path: &Path) -> Result<(), AdcArchiveFileError> {
    if path.exists() {
        return Err(error(format!(
            "ADC archive destination already exists: {}",
            path.display()
        )));
    }
    let parent = path
        .parent()
        .ok_or_else(|| error("ADC archive destination has no parent directory."))?;
    if !parent.is_dir() {
        return Err(error(format!(
            "ADC archive destination parent does not exist: {}",
            parent.display()
        )));
    }
    Ok(())
}

fn temporary_path(destination: &Path) -> Result<PathBuf, AdcArchiveFileError> {
    let parent = destination
        .parent()
        .ok_or_else(|| error("ADC archive destination has no parent directory."))?;
    let name = destination
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| error("ADC archive destination filename is not valid UTF-8."))?;
    for _ in 0..32 {
        let count = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{name}.{}.{count}.tmp", std::process::id()));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(error(
        "Cannot allocate a unique temporary ADC archive path.",
    ))
}

fn sync_parent(_path: &Path) -> Result<(), AdcArchiveFileError> {
    #[cfg(unix)]
    {
        let parent = _path
            .parent()
            .ok_or_else(|| error("ADC archive destination has no parent directory."))?;
        File::open(parent)
            .and_then(|file| file.sync_all())
            .map_err(|value| io_error("sync archive directory", value))?;
    }
    Ok(())
}
