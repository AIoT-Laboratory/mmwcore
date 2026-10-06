use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::writer::{require_new_destination, temporary_path};
use super::{CompressedAdcFileError, error, io_error, open_compressed_adc};

/// Restore the original ADC bytes and return the embedded decoding contract.
///
/// Decodes one restart group at a time. All chunk and complete-stream digests
/// must pass before the raw file is published; an existing destination is never replaced.
pub fn decompress_adc_file(
    source: &Path,
    destination: &Path,
) -> Result<String, CompressedAdcFileError> {
    let mut compressed = open_compressed_adc(source)?;
    require_new_destination(destination)?;
    let temporary = temporary_path(destination)?;
    let result = (|| {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|value| io_error("create decompressed ADC file", value))?;
        let mut digest = Sha256::new();
        for start in (0..compressed.frame_count()).step_by(compressed.restart_frames() as usize) {
            let stop =
                (start + u64::from(compressed.restart_frames())).min(compressed.frame_count());
            let raw = compressed.read_frames(start, stop, true)?;
            digest.update(&raw);
            output
                .write_all(&raw)
                .map_err(|value| io_error("write decompressed ADC bytes", value))?;
        }
        if digest.finalize().as_slice() != compressed.adc_sha256() {
            return Err(error("Decompressed ADC stream SHA-256 mismatch."));
        }
        output
            .sync_all()
            .map_err(|value| io_error("flush decompressed ADC file", value))?;
        drop(output);
        fs::hard_link(&temporary, destination)
            .map_err(|value| io_error("publish decompressed ADC file", value))?;
        Ok(compressed.capture_json().to_owned())
    })();
    let _ = fs::remove_file(temporary);
    result
}
