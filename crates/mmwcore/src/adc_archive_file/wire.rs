use crate::maximum_adc_archive_chunk_bytes;

use super::contract::{CaptureRecord, capture_frame_bytes};
use super::{
    AdcArchiveFileError, CODEC_I16_FRAME_DELTA_RICE, ChunkRecord, FIXED_HEADER_BYTES, FOOTER_BYTES,
    FOOTER_MAGIC, HEADER_MAGIC, INDEX_RECORD_BYTES, MAX_RESTART_FRAMES,
    METADATA_RADAR_CAPTURE_JSON, VERSION, error, push_u32, push_u64, read_u32, read_u64, sha256,
};

#[derive(Debug)]
pub(super) struct DecodedHeader {
    pub(super) header_bytes: u64,
    pub(super) metadata_bytes: u64,
    pub(super) frame_bytes: u64,
    pub(super) frame_count: u64,
    pub(super) block_samples: u32,
    pub(super) restart_frames: u32,
    pub(super) capture_sha256: [u8; 32],
}

#[derive(Debug)]
pub(super) struct DecodedFooter {
    pub(super) index_offset: u64,
    pub(super) index_bytes: u64,
    pub(super) index_sha256: [u8; 32],
    pub(super) adc_sha256: [u8; 32],
}

pub(super) fn encode_header(
    metadata: &[u8],
    frame_bytes: u64,
    frame_count: u64,
    block_samples: u32,
    restart_frames: u32,
    capture_sha256: [u8; 32],
) -> Result<Vec<u8>, AdcArchiveFileError> {
    validate_codec_dimensions(frame_bytes, block_samples, restart_frames)?;
    if metadata.is_empty() || metadata.len() as u64 > super::MAX_METADATA_BYTES {
        return Err(error(
            "ADC archive capture metadata size is outside v3 bounds.",
        ));
    }
    let header_bytes = FIXED_HEADER_BYTES as u64 + metadata.len() as u64;
    let mut header = Vec::with_capacity(header_bytes as usize);
    header.extend_from_slice(HEADER_MAGIC);
    push_u32(&mut header, VERSION);
    push_u32(&mut header, FIXED_HEADER_BYTES as u32);
    push_u64(&mut header, header_bytes);
    push_u64(&mut header, metadata.len() as u64);
    push_u64(&mut header, frame_bytes);
    push_u64(&mut header, frame_count);
    push_u32(&mut header, INDEX_RECORD_BYTES as u32);
    push_u32(&mut header, CODEC_I16_FRAME_DELTA_RICE);
    push_u32(&mut header, METADATA_RADAR_CAPTURE_JSON);
    push_u32(&mut header, 0);
    push_u32(&mut header, block_samples);
    push_u32(&mut header, restart_frames);
    push_u64(&mut header, 0);
    header.extend_from_slice(&capture_sha256);
    debug_assert_eq!(header.len(), FIXED_HEADER_BYTES);
    header.extend_from_slice(metadata);
    Ok(header)
}

pub(super) fn decode_fixed_header(
    bytes: &[u8; FIXED_HEADER_BYTES],
) -> Result<DecodedHeader, AdcArchiveFileError> {
    if &bytes[0..8] != HEADER_MAGIC {
        return Err(error("ADC archive header is not mmwcore.adc_archive.v3."));
    }
    if read_u32(bytes, 8)? != VERSION
        || read_u32(bytes, 12)? != FIXED_HEADER_BYTES as u32
        || read_u32(bytes, 48)? != INDEX_RECORD_BYTES as u32
        || read_u32(bytes, 52)? != CODEC_I16_FRAME_DELTA_RICE
        || read_u32(bytes, 56)? != METADATA_RADAR_CAPTURE_JSON
        || read_u32(bytes, 60)? != 0
        || read_u64(bytes, 72)? != 0
    {
        return Err(error("ADC archive v3 header contract is unsupported."));
    }
    let header_bytes = read_u64(bytes, 16)?;
    let metadata_bytes = read_u64(bytes, 24)?;
    let frame_bytes = read_u64(bytes, 32)?;
    let frame_count = read_u64(bytes, 40)?;
    let block_samples = read_u32(bytes, 64)?;
    let restart_frames = read_u32(bytes, 68)?;
    if metadata_bytes == 0
        || metadata_bytes > super::MAX_METADATA_BYTES
        || header_bytes != FIXED_HEADER_BYTES as u64 + metadata_bytes
        || frame_bytes == 0
        || !frame_bytes.is_multiple_of(2)
        || frame_bytes > super::MAX_FRAME_BYTES
        || frame_count == 0
    {
        return Err(error("ADC archive v3 header dimensions are invalid."));
    }
    validate_codec_dimensions(frame_bytes, block_samples, restart_frames)?;
    Ok(DecodedHeader {
        header_bytes,
        metadata_bytes,
        frame_bytes,
        frame_count,
        block_samples,
        restart_frames,
        capture_sha256: bytes[80..112].try_into().expect("fixed digest slice"),
    })
}

pub(super) fn decode_footer(
    footer: &[u8; FOOTER_BYTES],
    header: &[u8],
) -> Result<DecodedFooter, AdcArchiveFileError> {
    if &footer[0..8] != FOOTER_MAGIC
        || read_u32(footer, 8)? != VERSION
        || read_u32(footer, 12)? != FOOTER_BYTES as u32
    {
        return Err(error("ADC archive v3 commit footer is missing or invalid."));
    }
    let footer_sha256: [u8; 32] = footer[128..160].try_into().expect("footer digest slice");
    if footer_sha256 != sha256(&footer[..128]) {
        return Err(error("ADC archive footer SHA-256 does not match."));
    }
    let header_sha256: [u8; 32] = footer[32..64].try_into().expect("header digest slice");
    if header_sha256 != sha256(header) {
        return Err(error(
            "ADC archive header SHA-256 does not match the footer.",
        ));
    }
    Ok(DecodedFooter {
        index_offset: read_u64(footer, 16)?,
        index_bytes: read_u64(footer, 24)?,
        index_sha256: footer[64..96].try_into().expect("index digest slice"),
        adc_sha256: footer[96..128].try_into().expect("ADC digest slice"),
    })
}

pub(super) fn encode_index(records: &[ChunkRecord]) -> Vec<u8> {
    let mut index = Vec::with_capacity(records.len() * INDEX_RECORD_BYTES);
    for record in records {
        push_u64(&mut index, record.offset);
        push_u64(&mut index, record.stored_bytes);
        push_u32(&mut index, record.frame_count);
        push_u32(&mut index, 0);
        index.extend_from_slice(&record.raw_sha256);
    }
    index
}

pub(super) fn encode_footer(
    index_offset: u64,
    index_bytes: u64,
    header: &[u8],
    index_sha256: [u8; 32],
    adc_sha256: [u8; 32],
) -> [u8; FOOTER_BYTES] {
    let mut footer = Vec::with_capacity(FOOTER_BYTES);
    footer.extend_from_slice(FOOTER_MAGIC);
    push_u32(&mut footer, VERSION);
    push_u32(&mut footer, FOOTER_BYTES as u32);
    push_u64(&mut footer, index_offset);
    push_u64(&mut footer, index_bytes);
    footer.extend_from_slice(&sha256(header));
    footer.extend_from_slice(&index_sha256);
    footer.extend_from_slice(&adc_sha256);
    let digest = sha256(&footer);
    footer.extend_from_slice(&digest);
    footer.try_into().expect("footer size")
}

pub(super) fn parse_index(
    index: &[u8],
    header: &DecodedHeader,
    index_offset: u64,
) -> Result<Vec<ChunkRecord>, AdcArchiveFileError> {
    let expected_records = archive_chunk_count(header.frame_count, header.restart_frames)?;
    if index.len() != expected_records * INDEX_RECORD_BYTES {
        return Err(error("ADC archive v3 index record count is invalid."));
    }
    let mut records = Vec::with_capacity(expected_records);
    let mut expected_offset = header.header_bytes;
    let mut remaining_frames = header.frame_count;
    for record in index.as_chunks::<INDEX_RECORD_BYTES>().0 {
        let offset = read_u64(record, 0)?;
        let stored_bytes = read_u64(record, 8)?;
        let frame_count = read_u32(record, 16)?;
        let expected_frames = remaining_frames.min(u64::from(header.restart_frames)) as u32;
        let maximum_bytes = maximum_chunk_bytes(header, frame_count)?;
        if offset != expected_offset
            || frame_count != expected_frames
            || read_u32(record, 20)? != 0
            || stored_bytes == 0
            || stored_bytes > maximum_bytes
        {
            return Err(error(
                "ADC archive chunk index contains invalid payload bounds or frame counts.",
            ));
        }
        expected_offset = expected_offset
            .checked_add(stored_bytes)
            .ok_or_else(|| error("ADC archive payload offset overflows u64."))?;
        if expected_offset > index_offset {
            return Err(error("ADC archive payload extends into the chunk index."));
        }
        remaining_frames -= u64::from(frame_count);
        records.push(ChunkRecord {
            offset,
            stored_bytes,
            frame_count,
            raw_sha256: record[24..56].try_into().expect("chunk digest slice"),
        });
    }
    if expected_offset != index_offset || remaining_frames != 0 {
        return Err(error("ADC archive payload region is not contiguous."));
    }
    Ok(records)
}

pub(super) fn archive_chunk_count(
    frame_count: u64,
    restart_frames: u32,
) -> Result<usize, AdcArchiveFileError> {
    let count = frame_count.div_ceil(u64::from(restart_frames));
    usize::try_from(count).map_err(|_| error("ADC archive chunk count does not fit memory."))
}

pub(super) fn validate_header_capture(
    header: &DecodedHeader,
    capture: &CaptureRecord,
) -> Result<(), AdcArchiveFileError> {
    let frame_bytes = capture_frame_bytes(capture)?;
    let frame_count = capture.num_frames.expect("validated capture frame count");
    let expected_size = frame_bytes
        .checked_mul(frame_count)
        .ok_or_else(|| error("ADC archive capture size overflows u64."))?;
    if header.frame_bytes != frame_bytes
        || header.frame_count != frame_count
        || capture.expected_size_bytes != Some(expected_size)
    {
        return Err(error(
            "ADC archive Header and embedded capture metadata disagree.",
        ));
    }
    Ok(())
}

fn validate_codec_dimensions(
    frame_bytes: u64,
    block_samples: u32,
    restart_frames: u32,
) -> Result<(), AdcArchiveFileError> {
    let frame_bytes = usize::try_from(frame_bytes)
        .map_err(|_| error("ADC archive frame size does not fit memory."))?;
    maximum_adc_archive_chunk_bytes(frame_bytes, 1, block_samples as usize)
        .map_err(|value| error(value.to_string()))?;
    if restart_frames == 0 || restart_frames > MAX_RESTART_FRAMES {
        return Err(error("ADC archive restart_frames must be in [1, 64]."));
    }
    Ok(())
}

fn maximum_chunk_bytes(
    header: &DecodedHeader,
    frame_count: u32,
) -> Result<u64, AdcArchiveFileError> {
    let frame_bytes = usize::try_from(header.frame_bytes)
        .map_err(|_| error("ADC archive frame size does not fit memory."))?;
    let maximum = maximum_adc_archive_chunk_bytes(
        frame_bytes,
        frame_count as usize,
        header.block_samples as usize,
    )
    .map_err(|value| error(value.to_string()))?;
    u64::try_from(maximum).map_err(|_| error("ADC archive chunk size does not fit u64."))
}
