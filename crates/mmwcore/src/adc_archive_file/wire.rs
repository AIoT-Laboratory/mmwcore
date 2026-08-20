use super::contract::{CaptureRecord, capture_frame_bytes};
use super::{
    AdcArchiveFileError, CODEC_I16_SHUFFLE_ZLIB_1, FIXED_HEADER_BYTES, FOOTER_BYTES, FOOTER_MAGIC,
    FrameRecord, HEADER_MAGIC, INDEX_RECORD_BYTES, METADATA_RADAR_CAPTURE_JSON, VERSION, error,
    push_u32, push_u64, read_u32, read_u64, sha256,
};

#[derive(Debug)]
pub(super) struct DecodedHeader {
    pub(super) header_bytes: u64,
    pub(super) metadata_bytes: u64,
    pub(super) frame_bytes: u64,
    pub(super) frame_count: u64,
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
    capture_sha256: [u8; 32],
) -> Result<Vec<u8>, AdcArchiveFileError> {
    if metadata.is_empty() || metadata.len() as u64 > super::MAX_METADATA_BYTES {
        return Err(error(
            "ADC archive capture metadata size is outside v2 bounds.",
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
    push_u32(&mut header, CODEC_I16_SHUFFLE_ZLIB_1);
    push_u32(&mut header, METADATA_RADAR_CAPTURE_JSON);
    push_u32(&mut header, 0);
    header.extend_from_slice(&capture_sha256);
    debug_assert_eq!(header.len(), FIXED_HEADER_BYTES);
    header.extend_from_slice(metadata);
    Ok(header)
}

pub(super) fn decode_fixed_header(
    bytes: &[u8; FIXED_HEADER_BYTES],
) -> Result<DecodedHeader, AdcArchiveFileError> {
    if &bytes[0..8] != HEADER_MAGIC {
        return Err(error("ADC archive header is not mmwcore.adc_archive.v2."));
    }
    if read_u32(bytes, 8)? != VERSION
        || read_u32(bytes, 12)? != FIXED_HEADER_BYTES as u32
        || read_u32(bytes, 48)? != INDEX_RECORD_BYTES as u32
        || read_u32(bytes, 52)? != CODEC_I16_SHUFFLE_ZLIB_1
        || read_u32(bytes, 56)? != METADATA_RADAR_CAPTURE_JSON
        || read_u32(bytes, 60)? != 0
    {
        return Err(error("ADC archive v2 header contract is unsupported."));
    }
    let header_bytes = read_u64(bytes, 16)?;
    let metadata_bytes = read_u64(bytes, 24)?;
    let frame_bytes = read_u64(bytes, 32)?;
    let frame_count = read_u64(bytes, 40)?;
    if metadata_bytes == 0
        || metadata_bytes > super::MAX_METADATA_BYTES
        || header_bytes != FIXED_HEADER_BYTES as u64 + metadata_bytes
        || frame_bytes == 0
        || !frame_bytes.is_multiple_of(2)
        || frame_bytes > super::MAX_FRAME_BYTES
        || frame_count == 0
    {
        return Err(error("ADC archive v2 header dimensions are invalid."));
    }
    Ok(DecodedHeader {
        header_bytes,
        metadata_bytes,
        frame_bytes,
        frame_count,
        capture_sha256: bytes[64..96].try_into().expect("fixed digest slice"),
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
        return Err(error("ADC archive v2 commit footer is missing or invalid."));
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

pub(super) fn encode_index(records: &[FrameRecord]) -> Vec<u8> {
    let mut index = Vec::with_capacity(records.len() * INDEX_RECORD_BYTES);
    for record in records {
        push_u64(&mut index, record.offset);
        push_u64(&mut index, record.stored_bytes);
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
    header_bytes: u64,
    index_offset: u64,
    frame_bytes: u64,
) -> Result<Vec<FrameRecord>, AdcArchiveFileError> {
    let mut records = Vec::with_capacity(index.len() / INDEX_RECORD_BYTES);
    let mut expected_offset = header_bytes;
    for record in index.chunks_exact(INDEX_RECORD_BYTES) {
        let offset = read_u64(record, 0)?;
        let stored_bytes = read_u64(record, 8)?;
        if offset != expected_offset
            || stored_bytes == 0
            || stored_bytes > maximum_encoded_frame_bytes(frame_bytes)
        {
            return Err(error(
                "ADC archive frame index contains invalid payload bounds.",
            ));
        }
        expected_offset = expected_offset
            .checked_add(stored_bytes)
            .ok_or_else(|| error("ADC archive payload offset overflows u64."))?;
        if expected_offset > index_offset {
            return Err(error("ADC archive payload extends into the frame index."));
        }
        records.push(FrameRecord {
            offset,
            stored_bytes,
            raw_sha256: record[16..48].try_into().expect("frame digest slice"),
        });
    }
    if expected_offset != index_offset {
        return Err(error("ADC archive payload region is not contiguous."));
    }
    Ok(records)
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

fn maximum_encoded_frame_bytes(frame_bytes: u64) -> u64 {
    frame_bytes + frame_bytes / 16 + 1024
}
