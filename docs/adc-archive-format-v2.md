# Historical ADC Archive v2 Binary Format

This document specifies `mmwcore.adc_archive.v2`, normally stored as `.mmwa`. Rust writes,
opens, validates, and reads this format. Python does not reconstruct the container with a second
parser.

All integers are unsigned little-endian. SHA-256 values occupy 32 raw bytes. Offsets are absolute
from the start of the file. No padding or trailing bytes are allowed.

## Layout

```text
0                                                                         EOF
+----------------+----------------+------------------------+-----------+--------+
| fixed header   | capture JSON   | encoded frame payloads | index     | footer |
| 96 bytes       | M bytes        | variable               | 48*N      | 160    |
+----------------+----------------+------------------------+-----------+--------+
                 ^                ^                        ^           ^
                 96               header_bytes             index_offset EOF-160
```

The header is the 96-byte preamble followed immediately by the embedded metadata. Payloads and
index records appear in frame order.

## Fixed Header

| Offset | Size | Type | Field | Required value or meaning |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | ASCII `MMWADCA2` |
| 8 | 4 | `u32` | `version` | `2` |
| 12 | 4 | `u32` | `fixed_header_bytes` | `96` |
| 16 | 8 | `u64` | `header_bytes` | `96 + metadata_bytes` |
| 24 | 8 | `u64` | `metadata_bytes` | UTF-8 JSON length, in `[1, 1 MiB]` |
| 32 | 8 | `u64` | `frame_bytes` | Raw bytes per frame, positive, even, at most 64 MiB |
| 40 | 8 | `u64` | `frame_count` | Positive finalized frame count |
| 48 | 4 | `u32` | `index_record_bytes` | `48` |
| 52 | 4 | `u32` | `codec_id` | `1`: int16 byte shuffle plus zlib level 1 |
| 56 | 4 | `u32` | `metadata_format_id` | `1`: RadarCaptureSpec JSON |
| 60 | 4 | `u32` | `flags` | `0` |
| 64 | 32 | bytes | `capture_sha256` | SHA-256 of the exact embedded JSON bytes |

Unknown versions, sizes, codecs, metadata formats, flags, or magic values are rejected.

## Embedded Capture Metadata

Bytes `[96, header_bytes)` are one UTF-8 JSON object with schema
`mmwcore.radar_capture_spec.v1`. The object contains:

- the radar waveform profile and physical constants;
- ADC chirp, receiver, sample, and complex-layout dimensions;
- physical Tx order;
- frame periodicity;
- finalized frame count and expected logical byte count.

The Rust parser rejects missing and unknown fields, non-finite or invalid dimensions, duplicate Tx
identifiers, inconsistent profile/ADC dimensions, inconsistent frame counts, and an
`expected_size_bytes` value different from `frame_bytes * frame_count`. The header repeats the
minimal dimensions needed to bound parsing and requires exact agreement with the JSON.

This metadata is sufficient to reconstruct `RadarCaptureSpec` without a sidecar. It intentionally
does not claim packet coverage, antenna geometry, board orientation, calibration, labels, or data
provenance.

## Encoded Frames

For each raw little-endian `int16` frame:

1. rearrange `lo[0], hi[0], ...` into `lo[0..n] || hi[0..n]`;
2. encode the shuffled bytes as one zlib-wrapped DEFLATE stream at level 1;
3. append that stream without a per-frame header.

The index supplies each payload offset and stored length. Decoding must consume one complete zlib
stream, produce exactly `frame_bytes`, reject trailing compressed bytes, reverse the shuffle, and
optionally verify the decoded-frame digest. A stored payload is non-empty and no larger than:

```text
frame_bytes + floor(frame_bytes / 16) + 1024
```

Payload 0 starts at `header_bytes`; every later payload begins where its predecessor ends.

## Frame Index

The index contains `frame_count` records, each equivalent to `<QQ32s>`.

| Record offset | Size | Type | Field | Meaning |
|---:|---:|---|---|---|
| 0 | 8 | `u64` | `payload_offset` | Absolute encoded-payload offset |
| 8 | 8 | `u64` | `stored_bytes` | Exact encoded length |
| 16 | 32 | bytes | `raw_frame_sha256` | SHA-256 of the decoded raw frame |

```text
index_bytes = frame_count * 48
index[0].payload_offset = header_bytes
index[i + 1].payload_offset = index[i].payload_offset + index[i].stored_bytes
index[last].payload_offset + index[last].stored_bytes = index_offset
```

## Commit Footer

The final 160 bytes are equivalent to `<8sIIQQ32s32s32s32s>`.

| Offset | Size | Type | Field | Required value or meaning |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | ASCII `MMWACMT2` |
| 8 | 4 | `u32` | `version` | `2` |
| 12 | 4 | `u32` | `footer_bytes` | `160` |
| 16 | 8 | `u64` | `index_offset` | Absolute index start |
| 24 | 8 | `u64` | `index_bytes` | `frame_count * 48` |
| 32 | 32 | bytes | `header_sha256` | SHA-256 of fixed header plus capture JSON |
| 64 | 32 | bytes | `index_sha256` | SHA-256 of the complete index |
| 96 | 32 | bytes | `adc_sha256` | SHA-256 of all decoded frames concatenated |
| 128 | 32 | bytes | `footer_sha256` | SHA-256 of footer bytes `[0, 128)` |

`index_offset + index_bytes` must equal `file_size - 160`. The footer is the commit marker;
missing, displaced, truncated, or non-terminal footers are invalid.

## Verification

Structural open verifies fixed fields, metadata syntax and semantics, capture digest, header
digest, footer digest, index bounds and digest, contiguous payload offsets, and encoded-size
bounds. A normal frame read verifies each decoded frame digest. `verify_all()` streams through all
frames, verifies every frame digest and the logical ADC digest, then enables unverified reads on
that opened object until file identity changes.

These hashes provide corruption and mismatch detection, not origin authentication. Authenticity
requires a separately authenticated manifest or signature.

## Publication and Versioning

The writer validates the source size against the finalized embedded contract, reads the source once,
writes and flushes a same-directory temporary archive, structurally reopens it, and publishes with
atomic no-overwrite semantics. Version 2 has no extension negotiation. Readers reject v1 and unknown
formats; no compatibility parser is retained.
