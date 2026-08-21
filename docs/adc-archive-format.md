# ADC Archive v3 Binary Format

This document specifies the accepted `mmwcore.adc_archive.v3` format scheduled for the next
release, normally stored as `.mmwa`. Rust writes, opens, validates, and reads the complete
container. Version 3 replaces byte-shuffle zlib with bounded homologous-frame prediction and
adaptive Rice coding.

All integers are unsigned little-endian. SHA-256 values occupy 32 raw bytes. Offsets are absolute
from the start of the file. No padding or trailing bytes are allowed outside the zero bit padding
defined for a Rice block.

## Layout

```text
0                                                                            EOF
+----------------+----------------+------------------------+-------------+--------+
| fixed header   | capture JSON   | encoded chunk payloads | chunk index | footer |
| 112 bytes      | M bytes        | variable               | 56*K        | 160    |
+----------------+----------------+------------------------+-------------+--------+
                 ^                ^                        ^             ^
                 112              header_bytes             index_offset  EOF-160
```

One chunk contains at most four radar frames. Chunks and index records appear in frame order.

## Fixed Header

| Offset | Size | Type | Field | Required value or meaning |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | ASCII `MMWADCA3` |
| 8 | 4 | `u32` | `version` | `3` |
| 12 | 4 | `u32` | `fixed_header_bytes` | `112` |
| 16 | 8 | `u64` | `header_bytes` | `112 + metadata_bytes` |
| 24 | 8 | `u64` | `metadata_bytes` | UTF-8 JSON length, in `[1, 1 MiB]` |
| 32 | 8 | `u64` | `frame_bytes` | Raw bytes per frame, positive, even, at most 64 MiB |
| 40 | 8 | `u64` | `frame_count` | Positive finalized frame count |
| 48 | 4 | `u32` | `index_record_bytes` | `56` |
| 52 | 4 | `u32` | `codec_id` | `2`: homologous-frame delta plus adaptive Rice |
| 56 | 4 | `u32` | `metadata_format_id` | `1`: RadarCaptureSpec JSON |
| 60 | 4 | `u32` | `flags` | `0` |
| 64 | 4 | `u32` | `block_samples` | `512` in the current writer; decoder accepts powers of two in `[256, 1024]` |
| 68 | 4 | `u32` | `restart_frames` | `4` in the current writer; decoder accepts `[1, 64]` |
| 72 | 8 | `u64` | `reserved` | `0` |
| 80 | 32 | bytes | `capture_sha256` | SHA-256 of the exact embedded JSON bytes |

Unknown versions, sizes, codecs, metadata formats, flags, or nonzero reserved values are rejected.

## Embedded Capture Metadata

Bytes `[112, header_bytes)` are one canonical UTF-8 JSON object with schema
`mmwcore.radar_capture_spec.v1`. It records the waveform, ADC dimensions and layout, physical Tx
order, frame periodicity, finalized frame count, and expected logical byte count. Header dimensions
must agree exactly with this object. Packet coverage, antenna geometry, mounting, calibration,
labels, and provenance remain outside this decoding contract.

## Homologous-Frame Transform

The exact ADC layout and capture schedule induce a stable flattened coordinate for every `int16`
word in a radar frame. In physical notation:

```text
x[f, c, r, q, n]

  f = radar-frame index inside the independently decodable chunk
  c = chirp / TDM transmit position
  r = receive channel
  q = I or Q component
  n = fast-time ADC sample
```

The first frame of each chunk is an absolute restart. Later frames use the previous frame at the
same capture coordinate:

```text
d[0, c, r, q, n] = x[0, c, r, q, n]
d[f, c, r, q, n] = x[f, c, r, q, n] - x[f-1, c, r, q, n]  (f >= 1)
```

The subtraction is evaluated in `i32`, so the complete `int16` difference domain
`[-65535, 65535]` is represented without modulo ambiguity. Decoding uses:

```text
x[0, ...] = d[0, ...]
x[f, ...] = d[f, ...] + x[f-1, ...]
```

Only one previous raw frame is required. Restarting every four frames bounds random-read work and
corruption propagation; it intentionally differs from an unbounded dependency chain beginning at
archive frame zero.

## Adaptive Rice Blocks

Each frame is partitioned independently into consecutive blocks of `block_samples` flattened
`int16` coordinates. The final block may be shorter. A block never spans two frames.

Residuals use ZigZag mapping:

```text
0 -> 0, -1 -> 1, +1 -> 2, -2 -> 3, +2 -> 4, ...
```

For each block, the encoder evaluates Rice parameters `k = 0..16` and minimizes:

```text
sum((value >> k) + 1 + k)
```

The block starts on a byte boundary with one tag byte:

| Tag | Payload |
|---:|---|
| `0..16` | Rice parameter `k`, followed by coded ZigZag residuals |
| `255` | Exact raw little-endian `int16` samples for this block |

For Rice values, quotient `q = value >> k` is encoded as `q` zero bits followed by one bit. The
`k`-bit remainder follows most-significant bit first. The block ends at the next byte boundary;
padding bits must be zero. The raw representation is selected unless the Rice payload is strictly
shorter than the raw block. Consequently, a chunk is bounded by:

```text
raw_chunk_bytes + number_of_blocks
```

The decoder rejects unknown tags, truncated unary/remainder codes, quotients outside the full
`int16` delta domain, nonzero padding, reconstructed values outside `int16`, and trailing bytes.

## Chunk Index

The index contains `K = ceil(frame_count / restart_frames)` records. Each record is equivalent to
`<QQII32s>` and occupies 56 bytes.

| Record offset | Size | Type | Field | Meaning |
|---:|---:|---|---|---|
| 0 | 8 | `u64` | `payload_offset` | Absolute encoded-chunk offset |
| 8 | 8 | `u64` | `stored_bytes` | Exact encoded length |
| 16 | 4 | `u32` | `frame_count` | Frames in this chunk; normally 4, shorter only for the final chunk |
| 20 | 4 | `u32` | `reserved` | `0` |
| 24 | 32 | bytes | `raw_chunk_sha256` | SHA-256 of all decoded raw frames in this chunk |

Payload zero starts at `header_bytes`; payloads are contiguous; the final payload ends at
`index_offset`. Frame ranges decode only intersecting chunks.

## Commit Footer

The final 160 bytes are equivalent to `<8sIIQQ32s32s32s32s>`.

| Offset | Size | Type | Field | Required value or meaning |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | ASCII `MMWACMT3` |
| 8 | 4 | `u32` | `version` | `3` |
| 12 | 4 | `u32` | `footer_bytes` | `160` |
| 16 | 8 | `u64` | `index_offset` | Absolute index start |
| 24 | 8 | `u64` | `index_bytes` | `K * 56` |
| 32 | 32 | bytes | `header_sha256` | SHA-256 of fixed header plus capture JSON |
| 64 | 32 | bytes | `index_sha256` | SHA-256 of the complete chunk index |
| 96 | 32 | bytes | `adc_sha256` | SHA-256 of all decoded frames concatenated |
| 128 | 32 | bytes | `footer_sha256` | SHA-256 of footer bytes `[0, 128)` |

The footer is the terminal commit marker. Structural open verifies format fields, metadata,
digests, index bounds, contiguous chunk offsets, frame coverage, and codec size bounds. Verified
reads additionally validate each decoded chunk digest. `verify_all()` replays every chunk and
validates the logical ADC digest before trusted reads are enabled on that object.

These hashes detect corruption and mismatched artifacts; they do not authenticate origin.

## Versioning

Version 3 is intentionally incompatible with the published v2 byte-shuffle/zlib format. The v3
reader rejects v2 rather than carrying a compatibility decoder. The normative historical v2
layout remains documented in [ADC Archive v2 Binary Format](adc-archive-format-v2.md).
