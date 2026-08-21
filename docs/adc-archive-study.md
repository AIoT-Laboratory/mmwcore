# ADC Archive Study

This document records the codec study, corpus acceptance for historical v1 and published v2, and
the accepted v3 Rice format scheduled for the next release. Historical measurements remain
evidence for their admitted revisions; they are not relabeled as v3 evidence.

## Objective

Radar capture storage must preserve complete acquired ADC data while supporting bounded
random access and efficient repeated training. Point clouds, detections, dense radar tensors, and
model tokens are derived views and cannot replace the captured ADC payload.

The working data model has three layers:

1. **ADC source**: exact ADC bytes, frame boundaries, packet coverage, timing, radar configuration,
   calibration identity, and integrity digests.
2. **Reversible representation**: chunking, byte-plane transforms, integer prediction, and lossless
   entropy coding. Full decoding must reproduce the ADC bytes exactly.
3. **Research views**: range, Doppler, angle, Cartesian tensors, point clouds, crops, reduced
   precision tensors, and model inputs. These are content-addressed caches that may be deleted and
   rebuilt from the ADC source plus an explicit processing recipe.

FFT output, magnitude conversion, quantization, clipping, filtering, and learned reconstruction
are not lossless ADC storage operations.

## Hypotheses

- **H1 - axis-aware lossless coding**: reversible transforms that respect frame and sample layout
  reduce storage relative to direct general-purpose compression without changing a byte.
- **H2 - bounded chunks**: independently decodable frame chunks provide useful compression while
  retaining practical sequential and random-window throughput.
- **H3 - progressive views**: a small approximate base plus an exact residual may reduce routine
  training I/O; only the complete base and residual pair restores the source ADC bytes.
- **H4 - learned entropy models**: a learned probability model may improve lossless coding across
  held-out captures. Prediction errors may increase bitrate but must never alter decoded samples.

H3 and H4 are deferred until simple lossless controls establish a credible baseline.

## Research Workflow and AI Use

The intended workflow is broader than a compressor:

1. Acquisition publishes immutable ADC segments plus packet coverage, clocks, geometry,
   configuration, calibration identity, and an explicit commit outcome.
2. A validated mmwcore codec stores each exact segment in independently verifiable chunks.
3. Processing recipes create content-addressed views on demand. Dense RT/RD/RA tensors,
   overlapping windows, point clouds, and model tokens are caches rather than duplicate source data.
4. Training reads only the requested frame windows and regenerates missing views from the exact
   ADC source and recipe hash.

AI may optimize a lossless entropy model, identify low-quality or novel intervals, prioritize
derived caches, and select samples for annotation or adaptation. It must not synthesize missing
ADC bytes, silently repair packet loss, or decide that filtered point clouds replace the captured
ADC source. Model weights used by a learned lossless codec become part of the decoder identity and
must be versioned with the chunk contract.

## Offline Baseline

`benchmarks/adc_storage_benchmark_cli.py` compares a closed set of controls requiring no new compression
dependency on caller-owned ADC files:

- `raw`
- `zlib`
- `shuffle-zlib`: separate the low and high byte planes of each little-endian `int16` word before
  compression
- `frame-delta-shuffle-zlib`: apply modulo-`2^16` prediction at equal sample positions across
  frames, then byte-plane shuffle and compression
- `adaptive-shuffle-zlib`: encode both shuffle candidates per chunk, retain the shorter one, and
  store a one-byte reversible transform tag
- `frame-delta-rice`: call the Rust v3 codec for a bounded frame group, using homologous-coordinate
  prediction, ZigZag, adaptive Rice blocks, and exact raw-block fallback

The zlib controls use level 1 by default. The selected level and the compile/runtime zlib versions
are recorded in the report; change the level explicitly rather than relying on environment state.

The runner creates temporary chunk payloads and removes them after each case. It does not publish
an archive and does not modify its inputs. Each chunk is decoded and compared with the source
bytes. Random windows are also compared with direct source reads.

Random-window results separate two scopes over the same generated windows. `trusted` measures
seek, read, decode, and slice after the archive has passed sequential verification; it omits a
repeated SHA-256 calculation on every chunk read. `verified` additionally hashes every decoded
chunk. Both modes compare their output with direct source bytes outside the timed interval, and
sequential replay always verifies every chunk digest. The report records the mode order, so the
two numbers describe different read policies rather than an interchangeable headline latency.

Pack throughput measures buffered source read, hashing, transform, and payload write. It does not
issue `fsync` and is not a durable-acquisition throughput claim. An acquisition integration must
separately measure persistence, commit publication, recovery from interruption, and backpressure.

Smoke-test two frames of one capture:

```console
uv run --no-sync python -m benchmarks.adc_storage_benchmark_cli CAPTURE.bin \
  --frame-bytes 1572864 --case raw:1 --case adaptive-shuffle-zlib:2 --max-frames 2 \
  --random-windows 2 --window-frames 1 --output adc-storage-smoke.json
```

Run a corpus benchmark by passing a directory. The default discovery name is
`adc_data_Raw_0.bin`; use `--filename` for another acquisition convention.

```console
uv run --no-sync python -m benchmarks.adc_storage_benchmark_cli CAPTURE_ROOT \
  --frame-bytes 1572864 \
  --case raw:1 --case shuffle-zlib:1 \
  --case shuffle-zlib:4 --case adaptive-shuffle-zlib:4 \
  --zlib-level 1 --random-windows 128 --window-frames 4 \
  --output adc-storage-corpus.json
```

This is a long I/O benchmark. Run it on an idle workstation against a fixed corpus and filesystem.
Do not compare results across different cache states, storage devices, source selections, Python
versions, or revisions.

Use two stages rather than scanning a large matrix blindly:

1. **Pilot**: screen codecs and chunk sizes on fixed ranges from an empty scene, a mostly static
   person, and fast motion. Reject candidates that fail exact replay or the encode-throughput gate.
2. **Corpus**: run only the surviving one or two candidates on every retained take. Require stable
   results across scene classes before proposing a format.

The focused defaults are `raw:1`, `shuffle-zlib:1`, `shuffle-zlib:4`, and
`adaptive-shuffle-zlib:4`. Four frames match the current random training window; the matched
shuffle case separates transform gain from chunk-size gain. Direct zlib, fixed frame-delta
prediction, and longer chunks remain explicit controls, not defaults. The adaptive candidate is an
oracle baseline for future learned entropy models: a learned method must beat its storage/throughput
tradeoff without weakening exact replay or failure isolation.

## Development Pilot

The first pilot used 128 frames each from an empty scene, standing, and waving, with 64 random
four-frame reads. It is a development measurement from a dirty revision, not a publication result.

- Every tested case reproduced the selected ADC bytes exactly.
- Direct zlib expanded the payload to `1.01-1.02x`; it is rejected as a candidate.
- Byte-plane shuffle was stable near `0.70-0.71x` across all three scenes.
- Eight-frame delta plus shuffle reached `0.71x` for the empty scene, `0.68x` for standing, and
  `0.67x` for waving. Temporal prediction is useful but scene-dependent.
- Moving from 8 to 32 frames improved the aggregate ratio only from `0.69x` to `0.68x`, while the
  worst verified random-read P95 increased from `241 ms` to `1295 ms`; 16- and 32-frame chunks are
  rejected.
- A one-frame delta case is mathematically identical to shuffle and must not be benchmarked as a
  separate candidate.

The focused adaptive pilot then processed 128 frames per scene with matched four-frame controls:

- `shuffle-zlib:1` retained `70.73%` overall, with worst verified random-read P95 of `59 ms`.
- `shuffle-zlib:4` retained `70.72%` overall, confirming that chunk size alone did not improve
  compression; its worst verified random-read P95 increased to `104 ms`.
- `adaptive-shuffle-zlib:4` retained `68.59%` overall. Relative to matched `shuffle-zlib:4`, it
  reduced encoded bytes by `0%` for empty, `3.57%` for standing, and `5.41%` for waving.
- Adaptive encoding was `1.6-2.6x` slower than matched shuffle. Its minimum encode throughput was
  `41 MiB/s`, minimum end-to-end pack throughput was `38 MiB/s`, and worst verified random-read
  P95 was `132 ms`; it still exceeded the `30 MiB/s` development target.
- All 32 empty-scene chunks selected shuffle. All 32 standing and all 32 waving chunks selected
  frame-delta. The result is systematic across each selected interval rather than driven by a few
  outliers.

The candidates now have different roles. `shuffle-zlib:1` is the low-latency training-read
baseline. `adaptive-shuffle-zlib:4` is the cold-storage candidate. On the current nine
takes, adaptive coding projects to about `5.43 GiB` instead of `7.91 GiB` raw; it saves only about
`0.17 GiB` beyond simple shuffle. Therefore compression alone is not the new data paradigm. The
larger system gain must come from deleting regenerable dense tensors and overlapping windows, then
materializing content-addressed research views on demand.

The pilot reports were produced from a dirty development revision and cannot validate a format by
themselves.

## Full Corpus Result

The focused corpus run used clean revision
`34bdb9683e9796f8c5c50ab70054615613e3fd90`, 14 complete sources, 600 frames per source, and
`1,572,864` bytes per frame. It covered empty scenes, sitting, standing, walking, and waving across
the retained capture layouts. The logical corpus contained 8,400 frames and 13,212,057,600 bytes.
Each case replayed every source sequentially and compared 128 random four-frame windows with direct
source reads. Every chunk and window was byte-exact.

| Case | Payload ratio | Payload bytes | Minimum pack | Minimum replay | Worst verified random P95 |
|---|---:|---:|---:|---:|---:|
| `raw:1` | 1.0000 | 13,212,057,600 | 470.5 MiB/s | 446.9 MiB/s | 29.2 ms |
| `shuffle-zlib:1` | 0.7124 | 9,412,566,507 | 69.6 MiB/s | 111.9 MiB/s | 73.1 ms |
| `shuffle-zlib:4` | 0.7123 | 9,411,526,729 | 69.7 MiB/s | 117.4 MiB/s | 118.1 ms |
| `adaptive-shuffle-zlib:4` | 0.6905 | 9,122,669,306 | 35.5 MiB/s | 93.0 MiB/s | 152.9 ms |

Single-frame shuffle retained between `70.04%` and `73.06%` for every source and removed
3,799,491,093 bytes from the corpus. Four-frame shuffle saved only 1,039,778 additional bytes while
increasing the worst verified random-window P95 by about 62%; it is rejected. Adaptive coding saved
288,857,423 bytes beyond matched four-frame shuffle, but its worst pack throughput fell close to
the development target before durable-write costs. It remains a cold-archive and learned-entropy
control, not the first format.

The corpus satisfies the offline codec criteria for one-frame `shuffle-zlib` at level 1. This codec-only
run does not by itself validate a stable archive format: it excluded manifest/index overhead, `fsync`,
atomic publication, interruption recovery, malformed-index attacks, decompression bounds, and a
Rust implementation. Those properties belong to the format implementation and its acceptance
tests.

Top-level case summaries use total-byte-weighted storage ratio, the minimum throughput across
sources, separate maximum trusted and verified random-read P95 values, and an all-source
verification flag. They deliberately do not average away a bad capture.

## V3 Rice Format

The v3 format replaces zlib with a Rust-owned integer codec. Each independent
four-frame group uses the exact ADC schedule to align equal flattened `int16` coordinates across
frames. Its first frame is absolute; later frames are `i32` differences from the previous frame.
ZigZag residuals are divided into 512-sample blocks. Each block selects the minimum-bit Rice
parameter in `0..16` and falls back to its exact raw `int16` bytes unless Rice is strictly shorter.

The design has unit, small-artifact, and real-ADC corpus round-trip coverage. Both the codec and
complete container passed their fixed-corpus acceptance runs. Version 3 remains unpublished until
the next release is prepared.

The first Rust pilot used the first 16 frames from one empty-scene take, one standing take, and one
waving take. It compared the v3 codec with identical raw and historical zlib controls, replayed
every chunk, and compared eight random four-frame windows per source with direct source reads.
This was a dirty development revision and is evidence for direction only.

| Case | Payload ratio | Minimum pack | Minimum decode | Worst verified random P95 |
|---|---:|---:|---:|---:|
| `raw:1` | 1.0000 | 551.50 MiB/s | n/a | 13.71 ms |
| `shuffle-zlib:1` | 0.7125 | 120.24 MiB/s | 255.95 MiB/s | 35.40 ms |
| `adaptive-shuffle-zlib:4` | 0.6945 | 60.45 MiB/s | 182.67 MiB/s | 87.22 ms |
| `frame-delta-rice:4` | 0.4540 | 46.46 MiB/s | 174.92 MiB/s | 80.76 ms |

Rice retained `39.10%` for the empty scene, `45.11%` for standing, and `51.98%` for waving. Across
the selected 72 MiB, it used `34.63%` fewer payload bytes than adaptive zlib while retaining exact
round trips. Its minimum pack throughput remained above the current 30 MiB/s development gate.
The motion-dependent spread and limited frame range required the complete 14-source run below.

### Full codec corpus

The 2026-08-20 run used 14 complete sources, 600 frames per source, and `1,572,864` bytes per
frame. It covered empty scenes, sitting, standing, walking, and waving across both retained capture
layouts. The logical corpus contained 8,400 frames and 13,212,057,600 bytes. Every case replayed
every source exactly and matched 128 random four-frame windows per source, 1,792 in total, against
direct reads.

| Case | Payload ratio | Payload bytes | Minimum pack | Minimum decode | Minimum replay | Worst verified random P95 |
|---|---:|---:|---:|---:|---:|---:|
| `raw:1` | 1.0000 | 13,212,057,600 | 541.57 MiB/s | n/a | 664.85 MiB/s | 16.54 ms |
| `shuffle-zlib:1` | 0.7136 | 9,427,971,951 | 122.36 MiB/s | 271.43 MiB/s | 199.96 MiB/s | 37.07 ms |
| `adaptive-shuffle-zlib:4` | 0.6890 | 9,103,254,370 | 62.55 MiB/s | 207.69 MiB/s | 167.44 MiB/s | 81.93 ms |
| `frame-delta-rice:4` | 0.4792 | 6,331,864,426 | 46.80 MiB/s | 186.24 MiB/s | 150.61 MiB/s | 87.23 ms |

Rice removed 52.08% of the raw payload and used 2,771,389,944 fewer bytes, or 30.44% less, than
adaptive zlib. Its per-source ratio ranged from 0.3811 to 0.5610. It beat adaptive zlib on every
source; the relative saving ranged from 18.16% to 46.24%, so the aggregate result is not an
empty-scene artifact. Minimum encode throughput was 49.25 MiB/s and minimum end-to-end pack
throughput was 46.80 MiB/s, both above the 30 MiB/s development gate. The cost is a 10.0% lower
minimum sequential replay rate and a 5.30 ms higher worst verified random-window P95 than adaptive
zlib. This is an acceptable trade for 2.58 GiB less payload on the fixed corpus.

Random-window timing used a warm cache after sequential replay and a fixed four-frame request. It
does not establish cold-cache or concurrent training-loader performance. The corpus also uses one
IWR6843 ADC frame geometry, one device, and one person. Other chirp schedules, sampling rates,
devices, interference conditions, and long captures may change compression ratio; they must never
change exact reconstruction.

The report records base revision `d7012c2737baed24a5d6eb710de68ac88ff887a1` and a dirty
worktree. That limits publication provenance but does not reverse the engineering result: all 14
sources were exact, every source improved, and the throughput gate passed. Release evidence must
bind the unchanged implementation to a clean revision through the following container-level run.

Screen it on the fixed corpus against the admitted zlib controls with:

```console
uv run --no-sync python -m benchmarks.adc_storage_benchmark_cli CAPTURE_ROOT \
  --frame-bytes 1572864 \
  --case raw:1 --case shuffle-zlib:1 \
  --case adaptive-shuffle-zlib:4 --case frame-delta-rice:4 \
  --random-windows 128 --window-frames 4 \
  --scratch-dir D:/Shared --output D:/Shared/mmwcore-adc-rice-corpus.json
```

This is a long corpus task and must be run manually. Preserve it as the reproducibility command;
the completed development result above establishes the codec decision.

### Scalar hot-path optimization

The first post-corpus optimization preserved the v3 byte stream while replacing per-bit unary and
remainder loops with byte-batched operations, calculating all exact Rice parameter costs during
one residual traversal, and reusing residual and encoded-block buffers. Golden streams cover all
supported block sizes. A four-frame real-ADC anchor remained 3,267,266 bytes with SHA-256
`d69ddf71ce7ecbcc3a96995d57472d411fefa42eef80ddc481974550e8884cfe` before and after the change.

A same-process 64-frame development run improved codec-only encode throughput from 47.90 to
133.07 MiB/s and end-to-end pack throughput from 44.35 to 113.53 MiB/s. Decode improved from
192.45 to 216.25 MiB/s and sequential replay from 144.22 to 168.08 MiB/s. Payload ratio remained
exactly 0.543804 and every replay check passed. This short warm-cache result establishes the local
optimization direction; it does not replace the fixed-corpus or complete-container acceptance
runs.

The optimized implementation then repeated the complete 14-source codec corpus with identical
source paths, logical SHA-256 values, benchmark parameters, Python environment, and base revision.
Every source retained the same Rice payload bytes and passed exact sequential and random-window
replay. Total Rice payload remained 6,331,864,426 bytes with ratio 0.479249.

| Measurement | Before | Optimized | Change |
|---|---:|---:|---:|
| Minimum encode | 49.25 MiB/s | 129.92 MiB/s | +163.79% |
| Minimum pack | 46.80 MiB/s | 111.61 MiB/s | +138.46% |
| Minimum decode | 186.24 MiB/s | 213.75 MiB/s | +14.77% |
| Minimum sequential replay | 150.61 MiB/s | 167.81 MiB/s | +11.42% |
| Worst verified random P95 | 87.23 ms | 89.55 ms | +2.65% latency |

All 14 sources improved in encode, pack, and decode throughput. Paired encode improvement ranged
from 88.75% to 166.30%, and paired pack improvement ranged from 73.71% to 138.95%. Sequential
replay improved on 12 of 14 sources with a paired median gain of 4.78%. Random-window latency was
mixed and is classified as unchanged: these warm-cache measurements do not support a random-read
speedup claim.

Both corpus reports record `revision_dirty=true` at base revision
`d7012c2737baed24a5d6eb710de68ac88ff887a1`. They establish the engineering decision but do not
uniquely identify the native binary. A release result must be regenerated from the committed
candidate.

The crate forbids unsafe Rust, so the development implementation has no explicit AVX2/NEON
intrinsics. Compiler auto-vectorization is incidental and is not claimed as SIMD evidence. Add an
explicit SIMD path only through a maintainable safe abstraction with a scalar fallback and
separate throughput evidence; do not weaken the crate-wide unsafe prohibition for this codec.

## Required Measurements

Every candidate must report:

- stored payload ratio and reduction;
- encode and sequential-decode MiB/s;
- trusted and per-read-verified random-window p50 and p95 latency;
- windows/s and chunks decoded per window;
- selected source range and logical source SHA-256;
- successful byte-exact sequential and random-window replay.

The report also records all benchmark parameters, random seed, environment, repository revision,
and whether the worktree was dirty. Results intended for comparison should come from a clean,
committed revision.

The benchmark excludes archive index and manifest overhead. A format proposal must measure those
costs separately. Pure encode/decode throughput is not applicable to the `raw` no-codec control;
its end-to-end pack and sequential replay throughput remain available.

## Acceptance Criteria

A storage format may enter `mmwcore` only after the offline corpus shows all of the following:

- exact byte reconstruction for every tested chunk and window;
- sustained encode throughput above twice the target acquisition byte rate;
- independently verifiable chunks with bounded dependency length;
- useful sequential and random-window throughput for training;
- stable behavior across empty, static, and moving scenes;
- lower total retained storage after regenerable DSP and overlapping-window copies are removed.

Reject a candidate if it changes any ADC byte, hides missing data, requires decoding the whole
capture for a random frame, only works in one scene, or adds complexity without consistently
beating the simple controls.

The published v2 format was deliberately fixed: one complete frame per independently decodable
chunk, little-endian `int16` byte-plane shuffle, zlib-wrapped DEFLATE level 1, and a digest of each
decoded frame. There is no codec selector, adaptive transform, frame delta, learned decoder,
progressive layer, or compatibility negotiation. The archive is an offline representation of a
completed ADC file; it does not replace acquisition publication until durable-write and recovery
measurements exist.

The current development implementation exposes the same narrow `write_adc_archive()` and
`open_adc_archive()` surface with an intentionally incompatible v3 file contract. Version 3 has a
112-byte fixed header, canonical `RadarCaptureSpec` JSON, at most four frames per independently
decodable chunk, 512-sample adaptive Rice blocks, a 56-byte record per chunk, and a 160-byte commit
footer. The Header records every codec parameter needed for decoding. It does not replace packet
coverage, calibration, antenna geometry, or provenance records.

Rust owns the complete writer, parser, metadata validation, codec, index, digest checks, random
reads, full replay, and publication. Python only converts between the embedded JSON and
`RadarCaptureSpec`. The normative offsets and invariants are specified in
[ADC Archive v3 Binary Format](adc-archive-format.md). The published contract remains in
[Historical ADC Archive v2 Binary Format](adc-archive-format-v2.md).

The writer reads the source once while computing logical and per-chunk digests. It writes and
`fsync`s a same-directory temporary file, validates the header/index/footer chain, then publishes
without overwrite. Ordinary reads verify chunk digests. `verify_all()` streams a complete replay
and authorizes trusted reads on that object until its file identity changes.

Run the implemented-format acceptance pass only after the codec corpus. Publish throughput covers
the single source read, Rust transform/compression, source and chunk hashing,
temporary-file `fsync`, structural validation, and atomic publication. `full_verify` measures a
separate explicit full replay. Random windows report verified reads and trusted reads after that
full verification. Archive ratio includes header, index, footer, and every encoded payload.

```console
uv run --no-sync python -m benchmarks.adc_archive_acceptance_cli CAPTURE_ROOT \
  --capture-spec CAPTURE_SPEC.json --random-windows 128 --window-frames 4 \
  --scratch-dir D:/Shared --output D:/Shared/mmwcore-adc-archive-v3.json
```

The command writes only temporary archives under `--scratch-dir`, removes each after its source is
measured, and leaves the ADC inputs untouched. This is a long I/O task and should be run manually
on the fixed corpus.

## ADC Archive v3 Acceptance

The complete v3 container was validated on 2026-08-21 using clean revision
`a3c272b91174b2d5970d70fe30b34635a5ad63e2`. The run used the same 14 complete IWR6843 sources,
8,400 frames, and 13,212,057,600 logical bytes as the codec corpus. Every source passed complete
Rust replay and 128 direct-source comparisons of randomly selected four-frame windows.

| Measurement | Corpus result |
|---|---:|
| Raw ADC data | 12.3047 GiB |
| Complete v3 archive | 5.8971 GiB |
| Complete archive ratio | 0.479259 |
| Storage reduction | 6.4076 GiB / 52.07% |
| Rice payload | 6,331,864,426 bytes |
| Embedded capture metadata | 7,140 bytes total / 510 bytes per source |
| Index | 117,600 bytes total / 8,400 bytes per source |
| Total container overhead | 128,548 bytes / 0.000973% |
| Minimum atomic-publication throughput | 114.14 MiB/s |
| Minimum reopened full-verification throughput | 193.42 MiB/s |
| Worst verified four-frame random-read P95 | 74.08 ms |
| Worst trusted four-frame random-read P95 after full verification | 67.32 ms |
| Exact source round trips | 14 / 14 |

Per-source archive ratios ranged from 0.38113 to 0.56103. Atomic publication remained between
114.14 and 124.67 MiB/s, reopened full verification remained between 193.42 and 210.77 MiB/s, and
every source added exactly 9,182 container bytes. The report is bound to a clean revision and all
source logical SHA-256 values match the optimized codec corpus.

This result admits v3 as the next published ADC Archive format. Different ADC geometries,
cold-cache behavior, and concurrent training readers remain useful expansion measurements rather
than release blockers. The machine-readable report remains outside the repository because it
contains workstation paths.

## ADC Archive v2 Acceptance

The self-describing v2 archive was validated on 2026-08-20 using clean revision
`fb934d027969b19249d72a827da994dc302b2ab9`. The run used the same 14 complete IWR6843 sources,
8,400 frames, and 13,212,057,600 logical bytes as the historical archive acceptance. Every source
passed complete Rust replay and 128 direct-source comparisons of randomly selected four-frame
windows.

| Measurement | Corpus result |
|---|---:|
| Raw ADC data | 12.3047 GiB |
| Complete v2 archive | 8.0662 GiB |
| Total archive ratio | 0.65554 |
| Storage reduction | 4.2385 GiB / 34.45% |
| Embedded capture metadata | 7,140 bytes total / 510 bytes per source |
| Total container overhead | 413,924 bytes / 0.00313% |
| Minimum v2 atomic-publication throughput | 178.1 MiB/s |
| Minimum reopened full-verification throughput | 260.3 MiB/s |
| Worst verified four-frame random-read P95 | 26.73 ms |
| Worst trusted four-frame random-read P95 after full verification | 25.14 ms |
| Exact source round trips | 14 / 14 |

Per-source archive ratios remained between `0.64614` and `0.66332`. Atomic publication remained
between `178.1` and `199.8 MiB/s`, and reopened full verification remained between `260.3` and
`279.6 MiB/s`. Each archive embedded the same 510-byte canonical capture contract. Its per-source
container overhead was 29,566 bytes: the 96-byte fixed header, capture metadata, 600 48-byte index
records, and the 160-byte footer.

The v2 publication measurement covers one Rust source pass, logical and frame hashing, encoding,
payload write, temporary-file `fsync`, structural verification, and atomic publication. The
historical v1 admission path also repeated source hashing and complete decode verification before
publication. Their throughput figures therefore have different scopes and are not presented as a
direct performance comparison. The v2 run establishes its own acceptance result: self-contained
capture metadata adds negligible storage overhead while preserving exact replay and bounded
frame-window access on the tested corpus.

## Historical v1 Archive Acceptance

The fixed offline archive was validated on 2026-08-13 using clean revision
`9864cca55b9517d3bb80f80f4c3449a46174eee5`. The acceptance run used the same 14 complete sources,
8,400 frames, and 13,212,057,600 logical bytes as the codec corpus. It covered empty scenes,
sitting, standing, walking, and waving across both retained capture-directory layouts. Every source
passed complete replay and 128 direct-source comparisons of randomly selected four-frame windows.

| Measurement | Corpus result |
|---|---:|
| Raw ADC data | 12.3047 GiB |
| Complete archive | 8.0590 GiB |
| Total archive ratio | 0.6550 |
| Storage reduction | 4.2457 GiB / 34.50% |
| Header, index, and footer | 406,336 bytes / 0.00308% |
| Minimum admission-path atomic-publication throughput | 72.3 MiB/s |
| Minimum reopened full-verification throughput | 212.2 MiB/s |
| Worst verified four-frame random-read P95 | 32.93 ms |
| Worst trusted four-frame random-read P95 after full verification | 30.36 ms |
| Exact source round trips | 14 / 14 |

Per-source archive ratios remained between `0.6426` and `0.6633`. Admission-path publication
remained between `72.3` and `93.1 MiB/s`, and reopened full verification remained between `212.2`
and `235.1 MiB/s`. No scene or motion class was an outlier. These publication measurements belong
to the admitted revision and include source rehashing plus complete decode verification. Exact
replay having passed for every source, the maintained writer now removes those two redundant scans;
the historical throughput is not presented as a measurement of the optimized path.

This result validates the unchanged one-frame shuffle-zlib codec and the historical v1 container
only. Version 2 must pass the same 14-source exact-roundtrip and random-window procedure before its
container-level throughput and overhead are reported. Neither result validates inline acquisition,
capture backpressure, power-loss durability, or interrupted-device operation. The machine-readable
report remains outside the repository because it contains workstation paths.

## Repository Boundary

- Acquisition software owns hardware control, packet coverage, clock records, atomic publication,
  and capture backpressure.
- `mmwcore` owns a validated lossless chunk contract, deterministic codecs, integrity checks, and
  frame/window reads.
- Research platforms own labels, splits, processing recipes, disposable caches, and training
  scheduling.

The validated implementation remains offline. Acquisition must continue writing its current exact
ADC payload until inline encoding, backpressure, and interruption recovery are independently
implemented and measured. The archive reader is intended for post-capture processing, replay, and
training input; it does not silently change the source identity used by those workflows.
