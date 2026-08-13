# Evidence Storage Research

This document records the codec study and the acceptance path for the offline v1 evidence archive.

## Objective

Radar capture storage must preserve the complete acquired ADC evidence while supporting bounded
random access and efficient repeated training. Point clouds, detections, dense radar tensors, and
model tokens are derived views and cannot replace the captured ADC payload.

The working data model has three layers:

1. **Evidence**: exact ADC bytes, frame boundaries, packet coverage, timing, radar configuration,
   calibration identity, and integrity digests.
2. **Reversible representation**: chunking, byte-plane transforms, integer prediction, and lossless
   entropy coding. Full decoding must reproduce the evidence bytes exactly.
3. **Research views**: range, Doppler, angle, Cartesian tensors, point clouds, crops, reduced
   precision tensors, and model inputs. These are content-addressed caches that may be deleted and
   rebuilt from evidence plus an explicit processing recipe.

FFT output, magnitude conversion, quantization, clipping, filtering, and learned reconstruction
are not evidence-preserving operations.

## Hypotheses

- **H1 - axis-aware lossless coding**: reversible transforms that respect frame and sample layout
  reduce storage relative to direct general-purpose compression without changing a byte.
- **H2 - bounded chunks**: independently decodable frame chunks provide useful compression while
  retaining practical sequential and random-window throughput.
- **H3 - progressive views**: a small approximate base plus an exact residual may reduce routine
  training I/O; only the complete base and residual pair is evidence.
- **H4 - learned entropy models**: a learned probability model may improve lossless coding across
  held-out captures. Prediction errors may increase bitrate but must never alter decoded samples.

H3 and H4 are deferred until simple lossless controls establish a credible baseline.

## AI-assisted Research Data Plane

The intended workflow is broader than a compressor:

1. Acquisition publishes immutable evidence segments plus packet coverage, clocks, geometry,
   configuration, calibration identity, and an explicit commit outcome.
2. An admitted mmwcore codec stores each exact segment in independently verifiable chunks.
3. Processing recipes create content-addressed views on demand. Dense RT/RD/RA tensors,
   overlapping windows, point clouds, and model tokens are caches rather than duplicate truth.
4. Training reads only the requested frame windows and regenerates missing views from the exact
   evidence and recipe hash.

AI may optimize a lossless entropy model, identify low-quality or novel intervals, prioritize
derived caches, and select samples for annotation or adaptation. It must not synthesize missing
ADC bytes, silently repair packet loss, or decide that filtered point clouds are sufficient
evidence. Model weights used by a learned lossless codec become part of the decoder identity and
must be versioned with the chunk contract.

## Offline Baseline

`benchmarks/evidence_storage_cli.py` compares a closed set of controls requiring no new compression
dependency on caller-owned ADC files:

- `raw`
- `zlib`
- `shuffle-zlib`: separate the low and high byte planes of each little-endian `int16` word before
  compression
- `frame-delta-shuffle-zlib`: apply modulo-`2^16` prediction at equal sample positions across
  frames, then byte-plane shuffle and compression
- `adaptive-shuffle-zlib`: encode both shuffle candidates per chunk, retain the shorter one, and
  store a one-byte reversible transform tag

The zlib controls use level 1 by default. The selected level and the compile/runtime zlib versions
are recorded in the report; change the level explicitly rather than relying on environment state.

The runner creates temporary chunk payloads and removes them after each case. It does not publish
an archive and does not modify its inputs. Each chunk is decoded and compared with the source
bytes. Random windows are also compared with direct source reads.

Random-window results separate two scopes over the same generated windows. `trusted` measures
seek, read, decode, and slice after the archive has passed sequential admission; it omits a
repeated SHA-256 calculation on every chunk read. `verified` additionally hashes every decoded
chunk. Both modes compare their output with direct source bytes outside the timed interval, and
sequential replay always verifies every chunk digest. The report records the mode order, so the
two numbers describe different read policies rather than an interchangeable headline latency.

Pack throughput measures buffered source read, hashing, transform, and payload write. It does not
issue `fsync` and is not a durable-acquisition throughput claim. An acquisition integration must
separately measure persistence, commit publication, recovery from interruption, and backpressure.

Smoke-test two frames of one capture:

```console
uv run --no-sync python -m benchmarks.evidence_storage_cli CAPTURE.bin \
  --frame-bytes 1572864 --case raw:1 --case adaptive-shuffle-zlib:2 --max-frames 2 \
  --random-windows 2 --window-frames 1 --output evidence-smoke.json
```

Run a corpus benchmark by passing a directory. The default discovery name is
`adc_data_Raw_0.bin`; use `--filename` for another acquisition convention.

```console
uv run --no-sync python -m benchmarks.evidence_storage_cli CAPTURE_ROOT \
  --frame-bytes 1572864 \
  --case raw:1 --case shuffle-zlib:1 \
  --case shuffle-zlib:4 --case adaptive-shuffle-zlib:4 \
  --zlib-level 1 --random-windows 128 --window-frames 4 \
  --output evidence-corpus.json
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
four-frame reads. It is development evidence from a dirty revision, not a publication result.

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
  P95 was `132 ms`; it still passed the `30 MiB/s` development gate.
- All 32 empty-scene chunks selected shuffle. All 32 standing and all 32 waving chunks selected
  frame-delta. The result is systematic across each selected interval rather than driven by a few
  outliers.

The candidates now have different roles. `shuffle-zlib:1` is the low-latency training-read
baseline. `adaptive-shuffle-zlib:4` is the cold-evidence archive candidate. On the current nine
takes, adaptive coding projects to about `5.43 GiB` instead of `7.91 GiB` raw; it saves only about
`0.17 GiB` beyond simple shuffle. Therefore compression alone is not the new data paradigm. The
larger system gain must come from deleting regenerable dense tensors and overlapping windows, then
materializing content-addressed research views on demand.

The pilot reports were produced from a dirty development revision and cannot admit a format by
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
the development gate before durable-write costs. It remains a cold-archive and learned-entropy
control, not the first format.

The corpus passes the offline codec gate for one-frame `shuffle-zlib` at level 1. This codec-only
run does not by itself admit a stable archive format: it excluded manifest/index overhead, `fsync`,
atomic publication, interruption recovery, malformed-index attacks, decompression bounds, and a
Rust implementation. Those properties belong to the format implementation and its acceptance
tests.

Top-level case summaries use total-byte-weighted storage ratio, the minimum throughput across
sources, separate maximum trusted and verified random-read P95 values, and an all-source
verification flag. They deliberately do not average away a bad capture.

## Required Evidence

Every candidate must report:

- stored payload ratio and reduction;
- encode and sequential-decode MiB/s;
- trusted and per-read-verified random-window p50 and p95 latency;
- windows/s and chunks decoded per window;
- selected source range and logical source SHA-256;
- successful byte-exact sequential and random-window replay.

The report also records all benchmark parameters, random seed, environment, repository revision,
and whether the worktree was dirty. Evidence intended for comparison should come from a clean,
committed revision.

The benchmark excludes archive index and manifest overhead. A format proposal must measure those
costs separately. Pure encode/decode throughput is not applicable to the `raw` no-codec control;
its end-to-end pack and sequential replay throughput remain available.

## Admission Gate

A storage format may enter `mmwcore` only after the offline corpus shows all of the following:

- exact byte reconstruction for every tested chunk and window;
- sustained encode throughput above twice the target acquisition byte rate;
- independently verifiable chunks with bounded dependency length;
- useful sequential and random-window throughput for training;
- stable behavior across empty, static, and moving scenes;
- lower total retained storage after regenerable DSP and overlapping-window copies are removed.

Reject a candidate if it changes any evidence byte, hides missing data, requires decoding the whole
capture for a random frame, only works in one scene, or adds complexity without consistently
beating the simple controls.

The first format candidate is deliberately fixed: one complete frame per independently decodable
chunk, little-endian `int16` byte-plane shuffle, zlib-wrapped DEFLATE level 1, and a digest of each
decoded frame. There is no codec selector, adaptive transform, frame delta, learned decoder,
progressive layer, or compatibility negotiation. The archive is an offline representation of a
completed ADC file; it does not replace acquisition publication until durable-write and recovery
evidence exists.

The implementation exposes `write_evidence_archive()` and `open_evidence_archive()`. Its fixed
little-endian layout contains a 64-byte header, one encoded payload per frame, a 48-byte fixed index
record per frame, and a 160-byte commit footer at physical EOF. The header binds frame dimensions
and the caller-owned capture-contract SHA-256. Each index record binds its payload offset, encoded
length, and decoded-frame SHA-256. The self-digested footer binds the header, complete index, and
concatenated logical ADC SHA-256. Frame size and encoded payload length have explicit bounds.

For a declared `RadarCaptureSpec`, `write_adc_evidence_archive()` derives the frame size and
capture-contract digest from that contract and requires the SHA-256 of the original ADC file. The
matching `ADCEvidenceArchiveFrameReader.from_capture()` checks all three identities before exposing
random-access frames. This makes the archive a storage representation of one known ADC source, not a
replacement contract that infers layout or hardware metadata.

The writer uses a same-directory temporary file, flushes and `fsync`s the complete archive file,
reopens and fully verifies it, then atomically publishes it with no overwrite. POSIX publication
also `fsync`s the containing directory; Windows relies on the completed file flush plus atomic
hard-link publication. Structural open validates the complete header/index/footer chain. Ordinary
reads verify frame digests; trusted reads are available only after the same reader has completed
`verify_all()` and are revoked if the opened file identity, size, or modification time changes.

Run the implemented-format acceptance pass only after the codec corpus. Publish throughput covers
source read, per-frame Rust transform/compression, source and frame hashing, temporary-file
`fsync`, full pre-publication decode verification, and atomic publication. `full_verify` measures a
separate reopened full replay. Random windows report verified reads and trusted reads after that
full verification. Archive ratio includes header, index, footer, and every encoded payload.

```console
uv run --no-sync python -m benchmarks.evidence_archive_acceptance_cli CAPTURE_ROOT \
  --frame-bytes 1572864 --random-windows 128 --window-frames 4 \
  --scratch-dir D:/Shared --output D:/Shared/mmwcore-evidence-archive-v1.json
```

The command writes only temporary archives under `--scratch-dir`, removes each after its source is
measured, and leaves the ADC inputs untouched. This is a long I/O task and should be run manually
on the fixed corpus.

## Implemented Archive Acceptance

The fixed offline archive was admitted on 2026-08-13 using clean revision
`9864cca55b9517d3bb80f80f4c3449a46174eee5`. The acceptance run used the same 14 complete sources,
8,400 frames, and 13,212,057,600 logical bytes as the codec corpus. It covered empty scenes,
sitting, standing, walking, and waving across both retained capture-directory layouts. Every source
passed complete replay and 128 direct-source comparisons of randomly selected four-frame windows.

| Measurement | Corpus result |
|---|---:|
| Raw evidence | 12.3047 GiB |
| Complete archive | 8.0590 GiB |
| Total archive ratio | 0.6550 |
| Storage reduction | 4.2457 GiB / 34.50% |
| Header, index, and footer | 406,336 bytes / 0.00308% |
| Minimum verified atomic-publication throughput | 72.3 MiB/s |
| Minimum reopened full-verification throughput | 212.2 MiB/s |
| Worst verified four-frame random-read P95 | 32.93 ms |
| Worst trusted four-frame random-read P95 after full verification | 30.36 ms |
| Exact source round trips | 14 / 14 |

Per-source archive ratios remained between `0.6426` and `0.6633`. Verified atomic publication remained
between `72.3` and `93.1 MiB/s`, and reopened full verification remained between `212.2` and
`235.1 MiB/s`. No scene or motion class was an outlier. The minimum publication result is `2.41x`
the established `30 MiB/s` development gate and includes source hashing, Rust encoding, payload and
metadata writes, file `fsync`, source rehashing, complete decode verification, and atomic
publication.

This result admits evidence archive v1 only as an offline representation of finalized ADC files.
It does not admit inline acquisition encoding or make claims about capture backpressure, power-loss
durability, or interrupted-device operation. The external machine-readable report intentionally
remains outside the public repository because it contains workstation paths; this document records
the aggregate acceptance evidence and committed implementation revision.

## Repository Boundary

- Acquisition software owns hardware control, packet coverage, clock evidence, atomic publication,
  and capture backpressure.
- `mmwcore` owns an admitted lossless chunk contract, deterministic codecs, integrity checks, and
  frame/window reads.
- Research platforms own labels, splits, processing recipes, disposable caches, and training
  scheduling.

The admitted implementation remains offline. Acquisition must continue writing its current exact
ADC payload until inline encoding, backpressure, and interruption recovery are independently
implemented and measured. The archive reader is intended for post-capture processing, replay, and
training input; it does not silently change the source identity used by those workflows.
