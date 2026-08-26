# Changelog

All notable changes to mmwcore are documented here.

## Unreleased

## [0.7.1] - 2026-08-26

### Changed

- Open completed ADC and multi-sensor captures without scanning every payload by default; use
  `verify_payload=True` or `verify_artifacts=True` for an explicit full SHA-256 replay.
- Keep verified Archive reads as the default while allowing `verify=False` as an explicit local
  fast path without a preceding full replay.
- Simplify Archive publication to one source pass, same-directory staging, no-overwrite publish,
  and one structural open of the published file.
- Allow unrelated files beside declared multi-sensor artifacts without weakening artifact sizes,
  index structure, frame geometry, or explicit digest verification.

### Removed

- Remove repeated input-revalidation APIs and per-window file identity polling from completed
  capture, timeline, and ADC Archive readers.

## [0.7.0] - 2026-08-25

### Changed

- Replace the development archive writer and reader with the incompatible
  `mmwcore.adc_archive.v3` format.
- Encode independently decodable four-frame groups with homologous-coordinate `int16` prediction,
  ZigZag mapping, adaptive 512-sample Rice blocks, and exact raw-block fallback.
- Index and verify frame groups so random reads retain a bounded temporal dependency.
- Validate the Rice codec on 14 complete real-ADC sources with exact sequential and random-window
  replay, retaining 47.92% of the raw payload and 30.44% fewer bytes than adaptive zlib.
- Batch Rice bitstream reads and writes, compute exact parameter costs in one residual pass, and
  reuse block buffers while preserving the v3 byte stream. On the fixed 14-source corpus, minimum
  encode and pack throughput improve by 163.79% and 138.46%, respectively.
- Accept the complete v3 container on clean revision `a3c272b` after exact replay of 14 sources,
  durable publication, full verification, and verified and trusted random-window reads.
- Add ordered fixed-length batch reads that open the archive once and decode each distinct touched
  chunk once per call, with low-level byte windows and high-level `RawADCFrame` access.
- Preserve archive file I/O categories and sources across PyO3 so callers receive standard
  `FileNotFoundError`, `PermissionError`, or `OSError` separately from invalid archive data.
- Isolate the private Rice bitstream machinery and remove complexity and typing suppressions from
  the multi-sensor readers without changing their protocol validation.

### Removed

- Remove the single-frame byte-shuffle/zlib codec API and the Rust `flate2` dependency.
- Remove the public `ADC_RICE_RESTART_FRAMES` constant; restart grouping is an archive-writer
  policy rather than part of the standalone chunk codec API.

## [0.6.0] - 2026-08-20

### Changed

- Replace ADC Archive v1 with a self-describing v2 container implemented in Rust.
- Embed and validate the complete `RadarCaptureSpec`; opening an archive no longer requires a
  sidecar contract.
- Move archive writing, parsing, indexing, hashing, random reads, and full verification behind the
  PyO3 boundary.

## [0.5.1] - 2026-08-14

### Changed

- Write ADC archives with one source pass and structural pre-publication validation; retain
  per-frame verified reads and explicit `verify_all()` without repeating full replay on write.

## [0.5.0] - 2026-08-13

### Changed

- Rename the public raw-ADC storage API to `ADCArchive`, `ADCArchiveFrameReader`,
  `open_adc_archive()`, `write_adc_archive()`, and `write_capture_adc_archive()`.
- Rename the native frame codec to `encode_adc_archive_frame()` and
  `decode_adc_archive_frame()` and use the Rust error type `AdcArchiveCodecError`.
- Publish ADC archives with the `.mmwa` suffix and the `mmwcore.adc_archive.v1` format identity.
- Create a GitHub Release with wheel and source-distribution assets from the release workflow.

### Removed

- Remove the former evidence-named archive API and `.mmwe` format without compatibility aliases.

## [0.4.0] - 2026-08-13

### Added

- Add a fixed offline ADC evidence archive with Rust byte-shuffle/zlib coding, independent frame
  verification, strict structural opening, bounded random-window reads, and atomic publication.
- Add corpus acceptance reporting for total archive overhead, publication throughput, full
  verification throughput, and verified versus post-admission trusted window latency.
- Add metadata-only readers for finalized mmwcli capture contracts, multi-sensor session manifests,
  and per-source timelines so archived ADC can retain its validated capture and timing context.

### Changed

- Admit the fixed evidence archive v1 for finalized ADC files after a clean 14-source corpus passes
  exact replay, verified atomic publication, full verification, random-window, and scene-stability
  gates.
- Raise the Rust MSRV from 1.85 to 1.97 and validate CI and release builds against Rust 1.97.0.
- Read finalized open-ended mmwcli capture directories by deriving their positive actual frame
  count from complete ADC bytes, while keeping both live stream protocols finite.
- Revalidate opened archive, capture-metadata, and timeline inputs before publishing derived
  artifacts or starting long processing.

## [0.3.0] - 2026-08-08

### Added

- Add runnable capture-directory, raw-frame, explicit xWR18xx, radar-stream, and multi-sensor
  offline/live examples.
- Open integrity-checked versioned capture directories as lazy ADC readers and physical contracts.
- Decode finite `mmwcli.capture_stream.v1` records from caller-owned binary streams, keeping frames
  provisional until COMMIT and terminal EOF validate.
- Bind `open_capture` and the pull-driven `open_capture_stream` facade to an exact contract-derived
  Range-Doppler recipe or preset, with matching stdout producers from both mmwcli capture routes.
- Expose a frozen raw-capture descriptor and bind mmwcli directory/stream v1 decoding to its closed
  hardware, config-format, lane, byte-order, and layout tuple.
- Accept closed xWR16xx, xWR18xx, and xWR68xx mmwcli descriptors with explicit family-bound RF and
  transmitter validation, without inferring board geometry or a processing preset.
- Open published multi-sensor sessions, nested radar captures, and lazy causal training pairs with
  integrity-bound indices and conservative clock mappings.
- Decode aggregate live radar/camera streams, map RADAR_START and `delivery_observed` camera items
  onto one conservative host-relative time axis, and bind accepted results to source outcomes plus
  global COMMIT and EOF.
- Decode explicit TI complex16 two-lane and four-lane capture layouts.
- Add source-backed antenna geometries for XWR1642, standard XWR1843 EVM, IWR6843ISK,
  IWR6843 AOP, and AWR1843 AOP.
- Add a reproducible synthetic IWR6843 pipeline benchmark with versioned JSON output.

### Changed

- Support CPython 3.12–3.14 and use Python 3.12 language and typing features directly.
- Treat DCA1000 packet numbers as wrapping u32 and byte counters as wrapping u48.
- Require explicit trusted packet/frame origins; exact assembly rejects missing, duplicate,
  cross-frame, inexact-payload, and trailing packets instead of filling or truncating.
- Validate physical scalars, shapes, indices, thresholds, and public native integer domains before
  computation.

### Removed

- Remove the package command line and its preprocessing/export paths.
- Remove hardware configuration rendering.
- Remove live serial control, DCA control, UDP packet-source, and frame-reader APIs.
- Remove synchronized acquisition, radar-only capture sessions, legacy JSONL artifact
  writers/manifests, and the `mmwcore.session` package.
- Remove the permissive TI CLI shape-summary parser; finite capture contracts use strict parsing.
- Remove compatibility bridges and fallbacks for Python versions older than 3.12.

## [0.2.2] - 2026-08-06

- Fix Linux wheel builds by selecting CPython 3.12 inside the manylinux container.
- Validate release wheels against PyPI compatibility requirements.
- Exercise the release-equivalent Linux wheel build in ordinary CI.

## [0.2.1] - 2026-08-05

- Restore strict Clippy compatibility with the declared Rust 1.85 MSRV.
- Make native Cartesian test indexing portable across supported Python platforms.

## [0.2.0] - 2026-08-05

- Publish the Rust radar core, PyO3 bindings, and typed Python API as standalone packages.
- Add independent crates.io and PyPI packaging.
- Add public CI and controlled release workflows.
