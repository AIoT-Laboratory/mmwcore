# Changelog

All notable changes to mmwcore are documented here.

## Unreleased

### Added

- Add a fixed offline ADC evidence archive with Rust byte-shuffle/zlib coding, independent frame
  verification, strict structural opening, bounded random-window reads, and atomic publication.
- Add corpus acceptance reporting for total archive overhead, publication throughput, full
  verification throughput, and verified versus post-admission trusted window latency.

### Changed

- Read finalized open-ended mmwcli capture directories by deriving their positive actual frame
  count from complete ADC bytes, while keeping both live stream protocols finite.

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
