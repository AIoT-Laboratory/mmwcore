# Changelog

All notable changes to mmwcore are documented here.

## Unreleased

### Added

- Open integrity-checked versioned capture directories as lazy ADC readers and physical contracts.
- Decode finite `mmwcli.capture_stream.v1` records from caller-owned binary streams, keeping frames
  provisional until COMMIT and terminal EOF validate.
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
