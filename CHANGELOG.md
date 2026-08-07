# Changelog

All notable changes to mmwcore are documented here.

## Unreleased

- Parse the supported xWR68xx legacy raw-capture subset into validated `RadarCaptureSpec` values
  while preserving the existing permissive ADC-shape parser.
- Open integrity-checked `mmwcli.capture_session.v1` directories as validated ADC readers and
  physical capture contracts.
- Iterate capture frames lazily and run explicit, contract-matched range-Doppler recipes, including
  IWR6843 active-Tx subsets.
- Add a reproducible synthetic IWR6843 pipeline benchmark runner with versioned JSON results.
- Support Python 3.10–3.13 with compatible typing and standard-library fallbacks.
- Add TI SDK source-backed, offline-tested antenna geometries for XWR1642, the standard XWR1843
  EVM, IWR6843 AOP, and AWR1843 AOP.
- Decode the TI Studio complex16, four-lane, frame-continuous channel-interleaved word layout used
  by xWR12xx and xWR14xx raw captures through explicit `group4_i_then_q` contracts.

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
