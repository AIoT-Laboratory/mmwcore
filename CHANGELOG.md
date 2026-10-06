# Changelog

## Unreleased

This section describes `main`, not the published 0.7.1 artifacts. The next release needs a new
version; it must not reuse or replace 0.7.1.

### Compatibility

- Development now targets CPython 3.12 only; 0.7.1 supported CPython 3.12–3.14.
- The current Python package embeds the complete Rust TI GTRACK 3DA implementation. Its
  distribution license includes TI's device-only terms alongside Apache-2.0.
- Legacy `GTrack2D`/six-state `GTrack3D`, `PointTracker2D`, `ClusterTracker2D`, their adapters,
  ADC runners, and dedicated 2D benchmark are removed without aliases. Use `TiGTrack3D` or
  `ScatterBodyTracker`; their contracts differ. See [tracking](docs/ti-gtrack.md) and
  [comparison semantics](docs/architecture.md#ti--multiscale-tracking-comparison).
- Capture and verified-take APIs now follow the finite `mmwcli.take.v3` and
  `openmmw.take.v4` contracts; see [architecture](docs/architecture.md#data-boundary).

### Quality and maintenance

- ADC storage is organized as `io::adc_compression::{codec, container}`. Explicit compression
  names distinguish frame-group payloads, standardized files, and ADCFrame readers. Old archive
  imports remain compatibility aliases; the .mmwa v3 wire format and raw ADC bytes are unchanged.
- `decompress_adc_file` restores raw ADC in bounded groups, verifies chunk and full-stream
  digests before publishing, refuses overwrite, and returns the embedded capture contract.

- Rust implementation is grouped into storage (`io`), processing (`dsp`), and tracking
  domains while retaining existing import paths. Python binding registration and boundary
  contracts are separated.
- The full multiscale `ScatterBodyTracker.step_points` pipeline now runs in the Apache-2.0
  Rust crate. Python keeps the existing input/output and diagnostic state contracts; the
  earlier Python ablation classes remain available for comparisons. A frozen pre-migration
  Python oracle checks frame outputs and internal histories across configuration variants.

- FFT configurations retain physical resolution/bin-spacing distinctions and validate
  precomputed range-Doppler provenance.
- Raw ADC and archive readers share index validation, preserving accepted integer-like
  indices, errors, bounds, and frame ordering.
- TI oracle checks retain the existing floating-point tolerances, compare integral reference
  values exactly, and reject truncated frame sequences.
- TI GTRACK rounds inverse-tangent results through f64 while retaining f32 input/output,
  reducing host `atanf` differences without changing lookup-table trigonometry or test tolerances.
- CI checks independent wheel installation and runs Rust tests from the extracted source
  distribution. Its distribution check is also runnable locally.
- Public documentation distinguishes published releases, current development, supported
  environments, component licensing, and contribution/release requirements.

## 0.7.1 — 2026-08-26

Historical release: [artifacts](https://github.com/AIoT-Laboratory/mmwcore/releases/tag/v0.7.1),
[source and documentation](https://github.com/AIoT-Laboratory/mmwcore/tree/v0.7.1), and
[changes from 0.7.0](https://github.com/AIoT-Laboratory/mmwcore/compare/v0.7.0...v0.7.1).
These artifacts predate the unreleased API and license changes above.
