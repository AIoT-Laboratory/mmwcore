# mmwcore Agent Guide

mmwcore is the storage and compute layer for mmWave AI research.

## Boundary

- Read completed mmwcli takes, explicit raw ADC files, and `.mmwa` archives.
- Own ADC archive storage, deterministic DSP, classical tracking baselines, and quality benchmarks.
- Do not add hardware control, DCA packet reception, sockets, process launch, custom live protocols,
  experiment orchestration, model code, or web presentation.
- Do not add hostile-input security work, compatibility shims, generic plugin systems, or repeated
  validation without an observed research failure and explicit approval.
- Keep scientific contracts strict where they change results: byte-exact archives, ADC/frame
  geometry, timing, antenna geometry, calibration, axes, units, and lossless round trips.

## Responsibilities

- `crates/mmwcore`: deterministic Rust storage and compute kernels.
- `crates/mmwcore-python`: checked PyO3/NumPy boundary.
- `python/mmwcore`: explicit contracts, completed-file readers, DSP composition, and tracking.
- `benchmarks`: reproducible storage and DSP regression gates.

Do not add Python fallbacks for Rust-owned computation or hide lossy casts and ambiguous axes.

## Validation

Use the narrowest relevant offline check while editing. Before handoff run:

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uv run --no-sync ruff format --check python tests benchmarks examples
uv run --no-sync ruff check python tests benchmarks examples
uv run --no-sync pyright
uv run --no-sync pytest -q
```

Do not access hardware, install dependencies, publish, push, tag, or discard user changes.
