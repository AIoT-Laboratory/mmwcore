# mmwcore Agent Guide

This file governs automated work in this repository. Public behavior belongs in README.md,
examples/, and docs/.

## Supported toolchains

- Support the latest three stable CPython releases only: currently 3.12-3.14.
- Rust 1.97 is the MSRV. The Python package embeds the Rust core through PyO3.
- Python and Rust packages share one workspace version. Keep Cargo.toml, Cargo.lock, the
  mmwcore-python exact dependency, version tests, and the dated changelog section aligned.

## Agent roles

- Use Sol with the installed `grill-me` (`grilling`) skill for protocol or algorithm design,
  theoretical analysis, quality assessment, code audits, and release acceptance.
- Use Terra for implementation, documentation updates, and command generation.
- Use Luna for document or artifact cleanup and routine test execution.

## Project boundary

- mmwcore is a research compute library, not a hostile-input service. Do not add authentication,
  adversarial path handling, arbitrary research-scale quotas, repeated input revalidation, or
  crash-durability work without an observed failure and explicit user approval. Binary bounds,
  lossless round trips, physical contracts, and scientific provenance remain strict.
- Start from caller-owned capture directories, files, archived packet bytes, or BinaryIO streams.
- mmwcore reads completed radar and multi-sensor captures and finite live radar/multi-sensor
  streams. It does not configure hardware or own acquisition.
- Never restore mmwcore.session, serial/DCA control, UDP packet sources, device discovery, process
  launching, sockets, or caller-resource closure.
- Physical layout, frame geometry, timing, antenna geometry, calibration, and processing recipes
  stay explicit. A radar family does not select a board geometry or DSP preset.
- Live records remain provisional until their COMMIT and physical EOF validate. Respect source
  outcomes through commit.accepts for aggregate streams.

## Rust and Python responsibilities

- crates/mmwcore owns deterministic parsing and compute kernels.
- crates/mmwcore-python owns the checked PyO3/NumPy boundary.
- python/mmwcore owns explicit contracts, caller-owned IO readers, composition, metrics, and
  optional plotting.
- Do not add Python fallbacks for Rust-owned computation or hide lossy casts, truncation,
  non-finite values, or ambiguous axes at the binding boundary.

## Examples and documentation

- Keep examples runnable from caller-supplied files or stdin; examples must not start hardware.
- Cover generic capture/raw input, non-xWR68 explicit geometry, radar live streams, multi-sensor
  offline training, and multi-sensor live inference.
- Add examples to README.md and examples/README.md and keep them in Ruff and Pyright gates.
- State integrity, provenance, timing, provisional, and geometry limits honestly.

## Efficient validation

- For a small change, run the narrowest relevant Ruff/Pyright/pytest or Cargo check once.
- Do not repeat whole-repository audits or expand into unrelated numerical edge-case work.
- Before a release, run the full Python and Rust gates once:

  cargo fmt --all --check
  cargo clippy --workspace --all-targets --locked -- -D warnings
  cargo test --workspace --locked
  cargo publish -p mmwcore --dry-run --locked
  uv sync --extra dev --locked
  uv run --no-sync ruff format --check python tests benchmarks examples
  uv run --no-sync ruff check python tests benchmarks examples
  uv run --no-sync pyright
  uv run --no-sync pytest --cov=mmwcore -q

- Release workflow changes must preserve wheel/sdist smoke tests and the already configured PyPI
  and crates.io trusted publishing paths.

## Git and releases

- Develop ordinary changes on dev. main is the release integration branch.
- Release tags are v<workspace-version> and must match a dated CHANGELOG.md section.
- Do not merge, push, tag, publish, rewrite history, or discard user changes without explicit
  authorization.
- Use focused commits with messages in type(scope): description form.
