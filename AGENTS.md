# mmwcore

This file records only stable repository contracts. Keep one-off task decisions out of it.

## Skills

Use a matching `~/.codex/skills` skill only when it helps the current task. Common choices are
`simplify` (`code-simplifier`), `grill-me`, `code-review` (when installed), and `prototype`; these
are examples, not an allowlist. Read the selected `SKILL.md` first, and never run skills
mechanically or turn one-off outputs into permanent constraints.

## Role

- Own byte-exact ADC archives, deterministic DSP, classical tracking, and quality benchmarks.
- Accept completed `mmwcli.take.v3` captures, verified `openmmw.take.v3` takes, raw ADC files, and
  `.mmwa` archives.
- Expose Rust kernels through checked Python contracts. Acquisition, process control, models,
  experiments, and Web belong to mmwcli or OpenMMW.
- Online inference may reuse the DSP on in-memory frames; mmwcore does not own stream lifecycle.

## Preserve

- ADC bytes and lossless archive round trips.
- Frame geometry, timing, antenna geometry, calibration, axes, shapes, and units.
- Explicit casts and contracts; Rust kernels remain the authoritative implementation.

## Checks

Run only checks affected by the change. The full gate is `.github/workflows/ci.yml`.

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uv run --no-sync ruff format --check python tests benchmarks examples
uv run --no-sync ruff check --no-cache python tests benchmarks examples
uv run --no-sync pyright
uv run --no-sync python -m pytest -p no:cacheprovider -q
```

For storage or DSP changes, also run the benchmark smoke command from CI.
