# Contributing

Changes must preserve explicit axes, units, coordinate frames, counter origins, and radar capture
semantics. Numerical changes require focused tests and a physical derivation, source reference, or
redistributable validation vector.

## Compatibility policy

mmwcore targets the latest three stable CPython releases. The current matrix is 3.12–3.14 and the
current floor is Python 3.12. When the rolling matrix advances, update the declared floor and remove
dropped-version code; do not add compatibility shims or fallback implementations for older Python.

Rust 1.85 is the current minimum supported Rust version. New Rust features must compile on that
toolchain unless the declared minimum is intentionally raised in the same change.

## Boundary rules

- Start from caller-owned captured or live bytes and explicit physical contracts.
- Keep acquisition lifecycle, producer processes, hardware control, applications, experiments, and
  models outside mmwcore.
- Do not infer tensor axes, frame phase, byte layout, antenna geometry, or metric calibration.
- Do not add Python fallbacks for Rust-owned computation.
- Reject lossy integer casts, non-finite physical values, incomplete frames, and ambiguous shapes.
- Keep public examples deterministic and free of repository-private data.

## Checks

Run before opening a pull request:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uv sync --extra dev --locked
uv run --no-sync ruff format --check python tests benchmarks examples
uv run --no-sync ruff check python tests benchmarks examples
uv run --no-sync pyright
uv run --no-sync pytest --cov=mmwcore -q
uv run --no-sync python benchmarks/pipeline.py --warmups 0 --samples 1 --stream-frames 2
```

Benchmark smoke runs check that the maintained workload executes and emits its versioned schema.
They are not performance gates. Review [docs/benchmarking.md](docs/benchmarking.md) before comparing
measurements.

Keep commits small and independently verifiable. Separate API removal, numerical behavior, build
metadata, and documentation when they can be reviewed on their own.
