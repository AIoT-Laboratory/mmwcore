# Contributing

Changes must preserve explicit axes, units, coordinate frames, counter origins, and radar capture
semantics. Numerical changes require focused tests and a physical derivation, source reference, or
redistributable validation vector.

## Runtime

mmwcore uses CPython 3.12 throughout the local workspace and CI.

Rust 1.97 is the current minimum supported Rust version. New Rust features must compile on that
toolchain unless the declared minimum is intentionally raised in the same change.

## Boundary rules

- Start from caller-owned completed captures or archives and explicit physical contracts.
- Keep acquisition lifecycle, producer processes, hardware control, applications, experiments, and
  models outside mmwcore.
- Do not infer tensor axes, frame phase, byte layout, antenna geometry, or metric calibration.
- Reject lossy integer casts, non-finite physical values, incomplete frames, and ambiguous shapes.
- Keep public examples deterministic and free of repository-private data.

## Checks

Run before handing off a change:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uv sync --python 3.12 --extra dev --locked
uv run --python 3.12 ruff format --check python tests benchmarks examples
uv run --python 3.12 ruff check python tests benchmarks examples
uv run --python 3.12 pyright
uv run --python 3.12 pytest -q
uv run --python 3.12 python benchmarks/pipeline.py --warmups 0 --samples 1 --stream-frames 2
```

Benchmark smoke runs check that the maintained workload executes and emits its versioned schema.
They are not performance gates. Review [docs/benchmarking.md](docs/benchmarking.md) before comparing
measurements.

Keep commits small and independently verifiable. Separate API removal, numerical behavior, build
metadata, and documentation when they can be reviewed on their own.
