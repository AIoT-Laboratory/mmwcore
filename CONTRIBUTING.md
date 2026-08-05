# Contributing

Changes must preserve explicit tensor axes, units, coordinate frames, and radar
capture semantics. Numerical changes require focused unit tests and a physical
or reference-vector rationale.

Run before opening a pull request:

~~~bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uv sync --extra dev --locked
uv run --no-sync ruff format --check python tests
uv run --no-sync ruff check python tests
uv run --no-sync pyright
uv run --no-sync pytest --cov=mmwcore -q
~~~

Do not add silent shape inference, uncalibrated Cartesian claims, or Python
fallback implementations for Rust-owned computation.
