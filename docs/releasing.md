# Releasing

Python and Rust artifacts use the same version from the Cargo workspace.

## Before release

1. Update the workspace version and `CHANGELOG.md`.
2. Run `uv lock` and `cargo update --workspace` only when dependency changes require it.
3. Run CI, `cargo publish -p mmwcore --dry-run --locked`, and a wheel installation smoke test.
4. Verify that the Git tag is `v<version>`.

## PyPI

The `release.yml` workflow builds interpreter-specific wheels for ordinary, GIL-enabled CPython
3.12–3.14 on Linux, Windows, and macOS, plus an sdist. Every wheel is built and smoke-tested
independently; `abi3` and free-threaded CPython 3.14 (`cp314t`) are not built or promised.

The GitHub `pypi` environment and PyPI trusted publisher have completed a release successfully.

## crates.io

The GitHub `crates-io` environment and crates.io trusted publisher have completed a release
successfully. Release workflow dispatches may set `publish_crate=true`.

Publishing is intentionally separate from ordinary CI.
