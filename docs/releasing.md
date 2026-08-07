# Releasing

Python and Rust artifacts use the same version from the Cargo workspace.

## Before release

1. Update the workspace version and `CHANGELOG.md`.
2. Run `uv lock` and `cargo update --workspace` only when dependency changes require it.
3. Run CI, `cargo publish -p mmwcore --dry-run --locked`, and a wheel installation smoke test.
4. Verify that the Git tag is `v<version>`.

## PyPI

The `release.yml` workflow builds separate wheels for ordinary CPython 3.10, 3.11, 3.12, and 3.13
on Linux, Windows, and macOS, plus an sdist. These are interpreter-specific wheels, not one `abi3`
wheel: every CPython version is built and smoke-tested independently. CPython 3.13 support refers
to the standard GIL-enabled interpreter; free-threaded 3.13 (`cp313t`) is not built or promised.

Configure the GitHub `pypi` environment as a PyPI trusted publisher. A pending trusted publisher
can authorize the first release.

## crates.io

crates.io trusted publishing can be enabled only after the first crate version exists. Publish the
first version manually with `cargo publish -p mmwcore --locked`, then configure this repository as
the crate's trusted publisher. Future workflow dispatches may set `publish_crate=true`.

Publishing is intentionally separate from ordinary CI.
