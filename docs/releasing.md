# Releasing

Python and Rust artifacts use the same version from the Cargo workspace.

## Before release

1. Update the workspace version and `CHANGELOG.md`.
2. Run `uv lock` and `cargo update --workspace` only when dependency changes require it.
3. Run CI, `cargo publish -p mmwcore --dry-run --locked`, and a wheel installation smoke test.
4. Verify that the Git tag is `v<version>`.

## PyPI

The `release.yml` workflow builds Python 3.12 wheels for Linux, Windows, and macOS plus an sdist.
Configure the GitHub `pypi` environment as a PyPI trusted publisher. A pending trusted publisher
can authorize the first release.

## crates.io

crates.io trusted publishing can be enabled only after the first crate version exists. Publish the
first version manually with `cargo publish -p mmwcore --locked`, then configure this repository as
the crate's trusted publisher. Future workflow dispatches may set `publish_crate=true`.

Publishing is intentionally separate from ordinary CI.
