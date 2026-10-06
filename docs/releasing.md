# Releasing mmwcore

## Published versus development

GitHub `main` documents the development API. PyPI, crates.io, and GitHub release assets describe
the source at their version tag. The latest published Python version is 0.7.1. The current
0.8.0 candidate is not published; its compatibility and license changes are listed in
[CHANGELOG](../CHANGELOG.md) and the [candidate notes](releases/0.8.0.md).

The current workflow is a **validation gate**, not an automated publisher. It supports pushes,
pull requests, and manual CI runs. It builds a wheel from the source distribution, checks an
isolated installation, and runs the extracted source's Rust tests on Linux and Windows.
The old release workflow used removed APIs and Python versions outside the current contract;
do not run it from an old tag to publish current code.

## Preparing a release

1. Choose a new version and update `workspace.package.version`, internal exact-version
   dependencies, Cargo.lock, and uv.lock together. Do not overwrite a published version or
   reuse its tag. Record API migration, Python/platform support, license changes, and known
   limitations in a versioned changelog entry.
2. Update README installation/version statements and package metadata to match that release.
   Keep the GitHub About description consistent with the standalone compression/DSP/tracking
   library scope. Public examples and links must not depend on private application repositories.
3. Run the complete CI gate on the exact candidate commit. Both operating systems must pass.
   Local Windows success cannot substitute for the Linux numerical checks.
4. Build candidate artifacts in a fresh output directory. For each target, build from the sdist
   and run `tools/check_distribution.py` with one matching wheel and sdist. Check licenses,
   type stubs, metadata/version, native imports, archive round trips, and tracking behavior.
5. Tag the verified commit, attach those artifacts to a GitHub draft release, and write release
   notes describing the version's actual behavior and migration. Verify the target commit and
   asset versions before publishing.

Local build and distribution verification:

```console
uv run --no-sync maturin build --release --locked --sdist --interpreter python --out dist
uv run --no-sync python tools/check_distribution.py dist
```

For PyPI, build portable wheels with Maturin's `--compatibility pypi` and an appropriate
manylinux build environment or Zig on Linux. A native CI Linux wheel is a validation artifact,
not necessarily a PyPI-compatible wheel. Follow the [Maturin distribution guide](https://www.maturin.rs/distribution.html).
Publish only tested interpreter/platform combinations; macOS is not currently in CI.

Publish the verified distributions using the project's configured PyPI credentials or trusted
publisher. The existing PyPI description changes only when a new distribution is uploaded;
changing GitHub README or About does not rewrite old package metadata. Publish the standalone
Apache-2.0 Rust crate separately after `cargo package -p mmwcore --locked` validation.
The Python binding and TI tracking crates have `publish = false`.

After publication, verify installation from the registry in a fresh environment and check the
rendered README, links, Requires-Python, license files, release notes, and supported wheels.
Fix description-only mistakes in GitHub notes; fixes to packaged metadata need a new version.
