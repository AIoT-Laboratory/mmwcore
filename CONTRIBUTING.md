# Contributing to mmwcore

Start with a focused issue or pull request describing the concrete problem, expected behavior,
and a minimal reproduction. Include mmwcore/Python/Rust versions, OS, array shapes/dtypes/axes,
and relevant capture specifications. Prefer synthetic inputs; do not upload private recordings.

## Implementation boundaries

Rust owns archive codecs and numerical kernels. Python validates public contracts, composes
processing stages, and carries metadata. Reuse existing helpers and keep casts explicit.
Acquisition, hardware control, training, and visualization belong to calling applications.

Preserve ADC bytes, archive round trips, frame timing, antenna geometry, calibration,
coordinates, Doppler sign, axes, and units. A refactor must preserve public behavior and errors.
For an intentional API change, explain migration and add it to the unreleased changelog.
The project is pre-1.0, but compatibility changes still need explicit release notes.

Numerical fixes need a regression that reproduces the problem independently of the new
implementation. Tracking results require frame-aligned independent truth to claim accuracy;
synthetic branch tests and development recordings establish narrower behavior only.
Keep TI code and its license notices separate from Apache-2.0 code.

## Validation

Set up the checkout and run the affected checks from [README](README.md#validation).
The complete gate is [CI](.github/workflows/ci.yml), including Linux/Windows tests, benchmark
smoke, wheel installation outside the checkout, and Rust tests extracted from the source archive.
Storage or DSP changes also need the benchmark smoke. Tests require no radar hardware.

Pull requests should explain what changes for users, why, which checks passed, and any limits.
Avoid broad cleanup mixed with numerical or format changes. Keep documentation concise and
link to contracts and tests rather than copying implementation details.

For distribution changes, follow the [release process](docs/releasing.md). A checkout passing
tests is insufficient evidence that an installed wheel or source archive works.
