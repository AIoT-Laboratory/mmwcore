# Architecture

## Boundary

mmwcore begins at captured bytes and explicit offline contracts. It decodes and transforms radar
data without owning device configuration, transport lifecycle, experiment orchestration, models,
or application behavior.

```text
captured packet bytes / ADC files / versioned capture directory
                            |
                            v
              physical and integrity contracts
                            |
                            v
                 Rust parsing and kernels
                            |
                            v
              PyO3 boundary and Python APIs
                            |
                            v
       cubes / detections / point clouds / tracks
```

The Rust crate is independently usable. The Python wheel embeds it and adds offline readers,
composition, tracking utilities, and plotting.

## Layers

`crates/mmwcore` owns deterministic parsing, ADC decoding, FFT transforms, CFAR, calibration,
geometry, point-cloud projection, clustering primitives, assignment, and tracking kernels.

`crates/mmwcore-python` validates native boundary types and exposes Rust results as NumPy arrays.
It must not hide truncation, lossy casts, non-finite values, or ambiguous axes.

`python/mmwcore` owns immutable physical contracts, offline I/O, explicit recipes, stateful
composition, metrics, and optional visualization.

Downstream projects own datasets, experiments, learned models, product behavior, and deployment.

## Input contracts

Packet assembly distinguishes diagnostics from exact frame construction:

- diagnostic reordering requires an explicit wrapping u32 packet origin and may report loss;
- exact assembly requires a caller-proven u48 frame byte origin;
- exact payload size, packet count, byte slots, and packet sequence must all validate;
- a first packet or modulo relation never proves radar-frame phase.

ADC files require an explicit shape, complex layout, and timing when used by temporal algorithms.
Versioned capture directories additionally validate fixed file names, schema, hashes, byte counts,
and their embedded physical contract.

TI layouts and antenna geometries are selected explicitly. Firmware profiles do not prove lane
layout, board geometry, orientation, or capture provenance.

## Physical data path

Every maintained transform carries or requires:

1. ADC layout and frame geometry;
2. range and Doppler FFT conventions;
3. virtual-array mapping and TDM phase compensation;
4. antenna geometry and angle calibration;
5. detection scale and quality channels;
6. coordinate frame, units, and metric projection;
7. clustering and tracker timing.

Silent axis inference, uncalibrated metric claims, and dataset-specific normalization do not belong
in the core.

## Future acquisition boundary

Trusted real-time acquisition may later be provided by the dedicated `mmwcli` acquisition tool
through a versioned stream contract. Such a contract would need producer identity, schema version,
counter origins, frame boundaries, lifecycle state, and integrity rules.

This is an architectural direction, not an implemented mmwcore stream integration. Until that
contract exists, live transport and hardware control remain outside mmwcore.

## Validation

Unit tests cover dimensions, integer domains, finite values, counter wrap, integrity checks, and
known transform conventions. Synthetic fixtures establish determinism but not hardware provenance.
Physical claims require a documented capture or redistributable reference vector.

Benchmarks use a versioned generated workload. Compare results only when workload, build, runtime,
platform, thread settings, and cache mode match.

## Versioning

Python and Rust packages share one version and evolve their public contracts together. Recorded
schema identifiers remain stable wire values. Breaking removal of ambiguous or live-control APIs is
preferred over compatibility shims that preserve unclear semantics.
