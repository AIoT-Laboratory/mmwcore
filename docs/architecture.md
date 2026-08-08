# Architecture

## Boundary

mmwcore begins at captured bytes and explicit offline contracts. It decodes and transforms radar
data without owning device configuration, transport lifecycle, experiment orchestration, models,
or application behavior.

```text
captured packet bytes / ADC files / capture directory / capture stream
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
The listed TI-family layout decoders and board geometries remain caller-selected capabilities;
firmware profiles do not prove lane layout, board geometry, orientation, or provenance.

Versioned mmwcli directory and stream v1 readers additionally require one closed family tuple:
`ti`, `xwr16xx|xwr18xx|xwr68xx`, empty model/revision, `route_declaration`,
`ti_mmwave_legacy_cli.v1`, `int16`, little endian, two lanes, and `group2_i_then_q`. The strict CFG
binding accepts 76–81 GHz with at most two TX identifiers for xWR16xx, 76–81 GHz with at most three
for xWR18xx, and 57–64 GHz with at most three for xWR68xx; incompatible frequency or TX masks fail
closed. The parser has no default family. Public `raw_capture` exposes the route declaration, while
chirps expose `tx_order`; neither field proves device identity, board geometry, or a preset.

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

## Capture-stream boundary

`CaptureStreamReader` decodes finite `mmwcli.capture_stream.v1` records from a caller-owned
`BinaryIO`. It validates producer metadata, zero-based counters, complete frame boundaries, strict
JSON, payload bounds, record and ADC digests, the embedded radar configuration, terminal state, and
EOF. Frames remain provisional unless COMMIT and EOF both validate.

Both mmwcli capture routes, `studio-cli capture --stream` and `debug-cli capture --stream`, provide
the matching binary stdout producer. `open_capture_stream` validates the contract immediately and
offers pull-driven ADC-frame or contract-bound Range-Doppler iteration without prefetch or worker
threads. mmwcore does not open a process, socket, device, or hardware session and never closes the
source. The caller owns both the producer process and source and must make reads honor any required
deadline or cancellation.

## Multi-sensor boundary

`open_multisensor_capture` validates a published `mmwcli.multisensor_session.v1` directory,
artifact hashes, fixed indices, source outcomes, and clock mappings. Radar sources can open their
nested capture session with an explicit recipe or preset. `causal_pairs` joins source items by
conservative mapped intervals and an explicit lag window; it does not infer synchronization from
item indices.

`open_multisensor_stream` decodes caller-owned `mmwcli.multisensor_stream.v1` data. It exposes
cross-source provisional items, source outcomes, and final COMMIT+EOF validation. RADAR_START maps
radar tick zero to a conservative host interval, while `delivery_observed` camera ticks are exact
host-relative delivery observations. Neither is silently promoted to an exposure timestamp.
Failed or omitted optional-source items are rejected by the final commit lineage.

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
