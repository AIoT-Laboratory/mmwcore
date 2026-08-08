# mmwcore

mmwcore is an offline radar decoding and signal-processing library for captured mmWave data.
Its boundary starts at bytes plus explicit physical contracts, then produces range-Doppler cubes,
detections, calibrated point clouds, clusters, tracks, and vital-sign products.

[![PyPI](https://img.shields.io/pypi/v/mmwcore.svg?logo=pypi&logoColor=white)](https://pypi.org/project/mmwcore/) [![crates.io](https://img.shields.io/crates/v/mmwcore.svg?logo=rust)](https://crates.io/crates/mmwcore)
[![docs.rs](https://img.shields.io/docsrs/mmwcore.svg?logo=docs.rs)](https://docs.rs/mmwcore) [![CI](https://github.com/AIoT-Laboratory/mmwcore/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AIoT-Laboratory/mmwcore/actions/workflows/ci.yml) [![License](https://img.shields.io/github/license/AIoT-Laboratory/mmwcore.svg)](LICENSE)

## Install

CPython 3.12–3.14:

```console
pip install mmwcore
```

Rust 1.85 or newer:

```console
cargo add mmwcore
```

## Input boundary

Maintained inputs are:

- archived DCA1000 datagrams;
- headerless `int16` ADC files with an explicit frame contract;
- completed versioned capture directories opened with `open_capture`;
- finite `mmwcli.capture_stream.v1` data supplied through a caller-owned `BinaryIO`.

The library does not configure devices, render firmware commands, or manage live acquisition.
Callers must supply ADC layout, frame geometry, timing, antenna geometry, and trusted packet/frame
origins when the stored format does not prove them.

## TI capture contracts

- `group2_i_then_q`: complex16 two-lane layout documented by the TI mmWave Studio reader for
  xWR16xx, xWR18xx, and xWR68xx captures.
- `group4_i_then_q`: complex16 four-lane, channel-interleaved layout documented by the TI mmWave
  Studio reader for xWR12xx and xWR14xx captures.
- Source-backed antenna geometries: XWR1642, standard XWR1843 EVM, IWR6843ISK, IWR6843 AOP, and
  AWR1843 AOP.

These explicit decoders and geometry presets are usable only when the caller proves the actual
layout and board. They do not widen the versioned mmwcli contract: directory and stream v1 readers
currently accept only `vendor=ti`, `family=xwr68xx`, empty `model`/`revision`,
`identity_source=route_declaration`, `config_format=ti_mmwave_legacy_cli.v1`, `dtype=int16`,
`byte_order=little`, `lane_count=2`, and `layout=group2_i_then_q`.
`ADCFileCapture.raw_capture` and `CaptureStreamContract.raw_capture` expose that tuple;
`route_declaration` is not an observed device identity. See the
[mmwcli hardware-support matrix](https://github.com/AIoT-Laboratory/mmwcli/blob/main/docs/hardware-support.md).

## Python examples

### Open a versioned capture directory

`open_capture` validates the manifest schema, required regular files, hashes, byte count, and finite
physical contract. The directory must remain unchanged while it is open.

```python
from mmwcore.config import iwr6843_isk_range_doppler_recipe
from mmwcore.io import open_capture

capture = open_capture("capture-session")
contract = capture.radar_capture
recipe = iwr6843_isk_range_doppler_recipe(
    contract.profile,
    adc_layout=contract.adc.layout,
    tx_order=contract.tx_order,
)
range_doppler = capture.range_doppler(recipe, frame_index=0)
print(range_doppler.axes, range_doppler.data.shape)
```

The hash proves internal consistency, not provenance. The recipe selects IWR6843ISK geometry, so
use a different explicit recipe when the capture came from another board.

### Decode a finite capture stream

```python
from mmwcore.io import CaptureStreamReader

reader = CaptureStreamReader(source)  # caller-owned BinaryIO
contract = reader.read_contract()
for provisional in reader.provisional_frames():
    process(provisional.frame, contract.radar_capture)
commit = reader.require_commit()
```

Frames remain provisional until `require_commit` validates COMMIT and terminal EOF. mmwcore neither
closes the source nor opens a process, socket, or device. The source must make `read` honor any
required deadline or cancellation. The matching mmwcli transport and CLI producer are not wired yet.

### Read a headerless ADC file

```python
from mmwcore.core import ADCComplexLayout, ADCFrameSpec
from mmwcore.io import ADCFileFrameReader

spec = ADCFrameSpec(
    num_chirps=2,
    num_rx=4,
    num_samples=128,
    layout=ADCComplexLayout.GROUP2_I_THEN_Q,
)
reader = ADCFileFrameReader("adc.bin", spec, frame_periodicity_s=0.01)
raw = reader.read_frame(0)
print(raw.samples.shape)
```

`ADCFileFrameReader` rejects incomplete files by default and reads frames without loading the full
capture.

### Assemble archived datagrams

```python
from mmwcore.core import ADCFrameSpec
from mmwcore.io import assemble_dca1000_frame_bytes

spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)
raw, stats = assemble_dca1000_frame_bytes(
    [datagram_0, datagram_1],
    spec,
    frame_start_byte_count=trusted_frame_origin,
    payload_values_per_packet=2,
)
```

Stateless assembly requires exactly one complete frame: fixed payload lengths, contiguous u48 byte
slots, and contiguous wrapping u32 packet numbers. `frame_start_byte_count` must come from the
capture lifecycle; it is never inferred from the first packet or a modulo guess.

### Continue to point clouds and tracks

Use `iwr6843_isk_3d_point_cloud_recipe` with `process_adc_to_calibrated_point_cloud`, then
`cluster_point_cloud` and `ClusterTracker2D`. Thresholds, calibration, Tx order, and tracker timing
remain explicit. Keep one tracker instance for a sequence; recreating it discards temporal state.

## Rust example

```rust
use mmwcore::{AdcComplexLayout, AdcFrameSpec, decode_adc_i16};

let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::Group2IThenQ).expect("valid spec");
let cube = decode_adc_i16(&[1, 2, 3, 4], spec, false).expect("valid captured payload");
assert_eq!(cube.shape(), [1, 1, 1, 2]);
```

## Package map

- `mmwcore.core`: axes, units, ADC, cube, detection, point-cloud, and tracking contracts.
- `mmwcore.config`: finite radar profiles, capture contracts, antenna geometries, and recipes.
- `mmwcore.io`: packet, ADC-file, capture-directory, and finite capture-stream readers.
- `mmwcore.dsp`: FFT, clutter removal, CFAR, calibration, AoA, projection, and clustering.
- `mmwcore.tracking`: assignment, stateful trackers, runners, metrics, and validation artifacts.
- `mmwcore.plot`: optional research visualizations outside the Rust compute core.

## Validation boundaries

The retained laboratory capture used for the figures below has 5000 frames, 2 chirps, 4 receivers,
128 complex samples, `group2_i_then_q`, and a 10 ms period. It lacks slope and sample-rate metadata,
so range remains in bins. The source capture is not distributed; the figures are evidence, not a
runnable fixture.

![Range-Time magnitude before and after temporal-background suppression](https://raw.githubusercontent.com/AIoT-Laboratory/mmwcore/main/docs/assets/adc-range-time.png)

![Raw ADC I/Q and four-receiver range spectra](https://raw.githubusercontent.com/AIoT-Laboratory/mmwcore/main/docs/assets/adc-frame-diagnostics.png)

Synthetic tests verify shapes, axes, finite values, wrap behavior, and deterministic transforms.
They do not prove a board configuration, capture provenance, universal thresholds, or performance
superiority. Device documentation and redistributable reference vectors remain authoritative.

## Benchmarks and development

The reproducible synthetic pipeline, workload contract, and comparison rules are documented in
[docs/benchmarking.md](docs/benchmarking.md).

```console
uv sync --extra dev --locked
uv run pytest --cov=mmwcore
cargo test --workspace --locked
uv run python benchmarks/pipeline.py --warmups 0 --samples 1 --stream-frames 2
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/architecture.md](docs/architecture.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
