# mmwcore

mmwcore decodes captured and live mmWave data for offline training and real-time inference. Its
boundary starts at caller-owned bytes plus explicit physical contracts, then produces
range-Doppler cubes, detections, calibrated point clouds, clusters, tracks, and vital-sign products.

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
- finite `mmwcli.capture_stream.v1` data supplied through a caller-owned `BinaryIO`;
- published radar-plus-camera sessions opened with `open_multisensor_capture`;
- finite `mmwcli.multisensor_stream.v1` data opened with `open_multisensor_stream`.

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
layout and board. Versioned mmwcli directory and stream v1 readers accept the three closed family
tuples `family=xwr16xx`, `family=xwr18xx`, and `family=xwr68xx`; each also requires `vendor=ti`,
empty `model`/`revision`, `identity_source=route_declaration`,
`config_format=ti_mmwave_legacy_cli.v1`, `dtype=int16`, `byte_order=little`, `lane_count=2`, and
`layout=group2_i_then_q`.

The embedded legacy CFG must match the declared family: xWR16xx uses the 76–81 GHz range and up to
two TX identifiers, xWR18xx uses 76–81 GHz and up to three, and xWR68xx uses 57–64 GHz and up to
three. Chirp order determines the explicit `tx_order`; no family selects antenna geometry or a
preset. The standalone TI CLI parser likewise requires an explicit `family` keyword and has no
default. `ADCFileCapture.raw_capture` and `CaptureStreamContract.raw_capture` expose the declared
tuple, but decoder acceptance does not claim that an mmwcli acquisition route has been validated.
`route_declaration` is not an observed device identity. See the
[mmwcli hardware-support matrix](https://github.com/AIoT-Laboratory/mmwcli/blob/main/docs/hardware-support.md).

## Python examples

### Open a versioned capture directory

`open_capture` validates the manifest schema, required regular files, hashes, byte count, and finite
physical contract. The directory must remain unchanged while it is open.

```python
from mmwcore import open_capture
from mmwcore.config import iwr6843_isk_range_doppler_recipe

capture = open_capture("capture-session", range_doppler=iwr6843_isk_range_doppler_recipe)
range_doppler = capture.range_doppler(frame_index=0)
```

The hash proves internal consistency, not provenance. Passing the preset is the caller's explicit
declaration that the board uses IWR6843ISK geometry; the route-declared manifest does not identify
or guess a board. Pass a different preset or an explicit recipe for other verified hardware.

### Process a finite capture stream

```python
import mmwcore
from mmwcore.config import iwr6843_isk_range_doppler_recipe

stream = mmwcore.open_capture_stream(
    source,  # caller-owned BinaryIO
    range_doppler=iwr6843_isk_range_doppler_recipe,
)
for item in stream.range_doppler():
    infer(item.cube)
commit = stream.require_commit()
```

Frames remain provisional until `require_commit` validates COMMIT and terminal EOF. mmwcore neither
closes the source nor opens a process, socket, or device. The source must make `read` honor any
required deadline or cancellation. Processing is synchronous and pull-driven, without prefetch or
worker threads.

### Train from synchronized radar and camera data

```python
from mmwcore import open_multisensor_capture
from mmwcore.config import iwr6843_isk_range_doppler_recipe

session = open_multisensor_capture("training-session")
radar = session.source("radar-0")
radar_capture = radar.open_radar_capture(
    range_doppler=iwr6843_isk_range_doppler_recipe,
)
for camera_item, radar_item in session.causal_pairs(
    "camera-0", "radar-0", lag_min_ns=0, lag_max_ns=50_000_000
):
    train(
        camera_item.payload,
        radar_capture.range_doppler(frame_index=radar_item.item_index),
    )
```

The join uses conservative mapped time intervals, not equal frame numbers or nearest arrival time.
The preset is an explicit board-geometry declaration; choose the preset or recipe that matches the
actual radar.

### Consume a live aggregate stream

```python
from mmwcore import open_multisensor_stream

stream = open_multisensor_stream(source)  # caller-owned BinaryIO
provisional = list(stream.items())
commit = stream.require_commit()
accepted = [item for item in provisional if commit.accepts(item)]
```

Radar and `delivery_observed` camera items expose `mapped_time` on the same host-relative axis.
A camera `exposure_midpoint` remains unmapped in the live stream unless the producer supplies a
live mapping; mmwcore never substitutes delivery time for exposure time. Items and derived results
remain provisional until global COMMIT and EOF.

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
- `mmwcore.io`: packet, ADC-file, radar/multi-sensor directory, and finite live-stream readers.
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
