# mmwcore

Decode DCA1000 raw ADC data into range-Doppler products, detections, calibrated point clouds, and
tracks from Rust or Python.

mmwcore provides explicit capture and physical contracts, frame-by-frame file and packet ingestion,
Rust compute kernels, Python composition APIs, and plotting. It covers the physical data path from
raw samples to sensing products; learned models and experiment orchestration belong in downstream
projects.

[![PyPI](https://img.shields.io/pypi/v/mmwcore.svg?logo=pypi&logoColor=white)](https://pypi.org/project/mmwcore/)
[![crates.io](https://img.shields.io/crates/v/mmwcore.svg?logo=rust)](https://crates.io/crates/mmwcore)
[![docs.rs](https://img.shields.io/docsrs/mmwcore.svg?logo=docs.rs)](https://docs.rs/mmwcore)
[![CI](https://github.com/AIoT-Laboratory/mmwcore/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AIoT-Laboratory/mmwcore/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/AIoT-Laboratory/mmwcore.svg)](https://github.com/AIoT-Laboratory/mmwcore/blob/main/LICENSE)

## Install

Python 3.10–3.13 with native Rust kernels:

```console
pip install mmwcore
```

Rust 1.85 or newer:

```console
cargo add mmwcore
```

## TI capture contracts

- `group2_i_then_q` decodes the complex16, two-lane TI mmWave Studio layout used by xWR16xx,
  xWR18xx, and xWR68xx captures.
- `group4_i_then_q` decodes the complex16, four-lane, channel-interleaved TI mmWave Studio layout
  used by xWR12xx and xWR14xx captures.
- Source-backed antenna geometries are available for XWR1642, the standard XWR1843 EVM,
  IWR6843ISK, IWR6843 AOP, and AWR1843 AOP.

The legacy TI firmware-configuration parser and `mmwcli.capture_session.v1` consumer remain
xWR68xx-specific.
These formats and geometries are based on local TI SDK/Studio sources and offline tests; they do not
claim hardware control or hardware validation. Select the ADC layout and board geometry explicitly.

## Hardware-derived validation evidence

These figures were generated from a retained laboratory capture, not from synthetic fixtures. Its
explicit decode contract is 5000 frames, 2 chirps, 4 receivers, 128 complex samples,
`group2_i_then_q` layout, and a 10 ms frame period. The capture does not carry slope or sample-rate
metadata, so the vertical coordinate remains a range bin rather than a fabricated distance in
meters. The source capture is not distributed, so these figures are validation evidence rather
than a runnable example.

![Range-Time magnitude before and after temporal-background suppression](https://raw.githubusercontent.com/AIoT-Laboratory/mmwcore/main/docs/assets/adc-range-time.png)

The lower panel removes the complex temporal mean independently for each chirp, receiver, and
range bin before magnitude aggregation. It exposes changing returns while preserving the raw map
above it; it is a deterministic diagnostic, not a learned result.

![Raw ADC I/Q and four-receiver range spectra](https://raw.githubusercontent.com/AIoT-Laboratory/mmwcore/main/docs/assets/adc-frame-diagnostics.png)

## Python API

### Open and process an mmwcli capture directory

`mmwcli ... capture --session-dir` publishes `adc.bin`, `radar.cfg`, and `capture.json` together.
Open the completed directory and reuse its physical contract:

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
cube = capture.range_doppler(recipe, frame_index=0)
print(cube.axes, cube.data.shape)
```

`open_capture` verifies the v1 schema, hashes, byte count, and finite CFG-derived contract. Use the
completed directory and keep it unchanged while reading; SHA-256 verifies self-consistency, not
provenance. The recipe explicitly selects IWR6843ISK antenna geometry; use a recipe matching the
actual board. `capture.frames()` lazily yields validated raw frames without loading the full file.

### Open a capture from an xWR68xx CLI config

The strict parser accepts the supported legacy raw-capture subset: complex 16-bit ADC, legacy
`frameCfg`, one-hot TDM chirps, `adcbufCfg -1 0 1 1 1`, and headerless hardware ADC LVDS.

```python
from mmwcore.config import parse_ti_cli_capture_spec_file
from mmwcore.core import ADCComplexLayout
from mmwcore.io import ADCFileFrameReader

capture = parse_ti_cli_capture_spec_file(
    "radar.cfg",
    layout=ADCComplexLayout.GROUP2_I_THEN_Q,
)
reader = ADCFileFrameReader.from_capture("capture.bin", capture)
```

Choose `layout` from the actual DCA1000 write format; the TI CLI config does not prove it. This
extracts an offline waveform, frame, and decode contract. It does not validate device readiness or
execute the configuration.

### Decode one frame

```python
import numpy as np

from mmwcore.core import ADCFrameSpec, RawADCFrame
from mmwcore.dsp import organize_adc_samples

raw = RawADCFrame(np.zeros(4 * 128 * 2, dtype=np.int16))
cube = organize_adc_samples(
    raw,
    ADCFrameSpec(num_chirps=2, num_rx=4, num_samples=128),
)
print(cube.axes, cube.data.shape)
```

### Build point clouds and tracks

The high-level recipe composes ADC decoding, range/Doppler transforms, TDM virtual-array mapping,
angle estimation, detection, and calibrated Cartesian projection. Detection thresholds are tied to
the capture's ADC scale; the example value is not a universal default.

```python
from mmwcore.config import iwr6843_isk_3d_point_cloud_recipe
from mmwcore.core import DBSCANClusteringSpec, Tracker2DSpec, TrackGatingSpec
from mmwcore.dsp import cluster_point_cloud, process_adc_to_calibrated_point_cloud
from mmwcore.io import ADCFileFrameReader
from mmwcore.tracking import ClusterTracker2D

recipe = iwr6843_isk_3d_point_cloud_recipe(
    threshold=250_000.0,  # Tune for the capture's ADC scale.
    remove_static_clutter=True,
)
reader = ADCFileFrameReader(
    "capture.bin",
    recipe.detection.transform.decode.adc,
    frame_periodicity_s=0.1,
)
cluster_spec = DBSCANClusteringSpec(
    eps_m=0.35,
    min_samples=3,
    velocity_scale_s=0.2,
)
tracker = ClusterTracker2D(
    Tracker2DSpec(
        frame_period_s=0.1,
        gating=TrackGatingSpec(max_distance_m=0.8),
    )
)

for index in range(reader.num_frames):
    points = process_adc_to_calibrated_point_cloud(reader.read_frame(index), recipe)
    clusters = cluster_point_cloud(points, cluster_spec)
    tracks = tracker.step(clusters)
    print(tracks.track_ids.tolist(), tracks.positions.tolist())
```

Keep one tracker instance for a sequence. Recreating it for every frame discards temporal state.

### Python package map

| Package | Responsibility |
| --- | --- |
| `mmwcore.core` | Typed ADC, cube, detection, point-cloud, clustering, tracking, and vital-sign contracts |
| `mmwcore.config` | Radar profiles, capture contracts, presets, and configuration rendering |
| `mmwcore.io` | ADC files, packets, capture-session readers, and compatibility hardware adapters |
| `mmwcore.dsp` | FFT, clutter suppression, calibration, CFAR, AoA, point-cloud, and clustering pipelines |
| `mmwcore.tracking` | Stateful 2D trackers, assignment, runners, and tracking metrics |
| `mmwcore.session` | Radar/camera capture and causal timestamp-alignment contracts |
| `mmwcore.plot` | Research visualizations kept outside the Rust compute core |

## Rust API

Decode one ADC frame into the canonical `[frame, chirp, rx, sample]` cube:

```rust
use mmwcore::{AdcComplexLayout, AdcFrameSpec, decode_adc_i16};

let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::Group2IThenQ)
    .expect("valid ADC frame specification");
let cube = decode_adc_i16(&[1, 2, 3, 4], spec, false).expect("valid ADC payload");
assert_eq!(cube.shape(), [1, 1, 1, 2]);
```

Run one-dimensional cell-averaging CFAR:

```rust
use mmwcore::{Cfar1DConfig, CfarMode, detect_cfar_1d};

let power = [1.0, 1.0, 0.0, 20.0, 0.0, 5.0, 5.0];
let config = Cfar1DConfig::new(2, 0, 1.1, CfarMode::Ca, false, 0, 0)
    .expect("valid CFAR configuration");
let result = detect_cfar_1d(&power, config).expect("valid CFAR input");
assert_eq!(result.indices, [3]);
```

## Scope

- DCA1000 packet and file ingestion
- explicit ADC, FFT, calibration, antenna, detection, point-cloud, and tracking contracts
- range, Doppler, angle, CFAR, clutter suppression, calibration, TDM compensation, and
  deterministic Cartesian sparsification
- DBSCAN and stateful 2D tracking
- radar/camera capture-session synchronization contracts
- Rust kernels exposed through PyO3; plotting remains in Python

The project is alpha. Physical conventions are explicit and tested, but supported capture formats
and public validation vectors are still being expanded.

## Positioning

[OpenRadar](https://github.com/PreSenseRadar/OpenRadar) remains a useful Python reference for TI
mmWave ADC parsing and DSP. mmwcore is an independent implementation, not a source fork. It focuses
on explicit physical contracts, Rust-backed kernels, frame-by-frame ingestion, calibrated TDM and
point-cloud processing, stateful tracking, and native distribution through crates.io and PyPI.

Identical-input numerical comparisons, redistributable hardware fixtures, and published benchmark
results remain open validation work. Until those results are published, mmwcore does not claim
performance or feature superiority. Device documentation and reference vectors remain authoritative
for physical conventions.

## Development

```console
uv sync --extra dev
uv run pytest --cov=mmwcore
cargo test --workspace --locked
```

See [benchmarking](docs/benchmarking.md) for the reproducible synthetic pipeline runner.

See [CONTRIBUTING.md](https://github.com/AIoT-Laboratory/mmwcore/blob/main/CONTRIBUTING.md)
and [the architecture](https://github.com/AIoT-Laboratory/mmwcore/blob/main/docs/architecture.md).

## License

Apache-2.0.
