# mmwcore

Typed mmWave radar acquisition and signal-processing primitives in Rust and Python.

mmwcore owns the physical data path from raw ADC samples to range-Doppler products,
detections, calibrated point clouds, tracks, and vital-sign phase. Rust provides the
compute kernels and a native crate; Python provides typed research contracts, file and
device adapters, plotting, and PyO3 bindings. Learned models and experiment orchestration
belong in downstream projects.

[PyPI](https://pypi.org/project/mmwcore/) |
[crates.io](https://crates.io/crates/mmwcore) |
[Rust API](https://docs.rs/mmwcore) |
[source](https://github.com/AIoT-Laboratory/mmwcore)

## Install

Python 3.12 with native Rust kernels:

```console
pip install mmwcore
```

Rust 1.85 or newer:

```console
cargo add mmwcore
```

## Python API

### Inspect ADC geometry

Inspect a capture with an explicit frame shape:

```console
mmwcore inspect adc adc_data.bin --num-chirps 192 --num-rx 4 --num-samples 256 --json
```

When the shape is unknown, list byte-compatible candidates instead of silently selecting one:

```console
mmwcore inspect adc adc_data.bin --infer-shapes --json
```

Shape inference only checks storage compatibility. Chirp order, ADC layout, antenna geometry,
slope, sample rate, timing, and calibration must still come from the capture configuration.

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

### Stream range-Doppler frames

`ADCFileFrameReader` memory-maps fixed-size frames, so the full capture is never loaded at once.
This preset is valid only when the file was captured with the matching IWR6843ISK profile and Tx
order.

```python
from mmwcore.config import iwr6843_isk_range_doppler_recipe
from mmwcore.dsp import process_adc_to_range_doppler
from mmwcore.io import ADCFileFrameReader

recipe = iwr6843_isk_range_doppler_recipe(remove_static_clutter=True)
reader = ADCFileFrameReader(
    "adc_data.bin",
    recipe.decode.adc,
    frame_periodicity_s=0.1,
)

for index in range(min(reader.num_frames, 3)):
    cube = process_adc_to_range_doppler(reader.read_frame(index), recipe)
    print(index, cube.axes, cube.data.shape)
```

Pass an explicit `RadarProfile`, ADC layout, Tx order, and channel calibration when the capture
differs from the preset.

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
    "adc_data.bin",
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
| `mmwcore.io` | ADC files, DCA1000 packets and streams, capture controllers, and serial transport |
| `mmwcore.dsp` | FFT, clutter suppression, calibration, CFAR, AoA, point-cloud, and clustering pipelines |
| `mmwcore.tracking` | Stateful 2D trackers, assignment, runners, and tracking metrics |
| `mmwcore.session` | Radar/camera capture and causal timestamp-alignment contracts |
| `mmwcore.plot` | Research visualizations kept outside the Rust compute core |

Run `mmwcore --help` for command groups and operation-specific help.

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

The project is alpha. Physical conventions are explicit and tested, but hardware coverage and
public validation vectors are still being expanded.

## Positioning

[OpenRadar](https://github.com/PreSenseRadar/OpenRadar) remains a useful Python reference for TI
mmWave ADC parsing and DSP. mmwcore is an independent implementation, not a source fork. It targets
a higher-level replacement through explicit physical contracts, Rust kernels with Python bindings,
calibrated TDM and point-cloud processing, stateful tracking, capture synchronization, and native
distribution through both crates.io and PyPI.

mmwcore is not yet described as a strict feature superset. Capon/Bartlett/ZoomFFT coverage,
identical-input numerical comparisons, broader public hardware fixtures, and published throughput
and memory benchmarks remain open validation work. New surface area does not compensate for an
incorrect physical convention; device documentation and reference vectors remain authoritative.

## Development

```console
uv sync --extra dev
uv run pytest --cov=mmwcore
cargo test --workspace --locked
```

See [CONTRIBUTING.md](https://github.com/AIoT-Laboratory/mmwcore/blob/main/CONTRIBUTING.md)
and [the architecture](https://github.com/AIoT-Laboratory/mmwcore/blob/main/docs/architecture.md).

## License

Apache-2.0.
