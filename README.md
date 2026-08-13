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

Rust 1.97 or newer:

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

| mmwcli output | mmwcore entry point | Typical use |
| --- | --- | --- |
| radar capture-session directory | `open_capture` | reproducible offline radar processing |
| radar `--stream` stdout | `open_capture_stream` | pull-driven real-time radar inference |
| radar-plus-camera session directory | `open_multisensor_capture` | training data, indexed source access, and causal joins |
| aggregate `--stream` stdout | `open_multisensor_stream` | provisional multi-sensor inference followed by COMMIT/EOF validation |

The matching acquisition commands and camera-producer workflow are documented by
[mmwcli](https://github.com/AIoT-Laboratory/mmwcli) and its
[multi-sensor guide](https://github.com/AIoT-Laboratory/mmwcli/blob/main/docs/multisensor-sync.md).

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

The repository includes complete command-line examples that consume files or binary stdin and
never start hardware:

| Workflow | Example |
| --- | --- |
| capture directory or explicit raw frames | [capture_or_raw.py](examples/capture_or_raw.py) |
| explicit xWR18xx/XWR1843 EVM geometry and recipe | [xwr18_range_doppler.py](examples/xwr18_range_doppler.py) |
| finite radar live stream | [radar_live_stream.py](examples/radar_live_stream.py) |
| multi-sensor offline training pairs | [multisensor_offline_training.py](examples/multisensor_offline_training.py) |
| multi-sensor provisional live inference | [multisensor_live_inference.py](examples/multisensor_live_inference.py) |

See the [example index](examples/README.md) for copyable commands and finalization semantics.

### Open a versioned capture directory

`open_capture` validates the manifest schema, required regular files, hashes, byte count, and whole
ADC-frame geometry. A finalized CFG with `frameCfg numFrames=0` keeps its open-ended acquisition
semantics; mmwcore derives the positive actual frame count from the immutable ADC file instead of
inventing a planned length. The directory must remain unchanged while it is open.

```python
from mmwcore import open_capture

capture = open_capture("capture-session")
raw = capture.frame(0)
print(capture.raw_capture.family, raw.samples.shape)
```

The hash proves internal consistency, not provenance. A supplied preset is the caller's explicit
declaration of a processing contract; the route-declared manifest does not identify or guess a
board. Pass an exact `RangeDopplerRecipe` or callable preset to `open_capture(...,
range_doppler=...)` when Range-Doppler processing is wanted. Built-in board presets are conveniences,
not family defaults.

### Process a finite capture stream

```python
import mmwcore

stream = mmwcore.open_capture_stream(source)  # caller-owned BinaryIO
for item in stream.frames():
    infer(item.frame, stream.contract.radar_capture)
commit = stream.require_commit()
```

Frames remain provisional until `require_commit` validates COMMIT and terminal EOF. mmwcore neither
closes the source nor opens a process, socket, or device. The source must make `read` honor any
required deadline or cancellation. Processing is synchronous and pull-driven, without prefetch or
worker threads.

### Train from synchronized radar and camera data

```python
from mmwcore import open_multisensor_capture

session = open_multisensor_capture("training-session")
radar = session.source("radar-0")
radar_capture = radar.open_radar_capture()
for camera_item, radar_item in session.causal_pairs(
    "camera-0", "radar-0", lag_min_ns=0, lag_max_ns=50_000_000
):
    train(
        camera_item.payload,
        radar_capture.frame(radar_item.item_index),
    )
```

The join uses conservative mapped time intervals, not equal frame numbers or nearest arrival time.
If training needs Range-Doppler cubes instead of raw frames, bind the exact recipe or preset through
`open_radar_capture(range_doppler=...)` and call `range_doppler`. Choose processing geometry from
the actual board, never from the family string alone.

### Read session metadata after archiving ADC

```python
from mmwcore import (
    open_mmwcli_capture_metadata,
    open_multisensor_capture_metadata,
    open_multisensor_source_timeline,
)

capture = open_mmwcli_capture_metadata("training-session/sensors/radar-0")
session = open_multisensor_capture_metadata("training-session")
radar_timeline = open_multisensor_source_timeline("training-session", "radar-0")

print(capture.adc_sha256, session.session_id, radar_timeline.items[0].mapped_time)
```

These readers validate capture contracts, session metadata, and source timing without opening the
sensor payload. They keep synchronized training context usable after raw ADC is replaced by a
verified ADC archive. Call `revalidate_inputs()` before publishing a derived artifact when the
source files may have changed during processing.

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

### Read a capture-bound ADC archive

An ADC archive can be opened as the same finite frame-reader contract as a raw ADC file. The
reader requires both the immutable capture contract and the SHA-256 of the original logical ADC
bytes; an archive with a different layout, frame count, contract, or source digest is rejected.

```python
import hashlib
from pathlib import Path

from mmwcore.config import RadarCaptureSpec
from mmwcore.io import ADCArchiveFrameReader

capture = RadarCaptureSpec.from_record(capture_record)
with Path("adc.bin").open("rb") as stream:
    source_digest = hashlib.file_digest(stream, "sha256").hexdigest()
reader = ADCArchiveFrameReader(
    "adc.mmwa",
    capture,
    expected_adc_sha256=source_digest,
)
raw = reader.read_frame(100)
print(reader.num_frames, raw.samples.shape)
```

`read_frame()` verifies the selected frame before returning it. Use `reader.verify_all()` for an
explicit complete replay before a long processing or training run. `write_capture_adc_archive()`
creates the corresponding archive and refuses to publish when the source digest does not match the
caller-provided ADC identity. Use `reader.revalidate_input()` before publishing derived output
to confirm that the opened archive did not change during processing.

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

Use an explicit `PointCloudRecipe` with `process_adc_to_calibrated_point_cloud`, then
`cluster_point_cloud` and `ClusterTracker2D`. Source-backed geometry helpers cover XWR1642,
standard XWR1843 EVM, IWR6843ISK, IWR6843 AOP, and AWR1843 AOP; the IWR6843 processing recipes are
one optional preset family, not the mmwcore input boundary. Thresholds, calibration, Tx order, and
tracker timing remain explicit. Keep one tracker instance for a sequence; recreating it discards
temporal state.

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
so range remains in bins. The source capture is not distributed; the figures are validation assets,
not runnable fixtures.

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

### Archive completed ADC data

The offline ADC archive preserves every source byte in independently compressed and verified
frames. The capture contract remains caller-owned and is bound by its lowercase SHA-256 digest.

```python
from mmwcore.io import open_adc_archive, write_adc_archive

archive = write_adc_archive(
    "capture/adc.bin",
    "capture/adc.mmwa",
    frame_bytes=1_572_864,
    capture_contract_sha256="0123456789abcdef" * 4,
)
archive.verify_all()
four_frames = archive.read_frames(100, 104, verify=False)

reopened = open_adc_archive("capture/adc.mmwa")
one_verified_frame = reopened.read_frames(100, 101)
```

The fixed v1 codec is one-frame little-endian `int16` byte shuffle plus zlib level 1. Writes verify
the complete temporary archive before an atomic no-overwrite publication. Reads verify each frame
by default; `verify=False` is accepted only after `verify_all()` succeeds on that same reader.
See the [ADC archive study](docs/adc-archive-study.md) for corpus results, exact format scope,
and the acceptance benchmark. The fixed v1 format is admitted only for finalized ADC files;
acquisition still publishes exact raw payloads and conversion remains offline.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/architecture.md](docs/architecture.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
