# mmwcore

mmwcore is a standalone library for lossless mmWave ADC compression/decompression,
deterministic radar signal processing, and classical tracking through Rust-backed Python APIs.
Use it to turn completed raw captures or
in-memory ADC frames into radar tensors and point clouds with explicit geometry, axes, and units.

The maintained hardware contract is **TI IWR6843 ES2 with DCA1000**. Generic numerical kernels
can be composed with explicit specifications; other radar devices are not validated by this
repository. Acquisition, model training, stream management, and visualization belong to the
applications calling mmwcore.

## Version and installation

The current Python release is **0.8.0**. See the
[migration notes](docs/releases/0.8.0.md) before upgrading from 0.7.1.

Install with **CPython 3.12** on Windows x86-64 or Linux x86-64:

```console
python -m pip install "mmwcore==0.8.0"
```

For current development, install from source with **CPython 3.12 and Rust 1.97**:

```console
git clone https://github.com/AIoT-Laboratory/mmwcore.git
cd mmwcore
uv sync --python 3.12 --extra dev --locked
```

Source installation needs the platform's Rust linker/build tools. Current CI validates Linux
and Windows; macOS is not covered by the current CI matrix. Numerical kernels and the built-in
TI tracker require no TI SDK or external GTRACK library.

## Quick start

This synthetic example runs without hardware or recorded data:

```python
import numpy as np

from mmwcore.core import ADCDecodeSpec, ADCFrameSpec, RangeDopplerPipeline
from mmwcore.dsp import range_doppler

adc = ADCFrameSpec(num_chirps=2, num_rx=1, num_samples=8)
raw = np.zeros(adc.raw_values_per_frame, dtype=np.int16)
cube = range_doppler(raw, RangeDopplerPipeline(decode=ADCDecodeSpec(adc)))
print(cube.axes, cube.data.shape)
```

For real captures, specify the actual ADC layout, TDM order, waveform, and antenna geometry.
See [file examples](examples/README.md), [architecture](docs/architecture.md),
[ADC compression](docs/adc-compression.md), and [tracking](docs/ti-gtrack.md).

## Library scope

```text
raw ADC files -> compression/decompression -> frame readers -> radar DSP
in-memory ADC frames -> radar DSP -> tensors / point clouds -> tracking
```

Public APIs, examples, tests, and benchmarks can be used from this repository alone. Callers
supply raw ADC files, capture specifications, or in-memory frames. Acquisition, buffering,
model training, and visualization are application responsibilities.

## ADC files and capture packages

ADC storage uses explicit lossless compression/decompression APIs:

```python
from mmwcore.io import compress_adc_file, decompress_adc_file

# capture is the RadarCaptureSpec describing the complete raw ADC file.
compressed = compress_adc_file("adc.bin", "radar.mmwa", capture)
raw_frames = compressed.decompress_frames(0, 4)
restored_capture = decompress_adc_file("radar.mmwa", "restored.bin")
```

The standardized format carries frame geometry and the decoding contract; it preserves raw
ADC bytes exactly. See [file and frame-group APIs](docs/adc-compression.md). Take packaging
below adds the capture/setup/context contract around that compressed ADC file.

The optional take API packages a completed `mmwcli.take.v3` capture with verified setup and context metadata:

```python
from pathlib import Path

from mmwcore.io import read_capture, write_take

capture = read_capture("dataset/takes/subject/scene/action/take-001.capture")
context = Path("context.json").read_bytes()
take = write_take(capture, "dataset/takes/dataset/scenario/c01/take-001", context=context)
```

The published v4 take contains `session.json`, hashed `context.json`, the byte-exact immutable
`setup.json`, `radar.cfg`, and
`radar.mmwa`, plus `camera.mjpeg` and `camera.index.bin` when a camera participated. Mount height
and boresight pitch come only from the setup snapshot. The contract accepts downward pitch `0`,
`30`, or `90` degrees. DSP recipes use explicit sensor-to-level geometry. Open the verified take
for frame processing:

```python
from mmwcore.io import open_take

take = open_take("dataset/takes/dataset/scenario/c01/take-001")
frames = take.archive.decompress_frames(0, 4)
```

The `.mmwa` archive stores exact ADC bytes, frame geometry, capture specification, index, and
digests. Use `verify_all()` when a complete integrity check is required before processing.

DSP composition lives in `mmwcore.dsp`: ADC decoding, range/Doppler processing, TDM virtual-array
mapping, Cartesian projection, and bounded sparsification. Applications choose processing
windows and consume the resulting tensors, point clouds, or tracks.

An opt-in native [ISK dynamic Capon frontend](docs/isk-capon.md) provides the
complete default RA-Capon/CFAR/elevation/weighted-Doppler chain from ADC.
It retains stage diagnostics and source-specific behavior without changing
the existing RPC pipeline; host arithmetic is not TI DSP bit emulation.

## Tracking and benchmarks

`TiGTrack3D` implements the pinned IWR6843 **TI 3DA nine-state** tracker in Rust.
Normal installations need no TI SDK, C compiler or GTRACK DLL. Association, allocation,
update, lifecycle and static support run inside mmwcore's native extension.
This component retains TI's **TI-device-only** license, included in source and wheels;
other mmwcore code remains Apache-2.0. See [API and validation](docs/ti-gtrack.md).

The other maintained backend is `ScatterBodyTracker` in `mmwcore.tracking.multiscale`, backed by
`crates/mmwcore/src/tracking/multiscale`: multiscale clustering, scatter-component histories,
causal bulk-motion estimation, split-origin body hypotheses, and separate component/body limits.
Its ablation components remain available for comparison and subsequent improvements against
complete TI GTRACK. Both backends are exported from `mmwcore.tracking`.

The simplified `GTrack2D`/six-state `GTrack3D`, `PointTracker2D`, `ClusterTracker2D`, their
configuration adapters, ADC runners and dedicated 2D vector benchmark have been removed.
There are no compatibility aliases or legacy backend switch. Shared DBSCAN, assignment,
geometry, `TrackFrame` and comparison metrics remain; metric regions use `boundary_boxes`
directly instead of the retired `ScenerySpec`.

`benchmarks/pipeline.py` is the performance and regression gate for the fixed IWR6843 workload. It
uses deterministic synthetic ADC and requires no hardware or private data. See
[benchmarking](docs/benchmarking.md).

## Package map

- `mmwcore.io`: standardized ADC compression/decompression, frame readers, captures, and takes.
- `mmwcore.config`: IWR6843 capture parsing and processing presets.
- `mmwcore.core`: explicit array, geometry, DSP, and tracking contracts.
- `mmwcore.dsp`: deterministic radar processing and neural-input primitives.
- `mmwcore.tracking`: classical tracking baselines and metrics.
- `crates/mmwcore`: Rust ADC compression/decompression, DSP, and tracking kernels.
- `crates/mmwcore/src/tracking/multiscale`: independent Rust tracking backend.
- `crates/mmwcore-ti-gtrack`: TI GTRACK backend with its separate license.

## Validation

```console
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uv run --python 3.12 ruff format --check python tests benchmarks examples tools
uv run --python 3.12 ruff check --no-cache python tests benchmarks examples tools
uv run --python 3.12 pyright
uv run --python 3.12 python -m pytest -p no:cacheprovider -q
uv run --python 3.12 python benchmarks/pipeline.py --warmups 0 --samples 1 --stream-frames 2
```

These checks are local and do not access radar hardware.

CI also builds a source distribution and compiles the wheel from that archive on Windows and
Linux. It installs the wheel into a clean environment outside the checkout and runs
`tests/distribution_smoke.py` with `python -I`. The check verifies installed import paths,
license files, type stubs, FFT, ADC compression/decompression round trips and both tracking backends. The source
distribution includes the frozen TI oracle fixture; CI extracts the archive and runs its Rust
tests independently as well. The same check can be run locally:

```console
uv run --no-sync maturin build --release --locked --sdist --interpreter python --out dist
uv run --no-sync python tools/check_distribution.py dist
```

Use an output directory containing exactly one wheel and one source distribution.

## Contributing and licensing

Bug reports and focused contributions are welcome. Include a minimal reproduction, array
contracts, and platform/version information; synthetic data is sufficient for most numerical
issues. See [contribution guidelines](CONTRIBUTING.md) and the [release process](docs/releasing.md).

Current Python distributions include Apache-2.0 code and the TI-device-only GTRACK component;
their metadata declares `Apache-2.0 AND LicenseRef-TI-GTRACK`. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [TI-LICENSE.txt](crates/mmwcore-ti-gtrack/TI-LICENSE.txt). The standalone
`crates/mmwcore` Rust crate is Apache-2.0 and does not depend on the TI tracking crate.
