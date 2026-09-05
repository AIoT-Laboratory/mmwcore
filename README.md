# mmwcore

mmwcore is the storage and deterministic compute layer in the local OpenMMW research workspace.
It verifies finite IWR6843 captures, writes lossless takes, and turns ADC frames into repeatable
radar tensors. It also retains classical tracking and benchmarks as quality controls.

```text
finite: mmwcli.take.v3 -> mmwcore -> openmmw.take.v4 -> RT/RPC -> OpenMMW
online: mmwcli stream -> OpenMMW -> mmwcore DSP + tracking -> Web
quality: mmwcore DSP -> tracking baseline + benchmarks
```

Hardware setup, DCA1000 reception, training loops, checkpoints, and visualization remain outside
mmwcore. The maintained acquisition contract is IWR6843 ES2 with DCA1000. Finite storage begins
after mmwcli publishes a raw capture; online process and buffering remain in OpenMMW, which calls
the same mmwcore DSP on in-memory frames.

## Local setup

```console
uv sync --python 3.12 --extra dev --locked
```

CPython 3.12 is the maintained Python environment. Rust 1.97 supplies native storage and compute
kernels.

## Research path

Convert a completed `mmwcli.take.v3` raw capture into a verified take with immutable research context:

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
`30`, or `90` degrees;
OpenMMW applies the corresponding sensor-to-level transform. Open the verified take for dataset
construction or inference:

```python
from mmwcore.io import open_take

take = open_take("dataset/takes/dataset/scenario/c01/take-001")
frames = take.archive.read_frames(0, 4)
```

The `.mmwa` archive stores exact ADC bytes, frame geometry, capture specification, index, and
digests. Use `verify_all()` before a long training run when a complete replay is useful.

DSP composition lives in `mmwcore.dsp`: ADC decoding, range/Doppler processing, TDM virtual-array
mapping, Cartesian projection, and bounded sparsification. OpenMMW owns dataset policy, RT/RPC
windows, models, training, evaluation, and presentation.

## Tracking and benchmarks

`TiGTrack3D` runs the complete pinned IWR6843 **TI 3DA nine-state** source through a separately
built local plugin. It preserves original association, allocation, update, lifecycle and full
point/target reports. See [complete TI API, source provenance and validation](docs/ti-gtrack.md).
The plugin retains the TI-device-only license and is not bundled with the Apache package.

`mmwcore.tracking` is a deterministic classical baseline for learned temporal models. Keep it for
comparable association, state-estimation, and metric results.

`GTrack2D` keeps the TI-compatible 2D benchmark path. `GTrack3D` tracks sensor-frame
`[x,y,z,vx,vy,vz]` state from `[range,azimuth,elevation,radial velocity]` with an EKF. Both use
group dispersion, measurement noise, competitive Mahalanobis bidding, Doppler unwrapping,
lead-point allocation, and explicit tentative/confirmed/coasting lifecycles. GTrack3D reports full
3D position, velocity, position covariance, and reflection-extent covariance. Installation-pose
transforms belong at the application boundary because radial Doppler is defined about the radar,
not the room origin. These are compact, inspectable GTRACK implementations, not TI binary-library
or complete TI People Tracking equivalence claims.

For GTrack3D, the Cartesian distance gate bounds every association before the spherical
Mahalanobis/Doppler gate. A single associated point may conservatively correct a mature unit, but it
cannot confirm, reactivate, or keep a unit alive. Lifecycle evidence requires the configured
multi-point support. When a static-speed threshold is configured, a coasting unit additionally
requires renewed radial motion evidence before it can reactivate.

`benchmarks/pipeline.py` is the performance and regression gate for the fixed IWR6843 workload. It
uses deterministic synthetic ADC and requires no hardware or private data. See
[benchmarking](docs/benchmarking.md).

## Package map

- `mmwcore.io`: completed capture, take, raw ADC, and `.mmwa` access.
- `mmwcore.config`: IWR6843 capture parsing and processing presets.
- `mmwcore.core`: explicit array, geometry, DSP, and tracking contracts.
- `mmwcore.dsp`: deterministic radar processing and neural-input primitives.
- `mmwcore.tracking`: classical tracking baselines and metrics.
- `crates/mmwcore`: Rust archive and numerical kernels.

## Validation

```console
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uv run --python 3.12 ruff format --check python tests benchmarks examples
uv run --python 3.12 ruff check --no-cache python tests benchmarks examples
uv run --python 3.12 pyright
uv run --python 3.12 python -m pytest -p no:cacheprovider -q
uv run --python 3.12 python benchmarks/pipeline.py --warmups 0 --samples 1 --stream-frames 2
```

These checks are local and do not access radar hardware.
