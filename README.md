# mmwcore

mmwcore is the storage and deterministic compute layer in the local OpenMMW research workspace.
It verifies finite IWR6843 captures, writes lossless takes, and turns ADC frames into repeatable
radar tensors. It also retains classical tracking and benchmarks as quality controls.

```text
finite: mmwcli.take.v3 -> mmwcore -> openmmw.take.v3 -> RT/RPC -> OpenMMW
online: mmwcli stream -> OpenMMW -> mmwcore DSP -> checkpoint -> Web
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

Convert a completed `mmwcli.take.v3` raw capture into the fixed `openmmw.take.v3` verified take:

```python
from mmwcore.io import read_capture, write_take

capture = read_capture("dataset/takes/subject/scene/action/take-001.capture")
take = write_take(capture, "dataset/takes/subject/scene/action/take-001")
```

The published take contains `session.json`, the byte-exact immutable `setup.json`, `radar.cfg`, and
`radar.mmwa`, plus `camera.mjpeg` and `camera.index.bin` when a camera participated. Mount height
and boresight pitch come only from the setup snapshot. The contract accepts downward pitch `0`,
`30`, or `90` degrees;
OpenMMW applies the corresponding sensor-to-level transform. Open the verified take for dataset
construction or inference:

```python
from mmwcore.io import open_take

take = open_take("dataset/takes/subject/scene/action/take")
frames = take.archive.read_frames(0, 4)
```

The `.mmwa` archive stores exact ADC bytes, frame geometry, capture specification, index, and
digests. Use `verify_all()` before a long training run when a complete replay is useful.

DSP composition lives in `mmwcore.dsp`: ADC decoding, range/Doppler processing, TDM virtual-array
mapping, Cartesian projection, and bounded sparsification. OpenMMW owns dataset policy, RT/RPC
windows, models, training, evaluation, and presentation.

## Tracking and benchmarks

`mmwcore.tracking` is a deterministic classical baseline for learned temporal models. Keep it for
comparable association, state-estimation, and metric results.

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
