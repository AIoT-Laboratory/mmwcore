# mmwcore

mmwcore is the storage and deterministic compute layer in the local OpenMMW research workspace.
It turns a completed finite IWR6843 take into repeatable radar tensors for neural-network research.
It also retains classical tracking and benchmarks as quality controls.

```text
mmwcli capture -> completed take -> radar.mmwa -> RT/RPC -> OpenMMW model
                                                   +-----> tracking baseline
```

Hardware setup, DCA1000 reception, training loops, checkpoints, and visualization remain outside
mmwcore. The maintained acquisition contract is IWR6843 ES2 with DCA1000; mmwcore starts only after
mmwcli has published a complete take.

## Local setup

```console
uv sync --python 3.12 --extra dev --locked
```

CPython 3.12 is the maintained Python environment. Rust 1.97 supplies native storage and compute
kernels.

## Research path

Convert a completed mmwcli capture into the fixed OpenMMW take:

```python
from mmwcore.io import read_capture, write_take

capture = read_capture("dataset/captures/subject/scene/action/take")
take = write_take(capture, "dataset/takes/subject/scene/action/take")
```

The published take contains `session.json`, `radar.cfg`, and `radar.mmwa`, plus
`camera.mjpeg` and `camera.index.bin` when a camera participated. Open it for dataset construction
or inference:

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
uv run --python 3.12 ruff check python tests benchmarks examples
uv run --python 3.12 pyright
uv run --python 3.12 pytest -q
uv run --python 3.12 python benchmarks/pipeline.py --warmups 0 --samples 1 --stream-frames 2
```

These checks are local and do not access radar hardware.
