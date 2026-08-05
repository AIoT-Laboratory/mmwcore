# mmwcore

Typed mmWave radar acquisition and signal-processing primitives implemented in Rust, with a
Python API for research workflows.

mmwcore owns the physical data path from raw ADC samples to detections, calibrated point clouds,
tracking inputs, and vital-sign phase. Application models and experiment orchestration belong in
downstream projects.

## Install

Python 3.12:

```console
pip install mmwcore
```

Rust 1.85 or newer:

```console
cargo add mmwcore
```

## Python

```python
import numpy as np

from mmwcore.core import ADCFrameSpec, RawADCFrame
from mmwcore.dsp import organize_adc_samples

raw = RawADCFrame(np.zeros(4 * 128 * 2, dtype=np.int16))
cube = organize_adc_samples(
    raw,
    ADCFrameSpec(num_chirps=2, num_rx=4, num_samples=128),
)
print(cube.data.shape)
```

The `mmwcore` command exposes inspection and preprocessing operations:

```console
mmwcore --help
```

## Scope

- DCA1000 packet/file ingestion
- explicit ADC, FFT, calibration, antenna, detection, point-cloud, and tracking contracts
- range, Doppler, angle, CFAR, clutter suppression, calibration, TDM compensation, and
  deterministic Cartesian sparsification
- DBSCAN and stateful 2D tracking
- radar/camera capture-session synchronization contracts
- Rust kernels exposed through PyO3; plotting remains in Python

The project is alpha. Physical conventions are explicit and tested, but hardware coverage and
public validation vectors are still being expanded.

## OpenRadar relationship

mmwcore is designed as a typed, performance-oriented, dual-language successor to OpenRadar rather
than a source fork. It already covers the main raw-ADC-to-point-cloud path and adds calibrated TDM
processing, tracking, session contracts, and Rust distribution. Capon/Bartlett parity, broader
public device fixtures, and identical-input benchmarks remain release gates before claiming a
strict superset or a measured performance advantage.
See [the comparison](docs/openradar-comparison.md).

## Development

```console
uv sync --extra dev
uv run pytest --cov=mmwcore
cargo test --workspace --locked
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [the architecture](docs/architecture.md).

## License

Apache-2.0.
