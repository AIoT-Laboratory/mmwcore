# OpenRadar comparison

OpenRadar established an accessible Python path for TI mmWave ADC parsing and DSP. mmwcore keeps
that practical scope while making physical contracts explicit and moving compute kernels to Rust.

| Area | OpenRadar | mmwcore 0.2 |
| --- | --- | --- |
| Primary implementation | Python/NumPy | Rust with Python/PyO3 |
| DCA1000 raw ingestion | Yes | Yes |
| Range/Doppler FFT | Yes | Yes |
| Angle FFT | Yes | Yes |
| CFAR and peak processing | Yes | Yes |
| TDM phase compensation | Yes | Explicit typed contract and Rust kernel |
| Antenna/channel calibration | Yes | Typed calibration contracts and kernels |
| Cartesian point cloud | Yes | Calibrated typed output |
| Clustering | DBSCAN | DBSCAN |
| Tracking | EKF API | Stateful typed tracker and metrics |
| Capture synchronization | No core contract | Radar/camera session contracts |
| Rust crate | No | Yes |
| Python wheels | Yes | Release workflow prepared |
| Capon/Bartlett/ZoomFFT | Yes | Not yet feature-complete |

## Superset gate

mmwcore should be described as an intended upper-level replacement, not yet an unconditional
superset. That claim requires:

1. identical-input numerical comparisons for common range/Doppler/angle/CFAR paths;
2. maintained Capon/Bartlett equivalents or a documented reason to exclude them;
3. public real-device fixtures for supported capture formats;
4. published cross-platform wheels and a crates.io release;
5. benchmark results covering throughput, memory, and numerical tolerance.

New features do not compensate for an incorrect physical convention. Reference vectors and
device documentation remain authoritative for DSP acceptance.
