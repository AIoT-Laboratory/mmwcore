# mmwcore for Rust

The Rust crate contains deterministic storage and compute kernels used by the Python research API:

- lossless indexed ADC archives;
- raw ADC decoding;
- FFT, clutter removal, and TDM virtual-array transforms;
- Cartesian projection and sparsification;
- detection and clustering;
- multiscale scatter-component/body tracking, assignment, and comparison metrics.

Implementation follows three domains:

```text
src/io/        lossless archive codec and indexed archive files
src/dsp/       ADC decoding, FFT, arrays, detection, and projection
src/tracking/  assignment, metrics, and the independent multiscale backend
```

Existing root imports and module paths remain available through re-exports. New code can use
the domain paths directly. `tracking::multiscale::ScatterBodyTracker` owns typed configuration,
state, and per-frame output. Its inputs are `[x, y, z, radial_velocity, snr_db]` in level-world
forward/right/up coordinates; `step(points, dt)` advances the causal state.

The TI GTRACK backend belongs to the separate `mmwcore-ti-gtrack` crate because it retains
TI's device-only license. It is not a dependency of this Apache-2.0 crate.

It performs no hardware control, DCA packet reception, process launch, plotting, or experiment
management.

```rust
use mmwcore::{AdcComplexLayout, AdcFrameSpec, decode_adc_i16};

let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::Group2IThenQ)
    .expect("valid ADC frame specification");
let cube = decode_adc_i16(&[1, 2, 3, 4], spec, false).expect("valid ADC payload");
assert_eq!(cube.shape(), [1, 1, 1, 2]);
```

See the [repository README](https://github.com/AIoT-Laboratory/mmwcore) for the Python research
path and validation commands.
