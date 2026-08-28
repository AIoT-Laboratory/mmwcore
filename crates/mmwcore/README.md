# mmwcore for Rust

The Rust crate contains deterministic storage and compute kernels used by the Python research API:

- lossless indexed ADC archives;
- raw ADC decoding;
- FFT, clutter removal, and TDM virtual-array transforms;
- Cartesian projection and sparsification;
- detection, clustering, assignment, and tracking baselines.

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
