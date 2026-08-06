# mmwcore for Rust

`mmwcore` provides physically explicit mmWave radar primitives:

- raw ADC and DCA1000 decoding;
- radar-cube transforms, FFTs, clutter suppression, and TDM compensation;
- CFAR, peak processing, AoA, calibrated point projection, and Cartesian sparsification;
- clustering, assignment, stateful tracking, and tracking metrics;
- vital-sign phase unwrapping and displacement conversion.

The crate performs no device discovery and does not infer radar geometry from array shapes. Capture
parameters and physical conventions remain explicit inputs.

## ADC decoding

~~~rust
use mmwcore::{AdcComplexLayout, AdcFrameSpec, decode_adc_i16};

let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::Group2IThenQ)
    .expect("valid ADC frame specification");
let cube = decode_adc_i16(&[1, 2, 3, 4], spec, false).expect("valid ADC payload");
assert_eq!(cube.shape(), [1, 1, 1, 2]);
~~~

## CFAR

~~~rust
use mmwcore::{Cfar1DConfig, CfarMode, detect_cfar_1d};

let power = [1.0, 1.0, 0.0, 20.0, 0.0, 5.0, 5.0];
let config = Cfar1DConfig::new(2, 0, 1.1, CfarMode::Ca, false, 0, 0)
    .expect("valid CFAR configuration");
let result = detect_cfar_1d(&power, config).expect("valid CFAR input");
assert_eq!(result.indices, [3]);
~~~

See the [repository README](https://github.com/AIoT-Laboratory/mmwcore) for Python pipelines,
capture examples, project scope, and development commands. Python bindings and higher-level
contracts are published as [`mmwcore`](https://pypi.org/project/mmwcore/) on PyPI.
