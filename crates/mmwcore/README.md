# mmwcore for Rust

mmwcore provides physically explicit primitives for raw ADC decoding, radar
cube transforms, FFTs, CFAR, calibrated point projection, Cartesian RT/RPC
processing, clustering, assignment, tracking, and vital-sign phase operations.

~~~rust
use mmwcore::{AdcComplexLayout, AdcFrameSpec, decode_adc_i16};

let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::Group2IThenQ)?;
let cube = decode_adc_i16(&[1, 2, 3, 4], spec, false)?;
assert_eq!(cube.shape(), [1, 1, 1, 2]);
# Ok::<(), mmwcore::AdcDecodeError>(())
~~~

The crate performs no device discovery and does not infer radar geometry from
array shapes. Python bindings and higher-level contracts are published through
the mmwcore package on PyPI.
