# mmwcore for Rust

The Rust crate provides deterministic storage and compute kernels, also exposed through Python APIs:

- standardized ADC frame compression/decompression and indexed `.mmwa` files;
- raw ADC decoding;
- FFT, clutter removal, and TDM virtual-array transforms;
- Cartesian projection and sparsification;
- detection and clustering;
- multiscale scatter-component/body tracking, assignment, and comparison metrics.

Implementation follows three domains:

```text
src/io/adc_compression/   frame-group codec and standardized compressed ADC files
src/dsp/       ADC decoding, FFT, arrays, detection, and projection
src/tracking/  assignment, metrics, and the independent multiscale backend
```

Existing root imports and module paths remain available through re-exports. New code can use
the domain paths directly. `tracking::multiscale::ScatterBodyTracker` owns typed configuration,
state, and per-frame output. Its inputs are `[x, y, z, radial_velocity, snr_db]` in level-world
forward/right/up coordinates; `step(points, dt)` advances the causal state.

The TI GTRACK backend belongs to the separate `mmwcore-ti-gtrack` crate because it retains
TI's device-only license. It is not a dependency of this Apache-2.0 crate.

ADC operations are `compress_adc_frames` / `decompress_adc_frames` for byte payloads and
`compress_adc_file` / `decompress_adc_file` for self-describing files. `open_compressed_adc`
returns `CompressedAdcFile` for verified random access. See the
[compression API](https://github.com/AIoT-Laboratory/mmwcore/blob/main/docs/adc-compression.md).
Earlier archive-named imports remain compatibility aliases; the v3 wire bytes are unchanged.

It performs no hardware control, DCA packet reception, process launch, plotting, or experiment
management.

```rust
use mmwcore::{AdcComplexLayout, AdcFrameSpec, decode_adc_i16};

let spec = AdcFrameSpec::new(1, 1, 2, AdcComplexLayout::Group2IThenQ)
    .expect("valid ADC frame specification");
let cube = decode_adc_i16(&[1, 2, 3, 4], spec, false).expect("valid ADC payload");
assert_eq!(cube.shape(), [1, 1, 1, 2]);
```

See the [repository README](https://github.com/AIoT-Laboratory/mmwcore) for Python examples
and validation commands.
