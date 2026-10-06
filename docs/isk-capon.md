# ISK dynamic Capon frontend

`mmwcore.dsp.isk_capon_point_cloud` implements the complete supported default
dynamic chain, from ADC words to points, in native Rust. It is an opt-in research
frontend; existing RPC/Online processing is unchanged.

```python
from mmwcore.core import ADCDecodeSpec
from mmwcore.dsp import IskCaponSpec, isk_capon_point_cloud
from mmwcore.io import CompressedADCReader, open_take

take = open_take("path/to/take-001")
reader = CompressedADCReader.from_take(take)
capture = reader.capture
profile = capture.profile
spec = IskCaponSpec(
    range_resolution_m=profile.range_resolution_m,
    velocity_resolution_mps=profile.velocity_resolution_mps,
    doppler_bins=profile.num_chirps_per_tx,
)
frame = isk_capon_point_cloud(
    reader.read_frame(0), ADCDecodeSpec(capture.adc), spec,
    tx_order=capture.tx_order,
)
points = frame.points  # PointCloudFrame: x,y,z,velocity,snr
heatmap = frame.ra_power  # float32 (187, number_of_range_bins)
```

The example assumes a power-of-two loop count such as the actual 128-loop
capture. For 96 loops, choose FFT128 and multiply the unpadded velocity spacing
by 96/128. Range FFT size equals the power-of-two ADC sample count (>=64).
Exactly one complete frame is accepted; IQ decoding uses the capture contract.
TX order must explicitly be `(0, 1, 2)`, with four RX per TX. Other board
geometries, BPM, external nonunity calibration and alternate TI detector modes
are not represented by this API.

## Supported source contract

The algorithm selection is Radar Toolbox `4.00.00.05` IWR6843 ISK default
`CaponBF2D`, `detectionMethod=1`, `CFAR_LOW_BW` undefined, unity calibration.
It is a host mathematical implementation, **not HWA/C674 bit emulation** and
not an implementation of every Toolbox mode.

| Stage | Contract |
| --- | --- |
| Range | Symmetric Blackman window, full complex FFT, no fast-time DC subtraction |
| Clutter | Subtract each channel's mean across all measured loops |
| RA | Row `[0,1,2,3,8,9,10,11]`, `R=XXᴴ/N`, loading `.001*trace(R)/8` |
| RA spectrum | `1/(aᴴR_loaded⁻¹a)`, 187 direction-cosine samples, azimuth FOV70/step.75 |
| CFAR | Source raCAAll CASO, skips4/4 and2/2, windows8/12, guards4/8, thresholds5/8, neighbor/sidelobe fallback.4 |
| Capacity | First150 in source angle-major order, no power sort or deduplication |
| Elevation | Full12 channels, loading.03, 27 samples, FOV20/step1.5, one discrete peak |
| Interpolation | Three-point power centroid; a boundary peak is counted twice |
| Doppler | Discrete-peak `w=R_loaded⁻¹a`, unnormalized `wᴴX`, no slow-time window, padded FFT argmax |
| Output | Sensor forward/right/up in metres; approaching-negative/receding-positive m/s; linear RA power/noise SNR |

First range-bin initialization can test bin4 four times with partially
accumulated references. Angular edge references are asymmetric. Second-pass
acceptance is angle-CFAR **OR** strict neighboring angular peak above the
sidelobe fraction, after mandatory range CFAR. These source quirks are retained
and tested, not silently replaced with generic separable CFAR.

Native calculations use complex64 input, complex128 covariance/Cholesky and
beamforming, and float32 RA/CFAR power. Cholesky normalization is only for
numerical conditioning; the returned inverse and power retain their scale.
Zero covariance yields zero heatmap. An accepted candidate with zero noise is
dropped with `zero_noise_dropped`, instead of inventing a noise floor or
passing infinite SNR. Invalid/nonfinite input fails explicitly.

Differences from embedded arithmetic include float Blackman coefficients,
range-FFT scaling and saturation, mean-removal int16 packing, DSP reciprocal
approximations, steering rounding, and SNR x8/int16 quantization. Floating SNR
is intentional and recorded in point metadata. No dynamic TDM phase correction
is inserted into the audited source path. This differs from the existing
compensated RD frontend; fixed channel calibration is not a substitute.

## Geometry, diagnostics and reuse

RA row index `i` represents TI `nu=-sin(70°)+i*sin(70°)*.75/70`. TI-positive
nu points project-left. Elevation grid `mu=-sin(20°)+k*sin(20°)*1.5/20` has
27 samples, ending at about+18.96°, not+20°. Steering retains the source's
horizontal factor `.9813`. Project output direction is
`[sqrt(1-nu²-mu²), -nu, mu]`, with interpolated mu; the factor is not applied
again to XYZ. No mount rotation or ROI filtering is performed here.

Source Nyquist uses the positive endpoint: bin `N/2` reports `+N/2`, with bins
strictly above it wrapped negative. This is the ambiguous aliased endpoint,
not a reversal of the physical Doppler contract.

`CaponFrame` retains every emitted `CaponDetection`, RA power, the point cloud,
and counters for all range candidates, accepted candidates before capacity,
capacity drops, nonvisible drops, zero-noise drops and zero-covariance ranges.
The point cloud preserves frame/time/source, records units and frontend
metadata, and can be passed to the existing tracker without selecting IDs.

`isk_capon_from_range_cube(cube, spec)` provides the same native path for one
`(frame, loop, virtual_rx, range_bin)` cube. Its caller owns the uncompensated
TX0/1/2, unity-calibration provenance. An RD cube cannot be substituted.

The defaults are a source comparison, not a claim that ±20° elevation or TI's
point density matches a particular installation and tracker profile. The
frozen RPC tracker was configured for sparse RPC points; changing frontend
support and SNR changes its input statistics even with identical tracking
parameters. Validate the physical field of view and independent measurement
support before considering a new operational baseline.

## Validation

Rust checks cover complex inverse conjugation/loading, CFAR edge noise,
duplicate initial tests, capacity and ordering. Independent NumPy tests cover
the entire range-cube chain, Blackman FFT, known range/angle/Doppler from raw
int16 ADC, approaching/receding signs, positive Nyquist, zero padding, static
annihilation, zero-noise handling and invalid contracts. No extra numerical
dependency is introduced. Real-take evaluation and theory are owned by
OpenMMW, not this library's unit suite.
