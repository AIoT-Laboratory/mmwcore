# Architecture

## Research chains

```text
finite:
IWR6843 + DCA1000 -> mmwcli.take.v3 -> read_capture -> write_take + context
                  -> openmmw.take.v4 -> RT/RPC -> OpenMMW

online:
IWR6843 + DCA1000 -> mmwcli stream -> OpenMMW -> mmwcore DSP
                  -> RPC/RT checkpoint -> Web
```

mmwcore owns neither acquisition process. It does not configure hardware, receive DCA packets,
launch camera processes, train models, manage checkpoints, or serve results. OpenMMW imports its
DSP for both archived and in-memory ADC frames.

## Ownership

- `crates/mmwcore/src/io` owns lossless `.mmwa` storage;
  `src/dsp` owns deterministic numerical kernels; `src/tracking` owns assignment, metrics,
  and the independent multiscale scatter-component/body tracker. Old Rust module paths and
  root imports are preserved through re-exports.
- `crates/mmwcore-ti-gtrack` owns the complete Rust TI GTRACK 3DA implementation.
- `crates/mmwcore-python` exposes those kernels as checked NumPy operations. Its module entry
  point registers bindings; `boundary/` groups array conversion and configuration adapters by
  domain, separate from the processing bindings.
- `python/mmwcore` owns finite capture/take readers, physical contracts, DSP composition, and
  tracking adapters. `ScatterBodyTracker.step_points` executes the Rust multiscale backend and
  restores existing Python diagnostic attributes; its point rows, IDs, histories, and body
  reports retain their contracts. The earlier Python ablation classes remain explicit comparison
  implementations. Multiscale and TI GTRACK are the two maintained backends.
- `benchmarks` owns reproducible storage and DSP regression workloads.

The Python layer composes Rust kernels.

## Data boundary

`read_capture` accepts the fixed finite `mmwcli.take.v3` raw capture for IWR6843 ES2. `write_take`
replaces `adc.bin` with indexed, lossless `radar.mmwa` and publishes `openmmw.take.v4` when a
research context is supplied. `open_take`
is the normal dataset and finite-inference entry point.

A take has one radar stream and at most one directly recorded camera stream. Camera timestamps are
delivery observations rather than exposure timestamps. OpenMMW owns the downstream pairing policy.

Each raw and verified take references an immutable `mmwcli.snapshot.v1` `setup.json` by path, size,
and SHA-256. `write_take` copies those bytes unchanged. It also hashes `context.json` into v4 while
remaining able to read legacy v3 takes. The snapshot is the sole mount source and
requires downward boresight pitch `0`, `30`, or `90` degrees. Cartesian projection maps its level
grid into sensor coordinates while building the fixed sampling plan and emits the canonical
`level_forward_lateral_up` frame directly.

The archive preserves ADC layout and dimensions, frame count and period, waveform, TDM order, and
exact logical bytes. Antenna geometry, calibration, axes, units, and coordinate frames remain
explicit in recipes and products. See the [ADC archive format](adc-archive-format.md).

## Compute path

1. Decode raw `int16` ADC.
2. Apply the range FFT.
3. Map the TDM virtual array.
4. Apply the Doppler FFT and phase compensation.
5. Project dense Cartesian RT.
6. Optionally produce bounded sparse RPC.

OpenMMW chooses windows, labels, splits, tensor layouts, and neural networks. mmwcore supplies the
deterministic physical transformation beneath those choices.

`RadarProfile` rejects sampling that extends beyond the chirp ramp. Physical range/velocity
resolution is distinct from FFT-bin spacing: pass the actual `range_n_fft` and `doppler_bins`
to `to_point_cloud_projection_spec` when changing FFT lengths. Range FFT length means the full
transform length, before one-sided slicing. Precomputed RD reuse checks axes, dimensions,
FFT conventions, TDM channel mapping/compensation and calibration coefficients against the recipe.
Clutter-subtracted and full RD may share the same detection recipe; unrelated provenance is ignored.

## Quality boundary

Tracking remains a classical reference for learned temporal perception. Tests protect archive
round trips, tensor shapes and axes, numerical behavior, take semantics, and tracking results.
Benchmarks detect storage and DSP regressions on a fixed IWR6843 workload. Neither adds another
workflow or hardware path.

Tracking evaluation gates pairs before assignment, maximizing the number of valid matches and
then minimizing their total distance. This prevents a shorter out-of-gate assignment from hiding
otherwise valid pairs. Reported errors still require independently supplied, frame-aligned truth.

### TI / multiscale tracking comparison

Compare **body-level IDs and horizontal XY positions**, in metres in the same world
forward/right/up frame. TI produces a 3D filtered state; multiscale produces XY position and
bulk XY velocity. Its display height is the observed scatter centroid, its displayed vertical
velocity is zero, and it has no equivalent state covariance. Do not score those display values
as 3D body estimates or fabricate covariance to construct a `TrackFrame`.

`evaluate_track_frames(..., dimensions=2)` accepts TI's world-transformed `TrackFrame` and
`TrackingPredictionFrame.from_multiscale(bodies, ...)` against independent
`TrackingGroundTruthFrame` labels. Use the body reports from `step_points`, not scatter-component
IDs. Pass matching `frame_id`, `timestamp` and `coordinate_frame`; mismatches are rejected even
for empty frames. If both sides omit timing/IDs, list-index alignment is the caller's contract.
The default `include_statuses=(CONFIRMED, COASTING)` scores published continuity; select only
`CONFIRMED` explicitly for that subset. A confirmed status does not prove a centroid measurement
update in the current frame. Summary records include dimensions, statuses and coordinate frame.

Both runs must use the same RPC frames, mount transform, scene ROI, person-count prior and
frame schedule. TI advances by its configured frame period; pass that same period to multiscale,
including empty observations for missing frames, rather than host receipt-time jitter. For a
shared RPC comparison disable TI's supplemental static support; evaluate static support as a
separate observation condition. Different lifecycle and association rules remain backend
differences, not silently aligned settings.

`ScatterBodyTracker(max_bodies=person_count, max_components=...)` separates independent body
hypotheses from scatter history capacity. A body may contain several components only with the
existing split-origin evidence. The body limit includes tentative and coasting hypotheses:
existing bodies keep their slots until normal expiry, new roots use point count then SNR order,
and excess roots are removed from backend state and point associations. Independently supported
children are released, not forcibly kept in a body to satisfy the count. `max_components` is an
optional resource bound, not a person-count prior; leave enough capacity for split histories.
Neither limit changes the unlimited baseline's association or lifecycle rules.

## IQ and radial velocity

Radial velocity is `dr/dt`: approaching is negative, receding is positive. Correctly ordered
positive-slope FMCW ADC uses the same sign as the centered forward Doppler FFT bin. Swapped IQ
must be decoded correctly before the FFTs; negating only RPC velocity cannot repair it.

The fixed mmwcli IWR6843 configuration uses `iqSwapSel=1`, so its two-lane raw layout is
`GROUP2_Q_THEN_I` (`Q1 Q2 I1 I2`). New captures record that layout. Generic I-first decoders retain
their existing meaning. `ADCArchiveReader.from_take(take)` checks the verified take CFG and
corrects the historical I-first archive label only in the effective reader contract. It records
that correction in frame metadata and preserves every stored byte/hash. All other capture
mismatches are rejected; standalone archives continue to use their embedded contract.
