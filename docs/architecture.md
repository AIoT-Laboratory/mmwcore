# Architecture

## Fixed research chain

```text
IWR6843 ES2 + DCA1000
          |
       mmwcli
          |
  finite completed capture
          |
  read_capture -> write_take -> session.json + radar.mmwa [+ camera files]
                                      |
                        ADC windows -> RT/RPC -> OpenMMW model
                                      |
                              tracking baseline
```

mmwcore begins at completed files. It does not configure hardware, receive DCA packets, launch
camera processes, train models, manage checkpoints, or serve results.

## Ownership

- `crates/mmwcore` owns lossless `.mmwa` storage and deterministic numerical kernels.
- `crates/mmwcore-python` exposes those kernels as checked NumPy operations.
- `python/mmwcore` owns finite capture/take readers, physical contracts, DSP composition, and
  tracking baselines.
- `benchmarks` owns reproducible storage and DSP regression workloads.

The Python layer composes Rust kernels.

## Data boundary

`read_capture` accepts the fixed finite mmwcli take contract for IWR6843 ES2. `write_take` replaces
raw `adc.bin` with indexed, lossless `radar.mmwa` and publishes the OpenMMW take. `open_take` is the
normal dataset and inference entry point.

A take has one radar stream and at most one directly recorded camera stream. Camera timestamps are
delivery observations rather than exposure timestamps. OpenMMW owns the downstream pairing policy.

The archive preserves fields that change scientific meaning: ADC layout and dimensions, frame
count and period, waveform, TDM order, and exact logical bytes. Antenna geometry, calibration,
axes, units, and coordinate frames remain explicit in recipes and products. See the
[ADC archive format](adc-archive-format.md).

## Compute path

1. Decode raw `int16` ADC.
2. Apply the range FFT.
3. Map the TDM virtual array.
4. Apply the Doppler FFT and phase compensation.
5. Project dense Cartesian RT.
6. Optionally produce bounded sparse RPC.

OpenMMW chooses windows, labels, splits, tensor layouts, and neural networks. mmwcore supplies the
deterministic physical transformation beneath those choices.

## Quality boundary

Tracking remains a classical reference for learned temporal perception. Tests protect archive
round trips, tensor shapes and axes, numerical behavior, take semantics, and tracking results.
Benchmarks detect storage and DSP regressions on a fixed IWR6843 workload. Neither adds another
workflow or hardware path.
