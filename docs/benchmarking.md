# Benchmarking

`benchmarks/pipeline.py` measures the maintained IWR6843 ADC-to-RD and archive-read paths. It is
a repository tool, not a package command or workflow framework.

The deterministic fixture uses 3 transmitters, 4 receivers, 256 ADC samples, 128 loops, Tx order
`(0, 2, 1)`, and `group2_i_then_q` layout. Each frame is 1,572,864 bytes. Results record the frame
SHA-256 so separate runs can confirm identical input.

Run a quick regression check:

```console
uv run --python 3.12 python benchmarks/pipeline.py \
  --warmups 0 --samples 1 --stream-frames 2 --output benchmark.json
```

Run a more stable local measurement:

```console
uv run --python 3.12 python benchmarks/pipeline.py \
  --warmups 2 --samples 10 --stream-frames 100 --output benchmark.json
```

The cases isolate four costs:

- `decode`: raw `int16` to a complex ADC cube.
- `range_doppler`: decoded cube through range FFT, TDM mapping, Doppler FFT, and compensation.
- `adc_to_range_doppler`: raw ADC through the complete RD recipe.
- `stream_adc_to_rd`: finite file reads plus complete RD processing, one frame at a time.

For a recorded take, use the same runner with `--take`. The verified take CFG supplies IQ layout,
Tx order and waveform; the archive's logical SHA-256 identifies the input. `--stream-frames`
selects a prefix and must not exceed the recording length.

```console
uv run --python 3.12 python benchmarks/pipeline.py \
  --take /path/to/your/verified-take \
  --warmups 1 --samples 5 --stream-frames 32 --output recorded-benchmark.json
```

Recorded mode compares `archive_frames` with `archive_chunks`, then the corresponding
`archive_frames_to_rd` and `archive_chunks_to_rd` paths. Chunk batches still verify decoded
SHA-256, and DSP remains frame-by-frame. Timing excludes opening/verifying the take metadata,
but includes each read's decompression, hash check, Python frame construction and optional RD.
It does not include RPC, tracking, camera, pose inference or rendering. Divide `median_ns` by
`frames_per_sample` for per-frame latency; OS file-cache state is stated in every case.

Chunk-batched decoding avoids decoding the same restart group for each constituent frame.
Measure the resulting performance on your own workload; the synthetic runner is the
reproducible public baseline, and recorded mode uses caller-supplied data.

`CompressedADCReader.iter_frames(start, stop)` uses bounded, chunk-aligned `read_frames` calls;
it keeps frame IDs/timestamps and permits early close. Retained frames remain valid after the
iterator advances. Random `read_frame` calls retain their existing behavior and no hidden cache.
FFT scratch is reused within a cube, within each Cartesian projection worker, or across Capon
candidates; it is not shared across concurrent calls. Zero padding is cleared between candidates.

Warm-ups are excluded from samples. The `mmwcore.benchmark.v1` result records raw durations,
median, median absolute deviation, throughput, environment versions, source revision, workload,
and thread settings.

CI proves only that the benchmark executes and emits its contract. Compare performance only when
workload, Python, NumPy, mmwcore build, operating system, architecture, thread settings, and cache
mode match. Run serious comparisons on the same local machine; shared CI timing is not a gate.

Keep the benchmark fixed unless the maintained IWR6843 processing contract changes. Add a focused case
only when it protects a real storage, RT/RPC, or tracking regression; do not turn the runner into a
general benchmark framework.
