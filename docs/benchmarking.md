# Benchmarking

`benchmarks/pipeline.py` is the local quality gate for the maintained IWR6843 ADC-to-RD path. It is
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

Warm-ups are excluded from samples. The `mmwcore.benchmark.v1` result records raw durations,
median, median absolute deviation, throughput, environment versions, source revision, workload,
and thread settings.

CI proves only that the benchmark executes and emits its contract. Compare performance only when
workload, Python, NumPy, mmwcore build, operating system, architecture, thread settings, and cache
mode match. Run serious comparisons on the same local machine; shared CI timing is not a gate.

Keep the benchmark fixed unless the maintained IWR6843 research chain changes. Add a focused case
only when it protects a real storage, RT/RPC, or tracking regression; do not turn the runner into a
general benchmark framework.
