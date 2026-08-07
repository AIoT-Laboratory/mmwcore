# Benchmarking

`benchmarks/pipeline.py` measures the maintained ADC-to-range-Doppler path without adding a
package command or public API. It generates its input at run time and does not depend on the
repository-local `adc_data.bin` capture.

The fixed workload represents an IWR6843 TDM capture with 3 transmitters, 4 receivers, 256 ADC
samples, 128 loops, Tx order `(0, 2, 1)`, and `group2_i_then_q` raw layout. One generated frame is
1,572,864 bytes. The signed ADC words use the documented `lcg16_index_v1` integer formula, and the
result records the frame SHA-256 so separate runs can verify identical input.

Run the default suite from a release-profile development environment:

```console
uv sync --extra dev --locked
uv run python benchmarks/pipeline.py --output benchmark.json
```

The four cases have deliberately separate scopes:

- `decode`: one raw int16 frame to a complex64 ADC cube.
- `range_doppler`: one decoded cube through range FFT, TDM mapping, Doppler FFT, and compensation.
- `adc_to_range_doppler`: raw int16 through decode and the complete RD recipe.
- `stream_adc_to_rd`: frame-by-frame memmap reads and complete RD processing, discarding each
  product before reading the next frame.

Warm-up runs are never included in `samples_ns`. The runner reports each raw duration, median,
median absolute deviation, frames per second, input MiB/s, environment versions, source revision,
and workload contract using the `mmwcore.benchmark.v1` JSON schema. Stream fixture creation and
reader construction are outside the timed region. The stream result declares whether a complete
warm-up established a warm page-cache measurement.

For a longer streaming run:

```console
uv run python benchmarks/pipeline.py --warmups 2 --samples 10 --stream-frames 1000 --output benchmark.json
```

At this shape, 1000 frames require about 1.465 GiB of temporary disk and every warm-up or sample
processes the full file. The temporary fixture is removed after the run.

Compare results only when schema, workload, Python, NumPy, mmwcore build, operating system,
architecture, thread settings, and cache mode match. Shared CI hosts are suitable for smoke runs
and artifacts, not performance gates. Set NumPy-related thread environment variables consistently
before starting Python; the maintained Rust FFT path is single-threaded.

Peak RSS is intentionally absent from v1. Python allocation tracing omits some NumPy and native
Rust allocations, while dependency-free process RSS is not sufficiently consistent across the
supported operating systems. No OpenRadar comparison or relative performance claim is implied.
