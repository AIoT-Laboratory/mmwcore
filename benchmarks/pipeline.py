"""Reproducible synthetic IWR6843 pipeline benchmarks.

This repository-local runner is intentionally separate from the installed
``mmwcore`` command and public Python API.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import mmwcore
from mmwcore.config import iwr6843_isk_range_doppler_pipeline, iwr6843_profile
from mmwcore.core import ADCComplexLayout, ADCFrame, RadarCube, RangeDopplerPipeline
from mmwcore.dsp import (
    compensate_tdm_doppler_phase,
    doppler_fft,
    map_tdm_virtual_array,
    organize_adc_samples,
    range_doppler,
    range_fft,
)
from mmwcore.io import ADCFileReader

SCHEMA = "mmwcore.benchmark.v1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class _BenchmarkWorkload:
    name: str
    num_adc_samples: int
    num_loops: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Benchmark workload name must not be empty.")
        if self.num_adc_samples <= 0 or self.num_adc_samples % 2:
            raise ValueError("Benchmark ADC sample count must be positive and even.")
        if self.num_loops <= 0:
            raise ValueError("Benchmark loop count must be positive.")

    @property
    def recipe(self) -> RangeDopplerPipeline:
        profile = iwr6843_profile(
            num_adc_samples=self.num_adc_samples,
            num_chirps_per_tx=self.num_loops,
        )
        return iwr6843_isk_range_doppler_pipeline(
            profile,
            adc_layout=ADCComplexLayout.GROUP2_I_THEN_Q,
        )


_DEFAULT_WORKLOAD = _BenchmarkWorkload(
    name="iwr6843_3tx4rx_256samples_128loops",
    num_adc_samples=256,
    num_loops=128,
)


def _synthetic_frame(recipe: RangeDopplerPipeline) -> np.ndarray:
    """Return stable signed ADC words without relying on NumPy's RNG stream."""

    count = recipe.decode.adc.raw_values_per_frame
    indices = np.arange(count, dtype=np.uint64)
    words = ((indices * 25_173 + 13_849) & 0xFFFF).astype(np.uint16)
    return np.ascontiguousarray(words.view(np.int16))


def _little_endian_payload(frame: np.ndarray) -> bytes:
    return frame.astype(np.dtype("<i2"), copy=False).tobytes()


def _range_doppler_from_decoded(
    decoded: RadarCube,
    recipe: RangeDopplerPipeline,
) -> RadarCube:
    """Run the fixed benchmark RD stages without repeating ADC decoding."""

    if recipe.remove_static_clutter or recipe.channel_calibration is not None:
        raise ValueError("The fixed benchmark recipe must not enable optional preprocessing.")
    tdm = recipe.tdm_virtual_array
    if tdm is None:
        raise ValueError("The fixed benchmark recipe requires an explicit TDM virtual array.")

    range_cube = range_fft(decoded, recipe.range_fft)
    virtual_cube = map_tdm_virtual_array(range_cube, tdm)
    doppler_cube = doppler_fft(virtual_cube, recipe.doppler_fft)
    return compensate_tdm_doppler_phase(
        doppler_cube,
        tdm,
        fftshift=recipe.doppler_fft.fftshift,
    )


def _measure_case(
    *,
    name: str,
    scope: str,
    input_kind: str,
    operation: Callable[[], object],
    warmups: int,
    samples: int,
    frames_per_sample: int,
    input_bytes_per_frame: int,
    cache_mode: str | None = None,
) -> dict[str, object]:
    for _ in range(warmups):
        operation()

    gc.collect()
    gc_was_enabled = gc.isenabled()
    durations: list[int] = []
    try:
        gc.disable()
        for _ in range(samples):
            started = time.perf_counter_ns()
            operation()
            durations.append(time.perf_counter_ns() - started)
    finally:
        if gc_was_enabled:
            gc.enable()

    median_ns = float(statistics.median(durations))
    mad_ns = float(statistics.median(abs(value - median_ns) for value in durations))
    median_seconds = median_ns / 1_000_000_000.0
    result: dict[str, object] = {
        "name": name,
        "scope": scope,
        "input_kind": input_kind,
        "warmup_iterations": warmups,
        "sample_count": samples,
        "frames_per_sample": frames_per_sample,
        "input_bytes_per_frame": input_bytes_per_frame,
        "samples_ns": durations,
        "median_ns": median_ns,
        "mad_ns": mad_ns,
        "frames_per_second": frames_per_sample / median_seconds,
        "input_mib_per_second": (
            input_bytes_per_frame * frames_per_sample / (1024 * 1024) / median_seconds
        ),
    }
    if cache_mode is not None:
        result["cache_mode"] = cache_mode
    return result


def _write_stream_fixture(path: Path, payload: bytes, frames: int) -> None:
    with path.open("wb", buffering=1024 * 1024) as destination:
        for _ in range(frames):
            destination.write(payload)


def _stream_operation(
    reader: ADCFileReader,
    recipe: RangeDopplerPipeline,
) -> Callable[[], float]:
    def process_all_frames() -> float:
        checksum = 0.0
        for index in range(reader.num_frames):
            product = range_doppler(reader.read_frame(index), recipe)
            checksum += float(product.data[0, 0, 0, 0].real)
        return checksum

    return process_all_frames


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "mmwcore": mmwcore.__version__,
        "numpy": np.__version__,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "byte_order": sys.byteorder,
        "thread_environment": {
            name: os.environ[name] for name in _THREAD_ENVIRONMENT if name in os.environ
        },
    }


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _validate_counts(*, warmups: int, samples: int, stream_frames: int) -> None:
    if warmups < 0:
        raise ValueError("Benchmark warmups must be non-negative.")
    if samples <= 0:
        raise ValueError("Benchmark sample count must be positive.")
    if stream_frames <= 0:
        raise ValueError("Benchmark stream frame count must be positive.")


def run_benchmarks(
    *,
    warmups: int,
    samples: int,
    stream_frames: int,
    workload: _BenchmarkWorkload = _DEFAULT_WORKLOAD,
) -> dict[str, object]:
    """Run the repository-local suite and return its JSON-compatible record."""

    _validate_counts(warmups=warmups, samples=samples, stream_frames=stream_frames)
    recipe = workload.recipe
    adc_spec = recipe.decode.adc
    frame = _synthetic_frame(recipe)
    payload = _little_endian_payload(frame)
    raw = ADCFrame(frame, frame_id="synthetic-iwr6843", source="generated")

    decoded = organize_adc_samples(raw, adc_spec)
    rd_only = _range_doppler_from_decoded(decoded, recipe)
    end_to_end = range_doppler(raw, recipe)
    expected_shape = (1, workload.num_loops, 12, workload.num_adc_samples // 2 + 1)
    if rd_only.data.shape != expected_shape or rd_only.data.dtype != np.complex64:
        raise RuntimeError("Benchmark range-Doppler output contract changed unexpectedly.")
    np.testing.assert_array_equal(rd_only.data, end_to_end.data)

    raw_bytes = len(payload)
    cases = [
        _measure_case(
            name="decode",
            scope="one raw int16 frame to a complex64 ADC cube",
            input_kind="raw_adc_int16",
            operation=lambda: organize_adc_samples(raw, adc_spec),
            warmups=warmups,
            samples=samples,
            frames_per_sample=1,
            input_bytes_per_frame=raw_bytes,
        ),
        _measure_case(
            name="range_doppler",
            scope="decoded cube through range FFT, TDM mapping, Doppler FFT, and compensation",
            input_kind="decoded_adc_complex64",
            operation=lambda: _range_doppler_from_decoded(decoded, recipe),
            warmups=warmups,
            samples=samples,
            frames_per_sample=1,
            input_bytes_per_frame=decoded.data.nbytes,
        ),
        _measure_case(
            name="adc_to_range_doppler",
            scope="one raw int16 frame through decode and the complete range-Doppler recipe",
            input_kind="raw_adc_int16",
            operation=lambda: range_doppler(raw, recipe),
            warmups=warmups,
            samples=samples,
            frames_per_sample=1,
            input_bytes_per_frame=raw_bytes,
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="mmwcore-benchmark-") as temporary_directory:
        stream_path = Path(temporary_directory) / "adc.bin"
        _write_stream_fixture(stream_path, payload, stream_frames)
        reader = ADCFileReader(stream_path, adc_spec)
        cases.append(
            _measure_case(
                name="stream_adc_to_rd",
                scope="frame-by-frame memmap reads through the complete range-Doppler recipe",
                input_kind="adc_file_int16",
                operation=_stream_operation(reader, recipe),
                warmups=warmups,
                samples=samples,
                frames_per_sample=stream_frames,
                input_bytes_per_frame=raw_bytes,
                cache_mode=(
                    "warm_after_full_read_warmup" if warmups else "uncontrolled_os_page_cache"
                ),
            )
        )

    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "revision": _git_revision(),
        "environment": _environment(),
        "workload": {
            "name": workload.name,
            "num_tx": 3,
            "num_rx": 4,
            "num_adc_samples": workload.num_adc_samples,
            "num_loops": workload.num_loops,
            "num_chirps": workload.num_loops * 3,
            "tx_order": [0, 2, 1],
            "adc_layout": ADCComplexLayout.GROUP2_I_THEN_Q.value,
            "raw_values_per_frame": adc_spec.raw_values_per_frame,
            "raw_bytes_per_frame": raw_bytes,
            "range_doppler_shape": list(expected_shape),
            "stream_frames": stream_frames,
            "fixture_generator": "lcg16_index_v1",
            "fixture_frame_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "cases": cases,
    }


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=_non_negative_integer, default=1)
    parser.add_argument("--samples", type=_positive_integer, default=5)
    parser.add_argument("--stream-frames", type=_positive_integer, default=16)
    parser.add_argument("--output", type=Path, help="Write the versioned JSON result to this file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmarks(
        warmups=args.warmups,
        samples=args.samples,
        stream_frames=args.stream_frames,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
