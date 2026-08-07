from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pipeline import (  # noqa: E402
    SCHEMA,
    _BenchmarkWorkload,
    _little_endian_payload,
    _synthetic_frame,
    run_benchmarks,
)


def test_default_iwr6843_fixture_is_stable() -> None:
    workload = _BenchmarkWorkload(
        name="iwr6843_3tx4rx_256samples_128loops",
        num_adc_samples=256,
        num_loops=128,
    )
    frame = _synthetic_frame(workload.recipe)

    assert frame.size == 786_432
    assert frame.nbytes == 1_572_864
    assert (
        hashlib.sha256(_little_endian_payload(frame)).hexdigest()
        == "007ac7c62380a6a12f7b5f20ad88dda9557a60160bb3e7476bf7aaae2d648f66"
    )


def test_all_pipeline_benchmark_cases_emit_versioned_measurements() -> None:
    result = run_benchmarks(
        warmups=0,
        samples=1,
        stream_frames=2,
        workload=_BenchmarkWorkload(name="smoke", num_adc_samples=8, num_loops=4),
    )

    assert result["schema"] == SCHEMA
    revision = result["revision"]
    assert isinstance(revision, str)
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    workload = result["workload"]
    assert isinstance(workload, dict)
    assert workload["range_doppler_shape"] == [1, 4, 12, 5]
    assert workload["stream_frames"] == 2

    cases = result["cases"]
    assert isinstance(cases, list)
    assert [case["name"] for case in cases] == [
        "decode",
        "range_doppler",
        "adc_to_range_doppler",
        "stream_adc_to_rd",
    ]
    for case in cases:
        assert case["sample_count"] == 1
        assert len(case["samples_ns"]) == 1
        assert case["samples_ns"][0] > 0
        assert case["median_ns"] > 0
        assert case["mad_ns"] == 0
        assert case["frames_per_second"] > 0
        assert case["input_mib_per_second"] > 0
