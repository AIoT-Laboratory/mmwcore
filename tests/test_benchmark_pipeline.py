from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCDecodeSpec, ADCFrameSpec, RangeDopplerPipeline
from mmwcore.io import ADCArchiveReader, write_adc_archive

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pipeline import (  # noqa: E402
    SCHEMA,
    _archive_cases,
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


def test_archive_benchmark_exercises_both_read_paths(tmp_path):
    capture = RadarCaptureSpec(
        profile=RadarProfile(num_tx=1, num_rx=1, num_adc_samples=8, num_chirps_per_tx=4),
        adc=ADCFrameSpec(num_chirps=4, num_rx=1, num_samples=8),
        tx_order=(0,),
        frame_periodicity_s=0.1,
        num_frames=6,
    )
    source, archive = tmp_path / "adc.bin", tmp_path / "adc.mmwa"
    source.write_bytes(np.arange(capture.adc.raw_values_per_frame * 6, dtype="<i2").tobytes())
    write_adc_archive(source, archive, capture)
    cases = _archive_cases(
        ADCArchiveReader(archive),
        RangeDopplerPipeline(decode=ADCDecodeSpec(capture.adc)),
        warmups=0,
        samples=1,
        count=6,
    )
    assert [case["name"] for case in cases] == [
        "archive_frames",
        "archive_chunks",
        "archive_frames_to_rd",
        "archive_chunks_to_rd",
    ]
    assert all(case["frames_per_sample"] == 6 for case in cases)
    for case in cases:
        assert isinstance(case["median_ns"], float) and case["median_ns"] > 0
