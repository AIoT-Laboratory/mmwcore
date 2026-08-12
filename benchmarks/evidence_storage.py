"""Benchmark strictly reversible chunk storage for offline radar ADC evidence.

This repository-local tool measures storage and read behaviour only. It does
not define an archive format, change capture output, or expose a mmwcore API.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from benchmarks.evidence_archive import (
    ChunkRecord,
    read_frames,
    read_source_frames,
    replay_and_verify,
    write_payload,
)
from benchmarks.evidence_codecs import DEFAULT_ZLIB_LEVEL
from benchmarks.evidence_inputs import (
    EvidenceCase,
    discover_sources,
    source_selection,
    validate_options,
)
from benchmarks.evidence_report import build_report

SCHEMA = "mmwcore.evidence_storage_benchmark.v2"
DEFAULT_FILENAME = "adc_data_Raw_0.bin"
DEFAULT_CASES = (
    EvidenceCase(codec="raw", chunk_frames=1),
    EvidenceCase(codec="shuffle-zlib", chunk_frames=1),
    EvidenceCase(codec="shuffle-zlib", chunk_frames=4),
    EvidenceCase(codec="adaptive-shuffle-zlib", chunk_frames=4),
)


def run_benchmark(
    inputs: Sequence[Path],
    *,
    frame_bytes: int,
    filename: str = DEFAULT_FILENAME,
    cases: Sequence[EvidenceCase] = DEFAULT_CASES,
    start_frame: int = 0,
    max_frames: int | None = None,
    random_windows: int = 32,
    window_frames: int = 8,
    seed: int = 0,
    zlib_level: int = DEFAULT_ZLIB_LEVEL,
    scratch_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run the streaming benchmark and return a JSON-compatible record."""

    validate_options(
        inputs,
        frame_bytes=frame_bytes,
        cases=cases,
        start_frame=start_frame,
        max_frames=max_frames,
        random_windows=random_windows,
        window_frames=window_frames,
        zlib_level=zlib_level,
        scratch_dir=scratch_dir,
    )
    source_paths = discover_sources(inputs, filename=filename)
    case_total = len(source_paths) * len(cases)
    sources = []
    for source_index, source in enumerate(source_paths):
        sources.append(
            _benchmark_source(
                source,
                scratch_dir=scratch_dir,
                start_frame=start_frame,
                max_frames=max_frames,
                frame_bytes=frame_bytes,
                case_specs=cases,
                random_windows=random_windows,
                window_frames=window_frames,
                seed=seed,
                zlib_level=zlib_level,
                case_offset=source_index * len(cases),
                case_total=case_total,
                progress=progress,
            )
        )
    return build_report(
        schema=SCHEMA,
        sources=sources,
        benchmark_parameters={
            "filename": filename,
            "frame_bytes": frame_bytes,
            "cases": [case.as_record() for case in cases],
            "start_frame": start_frame,
            "max_frames": max_frames,
            "random_windows": random_windows,
            "window_frames": window_frames,
            "random_seed": seed,
            "zlib_level": zlib_level,
        },
    )


def _benchmark_source(
    source: Path,
    *,
    scratch_dir: Path | None,
    start_frame: int,
    max_frames: int | None,
    frame_bytes: int,
    case_specs: Sequence[EvidenceCase],
    random_windows: int,
    window_frames: int,
    seed: int,
    zlib_level: int,
    case_offset: int,
    case_total: int,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    total_frames, selected_frames = source_selection(
        source,
        frame_bytes=frame_bytes,
        start_frame=start_frame,
        max_frames=max_frames,
    )
    if window_frames > selected_frames:
        raise ValueError("Window frames must not exceed the selected source frames.")
    cases: list[dict[str, object]] = []
    logical_sha256: str | None = None
    for local_index, case_spec in enumerate(case_specs):
        case_index = case_offset + local_index + 1
        if progress is not None:
            progress(
                f"[{case_index}/{case_total}] start {source} codec={case_spec.codec} "
                f"chunk_frames={case_spec.chunk_frames}"
            )
        with tempfile.TemporaryDirectory(
            prefix="mmwcore-evidence-storage-",
            dir=scratch_dir,
        ) as temporary_directory:
            case_hash, case = _run_case(
                source,
                scratch_directory=Path(temporary_directory),
                start_frame=start_frame,
                selected_frames=selected_frames,
                frame_bytes=frame_bytes,
                chunk_frames=case_spec.chunk_frames,
                codec=case_spec.codec,
                zlib_level=zlib_level,
                random_windows=random_windows,
                window_frames=window_frames,
                seed=seed,
            )
        if logical_sha256 is None:
            logical_sha256 = case_hash
        elif case_hash != logical_sha256:
            raise RuntimeError("Evidence logical source hash changed between benchmark cases.")
        cases.append(case)
        if progress is not None:
            progress(
                f"[{case_index}/{case_total}] done {source} codec={case_spec.codec} "
                f"chunk_frames={case_spec.chunk_frames} payload_ratio={case['payload_ratio']:.4f}"
            )
    return {
        "path": str(source),
        "selection": {
            "total_frames": total_frames,
            "start_frame": start_frame,
            "selected_frames": selected_frames,
            "frame_bytes": frame_bytes,
            "logical_bytes": selected_frames * frame_bytes,
            "logical_sha256": logical_sha256,
        },
        "cases": cases,
    }


def _run_case(
    source: Path,
    *,
    scratch_directory: Path,
    start_frame: int,
    selected_frames: int,
    frame_bytes: int,
    chunk_frames: int,
    codec: str,
    zlib_level: int,
    random_windows: int,
    window_frames: int,
    seed: int,
) -> tuple[str, dict[str, object]]:
    payload_path = scratch_directory / "payload.bin"
    packed = write_payload(
        source,
        destination=payload_path,
        start_frame=start_frame,
        selected_frames=selected_frames,
        frame_bytes=frame_bytes,
        chunk_frames=chunk_frames,
        codec=codec,
        zlib_level=zlib_level,
    )
    replay_ns, decode_ns = replay_and_verify(
        source,
        payload_path=payload_path,
        records=packed.records,
        start_frame=start_frame,
        frame_bytes=frame_bytes,
        codec=codec,
    )
    window_starts = _random_window_starts(
        selected_frames=selected_frames,
        random_windows=random_windows,
        window_frames=window_frames,
        seed=seed,
    )
    trusted_durations, chunks_per_window = _measure_random_windows(
        source,
        payload_path=payload_path,
        records=packed.records,
        source_start_frame=start_frame,
        frame_bytes=frame_bytes,
        codec=codec,
        window_starts=window_starts,
        window_frames=window_frames,
        verify_digest=False,
    )
    verified_durations, verified_chunks_per_window = _measure_random_windows(
        source,
        payload_path=payload_path,
        records=packed.records,
        source_start_frame=start_frame,
        frame_bytes=frame_bytes,
        codec=codec,
        window_starts=window_starts,
        window_frames=window_frames,
        verify_digest=True,
    )
    if chunks_per_window != verified_chunks_per_window:
        raise RuntimeError("Random-window read modes decoded different chunk sets.")
    raw_bytes = selected_frames * frame_bytes
    stored_bytes = payload_path.stat().st_size
    return packed.logical_sha256, {
        "codec": codec,
        "chunk_frames": chunk_frames,
        "chunk_count": len(packed.records),
        "raw_bytes": raw_bytes,
        "payload_bytes": stored_bytes,
        "payload_ratio": stored_bytes / raw_bytes,
        "payload_reduction_fraction": 1.0 - stored_bytes / raw_bytes,
        "pack_mib_per_second": _mib_per_second(raw_bytes, packed.pack_ns),
        "encode_mib_per_second": (
            None if codec == "raw" else _mib_per_second(raw_bytes, packed.encode_ns)
        ),
        "sequential_replay_mib_per_second": _mib_per_second(raw_bytes, replay_ns),
        "decode_mib_per_second": None if codec == "raw" else _mib_per_second(raw_bytes, decode_ns),
        "throughput_scope": {
            "pack": "source_read_hash_transform_payload_write",
            "encode": (
                "not_applicable_no_codec"
                if codec == "raw"
                else "reversible_transform_and_codec_only"
            ),
            "sequential_replay": "payload_read_decode_digest_source_compare",
            "decode": "not_applicable_no_codec" if codec == "raw" else "codec_only",
        },
        "random_window": {
            "count": random_windows,
            "frames": window_frames,
            "average_chunks": (statistics.fmean(chunks_per_window) if chunks_per_window else 0.0),
            "cache_mode": "warm_after_sequential_replay",
            "mode_order": ["trusted", "verified"],
            "trusted": _random_window_metrics(
                trusted_durations,
                scope="payload_seek_read_decode_slice",
            ),
            "verified": _random_window_metrics(
                verified_durations,
                scope="payload_seek_read_decode_chunk_digest_slice",
            ),
        },
        "selected_transform_counts": dict(Counter(packed.selected_transforms)),
        "roundtrip_verified": True,
    }


def _measure_random_windows(
    source: Path,
    *,
    payload_path: Path,
    records: Sequence[ChunkRecord],
    source_start_frame: int,
    frame_bytes: int,
    codec: str,
    window_starts: Sequence[int],
    window_frames: int,
    verify_digest: bool,
) -> tuple[list[int], list[int]]:
    first_frames = tuple(record.first_frame for record in records)
    durations: list[int] = []
    chunks_per_window: list[int] = []
    with payload_path.open("rb", buffering=1024 * 1024) as payload:
        for first in window_starts:
            started = time.perf_counter_ns()
            actual, decoded_chunks = read_frames(
                payload,
                records=records,
                record_first_frames=first_frames,
                start_frame=first,
                stop_frame=first + window_frames,
                frame_bytes=frame_bytes,
                codec=codec,
                verify_digest=verify_digest,
            )
            durations.append(time.perf_counter_ns() - started)
            expected = read_source_frames(
                source,
                absolute_first_frame=source_start_frame + first,
                frame_count=window_frames,
                frame_bytes=frame_bytes,
            )
            if actual != expected:
                raise RuntimeError("Random evidence window differs from the source ADC bytes.")
            chunks_per_window.append(decoded_chunks)
    return durations, chunks_per_window


def _random_window_starts(
    *,
    selected_frames: int,
    random_windows: int,
    window_frames: int,
    seed: int,
) -> tuple[int, ...]:
    generator = np.random.default_rng(seed)
    return tuple(
        int(generator.integers(0, selected_frames - window_frames + 1))
        for _ in range(random_windows)
    )


def _random_window_metrics(durations: Sequence[int], *, scope: str) -> dict[str, object]:
    total_ns = sum(durations)
    return {
        "scope": scope,
        "p50_ns": _percentile(durations, 50.0),
        "p95_ns": _percentile(durations, 95.0),
        "windows_per_second": (len(durations) * 1_000_000_000.0 / total_ns if total_ns else 0.0),
    }


def _mib_per_second(raw_bytes: int, elapsed_ns: int) -> float:
    return raw_bytes * 1_000_000_000.0 / (1024 * 1024 * elapsed_ns) if elapsed_ns else 0.0


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
