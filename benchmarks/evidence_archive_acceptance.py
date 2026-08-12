"""Acceptance measurements for the implemented offline evidence archive."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchmarks.evidence_inputs import discover_sources, source_selection
from mmwcore.io import EvidenceArchive, open_evidence_archive, write_evidence_archive

SCHEMA = "mmwcore.evidence_archive_acceptance.v1"
DEFAULT_FILENAME = "adc_data_Raw_0.bin"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_archive_acceptance(
    inputs: Sequence[Path],
    *,
    frame_bytes: int,
    filename: str = DEFAULT_FILENAME,
    random_windows: int = 128,
    window_frames: int = 4,
    seed: int = 0,
    scratch_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Pack, reopen, verify, and sample the implemented archive for each source."""

    _validate_options(
        inputs,
        frame_bytes=frame_bytes,
        random_windows=random_windows,
        window_frames=window_frames,
        scratch_dir=scratch_dir,
    )
    sources = discover_sources(inputs, filename=filename)
    contract_sha256 = hashlib.sha256(f"{SCHEMA}\0frame_bytes={frame_bytes}".encode()).hexdigest()
    results = []
    for index, source in enumerate(sources, start=1):
        if progress is not None:
            progress(f"[{index}/{len(sources)}] start {source}")
        result = _measure_source(
            source,
            frame_bytes=frame_bytes,
            capture_contract_sha256=contract_sha256,
            random_windows=random_windows,
            window_frames=window_frames,
            seed=seed,
            scratch_dir=scratch_dir,
        )
        results.append(result)
        if progress is not None:
            progress(
                f"[{index}/{len(sources)}] done {source} "
                f"archive_ratio={result['archive_ratio']:.4f}"
            )
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "revision": _git_output("rev-parse", "HEAD"),
        "revision_dirty": _git_dirty(),
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "byte_order": sys.byteorder,
        },
        "parameters": {
            "filename": filename,
            "frame_bytes": frame_bytes,
            "random_windows": random_windows,
            "window_frames": window_frames,
            "random_seed": seed,
            "benchmark_capture_contract_sha256": contract_sha256,
        },
        "summary": _summary(results),
        "sources": results,
    }


def _measure_source(
    source: Path,
    *,
    frame_bytes: int,
    capture_contract_sha256: str,
    random_windows: int,
    window_frames: int,
    seed: int,
    scratch_dir: Path | None,
) -> dict[str, object]:
    total_frames, _ = source_selection(
        source,
        frame_bytes=frame_bytes,
        start_frame=0,
        max_frames=None,
    )
    if window_frames > total_frames:
        raise ValueError("Window frames must not exceed source frames.")
    raw_bytes = source.stat().st_size
    with tempfile.TemporaryDirectory(prefix="mmwcore-evidence-archive-", dir=scratch_dir) as root:
        destination = Path(root) / "evidence.mmwe"
        started = time.perf_counter_ns()
        archive = write_evidence_archive(
            source,
            destination,
            frame_bytes=frame_bytes,
            capture_contract_sha256=capture_contract_sha256,
        )
        publish_ns = time.perf_counter_ns() - started

        archive = open_evidence_archive(destination)
        started = time.perf_counter_ns()
        archive.verify_all()
        verify_ns = time.perf_counter_ns() - started

        window_starts = _window_starts(
            frame_count=total_frames,
            window_frames=window_frames,
            count=random_windows,
            seed=seed,
        )
        verified_ns = _measure_windows(
            archive,
            source=source,
            starts=window_starts,
            window_frames=window_frames,
            frame_bytes=frame_bytes,
            verify=True,
        )
        trusted_ns = _measure_windows(
            archive,
            source=source,
            starts=window_starts,
            window_frames=window_frames,
            frame_bytes=frame_bytes,
            verify=False,
        )
        return {
            "path": str(source),
            "frame_count": total_frames,
            "frame_bytes": frame_bytes,
            "raw_bytes": raw_bytes,
            "logical_sha256": archive.evidence_sha256,
            "archive_bytes": archive.archive_size,
            "payload_bytes": archive.payload_bytes,
            "index_bytes": archive.index_bytes,
            "metadata_bytes": archive.metadata_bytes,
            "archive_ratio": archive.archive_size / raw_bytes,
            "metadata_ratio": archive.metadata_bytes / raw_bytes,
            "publish_mib_per_second": _mib_per_second(raw_bytes, publish_ns),
            "full_verify_mib_per_second": _mib_per_second(raw_bytes, verify_ns),
            "throughput_scope": {
                "publish": (
                    "source_read_frame_and_logical_hash_native_encode_payload_write_fsync_"
                    "source_rehash_full_decode_verify_atomic_publish"
                ),
                "full_verify": "archive_read_native_decode_frame_digest_logical_digest",
            },
            "random_window": {
                "count": random_windows,
                "frames": window_frames,
                "mode_order": ["verified", "trusted_after_full_verify"],
                "verified": _latency(
                    verified_ns,
                    scope="archive_seek_read_native_decode_frame_digest",
                ),
                "trusted_after_full_verify": _latency(
                    trusted_ns,
                    scope="archive_seek_read_native_decode_after_same_reader_verify_all",
                ),
            },
            "roundtrip_verified": True,
        }


def _measure_windows(
    archive: EvidenceArchive,
    *,
    source: Path,
    starts: Sequence[int],
    window_frames: int,
    frame_bytes: int,
    verify: bool,
) -> list[int]:
    durations = []
    with source.open("rb") as stream:
        for start in starts:
            began = time.perf_counter_ns()
            actual = archive.read_frames(start, start + window_frames, verify=verify)
            durations.append(time.perf_counter_ns() - began)
            stream.seek(start * frame_bytes)
            expected = stream.read(window_frames * frame_bytes)
            if actual != expected:
                raise RuntimeError("Evidence archive random window differs from source bytes.")
    return durations


def _window_starts(
    *, frame_count: int, window_frames: int, count: int, seed: int
) -> tuple[int, ...]:
    generator = random.Random(seed)
    upper = frame_count - window_frames
    return tuple(generator.randint(0, upper) for _ in range(count))


def _latency(values: Sequence[int], *, scope: str) -> dict[str, str | float]:
    total = sum(values)
    return {
        "scope": scope,
        "p50_ns": _percentile(values, 50.0),
        "p95_ns": _percentile(values, 95.0),
        "windows_per_second": len(values) * 1_000_000_000.0 / total if total else 0.0,
    }


def _summary(sources: Sequence[dict[str, object]]) -> dict[str, object]:
    raw_bytes = sum(_integer_field(source, "raw_bytes") for source in sources)
    archive_bytes = sum(_integer_field(source, "archive_bytes") for source in sources)
    metadata_bytes = sum(_integer_field(source, "metadata_bytes") for source in sources)
    return {
        "source_count": len(sources),
        "raw_bytes": raw_bytes,
        "archive_bytes": archive_bytes,
        "metadata_bytes": metadata_bytes,
        "archive_ratio": archive_bytes / raw_bytes,
        "metadata_ratio": metadata_bytes / raw_bytes,
        "minimum_publish_mib_per_second": min(
            _float_field(source, "publish_mib_per_second") for source in sources
        ),
        "minimum_full_verify_mib_per_second": min(
            _float_field(source, "full_verify_mib_per_second") for source in sources
        ),
        "maximum_verified_random_p95_ns": max(
            _random_p95(source, "verified") for source in sources
        ),
        "maximum_trusted_random_p95_ns": max(
            _random_p95(source, "trusted_after_full_verify") for source in sources
        ),
        "all_roundtrip_verified": all(source["roundtrip_verified"] is True for source in sources),
    }


def _integer_field(source: Mapping[str, object], name: str) -> int:
    value = source[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Archive acceptance {name!r} must be an integer.")
    return value


def _float_field(source: Mapping[str, object], name: str) -> float:
    value = source[name]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"Archive acceptance {name!r} must be numeric.")
    return float(value)


def _random_p95(source: Mapping[str, object], mode: str) -> float:
    random_window = source["random_window"]
    if not isinstance(random_window, Mapping):
        raise TypeError("Archive acceptance random_window must be a mapping.")
    metrics = random_window[mode]
    if not isinstance(metrics, Mapping):
        raise TypeError(f"Archive acceptance random-window mode {mode!r} must be a mapping.")
    value = metrics["p95_ns"]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError("Archive acceptance random-window P95 must be numeric.")
    return float(value)


def _mib_per_second(size_bytes: int, elapsed_ns: int) -> float:
    return size_bytes * 1_000_000_000.0 / (1024 * 1024 * elapsed_ns) if elapsed_ns else 0.0


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _validate_options(
    inputs: Sequence[Path],
    *,
    frame_bytes: int,
    random_windows: int,
    window_frames: int,
    scratch_dir: Path | None,
) -> None:
    if not inputs:
        raise ValueError("At least one evidence source is required.")
    if frame_bytes <= 0 or frame_bytes % 2:
        raise ValueError("Frame bytes must be a positive multiple of two.")
    if random_windows < 0:
        raise ValueError("Random windows must be non-negative.")
    if window_frames <= 0:
        raise ValueError("Window frames must be positive.")
    if scratch_dir is not None and not scratch_dir.is_dir():
        raise FileNotFoundError(f"Scratch directory does not exist: {scratch_dir}")


def _git_output(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())
