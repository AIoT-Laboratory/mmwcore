"""Reproducibility metadata and conservative evidence benchmark summaries."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_report(
    *,
    schema: str,
    sources: Sequence[dict[str, object]],
    benchmark_parameters: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": schema,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "revision": _git_revision(),
        "revision_dirty": _git_dirty(),
        "environment": _environment(),
        "benchmark_parameters": dict(benchmark_parameters),
        "archive_metadata_overhead_included": False,
        "summary": {
            "source_count": len(sources),
            "case_summaries": _summarize_cases(sources),
        },
        "sources": list(sources),
    }


def _summarize_cases(sources: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for source in sources:
        for case in cast(list[dict[str, object]], source["cases"]):
            key = (cast(str, case["codec"]), cast(int, case["chunk_frames"]))
            grouped.setdefault(key, []).append(case)

    summaries = []
    for (codec, chunk_frames), cases in grouped.items():
        raw_bytes = sum(cast(int, case["raw_bytes"]) for case in cases)
        payload_bytes = sum(cast(int, case["payload_bytes"]) for case in cases)
        summaries.append(
            {
                "codec": codec,
                "chunk_frames": chunk_frames,
                "source_count": len(cases),
                "raw_bytes": raw_bytes,
                "payload_bytes": payload_bytes,
                "payload_ratio": payload_bytes / raw_bytes,
                "minimum_pack_mib_per_second": _minimum_metric(cases, "pack_mib_per_second"),
                "minimum_encode_mib_per_second": _minimum_metric(cases, "encode_mib_per_second"),
                "minimum_sequential_replay_mib_per_second": _minimum_metric(
                    cases, "sequential_replay_mib_per_second"
                ),
                "minimum_decode_mib_per_second": _minimum_metric(cases, "decode_mib_per_second"),
                "maximum_random_window_trusted_p95_ns": _maximum_random_p95(cases, "trusted"),
                "maximum_random_window_verified_p95_ns": _maximum_random_p95(cases, "verified"),
                "selected_transform_counts": _sum_transform_counts(cases),
                "all_roundtrip_verified": all(case["roundtrip_verified"] is True for case in cases),
            }
        )
    return summaries


def _minimum_metric(cases: Sequence[dict[str, object]], field: str) -> float | None:
    values = [cast(float, case[field]) for case in cases if case[field] is not None]
    return min(values) if values else None


def _sum_transform_counts(cases: Sequence[dict[str, object]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for case in cases:
        counts = cast(dict[str, int], case["selected_transform_counts"])
        total.update(counts)
    return dict(total)


def _maximum_random_p95(cases: Sequence[dict[str, object]], mode: str) -> float:
    values = []
    for case in cases:
        random_window = cast(dict[str, object], case["random_window"])
        metrics = cast(dict[str, object], random_window[mode])
        values.append(cast(float, metrics["p95_ns"]))
    return max(values)


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "byte_order": sys.byteorder,
        "zlib_compile_version": zlib.ZLIB_VERSION,
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
    }


def _git_revision() -> str | None:
    result = _run_git("rev-parse", "HEAD")
    return result.strip() or None if result is not None else None


def _git_dirty() -> bool | None:
    result = _run_git("status", "--porcelain", "--untracked-files=all")
    return bool(result) if result is not None else None


def _run_git(*arguments: str) -> str | None:
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
    return result.stdout
