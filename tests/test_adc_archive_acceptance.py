from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.adc_archive_acceptance import (  # noqa: E402
    SCHEMA,
    run_archive_acceptance,
)
from benchmarks.adc_archive_acceptance_cli import main  # noqa: E402


def _payload(frame_bytes: int, frame_count: int) -> bytes:
    values = np.arange(frame_bytes * frame_count // 2, dtype=np.uint16)
    return values.tobytes()


def test_archive_acceptance_measures_implemented_container(tmp_path: Path) -> None:
    source = tmp_path / "adc_data_Raw_0.bin"
    source.write_bytes(_payload(32, 7))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    report = run_archive_acceptance(
        [source],
        frame_bytes=32,
        random_windows=5,
        window_frames=3,
        seed=7,
        scratch_dir=scratch,
    )

    assert report["schema"] == SCHEMA
    summary = cast(dict[str, object], report["summary"])
    assert summary["source_count"] == 1
    assert summary["all_roundtrip_verified"] is True
    assert cast(int, summary["metadata_bytes"]) > 0
    source_report = cast(list[dict[str, object]], report["sources"])[0]
    assert source_report["frame_count"] == 7
    assert source_report["raw_bytes"] == 224
    assert cast(int, source_report["archive_bytes"]) == (
        cast(int, source_report["payload_bytes"]) + cast(int, source_report["metadata_bytes"])
    )
    random_window = cast(dict[str, object], source_report["random_window"])
    assert random_window["mode_order"] == ["verified", "trusted_after_full_verify"]
    assert not list(scratch.iterdir())


def test_archive_acceptance_cli_requires_and_writes_report(tmp_path: Path) -> None:
    source = tmp_path / "adc.bin"
    source.write_bytes(_payload(16, 5))
    output = tmp_path / "report.json"

    assert (
        main(
            [
                str(source),
                "--frame-bytes",
                "16",
                "--random-windows",
                "2",
                "--window-frames",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == SCHEMA
    assert report["summary"]["all_roundtrip_verified"] is True
    assert not list(tmp_path.glob(".*.tmp"))
