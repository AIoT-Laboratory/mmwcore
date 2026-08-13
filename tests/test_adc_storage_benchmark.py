from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.adc_storage_benchmark import SCHEMA, run_benchmark  # noqa: E402
from benchmarks.adc_storage_benchmark_cli import main  # noqa: E402
from benchmarks.adc_storage_chunks import ChunkRecord, read_frames  # noqa: E402
from benchmarks.adc_storage_codecs import (  # noqa: E402
    SUPPORTED_CODECS,
    decode,
    encode,
    selected_transform,
)
from benchmarks.adc_storage_inputs import StorageCase  # noqa: E402


def _frames(frame_bytes: int, count: int) -> bytes:
    words = np.array([0, 1, 32_767, 32_768, 65_535, 65_534, 12_345, 54_321], dtype=np.uint16)
    frame_words = frame_bytes // 2
    data = np.empty((count, frame_words), dtype=np.uint16)
    for index in range(count):
        data[index] = words[:frame_words] + np.uint16(index * 17)
    return data.tobytes()


@pytest.mark.parametrize("codec", SUPPORTED_CODECS)
def test_all_codecs_roundtrip_signed_word_boundaries(codec: str) -> None:
    payload = _frames(frame_bytes=16, count=4)

    assert (
        decode(
            encode(payload, codec=codec, frame_bytes=16, zlib_level=1),
            codec=codec,
            frame_bytes=16,
        )
        == payload
    )


def test_adaptive_codec_reports_its_selected_reversible_transform() -> None:
    payload = _frames(frame_bytes=16, count=4)
    encoded = encode(
        payload,
        codec="adaptive-shuffle-zlib",
        frame_bytes=16,
        zlib_level=1,
    )

    assert selected_transform("adaptive-shuffle-zlib", encoded) in {
        "shuffle-zlib",
        "frame-delta-shuffle-zlib",
    }
    assert decode(encoded, codec="adaptive-shuffle-zlib", frame_bytes=16) == payload


def test_trusted_read_skips_only_repeated_chunk_digest_verification() -> None:
    payload = _frames(frame_bytes=16, count=1)
    record = ChunkRecord(
        first_frame=0,
        frame_count=1,
        offset=0,
        stored_bytes=len(payload),
        raw_bytes=len(payload),
        sha256="0" * 64,
    )

    trusted, decoded_chunks = read_frames(
        BytesIO(payload),
        records=(record,),
        record_first_frames=(0,),
        start_frame=0,
        stop_frame=1,
        frame_bytes=16,
        codec="raw",
        verify_digest=False,
    )

    assert trusted == payload
    assert decoded_chunks == 1
    with pytest.raises(RuntimeError, match="recorded raw chunk"):
        read_frames(
            BytesIO(payload),
            records=(record,),
            record_first_frames=(0,),
            start_frame=0,
            stop_frame=1,
            frame_bytes=16,
            codec="raw",
            verify_digest=True,
        )


def test_benchmark_emits_schema_and_verifies_cross_chunk_random_windows(tmp_path: Path) -> None:
    source = tmp_path / "nested" / "adc_data_Raw_0.bin"
    source.parent.mkdir()
    source.write_bytes(_frames(frame_bytes=16, count=7))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = run_benchmark(
        [tmp_path],
        frame_bytes=16,
        cases=(StorageCase(codec="frame-delta-shuffle-zlib", chunk_frames=2),),
        random_windows=5,
        window_frames=3,
        seed=7,
        scratch_dir=scratch,
    )

    assert result["schema"] == SCHEMA
    assert result["archive_metadata_overhead_included"] is False
    parameters = cast(dict[str, object], result["benchmark_parameters"])
    assert parameters["random_seed"] == 7
    assert parameters["cases"] == [{"codec": "frame-delta-shuffle-zlib", "chunk_frames": 2}]
    summary = cast(dict[str, object], result["summary"])
    case_summaries = cast(list[dict[str, object]], summary["case_summaries"])
    assert summary["source_count"] == 1
    assert case_summaries[0]["all_roundtrip_verified"] is True
    assert case_summaries[0]["source_count"] == 1
    assert case_summaries[0]["selected_transform_counts"] == {"frame-delta-shuffle-zlib": 4}
    source_result = cast(list[dict[str, object]], result["sources"])[0]
    selection = cast(dict[str, object], source_result["selection"])
    cases = cast(list[dict[str, object]], source_result["cases"])
    assert selection["logical_sha256"]
    case = cases[0]
    assert case["roundtrip_verified"] is True
    assert case["chunk_count"] == 4
    random_window = cast(dict[str, object], case["random_window"])
    assert cast(float, random_window["average_chunks"]) >= 2.0
    assert random_window["mode_order"] == ["trusted", "verified"]
    trusted = cast(dict[str, object], random_window["trusted"])
    verified = cast(dict[str, object], random_window["verified"])
    assert trusted["scope"] == "payload_seek_read_decode_slice"
    assert verified["scope"] == "payload_seek_read_decode_chunk_digest_slice"
    assert not list(scratch.iterdir())


def test_cli_writes_atomic_json_and_rejects_invalid_source_shapes(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(_frames(frame_bytes=16, count=4))
    output = tmp_path / "result.json"

    assert (
        main(
            [
                str(source),
                "--frame-bytes",
                "16",
                "--case",
                "raw:2",
                "--random-windows",
                "1",
                "--window-frames",
                "3",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == SCHEMA
    raw_case = report["sources"][0]["cases"][0]
    assert raw_case["payload_ratio"] == 1.0
    assert raw_case["encode_mib_per_second"] is None
    assert raw_case["decode_mib_per_second"] is None
    random_window = raw_case["random_window"]
    assert random_window["trusted"]["scope"] == "payload_seek_read_decode_slice"
    assert random_window["verified"]["scope"] == "payload_seek_read_decode_chunk_digest_slice"

    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(b"not-a-complete-frame")
    with pytest.raises(ValueError, match="incomplete trailing frame"):
        run_benchmark([truncated], frame_bytes=16, cases=(StorageCase("raw", 1),))
    with pytest.raises(ValueError, match="outside"):
        run_benchmark([source], frame_bytes=16, start_frame=4, cases=(StorageCase("raw", 1),))
    with pytest.raises(ValueError, match="non-negative"):
        run_benchmark([source], frame_bytes=16, start_frame=-1, cases=(StorageCase("raw", 1),))
    with pytest.raises(ValueError, match="Maximum frames"):
        run_benchmark([source], frame_bytes=16, max_frames=0, cases=(StorageCase("raw", 1),))
    with pytest.raises(ValueError, match="must not exceed"):
        run_benchmark(
            [source],
            frame_bytes=16,
            cases=(StorageCase("raw", 1),),
            window_frames=5,
        )
