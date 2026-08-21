from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCFrameSpec
from mmwcore.io import ADCArchiveError, open_adc_archive, write_adc_archive

_FIXED_HEADER_BYTES = 112
_INDEX_RECORD_BYTES = 56
_FOOTER_BYTES = 160


def _capture(*, num_frames: int | None = 6) -> RadarCaptureSpec:
    return RadarCaptureSpec(
        profile=RadarProfile(
            num_tx=1,
            num_rx=1,
            num_adc_samples=4,
            num_chirps_per_tx=1,
        ),
        adc=ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=4),
        tx_order=(0,),
        frame_periodicity_s=0.1,
        num_frames=num_frames,
    )


def _write_source(tmp_path: Path, capture: RadarCaptureSpec) -> tuple[Path, bytes]:
    frame_count = capture.num_frames or 6
    raw = np.arange(capture.adc.raw_values_per_frame * frame_count, dtype=np.int16).tobytes()
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    return source, raw


def _archive(tmp_path: Path) -> tuple[Path, RadarCaptureSpec, bytes]:
    capture = _capture()
    source, raw = _write_source(tmp_path, capture)
    destination = tmp_path / "capture.mmwa"
    write_adc_archive(source, destination, capture)
    return destination, capture, raw


def test_v3_roundtrip_is_self_describing_and_random_accessible(tmp_path: Path) -> None:
    destination, capture, raw = _archive(tmp_path)
    archive = open_adc_archive(destination)

    assert destination.read_bytes()[:8] == b"MMWADCA3"
    assert archive.capture == capture
    assert archive.capture.num_frames == 6
    assert archive.frame_bytes == 16
    assert archive.frame_count == 6
    assert archive.block_samples == 512
    assert archive.restart_frames == 4
    assert archive.adc_sha256 == hashlib.sha256(raw).hexdigest()
    assert archive.read_frames(1, 4) == raw[16:64]
    assert archive.index_bytes == 2 * _INDEX_RECORD_BYTES
    assert archive.archive_size == archive.payload_bytes + archive.container_overhead_bytes
    assert archive.capture_metadata_bytes == archive.header_bytes - _FIXED_HEADER_BYTES
    assert archive.container_overhead_bytes == (
        archive.header_bytes + archive.index_bytes + _FOOTER_BYTES
    )


def test_open_ended_capture_is_finalized_inside_header(tmp_path: Path) -> None:
    capture = _capture(num_frames=None)
    source, _ = _write_source(tmp_path, capture)
    destination = tmp_path / "open-ended.mmwa"

    write_adc_archive(source, destination, capture)
    reopened = open_adc_archive(destination)

    assert reopened.capture.num_frames == 6
    assert reopened.capture.expected_size_bytes == source.stat().st_size


def test_trusted_read_requires_full_verification(tmp_path: Path) -> None:
    destination, _, raw = _archive(tmp_path)
    archive = open_adc_archive(destination)

    with pytest.raises(ADCArchiveError, match="verify_all"):
        archive.read_frames(0, 1, verify=False)
    archive.verify_all()
    assert archive.read_frames(2, 5, verify=False) == raw[32:80]


@pytest.mark.parametrize("area", ["header", "metadata", "index", "footer"])
def test_structural_regions_are_digest_bound(tmp_path: Path, area: str) -> None:
    destination, _, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    footer_offset = len(payload) - _FOOTER_BYTES
    index_offset = struct.unpack_from("<Q", payload, footer_offset + 16)[0]
    offsets = {
        "header": 16,
        "metadata": _FIXED_HEADER_BYTES,
        "index": index_offset,
        "footer": footer_offset + 32,
    }
    payload[offsets[area]] ^= 1
    destination.write_bytes(payload)

    with pytest.raises(ADCArchiveError):
        open_adc_archive(destination)


def test_payload_corruption_fails_verified_frame_read(tmp_path: Path) -> None:
    destination, _, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    metadata_bytes = struct.unpack_from("<Q", payload, 24)[0]
    payload[_FIXED_HEADER_BYTES + metadata_bytes + 1] ^= 0xFF
    destination.write_bytes(payload)

    archive = open_adc_archive(destination)
    with pytest.raises(ADCArchiveError):
        archive.read_frames(0, 1)


def test_self_consistent_wrong_logical_digest_fails_full_verification(tmp_path: Path) -> None:
    destination, _, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    footer_offset = len(payload) - _FOOTER_BYTES
    payload[footer_offset + 96] ^= 1
    payload[footer_offset + 128 :] = hashlib.sha256(
        payload[footer_offset : footer_offset + 128]
    ).digest()
    destination.write_bytes(payload)

    archive = open_adc_archive(destination)
    with pytest.raises(ADCArchiveError, match="logical raw SHA-256"):
        archive.verify_all()


@pytest.mark.parametrize("mutation", [lambda value: value[:-1], lambda value: value + b"tail"])
def test_truncation_and_trailing_bytes_fail_open(
    tmp_path: Path,
    mutation: Callable[[bytes], bytes],
) -> None:
    destination, _, _ = _archive(tmp_path)
    destination.write_bytes(mutation(destination.read_bytes()))
    with pytest.raises(ADCArchiveError):
        open_adc_archive(destination)


def test_writer_rejects_wrong_source_identity_and_never_overwrites(tmp_path: Path) -> None:
    capture = _capture()
    source, _ = _write_source(tmp_path, capture)
    destination = tmp_path / "capture.mmwa"

    with pytest.raises(ADCArchiveError, match="expected_adc_sha256"):
        write_adc_archive(
            source,
            destination,
            capture,
            expected_adc_sha256="0" * 64,
        )
    assert not destination.exists()

    destination.write_bytes(b"existing")
    with pytest.raises(ADCArchiveError, match="already exists"):
        write_adc_archive(source, destination, capture)
    assert destination.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".capture.mmwa.*.tmp"))


def test_v2_magic_is_rejected_without_compatibility_path(tmp_path: Path) -> None:
    destination, _, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    payload[:8] = b"MMWADCA2"
    destination.write_bytes(payload)
    with pytest.raises(ADCArchiveError, match="v3"):
        open_adc_archive(destination)
