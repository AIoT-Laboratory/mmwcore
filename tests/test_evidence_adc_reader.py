from __future__ import annotations

import hashlib
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mmwcore.config import RadarCaptureSpec, RadarProfile, capture_contract_sha256
from mmwcore.core import ADCDecodeRecipe, ADCFrameSpec, DopplerFFTSpec, RangeDopplerRecipe
from mmwcore.dsp import process_adc_to_range_doppler
from mmwcore.io import (
    ADCEvidenceArchiveFrameReader,
    ADCFileFrameReader,
    evidence_archive,
    write_adc_evidence_archive,
)
from mmwcore.io.evidence_archive import (
    EvidenceArchive,
    EvidenceArchiveError,
    write_evidence_archive,
)


@pytest.fixture(autouse=True)
def native_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    def encode(raw: bytes) -> bytes:
        return zlib.compress(raw, level=1)

    def decode(encoded: bytes, expected_raw_bytes: int) -> bytes:
        raw = zlib.decompress(encoded)
        if len(raw) != expected_raw_bytes:
            raise ValueError("decoded frame has the wrong size")
        return raw

    monkeypatch.setattr(
        evidence_archive,
        "_native",
        SimpleNamespace(encode_evidence_frame=encode, decode_evidence_frame=decode),
    )


def _capture(*, num_frames: int = 3) -> RadarCaptureSpec:
    profile = RadarProfile(
        num_tx=1,
        num_rx=1,
        num_adc_samples=2,
        num_chirps_per_tx=1,
    )
    return RadarCaptureSpec(
        profile=profile,
        adc=ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2),
        tx_order=(0,),
        frame_periodicity_s=0.1,
        num_frames=num_frames,
    )


def _archive(tmp_path: Path) -> tuple[Path, RadarCaptureSpec, bytes]:
    capture = _capture()
    assert capture.num_frames is not None
    raw = np.arange(capture.adc.raw_values_per_frame * capture.num_frames, dtype=np.int16).tobytes()
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    archive = tmp_path / "adc.mmwe"
    write_evidence_archive(
        source,
        archive,
        frame_bytes=capture.adc.raw_values_per_frame * np.dtype(np.int16).itemsize,
        capture_contract_sha256=capture_contract_sha256(capture),
    )
    return archive, capture, raw


def test_archive_reader_binds_capture_and_decodes_verified_raw_frames(tmp_path: Path) -> None:
    archive, capture, raw = _archive(tmp_path)
    evidence_sha256 = hashlib.sha256(raw).hexdigest()

    reader = ADCEvidenceArchiveFrameReader(
        archive,
        capture,
        expected_evidence_sha256=evidence_sha256,
        metadata={"session": "fixture"},
    )
    frame = reader.read_frame(2)

    assert reader.spec == capture.adc
    assert reader.num_frames == 3
    assert not isinstance(frame.samples, np.memmap)
    np.testing.assert_array_equal(frame.samples, np.array([8, 9, 10, 11], dtype=np.int16))
    assert frame.timestamp == pytest.approx(0.2)
    assert frame.profile["num_tx"] == 1
    assert frame.metadata == {
        "tx_order": [0],
        "session": "fixture",
        "frame_index": 2,
        "num_frames": 3,
        "evidence_sha256": evidence_sha256,
        "capture_contract_sha256": capture_contract_sha256(capture),
    }


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("expected_evidence_sha256", "0" * 64, "does not match"),
        ("expected_evidence_sha256", "A" * 64, "lowercase"),
    ],
)
def test_archive_reader_rejects_unbound_or_noncanonical_expected_digests(
    tmp_path: Path,
    keyword: str,
    value: str,
    match: str,
) -> None:
    archive, capture, _ = _archive(tmp_path)

    with pytest.raises(ValueError, match=match):
        ADCEvidenceArchiveFrameReader(
            archive,
            capture,
            expected_evidence_sha256=value,
        )


def test_archive_reader_rejects_contract_count_and_frame_integrity_mismatches(
    tmp_path: Path,
) -> None:
    archive, capture, raw = _archive(tmp_path)

    mismatched_capture = replace(capture, frame_periodicity_s=0.2)
    with pytest.raises(ValueError, match="capture_contract_sha256"):
        ADCEvidenceArchiveFrameReader(
            archive,
            mismatched_capture,
            expected_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        )

    wrong_count = replace(capture, num_frames=2)
    source = tmp_path / "wrong-count.bin"
    source.write_bytes(np.arange(12, dtype=np.int16).tobytes())
    wrong_count_archive = tmp_path / "wrong-count.mmwe"
    write_evidence_archive(
        source,
        wrong_count_archive,
        frame_bytes=wrong_count.adc.raw_values_per_frame * np.dtype(np.int16).itemsize,
        capture_contract_sha256=capture_contract_sha256(wrong_count),
    )
    with pytest.raises(ValueError, match="frame count"):
        ADCEvidenceArchiveFrameReader(
            wrong_count_archive,
            wrong_count,
            expected_evidence_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )

    payload = bytearray(archive.read_bytes())
    payload[64] ^= 0xFF
    archive.write_bytes(payload)
    with pytest.raises(EvidenceArchiveError, match="Native decode_evidence_frame"):
        ADCEvidenceArchiveFrameReader(
            archive,
            capture,
            expected_evidence_sha256=hashlib.sha256(_archive_raw(capture)).hexdigest(),
        ).read_frame(0)


def test_archive_reader_open_does_not_verify_all_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, capture, raw = _archive(tmp_path)

    def reject_eager_verification(self: EvidenceArchive) -> None:
        raise AssertionError("verify_all must remain explicit")

    monkeypatch.setattr(EvidenceArchive, "verify_all", reject_eager_verification)

    reader = ADCEvidenceArchiveFrameReader(
        archive,
        capture,
        expected_evidence_sha256=hashlib.sha256(raw).hexdigest(),
    )

    assert reader.num_frames == 3
    np.testing.assert_array_equal(reader.read_frame(0).samples, np.array([0, 1, 2, 3]))


def test_archive_reader_accepts_a_finalized_open_ended_capture(tmp_path: Path) -> None:
    capture = replace(_capture(), num_frames=None)
    raw = np.arange(capture.adc.raw_values_per_frame * 3, dtype=np.int16).tobytes()
    source = tmp_path / "open-ended.bin"
    source.write_bytes(raw)
    archive = tmp_path / "open-ended.mmwe"
    write_adc_evidence_archive(
        source,
        archive,
        capture,
        expected_evidence_sha256=hashlib.sha256(raw).hexdigest(),
    )

    reader = ADCEvidenceArchiveFrameReader(
        archive,
        capture,
        expected_evidence_sha256=hashlib.sha256(raw).hexdigest(),
    )

    assert reader.num_frames == 3


def test_capture_bound_writer_rejects_wrong_source_identity_before_publication(
    tmp_path: Path,
) -> None:
    capture = _capture()
    source = tmp_path / "adc.bin"
    source.write_bytes(_archive_raw(capture))
    destination = tmp_path / "adc.mmwe"

    with pytest.raises(EvidenceArchiveError, match="expected_evidence_sha256"):
        write_adc_evidence_archive(
            source,
            destination,
            capture,
            expected_evidence_sha256="0" * 64,
        )

    assert not destination.exists()


def test_capture_bound_writer_and_reader_preserve_the_logical_source(tmp_path: Path) -> None:
    capture = _capture()
    raw = _archive_raw(capture)
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    destination = tmp_path / "adc.mmwe"
    digest = hashlib.sha256(raw).hexdigest()

    written = write_adc_evidence_archive(
        source,
        destination,
        capture,
        expected_evidence_sha256=digest,
    )
    reader = ADCEvidenceArchiveFrameReader(
        destination,
        capture,
        expected_evidence_sha256=digest,
    )

    assert written.evidence_sha256 == digest
    np.testing.assert_array_equal(reader.read_frame(1).samples, np.array([4, 5, 6, 7]))


def test_raw_and_archive_readers_produce_identical_range_doppler_data(tmp_path: Path) -> None:
    capture = _capture()
    raw = _archive_raw(capture)
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    archive = tmp_path / "adc.mmwe"
    digest = hashlib.sha256(raw).hexdigest()
    write_adc_evidence_archive(
        source,
        archive,
        capture,
        expected_evidence_sha256=digest,
    )
    raw_reader = ADCFileFrameReader.from_capture(source, capture)
    archive_reader = ADCEvidenceArchiveFrameReader(
        archive,
        capture,
        expected_evidence_sha256=digest,
    )
    recipe = RangeDopplerRecipe(
        decode=ADCDecodeRecipe(capture.adc),
        doppler_fft=DopplerFFTSpec(fftshift=False),
    )

    for index in range(capture.num_frames or 0):
        raw_cube = process_adc_to_range_doppler(raw_reader.read_frame(index), recipe)
        archive_cube = process_adc_to_range_doppler(archive_reader.read_frame(index), recipe)
        np.testing.assert_array_equal(archive_cube.data, raw_cube.data)
        assert archive_cube.axes == raw_cube.axes
        assert archive_cube.frame_id == raw_cube.frame_id
        assert archive_cube.timestamp == raw_cube.timestamp


def _archive_raw(capture: RadarCaptureSpec) -> bytes:
    assert capture.num_frames is not None
    return np.arange(
        capture.adc.raw_values_per_frame * capture.num_frames,
        dtype=np.int16,
    ).tobytes()
