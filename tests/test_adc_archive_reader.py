from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCDecodeRecipe, ADCFrameSpec, DopplerFFTSpec, RangeDopplerRecipe
from mmwcore.dsp import process_adc_to_range_doppler
from mmwcore.io import ADCArchiveFrameReader, ADCFileFrameReader, write_adc_archive
from mmwcore.io.adc_archive import ADCArchive, ADCArchiveError


def _capture(*, num_frames: int | None = 3) -> RadarCaptureSpec:
    return RadarCaptureSpec(
        profile=RadarProfile(
            num_tx=1,
            num_rx=1,
            num_adc_samples=2,
            num_chirps_per_tx=1,
        ),
        adc=ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2),
        tx_order=(0,),
        frame_periodicity_s=0.1,
        num_frames=num_frames,
    )


def _raw(capture: RadarCaptureSpec, *, frame_count: int = 3) -> bytes:
    count = capture.num_frames or frame_count
    return np.arange(capture.adc.raw_values_per_frame * count, dtype=np.int16).tobytes()


def _archive(tmp_path: Path) -> tuple[Path, RadarCaptureSpec, bytes]:
    capture = _capture()
    raw = _raw(capture)
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    destination = tmp_path / "adc.mmwa"
    write_adc_archive(source, destination, capture)
    return destination, capture, raw


def test_reader_recovers_contract_and_decodes_frames_without_external_spec(tmp_path: Path) -> None:
    archive, capture, raw = _archive(tmp_path)
    reader = ADCArchiveFrameReader(archive, metadata={"session": "fixture"})
    frame = reader.read_frame(2)

    assert reader.capture == capture
    assert reader.spec == capture.adc
    assert reader.num_frames == 3
    assert not isinstance(frame.samples, np.memmap)
    np.testing.assert_array_equal(frame.samples, np.array([8, 9, 10, 11], dtype=np.int16))
    assert frame.timestamp == pytest.approx(0.2)
    assert frame.profile["num_tx"] == 1
    assert frame.metadata["tx_order"] == [0]
    assert frame.metadata["session"] == "fixture"
    assert frame.metadata["adc_sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(frame.metadata["capture_sha256"]) == 64


def test_reader_metadata_cannot_override_embedded_tx_order(tmp_path: Path) -> None:
    archive, _, _ = _archive(tmp_path)
    with pytest.raises(ValueError, match="tx_order"):
        ADCArchiveFrameReader(archive, metadata={"tx_order": [9]})


def test_reader_open_does_not_decode_all_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, _, _ = _archive(tmp_path)

    def reject_eager_verification(self: ADCArchive) -> None:
        raise AssertionError("verify_all must remain explicit")

    monkeypatch.setattr(ADCArchive, "verify_all", reject_eager_verification)
    reader = ADCArchiveFrameReader(archive)

    np.testing.assert_array_equal(reader.read_frame(0).samples, np.array([0, 1, 2, 3]))


def test_reader_revalidates_unchanged_input(tmp_path: Path) -> None:
    archive, _, _ = _archive(tmp_path)
    ADCArchiveFrameReader(archive).revalidate_input()


def test_open_ended_capture_records_final_frame_count(tmp_path: Path) -> None:
    capture = _capture(num_frames=None)
    raw = _raw(capture)
    source = tmp_path / "open-ended.bin"
    source.write_bytes(raw)
    destination = tmp_path / "open-ended.mmwa"

    write_adc_archive(source, destination, capture)
    reader = ADCArchiveFrameReader(destination)

    assert reader.capture.num_frames == 3
    assert reader.capture.expected_size_bytes == len(raw)


def test_writer_rejects_declared_frame_count_mismatch(tmp_path: Path) -> None:
    capture = replace(_capture(), num_frames=2)
    source = tmp_path / "adc.bin"
    source.write_bytes(_raw(_capture()))
    destination = tmp_path / "adc.mmwa"

    with pytest.raises(ValueError, match="frame count"):
        write_adc_archive(source, destination, capture)
    assert not destination.exists()


def test_writer_binds_expected_logical_source_identity(tmp_path: Path) -> None:
    capture = _capture()
    raw = _raw(capture)
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    destination = tmp_path / "adc.mmwa"
    digest = hashlib.sha256(raw).hexdigest()

    written = write_adc_archive(
        source,
        destination,
        capture,
        expected_adc_sha256=digest,
    )
    reader = ADCArchiveFrameReader(destination)

    assert written.adc_sha256 == digest
    np.testing.assert_array_equal(reader.read_frame(1).samples, np.array([4, 5, 6, 7]))


def test_raw_and_archive_readers_produce_identical_range_doppler_data(tmp_path: Path) -> None:
    capture = _capture()
    raw = _raw(capture)
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    archive = tmp_path / "adc.mmwa"
    write_adc_archive(source, archive, capture)
    raw_reader = ADCFileFrameReader.from_capture(source, capture)
    archive_reader = ADCArchiveFrameReader(archive)
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


def test_changed_archive_is_rejected_after_open(tmp_path: Path) -> None:
    archive, _, _ = _archive(tmp_path)
    reader = ADCArchiveFrameReader(archive)
    archive.write_bytes(archive.read_bytes())

    with pytest.raises(ADCArchiveError, match="changed"):
        reader.read_frame(0)
