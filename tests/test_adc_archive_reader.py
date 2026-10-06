from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCDecodeSpec, ADCFrameSpec, DopplerFFTSpec, RangeDopplerPipeline
from mmwcore.dsp import range_doppler
from mmwcore.io import ADCArchiveReader, ADCFileReader, write_adc_archive
from mmwcore.io.adc_archive import ADCArchive


@pytest.mark.parametrize("reader_kind", ["raw", "archive"])
def test_readers_share_integer_index_contract(tmp_path: Path, reader_kind: str) -> None:
    archive, capture, _ = _archive(tmp_path)
    reader = (
        ADCFileReader.from_capture(tmp_path / "adc.bin", capture)
        if reader_kind == "raw"
        else ADCArchiveReader(archive)
    )
    frame = reader.read_frame(cast(int, np.int64(2)))
    assert frame.frame_id == 2
    np.testing.assert_array_equal(frame.samples, [8, 9, 10, 11])
    for index in (True, 1.5, "1", None):
        with pytest.raises(TypeError, match="ADC frame index must be an integer"):
            reader.read_frame(cast(int, index))
    for index in (-1, 3, np.int64(3)):
        with pytest.raises(IndexError, match="outside"):
            reader.read_frame(cast(int, index))


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
    reader = ADCArchiveReader(archive, metadata={"session": "fixture"})
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


def test_reader_batches_frames_in_caller_order(tmp_path: Path) -> None:
    archive, _, _ = _archive(tmp_path)
    reader = ADCArchiveReader(archive)

    frames = reader.read_frames([2, 0, 2, 1])

    assert [frame.frame_id for frame in frames] == [2, 0, 2, 1]
    assert [frame.timestamp for frame in frames] == pytest.approx([0.2, 0.0, 0.2, 0.1])
    np.testing.assert_array_equal(frames[0].samples, np.array([8, 9, 10, 11]))
    np.testing.assert_array_equal(frames[1].samples, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(frames[2].samples, frames[0].samples)
    np.testing.assert_array_equal(frames[3].samples, np.array([4, 5, 6, 7]))
    assert reader.read_frames([]) == ()


def test_stream_batches_at_chunk_boundaries_and_keeps_yielded_bytes(tmp_path, monkeypatch):
    capture = _capture(num_frames=10)
    raw = _raw(capture)
    source, path = tmp_path / "adc.bin", tmp_path / "adc.mmwa"
    source.write_bytes(raw)
    write_adc_archive(source, path, capture)
    reader = ADCArchiveReader(path)
    assert reader.archive.restart_frames == 4
    calls = []
    read = ADCArchiveReader.read_frames

    def record(self, indices):
        calls.append(tuple(indices))
        return read(self, indices)

    monkeypatch.setattr(ADCArchiveReader, "read_frames", record)
    stream = reader.iter_frames(3, 9)
    assert calls == []
    first = next(stream)
    assert calls == [(3,)]
    frames = [first, *stream]
    assert calls == [(3,), (4, 5, 6, 7), (8,)]
    assert [frame.frame_id for frame in frames] == list(range(3, 9))
    for index, frame in enumerate(frames, start=3):
        expected = reader.read_frame(index)
        np.testing.assert_array_equal(frame.samples, expected.samples)
        assert (frame.timestamp, frame.profile, frame.metadata) == (
            expected.timestamp,
            expected.profile,
            expected.metadata,
        )
    assert list(reader.iter_frames(10, 10)) == []
    partial = reader.iter_frames()
    retained = next(partial)
    partial.close()
    np.testing.assert_array_equal(retained.samples, reader.read_frame(0).samples)


@pytest.mark.parametrize(
    "start,stop,error",
    [
        (-1, 1, IndexError),
        (2, 1, IndexError),
        (0, 4, IndexError),
        (True, 2, TypeError),
        (0, False, TypeError),
        (0.5, 2, TypeError),
    ],
)
def test_stream_rejects_invalid_interval(tmp_path, start, stop, error):
    archive, _, _ = _archive(tmp_path)
    with pytest.raises(error):
        list(ADCArchiveReader(archive).iter_frames(start, stop))


def test_reader_metadata_cannot_override_embedded_tx_order(tmp_path: Path) -> None:
    archive, _, _ = _archive(tmp_path)
    with pytest.raises(ValueError, match="tx_order"):
        ADCArchiveReader(archive, metadata={"tx_order": [9]})


def test_reader_open_does_not_decode_all_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, _, _ = _archive(tmp_path)

    def reject_eager_verification(self: ADCArchive) -> None:
        raise AssertionError("verify_all must remain explicit")

    monkeypatch.setattr(ADCArchive, "verify_all", reject_eager_verification)
    reader = ADCArchiveReader(archive)

    np.testing.assert_array_equal(reader.read_frame(0).samples, np.array([0, 1, 2, 3]))


def test_open_ended_capture_records_final_frame_count(tmp_path: Path) -> None:
    capture = _capture(num_frames=None)
    raw = _raw(capture)
    source = tmp_path / "open-ended.bin"
    source.write_bytes(raw)
    destination = tmp_path / "open-ended.mmwa"

    write_adc_archive(source, destination, capture)
    reader = ADCArchiveReader(destination)

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
    reader = ADCArchiveReader(destination)

    assert written.adc_sha256 == digest
    np.testing.assert_array_equal(reader.read_frame(1).samples, np.array([4, 5, 6, 7]))


def test_raw_and_archive_readers_produce_identical_range_doppler_data(tmp_path: Path) -> None:
    capture = _capture()
    raw = _raw(capture)
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    archive = tmp_path / "adc.mmwa"
    write_adc_archive(source, archive, capture)
    raw_reader = ADCFileReader.from_capture(source, capture)
    archive_reader = ADCArchiveReader(archive)
    recipe = RangeDopplerPipeline(
        decode=ADCDecodeSpec(capture.adc),
        doppler_fft=DopplerFFTSpec(fftshift=False),
    )

    for index in range(capture.num_frames or 0):
        raw_cube = range_doppler(raw_reader.read_frame(index), recipe)
        archive_cube = range_doppler(archive_reader.read_frame(index), recipe)
        np.testing.assert_array_equal(archive_cube.data, raw_cube.data)
        assert archive_cube.axes == raw_cube.axes
        assert archive_cube.frame_id == raw_cube.frame_id
        assert archive_cube.timestamp == raw_cube.timestamp
