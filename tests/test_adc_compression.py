import hashlib
from pathlib import Path

import numpy as np
import pytest
from test_adc_archive import _capture, _write_source

from mmwcore.io import (
    ADCArchive,
    ADCArchiveError,
    ADCArchiveReader,
    ADCCompressionError,
    CompressedADC,
    CompressedADCReader,
    compress_adc_file,
    compress_adc_frames,
    decompress_adc_file,
    decompress_adc_frames,
    open_adc_archive,
    open_compressed_adc,
    write_adc_archive,
)


def test_compress_decompress_file_restores_bytes_and_capture(tmp_path: Path):
    capture = _capture(num_frames=9)  # More than one restart group and a partial last group.
    source, raw = _write_source(tmp_path, capture)
    compressed = compress_adc_file(source, tmp_path / "radar.mmwa", capture)
    restored = tmp_path / "restored.bin"
    assert isinstance(compressed, CompressedADC)
    assert compressed.compressed_size_bytes == compressed.path.stat().st_size
    assert decompress_adc_file(compressed.path, restored) == capture
    assert restored.read_bytes() == raw
    assert source.read_bytes() == raw
    assert compressed.decompress_frames(0, 9) == raw
    assert compressed.decompress_windows([8, 0, 8], 1) == raw[-16:] + raw[:16] + raw[-16:]
    reader = CompressedADCReader(compressed.path)
    assert reader.compressed_adc is reader.archive
    np.testing.assert_array_equal(reader.read_frame(8).samples, np.frombuffer(raw[-16:], "<i2"))


def test_frame_codec_is_distinct_from_self_describing_file():
    raw = np.tile(np.arange(512, dtype="<i2"), 4).tobytes()
    payload = compress_adc_frames(raw, 1024)
    assert decompress_adc_frames(payload, 1024, 4) == raw
    with pytest.raises(ADCCompressionError):
        decompress_adc_frames(payload + b"extra", 1024, 4)
    with pytest.raises(TypeError, match="frame_bytes"):
        compress_adc_frames(raw, True)


def test_old_names_remain_compatible():
    assert ADCArchive is CompressedADC
    assert ADCArchiveError is ADCCompressionError
    assert ADCArchiveReader is CompressedADCReader
    assert write_adc_archive is compress_adc_file
    assert open_adc_archive is open_compressed_adc


def test_decompression_never_overwrites_existing_destination(tmp_path: Path):
    capture = _capture()
    source, _ = _write_source(tmp_path, capture)
    compressed = compress_adc_file(source, tmp_path / "radar.mmwa", capture)
    output = tmp_path / "existing.bin"
    output.write_bytes(b"existing")
    with pytest.raises(ADCCompressionError, match="already exists"):
        decompress_adc_file(compressed.path, output)
    assert output.read_bytes() == b"existing"


def test_corrupt_payload_cannot_publish_decompressed_file(tmp_path: Path):
    capture = _capture()
    source, _ = _write_source(tmp_path, capture)
    compressed = compress_adc_file(source, tmp_path / "radar.mmwa", capture)
    content = bytearray(compressed.path.read_bytes())
    content[compressed.header_bytes + 1] ^= 1
    compressed.path.write_bytes(content)
    output = tmp_path / "restored.bin"
    with pytest.raises(ADCCompressionError):
        decompress_adc_file(compressed.path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_missing_source_preserves_io_exception(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        decompress_adc_file(tmp_path / "missing.mmwa", tmp_path / "restored.bin")


def test_full_stream_digest_is_checked_before_publication(tmp_path: Path):
    capture = _capture()
    source, _ = _write_source(tmp_path, capture)
    compressed = compress_adc_file(source, tmp_path / "radar.mmwa", capture)
    content = bytearray(compressed.path.read_bytes())
    footer = len(content) - 160
    content[footer + 96] ^= 1
    content[footer + 128 :] = hashlib.sha256(content[footer : footer + 128]).digest()
    compressed.path.write_bytes(content)
    output = tmp_path / "restored.bin"
    with pytest.raises(ADCCompressionError, match="stream SHA-256"):
        decompress_adc_file(compressed.path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))
