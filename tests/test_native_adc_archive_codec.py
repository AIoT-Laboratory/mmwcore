from __future__ import annotations

from pathlib import Path

import pytest

from mmwcore import _native
from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCFrameSpec
from mmwcore.io import write_adc_archive


def test_native_adc_archive_codec_round_trip_uses_homologous_frame_delta() -> None:
    first = bytes((index * 37) % 256 for index in range(1024))
    second = bytes((value + 1) % 256 for value in first)
    raw = first + second

    encoded = _native.encode_adc_archive_chunk(raw, 1024)
    assert len(encoded) < len(raw)
    assert _native.decode_adc_archive_chunk(encoded, 1024, 2) == raw


def test_native_adc_archive_codec_rejects_invalid_and_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _native.encode_adc_archive_chunk(b"", 1024)
    with pytest.raises(ValueError, match="positive multiple of two"):
        _native.encode_adc_archive_chunk(b"abc", 3)
    with pytest.raises(ValueError, match="not a multiple"):
        _native.encode_adc_archive_chunk(b"abcdef", 4)
    with pytest.raises(ValueError, match="power of two"):
        _native.encode_adc_archive_chunk(bytes(1024), 1024, 128)
    with pytest.raises(ValueError, match="unsupported"):
        _native.decode_adc_archive_chunk(b"\x11", 1024, 1)
    encoded = _native.encode_adc_archive_chunk(bytes(1024), 1024)
    with pytest.raises(ValueError, match="trailing"):
        _native.decode_adc_archive_chunk(encoded + b"tail", 1024, 1)


def test_public_archive_round_trip_uses_native_codec(tmp_path: Path) -> None:
    raw = bytes((index * 19) % 256 for index in range(64))
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)

    archive = write_adc_archive(
        source,
        tmp_path / "adc.mmwa",
        RadarCaptureSpec(
            profile=RadarProfile(
                num_tx=1,
                num_rx=1,
                num_adc_samples=8,
                num_chirps_per_tx=1,
            ),
            adc=ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=8),
            tx_order=(0,),
            num_frames=2,
        ),
    )
    archive.verify_all()

    assert archive.block_samples == 512
    assert archive.restart_frames == 4
    assert archive.read_frames(0, 2, verify=False) == raw
