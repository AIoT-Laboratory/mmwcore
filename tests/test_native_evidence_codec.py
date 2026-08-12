from __future__ import annotations

import zlib
from pathlib import Path

import pytest

from mmwcore import _native
from mmwcore.io import write_evidence_archive


def _shuffle_i16_bytes(raw: bytes) -> bytes:
    return raw[::2] + raw[1::2]


def test_native_evidence_codec_round_trip_and_python_zlib_golden() -> None:
    raw = bytes((index * 37) % 256 for index in range(2048))

    encoded = _native.encode_evidence_frame(raw)
    assert zlib.decompress(encoded) == _shuffle_i16_bytes(raw)
    assert _native.decode_evidence_frame(encoded, len(raw)) == raw

    python_zlib_golden = zlib.compress(_shuffle_i16_bytes(raw), level=1)
    assert _native.decode_evidence_frame(python_zlib_golden, len(raw)) == raw


def test_native_evidence_codec_rejects_invalid_and_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _native.encode_evidence_frame(b"")
    with pytest.raises(ValueError, match="even"):
        _native.encode_evidence_frame(b"x")

    encoded = _native.encode_evidence_frame(b"\x01\x00\x02\x00")
    with pytest.raises(ValueError, match="expected raw byte count"):
        _native.decode_evidence_frame(encoded, 3)
    with pytest.raises(ValueError, match="beyond expected"):
        _native.decode_evidence_frame(encoded, 2)
    with pytest.raises(ValueError, match="trailing"):
        _native.decode_evidence_frame(encoded + b"tail", 4)
    with pytest.raises(ValueError, match="valid zlib"):
        _native.decode_evidence_frame(b"not a zlib frame", 4)


def test_public_archive_round_trip_uses_native_codec(tmp_path: Path) -> None:
    raw = bytes((index * 19) % 256 for index in range(64))
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)

    archive = write_evidence_archive(
        source,
        tmp_path / "adc.mmwe",
        frame_bytes=32,
        capture_contract_sha256="0123456789abcdef" * 4,
    )
    archive.verify_all()

    assert archive.read_frames(0, 2, verify=False) == raw
