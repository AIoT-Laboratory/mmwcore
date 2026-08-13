from __future__ import annotations

import hashlib
import os
import struct
import zlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmwcore.io import evidence_archive
from mmwcore.io.evidence_archive import (
    EvidenceArchiveError,
    open_evidence_archive,
    write_evidence_archive,
)

_HEADER_SIZE = 64
_INDEX_SIZE = 48
_FOOTER_SIZE = 160


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
        SimpleNamespace(
            encode_evidence_frame=encode,
            decode_evidence_frame=decode,
        ),
    )


def _raw_frames(frame_bytes: int = 32, count: int = 6) -> bytes:
    return b"".join(bytes([index]) * frame_bytes for index in range(count))


def _write_source(tmp_path: Path, raw: bytes) -> Path:
    source = tmp_path / "adc.bin"
    source.write_bytes(raw)
    return source


def _contract() -> str:
    return hashlib.sha256(b"capture-contract").hexdigest()


def _archive(tmp_path: Path, raw: bytes | None = None) -> tuple[Path, bytes]:
    payload = _raw_frames() if raw is None else raw
    source = _write_source(tmp_path, payload)
    destination = tmp_path / "capture.mmwe"
    write_evidence_archive(
        source,
        destination,
        frame_bytes=32,
        capture_contract_sha256=_contract(),
    )
    return destination, payload


def test_roundtrip_random_windows_and_metadata_overhead(tmp_path: Path) -> None:
    destination, raw = _archive(tmp_path)
    archive = open_evidence_archive(destination)

    assert archive.frame_bytes == 32
    assert archive.frame_count == 6
    assert archive.capture_contract_sha256 == _contract()
    assert archive.evidence_sha256 == hashlib.sha256(raw).hexdigest()
    assert archive.read_frames(0, 6) == raw
    assert archive.read_frames(1, 4) == raw[32:128]
    assert archive.read_frames(5, 6) == raw[160:]
    assert archive.index_bytes == 6 * _INDEX_SIZE
    assert archive.metadata_bytes == _HEADER_SIZE + archive.index_bytes + _FOOTER_SIZE
    assert archive.archive_size == archive.payload_bytes + archive.metadata_bytes


def test_trusted_read_requires_a_successful_full_verification(tmp_path: Path) -> None:
    destination, raw = _archive(tmp_path)
    archive = open_evidence_archive(destination)

    with pytest.raises(EvidenceArchiveError, match="verify_all"):
        archive.read_frames(0, 1, verify=False)
    archive.verify_all()
    assert archive.read_frames(2, 5, verify=False) == raw[64:160]

    destination.write_bytes(destination.read_bytes())
    with pytest.raises(EvidenceArchiveError, match="changed"):
        archive.read_frames(0, 1, verify=False)
    with pytest.raises(EvidenceArchiveError, match="verify_all"):
        archive.read_frames(0, 1, verify=False)


def test_failed_verified_read_revokes_trusted_state(tmp_path: Path) -> None:
    destination, _ = _archive(tmp_path)
    archive = open_evidence_archive(destination)
    archive.verify_all()

    records = vars(archive)["_records"]
    original = records[0]
    object.__setattr__(
        archive,
        "_records",
        (
            type(original)(original.offset, original.stored_bytes, b"\x00" * 32),
            *records[1:],
        ),
    )
    with pytest.raises(EvidenceArchiveError, match="Decoded frame SHA-256"):
        archive.read_frames(0, 1)
    with pytest.raises(EvidenceArchiveError, match="verify_all"):
        archive.read_frames(1, 2, verify=False)


@pytest.mark.parametrize("area", ["header", "index", "footer"])
def test_header_index_and_footer_tampering_fail_structural_open(
    tmp_path: Path,
    area: str,
) -> None:
    destination, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    index_offset = struct.unpack_from("<Q", payload, len(payload) - _FOOTER_SIZE + 16)[0]
    footer_offset = len(payload) - _FOOTER_SIZE
    offsets = {
        "header": 0,
        "index": index_offset,
        "footer": footer_offset + 32,
    }
    actual_offset = offsets[area]
    payload[actual_offset] ^= 0x01
    destination.write_bytes(payload)

    with pytest.raises(EvidenceArchiveError):
        open_evidence_archive(destination)


def test_payload_tampering_and_decode_error_fail_verified_read(tmp_path: Path) -> None:
    destination, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    payload[_HEADER_SIZE + 1] ^= 0xFF
    destination.write_bytes(payload)
    archive = open_evidence_archive(destination)

    with pytest.raises(EvidenceArchiveError, match="Native decode_evidence_frame"):
        archive.read_frames(0, 1)


def test_logical_footer_digest_tampering_fails_structural_open(tmp_path: Path) -> None:
    destination, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    payload[-1] ^= 0x01
    destination.write_bytes(payload)

    with pytest.raises(EvidenceArchiveError, match="footer SHA-256"):
        open_evidence_archive(destination)


def test_self_consistent_wrong_logical_digest_fails_full_verification(tmp_path: Path) -> None:
    destination, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    footer_offset = len(payload) - _FOOTER_SIZE
    payload[footer_offset + 96] ^= 0x01
    footer_body = bytes(payload[footer_offset : footer_offset + 128])
    payload[footer_offset + 128 :] = hashlib.sha256(footer_body).digest()
    destination.write_bytes(payload)

    archive = open_evidence_archive(destination)
    with pytest.raises(EvidenceArchiveError, match="logical raw SHA-256"):
        archive.verify_all()


@pytest.mark.parametrize(
    "mutator",
    [lambda value: value[:-1], lambda value: value + b"tail"],
)
def test_truncation_and_trailing_bytes_fail_open(
    tmp_path: Path,
    mutator: Callable[[bytes], bytes],
) -> None:
    destination, _ = _archive(tmp_path)
    destination.write_bytes(mutator(destination.read_bytes()))

    with pytest.raises(EvidenceArchiveError):
        open_evidence_archive(destination)


def test_write_is_atomic_and_cleans_temporary_file_after_codec_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_frames(count=2)
    source = _write_source(tmp_path, raw)
    destination = tmp_path / "capture.mmwe"
    calls = 0

    def fail_after_one_frame(value: bytes) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("native encoder failed")
        return zlib.compress(value, level=1)

    monkeypatch.setattr(evidence_archive, "_encode_evidence_frame", fail_after_one_frame)

    with pytest.raises(RuntimeError, match="native encoder failed"):
        write_evidence_archive(
            source,
            destination,
            frame_bytes=32,
            capture_contract_sha256=_contract(),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".capture.mmwe.*.tmp"))


def test_atomic_publication_never_overwrites_a_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path, _raw_frames(count=2))
    destination = tmp_path / "capture.mmwe"

    def race(_source: Path, target: Path) -> None:
        target.write_bytes(b"racing-writer")
        raise FileExistsError

    monkeypatch.setattr(os, "link", race)

    with pytest.raises(FileExistsError, match="already exists"):
        write_evidence_archive(
            source,
            destination,
            frame_bytes=32,
            capture_contract_sha256=_contract(),
        )
    assert destination.read_bytes() == b"racing-writer"
    assert not list(tmp_path.glob(".capture.mmwe.*.tmp"))


def test_source_change_aborts_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_frames(count=2)
    source = _write_source(tmp_path, raw)
    destination = tmp_path / "capture.mmwe"
    calls = 0

    def alter_source(value: bytes) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            source.write_bytes(b"z" * len(raw))
            os.utime(source, None)
        return zlib.compress(value, level=1)

    monkeypatch.setattr(evidence_archive, "_encode_evidence_frame", alter_source)

    with pytest.raises(EvidenceArchiveError, match="Source changed"):
        write_evidence_archive(
            source,
            destination,
            frame_bytes=32,
            capture_contract_sha256=_contract(),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".capture.mmwe.*.tmp"))


def test_expected_logical_digest_mismatch_aborts_before_publication(tmp_path: Path) -> None:
    source = _write_source(tmp_path, _raw_frames(count=2))
    destination = tmp_path / "capture.mmwe"

    with pytest.raises(EvidenceArchiveError, match="expected_evidence_sha256"):
        write_evidence_archive(
            source,
            destination,
            frame_bytes=32,
            capture_contract_sha256=_contract(),
            expected_evidence_sha256="0" * 64,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".capture.mmwe.*.tmp"))


def test_rejects_strict_paths_integers_and_digests(tmp_path: Path) -> None:
    source = _write_source(tmp_path, _raw_frames())
    destination = tmp_path / "capture.mmwe"
    with pytest.raises(TypeError, match="not bool"):
        write_evidence_archive(
            source, destination, frame_bytes=True, capture_contract_sha256=_contract()
        )
    with pytest.raises(ValueError, match="64 lowercase"):
        write_evidence_archive(source, destination, frame_bytes=32, capture_contract_sha256="short")
    with pytest.raises(ValueError, match="multiple of two"):
        write_evidence_archive(
            source, destination, frame_bytes=31, capture_contract_sha256=_contract()
        )
    with pytest.raises(FileNotFoundError, match="parent"):
        write_evidence_archive(
            source,
            tmp_path / "missing" / "capture.mmwe",
            frame_bytes=32,
            capture_contract_sha256=_contract(),
        )
    source.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="trailing"):
        write_evidence_archive(
            source,
            destination,
            frame_bytes=4,
            capture_contract_sha256=_contract(),
        )


def test_index_offsets_must_be_contiguous(tmp_path: Path) -> None:
    destination, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    index_offset = struct.unpack_from("<Q", payload, len(payload) - _FOOTER_SIZE + 16)[0]
    first_offset = struct.unpack_from("<Q", payload, index_offset)[0]
    struct.pack_into("<Q", payload, index_offset, first_offset + 1)
    index_bytes = bytes(payload[index_offset : len(payload) - _FOOTER_SIZE])
    struct.pack_into(
        "<32s",
        payload,
        len(payload) - _FOOTER_SIZE + 64,
        hashlib.sha256(index_bytes).digest(),
    )
    footer_offset = len(payload) - _FOOTER_SIZE
    footer_body = bytes(payload[footer_offset : footer_offset + 128])
    payload[footer_offset + 128 :] = hashlib.sha256(footer_body).digest()
    destination.write_bytes(payload)

    with pytest.raises(EvidenceArchiveError, match="contiguous"):
        open_evidence_archive(destination)


def test_index_rejects_encoded_frame_larger_than_fixed_bound(tmp_path: Path) -> None:
    destination, _ = _archive(tmp_path)
    payload = bytearray(destination.read_bytes())
    footer_offset = len(payload) - _FOOTER_SIZE
    index_offset = struct.unpack_from("<Q", payload, footer_offset + 16)[0]
    first_offset = struct.unpack_from("<Q", payload, index_offset)[0]
    struct.pack_into("<Q", payload, index_offset + 8, 4096)
    struct.pack_into("<Q", payload, index_offset + _INDEX_SIZE, first_offset + 4096)
    index_bytes = bytes(payload[index_offset:footer_offset])
    struct.pack_into("<32s", payload, footer_offset + 64, hashlib.sha256(index_bytes).digest())
    footer_body = bytes(payload[footer_offset : footer_offset + 128])
    payload[footer_offset + 128 :] = hashlib.sha256(footer_body).digest()
    destination.write_bytes(payload)

    with pytest.raises(EvidenceArchiveError, match="fixed bound"):
        open_evidence_archive(destination)
