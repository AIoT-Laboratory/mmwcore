"""Immutable, verified storage for finite raw ADC frames."""

from __future__ import annotations

import hashlib
import hmac
import operator
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final, cast

from mmwcore import _native

_HEADER = struct.Struct("<8sIIQQ32s")
_INDEX_RECORD = struct.Struct("<QQ32s")
_FOOTER_BODY = struct.Struct("<8sIIQQ32s32s32s")
_FOOTER = struct.Struct("<8sIIQQ32s32s32s32s")

_HEADER_MAGIC: Final = b"MMWADCA1"
_FOOTER_MAGIC: Final = b"MMWACMT1"
_VERSION: Final = 1
_HEADER_SIZE: Final = _HEADER.size
_INDEX_RECORD_SIZE: Final = _INDEX_RECORD.size
_FOOTER_SIZE: Final = _FOOTER.size
_SHA256_BYTES: Final = hashlib.sha256().digest_size
_MAX_FRAME_BYTES: Final = 64 * 1024 * 1024


class ADCArchiveError(ValueError):
    """Raised when an ADC archive is malformed, incomplete, or untrusted."""


@dataclass(frozen=True)
class _FrameRecord:
    offset: int
    stored_bytes: int
    raw_sha256: bytes


@dataclass(frozen=True)
class _Fingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int


class ADCArchive:
    """A read-only archive of independently encoded ADC frames."""

    def __init__(
        self,
        path: Path,
        *,
        header: bytes,
        footer: bytes,
        index: bytes,
        frame_bytes: int,
        frame_count: int,
        capture_contract_sha256: bytes,
        adc_sha256: bytes,
        index_offset: int,
        records: tuple[_FrameRecord, ...],
        fingerprint: _Fingerprint,
    ) -> None:
        self._path = path
        self._header = header
        self._footer = footer
        self._index = index
        self._frame_bytes = frame_bytes
        self._frame_count = frame_count
        self._capture_contract_sha256 = capture_contract_sha256
        self._adc_sha256 = adc_sha256
        self._index_offset = index_offset
        self._records = records
        self._fingerprint = fingerprint
        self._verified_all = False

    @property
    def frame_bytes(self) -> int:
        """The exact decoded byte length of every frame."""

        return self._frame_bytes

    @property
    def path(self) -> Path:
        """The resolved archive file path."""

        return self._path

    @property
    def frame_count(self) -> int:
        """The number of fixed-size raw frames."""

        return self._frame_count

    @property
    def adc_sha256(self) -> str:
        """SHA-256 of the concatenated logical raw ADC bytes."""

        return self._adc_sha256.hex()

    @property
    def capture_contract_sha256(self) -> str:
        """SHA-256 identifying the external capture contract."""

        return self._capture_contract_sha256.hex()

    @property
    def archive_size(self) -> int:
        """Total physical archive size in bytes."""

        return self._fingerprint.size

    @property
    def payload_bytes(self) -> int:
        """Total encoded payload size in bytes."""

        return self._index_offset - _HEADER_SIZE

    @property
    def index_bytes(self) -> int:
        """Total fixed index size in bytes."""

        return len(self._index)

    @property
    def metadata_bytes(self) -> int:
        """Header, index, and commit footer size in bytes."""

        return _HEADER_SIZE + self.index_bytes + _FOOTER_SIZE

    def read_frames(self, start: int, stop: int, *, verify: bool = True) -> bytes:
        """Decode the half-open frame interval, verifying every frame by default."""

        start, stop = _validate_frame_interval(start, stop, self._frame_count)
        if not isinstance(verify, bool):
            raise TypeError("verify must be a bool.")
        if not verify and not self._verified_all:
            raise ADCArchiveError(
                "Trusted reads require a successful verify_all() on this archive object."
            )
        try:
            if start == stop:
                self._open_checked().close()
                return b""

            decoded: list[bytes] = []
            with self._open_checked() as archive:
                for record in self._records[start:stop]:
                    decoded.append(self._decode_record(archive, record, verify=verify))
                self._require_unchanged_stream(archive)
            return b"".join(decoded)
        except Exception:
            self._verified_all = False
            raise

    def verify_all(self) -> None:
        """Verify all frame digests and the archive-wide logical ADC digest."""

        self._verified_all = False
        logical_digest = hashlib.sha256()
        with self._open_checked() as archive:
            for record in self._records:
                logical_digest.update(self._decode_record(archive, record, verify=True))
            self._require_unchanged_stream(archive)
        if not hmac.compare_digest(logical_digest.digest(), self._adc_sha256):
            raise ADCArchiveError("Archive logical raw SHA-256 does not match the footer.")
        self._verified_all = True

    def revalidate_input(self) -> None:
        """Confirm that the opened archive file still has the same identity."""

        with self._open_checked() as archive:
            archive.seek(self._index_offset)
            if archive.read(len(self._index)) != self._index:
                self._verified_all = False
                raise ADCArchiveError("ADC archive index changed after it was opened.")
            self._require_unchanged_stream(archive)

    def _open_checked(self) -> BinaryIO:
        archive = self._path.open("rb")
        try:
            self._require_unchanged_stream(archive)
            if archive.read(_HEADER_SIZE) != self._header:
                raise ADCArchiveError("ADC archive header changed after it was opened.")
            archive.seek(self.archive_size - _FOOTER_SIZE)
            if archive.read(_FOOTER_SIZE) != self._footer:
                raise ADCArchiveError("ADC archive footer changed after it was opened.")
            self._require_unchanged_stream(archive)
            return archive
        except Exception:
            archive.close()
            raise

    def _require_unchanged_stream(self, archive: BinaryIO) -> None:
        if _fingerprint_stream(archive) != self._fingerprint:
            self._verified_all = False
            raise ADCArchiveError("ADC archive changed after it was opened.")

    def _decode_record(
        self,
        archive: BinaryIO,
        record: _FrameRecord,
        *,
        verify: bool,
    ) -> bytes:
        archive.seek(record.offset)
        encoded = _read_exact(archive, record.stored_bytes, "encoded frame payload")
        raw = _decode_adc_archive_frame(encoded, self._frame_bytes)
        if verify and not hmac.compare_digest(hashlib.sha256(raw).digest(), record.raw_sha256):
            raise ADCArchiveError("Decoded frame SHA-256 does not match the archive index.")
        return raw


def write_adc_archive(
    source: str | Path,
    destination: str | Path,
    *,
    frame_bytes: int,
    capture_contract_sha256: str,
    expected_adc_sha256: str | None = None,
) -> ADCArchive:
    """Encode a finite ADC file into an atomically committed ADC archive."""

    source_path = _require_regular_file(source, "source")
    destination_path = _require_new_destination(destination)
    frame_bytes = _require_frame_bytes(frame_bytes)
    capture_contract_digest = _require_sha256(
        capture_contract_sha256,
        "capture_contract_sha256",
    )
    expected_adc_digest = (
        None
        if expected_adc_sha256 is None
        else _require_sha256(expected_adc_sha256, "expected_adc_sha256")
    )

    source_fingerprint = _fingerprint(source_path)
    frame_count, remainder = divmod(source_fingerprint.size, frame_bytes)
    if remainder:
        raise ValueError("Source contains an incomplete trailing frame.")
    if frame_count == 0:
        raise ValueError("Source must contain at least one complete frame.")

    header = _HEADER.pack(
        _HEADER_MAGIC,
        _VERSION,
        _HEADER_SIZE,
        frame_bytes,
        frame_count,
        capture_contract_digest,
    )
    temporary_path = destination_path.with_name(f".{destination_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        records, logical_sha256, index_offset = _write_payloads(
            source_path,
            temporary_path,
            header,
            frame_bytes,
            frame_count,
            source_fingerprint,
        )
        if expected_adc_digest is not None and not hmac.compare_digest(
            logical_sha256,
            expected_adc_digest,
        ):
            raise ADCArchiveError("Source logical SHA-256 does not match expected_adc_sha256.")
        index = b"".join(
            _INDEX_RECORD.pack(record.offset, record.stored_bytes, record.raw_sha256)
            for record in records
        )
        index_sha256 = hashlib.sha256(index).digest()
        footer_body = _FOOTER_BODY.pack(
            _FOOTER_MAGIC,
            _VERSION,
            _FOOTER_SIZE,
            index_offset,
            len(index),
            hashlib.sha256(header).digest(),
            index_sha256,
            logical_sha256,
        )
        footer = footer_body + hashlib.sha256(footer_body).digest()
        with temporary_path.open("ab") as archive:
            archive.write(index)
            archive.write(footer)
            archive.flush()
            os.fsync(archive.fileno())

        verified = open_adc_archive(temporary_path)
        verified.verify_all()
        try:
            os.link(temporary_path, destination_path)
        except FileExistsError:
            raise FileExistsError(
                f"ADC archive destination already exists: {destination_path}"
            ) from None
        try:
            committed = open_adc_archive(destination_path)
            if not os.path.samefile(temporary_path, destination_path):
                raise ADCArchiveError(
                    "Published ADC archive no longer identifies the verified temporary file."
                )
            _fsync_directory(destination_path.parent)
        except Exception:
            _unlink_if_same_file(temporary_path, destination_path)
            raise
        try:
            temporary_path.unlink()
        except OSError:
            # The committed hard link is complete and verified; a stale temporary name is harmless.
            pass
        return committed
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def open_adc_archive(path: str | Path) -> ADCArchive:
    """Open a completely committed archive after strict structural validation."""

    archive_path = _require_regular_file(path, "archive")
    fingerprint = _fingerprint(archive_path)
    minimum_size = _HEADER_SIZE + _FOOTER_SIZE
    if fingerprint.size < minimum_size:
        raise ADCArchiveError("ADC archive is too small for a header and footer.")

    with archive_path.open("rb") as archive:
        if _fingerprint_stream(archive) != fingerprint:
            raise ADCArchiveError("ADC archive changed while it was being opened.")
        header, frame_bytes, frame_count, capture_contract_sha256 = _read_header(archive)
        footer, index_offset, index_bytes, index_sha256, adc_sha256 = _read_footer(
            archive,
            archive_size=fingerprint.size,
            header=header,
        )
        index = _read_index(
            archive,
            archive_size=fingerprint.size,
            frame_count=frame_count,
            index_offset=index_offset,
            index_bytes=index_bytes,
            index_sha256=index_sha256,
        )
        if _fingerprint_stream(archive) != fingerprint:
            raise ADCArchiveError("ADC archive changed while it was being opened.")

    if _fingerprint(archive_path) != fingerprint:
        raise ADCArchiveError("ADC archive changed while it was being opened.")
    records = _parse_records(index, index_offset, frame_bytes)
    return ADCArchive(
        archive_path,
        header=header,
        footer=footer,
        index=index,
        frame_bytes=frame_bytes,
        frame_count=frame_count,
        capture_contract_sha256=capture_contract_sha256,
        adc_sha256=adc_sha256,
        index_offset=index_offset,
        records=records,
        fingerprint=fingerprint,
    )


def _read_header(archive: BinaryIO) -> tuple[bytes, int, int, bytes]:
    header = _read_exact(archive, _HEADER_SIZE, "header")
    magic, version, header_size, frame_bytes, frame_count, contract_sha256 = _HEADER.unpack(header)
    if magic != _HEADER_MAGIC or version != _VERSION or header_size != _HEADER_SIZE:
        raise ADCArchiveError("ADC archive header is not mmwcore.adc_archive.v1.")
    if frame_bytes == 0 or frame_count == 0:
        raise ADCArchiveError("ADC archive must declare positive frame dimensions.")
    if frame_bytes % 2:
        raise ADCArchiveError("ADC archive frame byte count must be a multiple of two.")
    if frame_bytes > _MAX_FRAME_BYTES:
        raise ADCArchiveError(f"ADC archive frame byte count exceeds {_MAX_FRAME_BYTES}.")
    return header, frame_bytes, frame_count, contract_sha256


def _read_footer(
    archive: BinaryIO,
    *,
    archive_size: int,
    header: bytes,
) -> tuple[bytes, int, int, bytes, bytes]:
    archive.seek(archive_size - _FOOTER_SIZE)
    footer = _read_exact(archive, _FOOTER_SIZE, "commit footer")
    (
        magic,
        version,
        footer_size,
        index_offset,
        index_bytes,
        header_sha256,
        index_sha256,
        adc_sha256,
        footer_sha256,
    ) = _FOOTER.unpack(footer)
    if magic != _FOOTER_MAGIC or version != _VERSION or footer_size != _FOOTER_SIZE:
        raise ADCArchiveError("ADC archive footer is missing or invalid.")
    if not hmac.compare_digest(
        footer_sha256,
        hashlib.sha256(footer[:-_SHA256_BYTES]).digest(),
    ):
        raise ADCArchiveError("ADC archive footer SHA-256 does not match.")
    if not hmac.compare_digest(header_sha256, hashlib.sha256(header).digest()):
        raise ADCArchiveError("ADC archive header SHA-256 does not match the footer.")
    return footer, index_offset, index_bytes, index_sha256, adc_sha256


def _read_index(
    archive: BinaryIO,
    *,
    archive_size: int,
    frame_count: int,
    index_offset: int,
    index_bytes: int,
    index_sha256: bytes,
) -> bytes:
    expected_index_bytes = frame_count * _INDEX_RECORD_SIZE
    footer_offset = archive_size - _FOOTER_SIZE
    if index_bytes != expected_index_bytes:
        raise ADCArchiveError("ADC archive index length does not match frame count.")
    if index_offset < _HEADER_SIZE or index_offset + index_bytes != footer_offset:
        raise ADCArchiveError("ADC archive index and footer must terminate at physical EOF.")
    archive.seek(index_offset)
    index = _read_exact(archive, index_bytes, "index")
    if not hmac.compare_digest(index_sha256, hashlib.sha256(index).digest()):
        raise ADCArchiveError("ADC archive index SHA-256 does not match the footer.")
    return index


def _write_payloads(
    source_path: Path,
    temporary_path: Path,
    header: bytes,
    frame_bytes: int,
    frame_count: int,
    source_fingerprint: _Fingerprint,
) -> tuple[tuple[_FrameRecord, ...], bytes, int]:
    records: list[_FrameRecord] = []
    logical_digest = hashlib.sha256()
    with source_path.open("rb") as source_file, temporary_path.open("xb") as archive:
        archive.write(header)
        offset = _HEADER_SIZE
        for _ in range(frame_count):
            raw = _read_exact(source_file, frame_bytes, "source frame")
            logical_digest.update(raw)
            encoded = _encode_adc_archive_frame(raw)
            archive.write(encoded)
            records.append(
                _FrameRecord(
                    offset=offset,
                    stored_bytes=len(encoded),
                    raw_sha256=hashlib.sha256(raw).digest(),
                )
            )
            offset += len(encoded)
        if source_file.read(1):
            raise ADCArchiveError("Source changed while the ADC archive was being written.")
        archive.flush()
        os.fsync(archive.fileno())

    if _fingerprint(source_path) != source_fingerprint:
        raise ADCArchiveError("Source changed while the ADC archive was being written.")
    logical_sha256 = logical_digest.digest()
    if _sha256_file(source_path) != logical_sha256:
        raise ADCArchiveError("Source changed while the ADC archive was being written.")
    return tuple(records), logical_sha256, offset


def _parse_records(
    index: bytes,
    index_offset: int,
    frame_bytes: int,
) -> tuple[_FrameRecord, ...]:
    records: list[_FrameRecord] = []
    expected_offset = _HEADER_SIZE
    for position in range(0, len(index), _INDEX_RECORD_SIZE):
        offset, stored_bytes, raw_sha256 = _INDEX_RECORD.unpack_from(index, position)
        if stored_bytes == 0:
            raise ADCArchiveError("ADC archive frame payload must not be empty.")
        if stored_bytes > _maximum_encoded_frame_bytes(frame_bytes):
            raise ADCArchiveError("ADC archive frame payload exceeds its fixed bound.")
        if offset != expected_offset:
            raise ADCArchiveError("ADC archive payload offsets must be contiguous.")
        expected_offset += stored_bytes
        if expected_offset > index_offset:
            raise ADCArchiveError("ADC archive payload extends into the index.")
        records.append(_FrameRecord(offset, stored_bytes, raw_sha256))
    if expected_offset != index_offset:
        raise ADCArchiveError("ADC archive payload does not end at the index.")
    return tuple(records)


def _encode_adc_archive_frame(raw: bytes) -> bytes:
    return _native_bytes("encode_adc_archive_frame", raw)


def _decode_adc_archive_frame(encoded: bytes, expected_raw_bytes: int) -> bytes:
    raw = _native_bytes("decode_adc_archive_frame", encoded, expected_raw_bytes)
    if len(raw) != expected_raw_bytes:
        raise ADCArchiveError("Native decode_adc_archive_frame() returned the wrong byte length.")
    return raw


def _native_bytes(name: str, *args: object) -> bytes:
    function = getattr(_native, name, None)
    if not callable(function):
        raise RuntimeError(f"mmwcore native extension does not provide {name}().")
    try:
        result = function(*args)
    except Exception as exc:
        raise ADCArchiveError(f"Native {name}() failed.") from exc
    if type(result) is not bytes:
        raise ADCArchiveError(f"Native {name}() must return bytes.")
    if not result:
        raise ADCArchiveError(f"Native {name}() returned an empty payload.")
    return cast(bytes, result)


def _read_exact(stream: BinaryIO, length: int, label: str) -> bytes:
    value = stream.read(length)
    if len(value) != length:
        raise ADCArchiveError(f"ADC archive has a truncated {label}.")
    return value


def _validate_frame_interval(start: int, stop: int, frame_count: int) -> tuple[int, int]:
    start = _require_non_negative_int(start, "start")
    stop = _require_non_negative_int(stop, "stop")
    if start > stop or stop > frame_count:
        raise IndexError(f"Frame interval [{start}, {stop}) is outside [0, {frame_count}).")
    return start, stop


def _require_positive_int(value: int, name: str) -> int:
    value = _require_non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _require_frame_bytes(value: int) -> int:
    result = _require_positive_int(value, "frame_bytes")
    if result % 2:
        raise ValueError("frame_bytes must be a multiple of two.")
    if result > _MAX_FRAME_BYTES:
        raise ValueError(f"frame_bytes must not exceed {_MAX_FRAME_BYTES}.")
    return result


def _maximum_encoded_frame_bytes(frame_bytes: int) -> int:
    return frame_bytes + frame_bytes // 16 + 1024


def _require_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _require_sha256(value: str, name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a lowercase hexadecimal string.")
    if len(value) != _SHA256_BYTES * 2 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{name} must be exactly {_SHA256_BYTES * 2} lowercase hexadecimal characters."
        )
    return bytes.fromhex(value)


def _require_regular_file(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a path.")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{name} must be an existing regular file: {path}")
    return path.resolve(strict=True)


def _require_new_destination(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("destination must be a path.")
    path = Path(value)
    if path.exists():
        raise FileExistsError(f"ADC archive destination already exists: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"ADC archive destination parent does not exist: {path.parent}")
    return path.parent.resolve(strict=True) / path.name


def _fingerprint(path: Path) -> _Fingerprint:
    stat = path.stat()
    if not path.is_file():
        raise ADCArchiveError(f"ADC archive path is no longer a regular file: {path}")
    return _Fingerprint(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _fingerprint_stream(stream: BinaryIO) -> _Fingerprint:
    stat = os.fstat(stream.fileno())
    return _Fingerprint(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.digest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_same_file(source: Path, destination: Path) -> None:
    try:
        if os.path.samefile(source, destination):
            destination.unlink(missing_ok=True)
    except OSError:
        pass
