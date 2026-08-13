"""Temporary chunk payloads used only by the ADC storage benchmark."""

from __future__ import annotations

import bisect
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from benchmarks.adc_storage_codecs import decode, encode, selected_transform


@dataclass(frozen=True)
class ChunkRecord:
    first_frame: int
    frame_count: int
    offset: int
    stored_bytes: int
    raw_bytes: int
    sha256: str


@dataclass(frozen=True)
class PackResult:
    records: tuple[ChunkRecord, ...]
    logical_sha256: str
    pack_ns: int
    encode_ns: int
    selected_transforms: tuple[str, ...]


def write_payload(
    source: Path,
    *,
    destination: Path,
    start_frame: int,
    selected_frames: int,
    frame_bytes: int,
    chunk_frames: int,
    codec: str,
    zlib_level: int,
) -> PackResult:
    """Stream selected source frames into one temporary compressed payload."""

    if chunk_frames <= 0:
        raise ValueError("Chunk frames must be positive.")
    records: list[ChunkRecord] = []
    logical_hash = hashlib.sha256()
    encode_ns = 0
    selected_transforms: list[str] = []
    started = time.perf_counter_ns()
    with (
        source.open("rb", buffering=1024 * 1024) as source_stream,
        destination.open("wb", buffering=1024 * 1024) as output,
    ):
        source_stream.seek(start_frame * frame_bytes)
        for first_frame in range(0, selected_frames, chunk_frames):
            frame_count = min(chunk_frames, selected_frames - first_frame)
            raw = source_stream.read(frame_count * frame_bytes)
            if len(raw) != frame_count * frame_bytes:
                raise RuntimeError("Source changed while ADC storage benchmark was reading it.")
            logical_hash.update(raw)
            codec_started = time.perf_counter_ns()
            encoded = encode(
                raw,
                codec=codec,
                frame_bytes=frame_bytes,
                zlib_level=zlib_level,
            )
            encode_ns += time.perf_counter_ns() - codec_started
            selected_transforms.append(selected_transform(codec, encoded))
            records.append(
                ChunkRecord(
                    first_frame=first_frame,
                    frame_count=frame_count,
                    offset=output.tell(),
                    stored_bytes=len(encoded),
                    raw_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
            output.write(encoded)
    return PackResult(
        records=tuple(records),
        logical_sha256=logical_hash.hexdigest(),
        pack_ns=time.perf_counter_ns() - started,
        encode_ns=encode_ns,
        selected_transforms=tuple(selected_transforms),
    )


def replay_and_verify(
    source: Path,
    *,
    payload_path: Path,
    records: Sequence[ChunkRecord],
    start_frame: int,
    frame_bytes: int,
    codec: str,
) -> tuple[int, int]:
    """Sequentially replay every chunk and compare it with direct source reads."""

    replay_started = time.perf_counter_ns()
    decode_ns = 0
    with (
        payload_path.open("rb", buffering=1024 * 1024) as payload,
        source.open("rb", buffering=1024 * 1024) as source_stream,
    ):
        source_stream.seek(start_frame * frame_bytes)
        for record in records:
            decoded, record_decode_ns = decode_record(
                payload,
                record,
                codec=codec,
                frame_bytes=frame_bytes,
            )
            decode_ns += record_decode_ns
            expected = source_stream.read(record.raw_bytes)
            if len(expected) != record.raw_bytes:
                raise RuntimeError("Source changed while ADC storage benchmark was replaying it.")
            if decoded != expected:
                raise RuntimeError("Sequential ADC storage replay differs from source ADC bytes.")
    return time.perf_counter_ns() - replay_started, decode_ns


def read_frames(
    payload: BinaryIO,
    *,
    records: Sequence[ChunkRecord],
    record_first_frames: Sequence[int],
    start_frame: int,
    stop_frame: int,
    frame_bytes: int,
    codec: str,
    verify_digest: bool = True,
) -> tuple[bytes, int]:
    """Read one logical frame window, decoding each intersecting chunk once."""

    if start_frame < 0 or stop_frame <= start_frame:
        raise ValueError("Frame window must have non-negative start and positive length.")
    first_record = max(0, bisect.bisect_right(record_first_frames, start_frame) - 1)
    output = bytearray()
    decoded_chunks = 0
    for record in records[first_record:]:
        record_stop = record.first_frame + record.frame_count
        if record.first_frame >= stop_frame:
            break
        if record_stop <= start_frame:
            continue
        decoded, _ = decode_record(
            payload,
            record,
            codec=codec,
            frame_bytes=frame_bytes,
            verify_digest=verify_digest,
        )
        decoded_chunks += 1
        local_start = max(start_frame, record.first_frame) - record.first_frame
        local_stop = min(stop_frame, record_stop) - record.first_frame
        output.extend(decoded[local_start * frame_bytes : local_stop * frame_bytes])
    expected_bytes = (stop_frame - start_frame) * frame_bytes
    if len(output) != expected_bytes:
        raise RuntimeError("ADC storage payload index does not cover the requested frame window.")
    return bytes(output), decoded_chunks


def decode_record(
    reader: BinaryIO,
    record: ChunkRecord,
    *,
    codec: str,
    frame_bytes: int,
    verify_digest: bool = True,
) -> tuple[bytes, int]:
    reader.seek(record.offset)
    encoded = reader.read(record.stored_bytes)
    if len(encoded) != record.stored_bytes:
        raise RuntimeError("Temporary ADC storage payload was truncated.")
    started = time.perf_counter_ns()
    raw = decode(encoded, codec=codec, frame_bytes=frame_bytes)
    decode_ns = time.perf_counter_ns() - started
    if len(raw) != record.raw_bytes:
        raise RuntimeError("ADC storage codec produced an unexpected raw chunk size.")
    if verify_digest and hashlib.sha256(raw).hexdigest() != record.sha256:
        raise RuntimeError("ADC storage codec did not reproduce its recorded raw chunk.")
    return raw, decode_ns


def read_source_frames(
    source: Path,
    *,
    absolute_first_frame: int,
    frame_count: int,
    frame_bytes: int,
) -> bytes:
    with source.open("rb") as handle:
        handle.seek(absolute_first_frame * frame_bytes)
        payload = handle.read(frame_count * frame_bytes)
    if len(payload) != frame_count * frame_bytes:
        raise RuntimeError("Source changed while ADC storage benchmark was reading it.")
    return payload
