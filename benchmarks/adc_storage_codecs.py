"""Strictly reversible transforms used by the ADC storage benchmark."""

from __future__ import annotations

import zlib

import numpy as np

SUPPORTED_CODECS = (
    "raw",
    "zlib",
    "shuffle-zlib",
    "frame-delta-shuffle-zlib",
    "adaptive-shuffle-zlib",
)
DEFAULT_ZLIB_LEVEL = 1
_SHUFFLE_TAG = 0
_FRAME_DELTA_TAG = 1


def encode(payload: bytes, *, codec: str, frame_bytes: int, zlib_level: int) -> bytes:
    """Encode one complete-frame chunk without changing its logical bytes."""

    if codec == "raw":
        return payload
    if codec == "zlib":
        return zlib.compress(payload, level=zlib_level)
    if codec == "shuffle-zlib":
        return zlib.compress(_shuffle_words(payload), level=zlib_level)
    if codec == "frame-delta-shuffle-zlib":
        transformed = _shuffle_words(_frame_delta(payload, frame_bytes=frame_bytes))
        return zlib.compress(transformed, level=zlib_level)
    if codec == "adaptive-shuffle-zlib":
        shuffled = zlib.compress(_shuffle_words(payload), level=zlib_level)
        if len(payload) == frame_bytes:
            return bytes((_SHUFFLE_TAG,)) + shuffled
        delta_shuffled = zlib.compress(
            _shuffle_words(_frame_delta(payload, frame_bytes=frame_bytes)),
            level=zlib_level,
        )
        if len(delta_shuffled) < len(shuffled):
            return bytes((_FRAME_DELTA_TAG,)) + delta_shuffled
        return bytes((_SHUFFLE_TAG,)) + shuffled
    raise ValueError(f"Unsupported ADC storage codec: {codec!r}.")


def decode(payload: bytes, *, codec: str, frame_bytes: int) -> bytes:
    """Decode one chunk to its byte-exact little-endian ADC representation."""

    if codec == "raw":
        return payload
    if codec == "zlib":
        return zlib.decompress(payload)
    if codec == "shuffle-zlib":
        return _unshuffle_words(zlib.decompress(payload))
    if codec == "frame-delta-shuffle-zlib":
        deltas = _unshuffle_words(zlib.decompress(payload))
        return _restore_frame_delta(deltas, frame_bytes=frame_bytes)
    if codec == "adaptive-shuffle-zlib":
        if not payload:
            raise ValueError("Adaptive ADC storage payload is missing its transform tag.")
        transformed = _unshuffle_words(zlib.decompress(payload[1:]))
        if payload[0] == _SHUFFLE_TAG:
            return transformed
        if payload[0] == _FRAME_DELTA_TAG:
            return _restore_frame_delta(transformed, frame_bytes=frame_bytes)
        raise ValueError(f"Unsupported adaptive ADC storage transform tag: {payload[0]}.")
    raise ValueError(f"Unsupported ADC storage codec: {codec!r}.")


def selected_transform(codec: str, payload: bytes) -> str:
    """Return the reversible transform selected for one encoded chunk."""

    if codec != "adaptive-shuffle-zlib":
        return codec
    if not payload:
        raise ValueError("Adaptive ADC storage payload is missing its transform tag.")
    if payload[0] == _SHUFFLE_TAG:
        return "shuffle-zlib"
    if payload[0] == _FRAME_DELTA_TAG:
        return "frame-delta-shuffle-zlib"
    raise ValueError(f"Unsupported adaptive ADC storage transform tag: {payload[0]}.")


def _shuffle_words(payload: bytes) -> bytes:
    if len(payload) % 2:
        raise ValueError("ADC payload byte count must be divisible by two for word shuffle.")
    words = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 2)
    return np.ascontiguousarray(words.T).tobytes()


def _unshuffle_words(payload: bytes) -> bytes:
    if len(payload) % 2:
        raise ValueError("Shuffled ADC payload byte count must be divisible by two.")
    planes = np.frombuffer(payload, dtype=np.uint8).reshape(2, -1)
    return np.ascontiguousarray(planes.T).tobytes()


def _frame_delta(payload: bytes, *, frame_bytes: int) -> bytes:
    _validate_complete_frames(payload, frame_bytes=frame_bytes, label="Chunk")
    frames = np.frombuffer(payload, dtype=np.dtype("<u2")).reshape(-1, frame_bytes // 2)
    deltas = np.empty_like(frames)
    deltas[0] = frames[0]
    if len(frames) > 1:
        np.subtract(frames[1:], frames[:-1], dtype=np.dtype("<u2"), out=deltas[1:])
    return deltas.tobytes()


def _restore_frame_delta(payload: bytes, *, frame_bytes: int) -> bytes:
    _validate_complete_frames(payload, frame_bytes=frame_bytes, label="Delta")
    deltas = np.frombuffer(payload, dtype=np.dtype("<u2")).reshape(-1, frame_bytes // 2)
    restored = np.empty_like(deltas)
    np.add.accumulate(deltas, axis=0, dtype=np.dtype("<u2"), out=restored)
    return restored.tobytes()


def _validate_complete_frames(payload: bytes, *, frame_bytes: int, label: str) -> None:
    if frame_bytes <= 0 or frame_bytes % 2:
        raise ValueError("Frame bytes must be a positive multiple of two.")
    if not payload or len(payload) % frame_bytes:
        raise ValueError(f"{label} payload does not contain complete frames.")
