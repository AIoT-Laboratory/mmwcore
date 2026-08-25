"""Self-describing ADC Archive v3 access backed by the Rust core."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from mmwcore import _native
from mmwcore.config import RadarCaptureSpec


class ADCArchiveError(ValueError):
    """Raised when an ADC archive is malformed, incomplete, or inconsistent."""


class ADCArchive:
    """Read-only access to one self-describing ADC Archive v3 file."""

    __slots__ = ("_capture", "_native")

    def __init__(self, native: _native.ADCArchiveFile) -> None:
        self._native = native
        try:
            record = json.loads(native.capture_json)
            self._capture = RadarCaptureSpec.from_record(record)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ADCArchiveError("Embedded ADC archive capture metadata is invalid.") from exc
        expected_frame_bytes = self._capture.adc.raw_values_per_frame * np.dtype(np.int16).itemsize
        if (
            native.frame_bytes != expected_frame_bytes
            or native.frame_count != self._capture.num_frames
        ):
            raise ADCArchiveError("Embedded ADC archive Header and capture metadata disagree.")

    @property
    def path(self) -> Path:
        return Path(self._native.path)

    @property
    def capture(self) -> RadarCaptureSpec:
        """Complete decoding contract restored from the archive Header."""

        return self._capture

    @property
    def capture_sha256(self) -> str:
        """SHA-256 of the exact embedded capture metadata bytes."""

        return self._native.capture_sha256

    @property
    def adc_sha256(self) -> str:
        """SHA-256 of all logical raw ADC frames in order."""

        return self._native.adc_sha256

    @property
    def frame_bytes(self) -> int:
        return self._native.frame_bytes

    @property
    def frame_count(self) -> int:
        return self._native.frame_count

    @property
    def block_samples(self) -> int:
        """Number of int16 residuals in each adaptive Rice block."""

        return self._native.block_samples

    @property
    def restart_frames(self) -> int:
        """Maximum temporal dependency length in radar frames."""

        return self._native.restart_frames

    @property
    def archive_size(self) -> int:
        return self._native.archive_size

    @property
    def payload_bytes(self) -> int:
        return self._native.payload_bytes

    @property
    def index_bytes(self) -> int:
        return self._native.index_bytes

    @property
    def header_bytes(self) -> int:
        return self._native.header_bytes

    @property
    def capture_metadata_bytes(self) -> int:
        return self._native.capture_metadata_bytes

    @property
    def container_overhead_bytes(self) -> int:
        return self._native.container_overhead_bytes

    def read_frames(self, start: int, stop: int, *, verify: bool = True) -> bytes:
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("start must be an integer.")
        if isinstance(stop, bool) or not isinstance(stop, int):
            raise TypeError("stop must be an integer.")
        if start < 0 or stop < 0:
            raise ValueError("ADC frame interval must be non-negative.")
        try:
            return self._native.read_frames(start, stop, verify=verify)
        except (TypeError, ValueError) as exc:
            raise ADCArchiveError(str(exc)) from exc

    def read_windows(
        self,
        starts: Sequence[int],
        window_frames: int,
        *,
        verify: bool = True,
    ) -> bytes:
        """Read equal-length windows in caller order, decoding shared chunks once."""

        if isinstance(starts, str | bytes) or not isinstance(starts, Sequence):
            raise TypeError("starts must be a sequence of integers.")
        if isinstance(window_frames, bool) or not isinstance(window_frames, int):
            raise TypeError("window_frames must be an integer.")
        if window_frames <= 0:
            raise ValueError("window_frames must be greater than zero.")
        normalized_starts = []
        for index, start in enumerate(starts):
            if isinstance(start, bool) or not isinstance(start, int):
                raise TypeError(f"starts[{index}] must be an integer.")
            if start < 0:
                raise ValueError(f"starts[{index}] must be non-negative.")
            normalized_starts.append(start)
        try:
            return self._native.read_windows(
                normalized_starts,
                window_frames,
                verify=verify,
            )
        except (OverflowError, TypeError, ValueError) as exc:
            raise ADCArchiveError(str(exc)) from exc

    def verify_all(self) -> None:
        try:
            self._native.verify_all()
        except ValueError as exc:
            raise ADCArchiveError(str(exc)) from exc

    def revalidate_input(self) -> None:
        try:
            self._native.revalidate_input()
        except ValueError as exc:
            raise ADCArchiveError(str(exc)) from exc


def write_adc_archive(
    source: str | Path,
    destination: str | Path,
    capture: RadarCaptureSpec,
    *,
    expected_adc_sha256: str | None = None,
) -> ADCArchive:
    """Write one self-describing v3 archive through the Rust implementation."""

    if not isinstance(capture, RadarCaptureSpec):
        raise TypeError("capture must be a RadarCaptureSpec.")
    source_path = _path(source, "source")
    destination_path = _path(destination, "destination")
    frame_bytes = capture.adc.raw_values_per_frame * np.dtype(np.int16).itemsize
    frame_count, remainder = divmod(source_path.stat().st_size, frame_bytes)
    if remainder:
        raise ValueError("ADC source contains an incomplete trailing frame.")
    if frame_count == 0:
        raise ValueError("ADC source must contain at least one complete frame.")
    if capture.num_frames is not None and capture.num_frames != frame_count:
        raise ValueError("ADC source frame count does not match RadarCaptureSpec.num_frames.")
    embedded_capture = replace(capture, num_frames=frame_count)
    capture_json = json.dumps(
        embedded_capture.to_record(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        native = _native.write_adc_archive_file(
            str(source_path),
            str(destination_path),
            capture_json,
            expected_adc_sha256,
        )
    except ValueError as exc:
        raise ADCArchiveError(str(exc)) from exc
    return ADCArchive(native)


def open_adc_archive(path: str | Path) -> ADCArchive:
    """Open one complete v3 archive without an external decoding contract."""

    archive_path = _path(path, "archive")
    try:
        native = _native.open_adc_archive_file(str(archive_path))
    except ValueError as exc:
        raise ADCArchiveError(str(exc)) from exc
    return ADCArchive(native)


def _path(value: str | Path, name: str) -> Path:
    if not isinstance(value, str | Path):
        raise TypeError(f"{name} must be a path.")
    return Path(value).resolve(strict=False)


__all__ = ["ADCArchive", "ADCArchiveError", "open_adc_archive", "write_adc_archive"]
