"""Verified evidence-archive reader for finite raw ADC frames."""

from __future__ import annotations

import hmac
import operator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mmwcore.config import RadarCaptureSpec, capture_contract_sha256
from mmwcore.core import ADCFrameSpec, RawADCFrame

from .evidence_archive import EvidenceArchive, open_evidence_archive, write_evidence_archive


class ADCEvidenceArchiveFrameReader:
    """Random-access ADC reader backed by a verified evidence archive, never a memmap."""

    __slots__ = ("_archive", "_capture", "_metadata")

    def __init__(
        self,
        path: str | Path,
        capture: RadarCaptureSpec,
        *,
        expected_evidence_sha256: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(capture, RadarCaptureSpec):
            raise TypeError("capture must be a RadarCaptureSpec.")
        if metadata is not None and "tx_order" in metadata:
            raise ValueError("Archive reader metadata must not override capture tx_order.")
        archive = open_evidence_archive(path)
        _require_digest_match(
            capture_contract_sha256(capture),
            archive.capture_contract_sha256,
            "Evidence archive capture_contract_sha256",
        )
        _require_digest_match(
            expected_evidence_sha256,
            archive.evidence_sha256,
            "expected_evidence_sha256",
        )
        expected_frame_bytes = capture.adc.raw_values_per_frame * np.dtype(np.int16).itemsize
        if archive.frame_bytes != expected_frame_bytes:
            raise ValueError(
                "Evidence archive frame_bytes does not match ADCFrameSpec: "
                f"{archive.frame_bytes} != {expected_frame_bytes}."
            )
        if capture.num_frames is not None and archive.frame_count != capture.num_frames:
            raise ValueError(
                "Evidence archive frame count does not match RadarCaptureSpec: "
                f"{archive.frame_count} != {capture.num_frames}."
            )
        self._archive = archive
        self._capture = capture
        self._metadata = {"tx_order": list(capture.tx_order), **(metadata or {})}

    @property
    def archive(self) -> EvidenceArchive:
        return self._archive

    @property
    def spec(self) -> ADCFrameSpec:
        return self._capture.adc

    @property
    def frame_periodicity_s(self) -> float | None:
        return self._capture.frame_periodicity_s

    @property
    def num_frames(self) -> int:
        return self._archive.frame_count

    @property
    def path(self) -> Path:
        return self._archive.path

    def read_frame(self, index: int) -> RawADCFrame:
        """Decode and verify one frame by zero-based index."""

        if isinstance(index, bool):
            raise TypeError("ADC frame index must be an integer, not bool.")
        try:
            index = operator.index(index)
        except TypeError as exc:
            raise TypeError("ADC frame index must be an integer.") from exc
        if not 0 <= index < self.num_frames:
            raise IndexError(f"ADC frame index {index} is outside [0, {self.num_frames}).")
        samples = np.frombuffer(self._archive.read_frames(index, index + 1), dtype=np.dtype("<i2"))
        timestamp = (
            index * self.frame_periodicity_s if self.frame_periodicity_s is not None else None
        )
        return RawADCFrame(
            samples,
            frame_id=index,
            timestamp=timestamp,
            source=str(self._archive.path),
            profile=asdict(self._capture.profile),
            metadata={
                **self._metadata,
                "frame_index": index,
                "num_frames": self.num_frames,
                "evidence_sha256": self._archive.evidence_sha256,
                "capture_contract_sha256": self._archive.capture_contract_sha256,
            },
        )

    def verify_all(self) -> None:
        """Explicitly verify every frame and the logical ADC digest."""

        self._archive.verify_all()


def write_adc_evidence_archive(
    source: str | Path,
    destination: str | Path,
    capture: RadarCaptureSpec,
    *,
    expected_evidence_sha256: str,
) -> EvidenceArchive:
    """Atomically archive ADC bytes bound to their capture and source identities."""

    if not isinstance(capture, RadarCaptureSpec):
        raise TypeError("capture must be a RadarCaptureSpec.")
    source_path = Path(source)
    if capture.expected_size_bytes is not None and source_path.stat().st_size != (
        capture.expected_size_bytes
    ):
        raise ValueError("ADC source size does not match RadarCaptureSpec.expected_size_bytes.")
    return write_evidence_archive(
        source_path,
        destination,
        frame_bytes=capture.adc.raw_values_per_frame * np.dtype(np.int16).itemsize,
        capture_contract_sha256=capture_contract_sha256(capture),
        expected_evidence_sha256=expected_evidence_sha256,
    )


def _require_digest_match(expected: str, actual: str, name: str) -> None:
    _require_sha256(expected, name)
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"{name} does not match the verified archive value.")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string.")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters.")


__all__ = ["ADCEvidenceArchiveFrameReader", "write_adc_evidence_archive"]
