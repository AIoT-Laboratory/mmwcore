"""ADC frame-reader facade for self-describing archives."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from mmwcore.config import RadarCaptureSpec
from mmwcore.core import ADCComplexLayout, ADCFrame, ADCFrameSpec

from .adc_archive import ADCArchive, open_adc_archive
from .capture import _radar_capture

if TYPE_CHECKING:
    from .take import Take


class ADCArchiveReader:
    """Random-access ADC reader whose contract comes from the archive Header."""

    __slots__ = ("_archive", "_capture", "_metadata")

    def __init__(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if metadata is not None and "tx_order" in metadata:
            raise ValueError("Archive reader metadata must not override embedded tx_order.")
        self._archive = open_adc_archive(path)
        self._capture = self._archive.capture
        self._metadata = {"tx_order": list(self._capture.tx_order), **(metadata or {})}

    @classmethod
    def from_take(cls, take: Take) -> Self:
        """Use a verified take's CFG to correct the historical Q-first layout label.

        The embedded archive contract and raw bytes remain unchanged. Standalone
        archives have no CFG evidence and keep their original decoding contract.
        """
        reader = cls(take.archive.path)
        effective = _radar_capture(take.config_path.read_bytes())
        stored = reader.capture
        if stored != effective:
            corrected = replace(
                stored, adc=replace(stored.adc, layout=ADCComplexLayout.GROUP2_Q_THEN_I)
            )
            if stored.adc.layout is not ADCComplexLayout.GROUP2_I_THEN_Q or corrected != effective:
                raise ValueError("Take radar CFG disagrees with the archive capture contract.")
            reader._capture = effective
            reader._metadata["adc_layout_correction"] = {
                "stored": stored.adc.layout.value,
                "effective": effective.adc.layout.value,
                "source": "verified_take_radar_cfg_iq_swap_1",
            }
        return reader

    @property
    def archive(self) -> ADCArchive:
        return self._archive

    @property
    def capture(self) -> RadarCaptureSpec:
        return self._capture

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

    def read_frame(self, index: int) -> ADCFrame:
        index = self._frame_index(index)
        samples = np.frombuffer(self._archive.read_frames(index, index + 1), dtype=np.dtype("<i2"))
        return self._raw_frame(index, samples)

    def read_frames(self, indices: Sequence[int]) -> tuple[ADCFrame, ...]:
        """Read frames in caller order while sharing archive chunk decoding."""

        if isinstance(indices, str | bytes) or not isinstance(indices, Sequence):
            raise TypeError("ADC frame indices must be a sequence of integers.")
        normalized = tuple(self._frame_index(index) for index in indices)
        if not normalized:
            return ()
        samples = np.frombuffer(
            self._archive.read_windows(normalized, 1),
            dtype=np.dtype("<i2"),
        ).reshape(len(normalized), self.spec.raw_values_per_frame)
        return tuple(
            self._raw_frame(index, frame_samples)
            for index, frame_samples in zip(normalized, samples, strict=True)
        )

    def _frame_index(self, index: int) -> int:
        if isinstance(index, bool):
            raise TypeError("ADC frame index must be an integer, not bool.")
        try:
            index = operator.index(index)
        except TypeError as exc:
            raise TypeError("ADC frame index must be an integer.") from exc
        if not 0 <= index < self.num_frames:
            raise IndexError(f"ADC frame index {index} is outside [0, {self.num_frames}).")
        return index

    def _raw_frame(self, index: int, samples: np.ndarray) -> ADCFrame:
        timestamp = (
            index * self.frame_periodicity_s if self.frame_periodicity_s is not None else None
        )
        return ADCFrame(
            samples,
            frame_id=index,
            timestamp=timestamp,
            source=str(self._archive.path),
            profile=asdict(self._capture.profile),
            metadata={
                **self._metadata,
                "frame_index": index,
                "num_frames": self.num_frames,
                "adc_sha256": self._archive.adc_sha256,
                "capture_sha256": self._archive.capture_sha256,
            },
        )

    def verify_all(self) -> None:
        self._archive.verify_all()


__all__ = ["ADCArchiveReader"]
