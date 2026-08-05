"""ADC binary file loading helpers for offline mmwcore workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mmwcore.core import ADCFrameSpec, RadarCube, RawADCFrame
from mmwcore.dsp.adc import organize_adc_samples

if TYPE_CHECKING:
    from mmwcore.config import RadarCaptureSpec


@dataclass(frozen=True)
class ADCFileFrameReader:
    """Random-access memmap reader for fixed-shape int16 ADC frames."""

    path: str | Path
    spec: ADCFrameSpec
    frame_periodicity_s: float | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _num_frames: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.path)
        size_bytes = path.stat().st_size
        int16_values, trailing_bytes = divmod(size_bytes, np.dtype(np.int16).itemsize)
        complete_frames, leftover_values = divmod(
            int16_values,
            self.spec.raw_values_per_frame,
        )
        if trailing_bytes:
            raise ValueError(f"ADC file has {trailing_bytes} trailing byte(s).")
        if leftover_values:
            raise ValueError(
                "ADC file does not contain a whole number of frames; "
                f"got {leftover_values} leftover int16 value(s)."
            )
        if complete_frames == 0:
            raise ValueError("ADC file contains no complete frames.")
        if self.frame_periodicity_s is not None and self.frame_periodicity_s <= 0:
            raise ValueError("ADCFileFrameReader.frame_periodicity_s must be positive.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "profile", dict(self.profile))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "_num_frames", complete_frames)

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @classmethod
    def from_capture(
        cls,
        path: str | Path,
        capture: RadarCaptureSpec,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ADCFileFrameReader:
        """Open a file and validate it against an explicit capture contract."""

        reader = cls(
            path=path,
            spec=capture.adc,
            frame_periodicity_s=capture.frame_periodicity_s,
            profile=asdict(capture.profile),
            metadata={"tx_order": list(capture.tx_order), **(metadata or {})},
        )
        if capture.num_frames is not None and reader.num_frames != capture.num_frames:
            raise ValueError(
                "ADC file frame count does not match RadarCaptureSpec: "
                f"{reader.num_frames} != {capture.num_frames}."
            )
        return reader

    def read_frame(self, index: int) -> RawADCFrame:
        """Map one frame by zero-based index without reading the full file."""

        if not 0 <= index < self.num_frames:
            raise IndexError(f"ADC frame index {index} is outside [0, {self.num_frames}).")
        values = self.spec.raw_values_per_frame
        samples = np.memmap(
            self.path,
            dtype=np.int16,
            mode="r",
            offset=index * values * np.dtype(np.int16).itemsize,
            shape=(values,),
        )
        timestamp = (
            index * self.frame_periodicity_s if self.frame_periodicity_s is not None else None
        )
        return RawADCFrame(
            samples,
            frame_id=index,
            timestamp=timestamp,
            source=str(self.path),
            profile=self.profile,
            metadata={
                **self.metadata,
                "frame_index": index,
                "num_frames": self.num_frames,
            },
        )


def load_adc_file(
    path: str | Path,
    *,
    frame_id: str | int | None = None,
    profile: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    mmap: bool = False,
) -> RawADCFrame:
    """Load raw int16 ADC samples from a binary file."""

    adc_path = Path(path)
    if mmap:
        samples = np.memmap(adc_path, dtype=np.int16, mode="r")
    else:
        samples = np.fromfile(adc_path, dtype=np.int16)

    return RawADCFrame(
        samples=samples,
        frame_id=frame_id,
        source=str(adc_path),
        profile=profile or {},
        metadata=metadata or {},
    )


def load_adc_cube(
    path: str | Path,
    spec: ADCFrameSpec,
    *,
    frame_id: str | int | None = None,
    profile: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    mmap: bool = False,
    drop_incomplete: bool = False,
) -> RadarCube:
    """Load an ADC binary file and organize it into a complex radar cube."""

    raw = load_adc_file(
        path,
        frame_id=frame_id,
        profile=profile,
        metadata=metadata,
        mmap=mmap,
    )
    return organize_adc_samples(raw, spec, drop_incomplete=drop_incomplete)
