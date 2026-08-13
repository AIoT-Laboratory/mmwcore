"""Input discovery and option validation for ADC storage benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.adc_storage_codecs import SUPPORTED_CODECS


@dataclass(frozen=True)
class StorageCase:
    codec: str
    chunk_frames: int

    def as_record(self) -> dict[str, object]:
        return {"codec": self.codec, "chunk_frames": self.chunk_frames}


def discover_sources(inputs: Iterable[Path], *, filename: str) -> list[Path]:
    if not filename:
        raise ValueError("Discovery filename must not be empty.")
    found: dict[Path, None] = {}
    for input_path in inputs:
        if input_path.is_file():
            found[input_path.resolve()] = None
        elif input_path.is_dir():
            for candidate in sorted(input_path.rglob(filename)):
                if candidate.is_file():
                    found[candidate.resolve()] = None
        else:
            raise FileNotFoundError(f"ADC storage benchmark input does not exist: {input_path}")
    sources = sorted(found)
    if not sources:
        raise FileNotFoundError(f"No files named {filename!r} were found in the requested inputs.")
    return sources


def source_selection(
    source: Path,
    *,
    frame_bytes: int,
    start_frame: int,
    max_frames: int | None,
) -> tuple[int, int]:
    size = source.stat().st_size
    remainder = size % frame_bytes
    if remainder:
        raise ValueError(
            f"ADC source has an incomplete trailing frame: {source} has {size} bytes, "
            f"which is not divisible by frame_bytes={frame_bytes}."
        )
    total_frames = size // frame_bytes
    if start_frame >= total_frames:
        raise ValueError(
            f"Start frame {start_frame} is outside {source}, which contains {total_frames} frames."
        )
    selected_frames = total_frames - start_frame
    if max_frames is not None:
        selected_frames = min(selected_frames, max_frames)
    return total_frames, selected_frames


def validate_options(
    inputs: Sequence[Path],
    *,
    frame_bytes: int,
    cases: Sequence[StorageCase],
    start_frame: int,
    max_frames: int | None,
    random_windows: int,
    window_frames: int,
    zlib_level: int,
    scratch_dir: Path | None,
) -> None:
    if not inputs:
        raise ValueError("At least one ADC source file or directory is required.")
    if frame_bytes <= 0 or frame_bytes % 2:
        raise ValueError("Frame bytes must be a positive multiple of two.")
    _validate_cases(cases)
    _validate_selection(
        start_frame=start_frame,
        max_frames=max_frames,
        random_windows=random_windows,
        window_frames=window_frames,
    )
    if not 0 <= zlib_level <= 9:
        raise ValueError("Zlib level must be in [0, 9].")
    if scratch_dir is not None and not scratch_dir.is_dir():
        raise FileNotFoundError(f"Scratch directory does not exist: {scratch_dir}")


def _validate_cases(cases: Sequence[StorageCase]) -> None:
    if not cases:
        raise ValueError("At least one ADC storage benchmark case is required.")
    if any(case.chunk_frames <= 0 for case in cases):
        raise ValueError("Case chunk frames must be positive.")
    unsupported = {case.codec for case in cases}.difference(SUPPORTED_CODECS)
    if unsupported:
        raise ValueError(f"Unsupported ADC storage codecs: {sorted(unsupported)!r}.")
    identities = {(case.codec, case.chunk_frames) for case in cases}
    if len(identities) != len(cases):
        raise ValueError("ADC storage benchmark cases must be unique.")


def _validate_selection(
    *,
    start_frame: int,
    max_frames: int | None,
    random_windows: int,
    window_frames: int,
) -> None:
    if start_frame < 0:
        raise ValueError("Start frame must be non-negative.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("Maximum frames must be positive when provided.")
    if random_windows < 0:
        raise ValueError("Random windows must be non-negative.")
    if window_frames <= 0:
        raise ValueError("Window frames must be positive.")
