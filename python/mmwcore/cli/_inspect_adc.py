"""ADC file inspection helpers for the mmwcore inspect CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from mmwcore.config import parse_ti_cli_config_file
from mmwcore.core import ADCFrameSpec


@dataclass(frozen=True)
class ADCShapeCandidate:
    """Candidate ADC frame shape derived from file size arithmetic."""

    num_chirps: int
    num_rx: int
    num_samples: int
    raw_values_per_frame: int
    complete_frames: int
    leftover_values: int

    @property
    def is_complete(self) -> bool:
        return self.leftover_values == 0

    def to_record(self) -> dict[str, bool | int]:
        return {
            "num_chirps": self.num_chirps,
            "num_rx": self.num_rx,
            "num_samples": self.num_samples,
            "raw_values_per_frame": self.raw_values_per_frame,
            "complete_frames": self.complete_frames,
            "leftover_values": self.leftover_values,
            "is_complete": self.is_complete,
        }


@dataclass(frozen=True)
class ADCSpecRecord:
    """Serializable ADC frame shape used for inspection."""

    num_chirps: int
    num_rx: int
    num_samples: int
    layout: str

    def to_record(self) -> dict[str, int | str]:
        return {
            "num_chirps": self.num_chirps,
            "num_rx": self.num_rx,
            "num_samples": self.num_samples,
            "layout": self.layout,
        }


@dataclass(frozen=True)
class ADCFileInspection:
    """Summary of an int16 ADC binary file."""

    path: str
    size_bytes: int
    int16_values: int
    trailing_bytes: int
    raw_values_per_frame: int | None = None
    complete_frames: int | None = None
    leftover_values: int | None = None
    adc_spec: ADCSpecRecord | None = None
    shape_candidates: tuple[ADCShapeCandidate, ...] = ()

    def to_record(
        self,
    ) -> dict[
        str,
        int | str | dict[str, int | str] | list[dict[str, bool | int]] | None,
    ]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "int16_values": self.int16_values,
            "trailing_bytes": self.trailing_bytes,
            "raw_values_per_frame": self.raw_values_per_frame,
            "complete_frames": self.complete_frames,
            "leftover_values": self.leftover_values,
            "adc_spec": self.adc_spec.to_record() if self.adc_spec is not None else None,
            "shape_candidates": [candidate.to_record() for candidate in self.shape_candidates],
        }


def inspect_adc_file(
    path: str | Path,
    spec: ADCFrameSpec | None = None,
    *,
    infer_shapes: bool = False,
    candidate_num_chirps: list[int] | tuple[int, ...] = (
        1,
        2,
        3,
        4,
        6,
        8,
        16,
        32,
        64,
        96,
        128,
        192,
    ),
    candidate_num_rx: list[int] | tuple[int, ...] = (1, 2, 4),
    candidate_num_samples: list[int] | tuple[int, ...] = (64, 128, 256),
    max_candidates: int = 12,
) -> ADCFileInspection:
    """Inspect an int16 ADC binary file without loading it into memory."""

    adc_path = Path(path)
    size_bytes = adc_path.stat().st_size
    int16_values, trailing_bytes = divmod(size_bytes, 2)
    shape_candidates = (
        infer_adc_shapes(
            int16_values,
            candidate_num_chirps=candidate_num_chirps,
            candidate_num_rx=candidate_num_rx,
            candidate_num_samples=candidate_num_samples,
            max_candidates=max_candidates,
        )
        if infer_shapes
        else ()
    )
    if spec is None:
        return ADCFileInspection(
            path=str(adc_path),
            size_bytes=size_bytes,
            int16_values=int16_values,
            trailing_bytes=trailing_bytes,
            shape_candidates=shape_candidates,
        )

    complete_frames, leftover_values = divmod(int16_values, spec.raw_values_per_frame)
    return ADCFileInspection(
        path=str(adc_path),
        size_bytes=size_bytes,
        int16_values=int16_values,
        trailing_bytes=trailing_bytes,
        raw_values_per_frame=spec.raw_values_per_frame,
        complete_frames=complete_frames,
        leftover_values=leftover_values,
        adc_spec=_adc_spec_record(spec),
        shape_candidates=shape_candidates,
    )


def infer_adc_shapes(
    int16_values: int,
    *,
    candidate_num_chirps: list[int] | tuple[int, ...],
    candidate_num_rx: list[int] | tuple[int, ...],
    candidate_num_samples: list[int] | tuple[int, ...],
    max_candidates: int,
) -> tuple[ADCShapeCandidate, ...]:
    """Infer plausible ADC frame shapes from int16 value count."""

    if max_candidates <= 0:
        raise ValueError(f"max_candidates must be positive; got {max_candidates}.")

    candidates: list[ADCShapeCandidate] = []
    for num_chirps in candidate_num_chirps:
        for num_rx in candidate_num_rx:
            for num_samples in candidate_num_samples:
                spec = ADCFrameSpec(
                    num_chirps=num_chirps,
                    num_rx=num_rx,
                    num_samples=num_samples,
                )
                complete_frames, leftover_values = divmod(
                    int16_values,
                    spec.raw_values_per_frame,
                )
                if complete_frames == 0:
                    continue
                candidates.append(
                    ADCShapeCandidate(
                        num_chirps=num_chirps,
                        num_rx=num_rx,
                        num_samples=num_samples,
                        raw_values_per_frame=spec.raw_values_per_frame,
                        complete_frames=complete_frames,
                        leftover_values=leftover_values,
                    )
                )

    candidates.sort(
        key=lambda candidate: (
            candidate.leftover_values != 0,
            candidate.leftover_values,
            -candidate.complete_frames,
            candidate.raw_values_per_frame,
        )
    )
    return tuple(candidates[:max_candidates])


def adc_spec_from_args(args: argparse.Namespace) -> ADCFrameSpec | None:
    values = (args.num_chirps, args.num_rx, args.num_samples)
    if all(value is None for value in values):
        if args.ti_cfg is not None:
            try:
                return parse_ti_cli_config_file(args.ti_cfg).to_adc_frame_spec()
            except (OSError, ValueError) as exc:
                raise SystemExit(f"--ti-cfg: {exc}") from exc
        return None
    if any(value is None for value in values):
        raise SystemExit("--num-chirps, --num-rx, and --num-samples must be provided together")
    if args.ti_cfg is not None:
        raise SystemExit("--ti-cfg cannot be combined with explicit ADC shape arguments")
    return ADCFrameSpec(
        num_chirps=args.num_chirps,
        num_rx=args.num_rx,
        num_samples=args.num_samples,
    )


def _adc_spec_record(spec: ADCFrameSpec) -> ADCSpecRecord:
    return ADCSpecRecord(
        num_chirps=spec.num_chirps,
        num_rx=spec.num_rx,
        num_samples=spec.num_samples,
        layout=spec.layout.value,
    )
