"""Radar-only capture orchestration built from narrow IO interfaces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mmwcore.config import parse_ti_cli_config_file
from mmwcore.core import RawADCFrame
from mmwcore.io import PacketLossStats
from mmwcore.session.manifest import RawADCArtifact, write_raw_adc_artifact


class RadarCliPort(Protocol):
    """Minimal radar CLI protocol required by RadarOnlyCaptureSession."""

    def open(self) -> object: ...

    def close(self) -> None: ...

    def load_config_lines(
        self,
        lines: Iterable[str],
        *,
        include_sensor_start: bool = False,
    ) -> tuple[str, ...]: ...

    def sensor_start(self) -> str: ...

    def sensor_stop(self) -> str: ...


class DCA1000Recorder(Protocol):
    """Minimal DCA1000 recorder protocol required by RadarOnlyCaptureSession."""

    def configure_fpga(self) -> object: ...

    def start_record(self) -> object: ...

    def stop_record(self) -> object: ...


class RawFrameReader(Protocol):
    """Minimal frame reader protocol required by RadarOnlyCaptureSession."""

    def open(self) -> object: ...

    def close(self) -> None: ...

    def read_frame(
        self,
        *,
        frame_id: str | int | None = None,
    ) -> tuple[RawADCFrame, PacketLossStats]: ...


@dataclass(frozen=True)
class RadarCaptureResult:
    """Result from one radar-only capture frame."""

    raw: RawADCFrame
    packet_stats: PacketLossStats
    loaded_commands: tuple[str, ...] = ()


class RadarOnlyCaptureSession:
    """Coordinate radar CLI, DCA1000 recording, and frame reading for one capture."""

    def __init__(
        self,
        *,
        radar: RadarCliPort,
        recorder: DCA1000Recorder,
        frame_reader: RawFrameReader,
    ) -> None:
        self.radar = radar
        self.recorder = recorder
        self.frame_reader = frame_reader

    def open(self) -> RadarOnlyCaptureSession:
        self.radar.open()
        self.frame_reader.open()
        return self

    def close(self) -> None:
        self.frame_reader.close()
        self.radar.close()

    def prepare(
        self,
        radar_config_lines: Iterable[str],
        *,
        include_sensor_start: bool = False,
    ) -> tuple[str, ...]:
        loaded = self.radar.load_config_lines(
            radar_config_lines,
            include_sensor_start=include_sensor_start,
        )
        self.recorder.configure_fpga()
        return loaded

    def capture_frame(
        self,
        *,
        frame_id: str | int | None = None,
        radar_config_lines: Iterable[str] | None = None,
        include_sensor_start: bool = False,
    ) -> RadarCaptureResult:
        loaded = ()
        if radar_config_lines is not None:
            loaded = self.prepare(
                radar_config_lines,
                include_sensor_start=include_sensor_start,
            )

        self.recorder.start_record()
        try:
            if not include_sensor_start:
                self.radar.sensor_start()
            raw, stats = self.frame_reader.read_frame(frame_id=frame_id)
        finally:
            self.radar.sensor_stop()
            self.recorder.stop_record()

        return RadarCaptureResult(raw=raw, packet_stats=stats, loaded_commands=loaded)

    def __enter__(self) -> RadarOnlyCaptureSession:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def save_raw_adc_capture(
    result: RadarCaptureResult,
    adc_path: str | Path,
    *,
    sample_id: str,
    artifact_manifest: str | Path | None = None,
    root: str | Path | None = None,
    ti_cfg: str | Path | None = None,
    metadata: dict[str, object] | None = None,
) -> RawADCArtifact | None:
    """Write a captured RawADCFrame to disk and optionally index it in JSONL."""

    output = Path(adc_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.raw.samples.tofile(output)

    if artifact_manifest is None:
        return None
    artifact_metadata = _raw_adc_metadata(ti_cfg=ti_cfg, metadata=metadata)
    return write_raw_adc_artifact(
        artifact_manifest,
        sample_id=sample_id,
        adc_path=output,
        raw=result.raw,
        root=root,
        metadata=artifact_metadata,
    )


def _raw_adc_metadata(
    *,
    ti_cfg: str | Path | None,
    metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    if ti_cfg is None:
        return metadata

    summary = parse_ti_cli_config_file(ti_cfg)
    spec = summary.to_adc_frame_spec()
    merged: dict[str, object] = {
        "ti_cfg": str(ti_cfg),
        "adc_spec": {
            "num_chirps": spec.num_chirps,
            "num_rx": spec.num_rx,
            "num_samples": spec.num_samples,
            "layout": spec.layout.value,
        },
    }
    if metadata is not None:
        merged.update(metadata)
    return merged
