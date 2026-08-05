"""Artifact manifest helpers for mmwcore session outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmwcore.core import PointCloudFrame, RawADCFrame


@dataclass(frozen=True)
class RawADCArtifact:
    """Metadata record for a captured raw ADC artifact."""

    sample_id: str
    adc: str
    dtype: str = "int16"
    num_values: int = 0
    frame_id: str | int | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "adc": self.adc,
            "dtype": self.dtype,
            "num_values": self.num_values,
            "frame_id": self.frame_id,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PointCloudArtifact:
    """Metadata record for a generated point-cloud artifact."""

    sample_id: str
    point_cloud: str
    point_channels: tuple[str, ...]
    raw_sources: dict[str, str] = field(default_factory=dict)
    timestamp: float | None = None
    source: str | None = None
    coordinate_frame: str = "radar"
    units: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "point_cloud": self.point_cloud,
            "point_channels": list(self.point_channels),
            "raw_sources": dict(self.raw_sources),
            "timestamp": self.timestamp,
            "source": self.source,
            "point_coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "metadata": dict(self.metadata),
        }


def write_point_cloud_artifact(
    manifest_path: str | Path,
    *,
    sample_id: str,
    point_cloud_path: str | Path,
    point_cloud: PointCloudFrame,
    root: str | Path | None = None,
    raw_adc_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    append: bool = True,
) -> PointCloudArtifact:
    """Write one JSONL artifact record for a generated point cloud."""

    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    base = Path(root) if root is not None else manifest.parent.parent
    point_cloud_ref = _relative_or_absolute(Path(point_cloud_path), base)
    raw_sources = {}
    if raw_adc_path is not None:
        raw_sources["adc"] = _relative_or_absolute(Path(raw_adc_path), base)
    artifact_metadata = dict(point_cloud.metadata)
    if metadata is not None:
        artifact_metadata.update(metadata)

    artifact = PointCloudArtifact(
        sample_id=sample_id,
        point_cloud=point_cloud_ref,
        point_channels=point_cloud.channels,
        raw_sources=raw_sources,
        timestamp=point_cloud.timestamp,
        source=point_cloud.source,
        coordinate_frame=point_cloud.coordinate_frame,
        units=point_cloud.units,
        metadata=artifact_metadata,
    )

    mode = "a" if append else "w"
    with manifest.open(mode, encoding="utf-8") as file:
        file.write(json.dumps(artifact.to_record()) + "\n")
    return artifact


def write_raw_adc_artifact(
    manifest_path: str | Path,
    *,
    sample_id: str,
    adc_path: str | Path,
    raw: RawADCFrame,
    root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    append: bool = True,
) -> RawADCArtifact:
    """Write one JSONL artifact record for a captured raw ADC frame."""

    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    base = Path(root) if root is not None else manifest.parent.parent
    artifact_metadata = dict(raw.metadata)
    if metadata is not None:
        artifact_metadata.update(metadata)
    artifact = RawADCArtifact(
        sample_id=sample_id,
        adc=_relative_or_absolute(Path(adc_path), base),
        dtype=str(raw.samples.dtype),
        num_values=int(raw.samples.size),
        frame_id=raw.frame_id,
        source=raw.source,
        metadata=artifact_metadata,
    )

    mode = "a" if append else "w"
    with manifest.open(mode, encoding="utf-8") as file:
        file.write(json.dumps(artifact.to_record()) + "\n")
    return artifact


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
