"""Output handling for the ADC preprocessing CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmwcore.core import PointCloudFrame, PointCloudRecipe
from mmwcore.session import PointCloudArtifact, write_point_cloud_artifact


@dataclass(frozen=True)
class PreprocessOutput:
    metadata_path: Path | None
    artifact: PointCloudArtifact | None


def write_preprocess_outputs(
    point_cloud: PointCloudFrame,
    *,
    args: argparse.Namespace,
    recipe: PointCloudRecipe,
) -> PreprocessOutput:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, point_cloud.points)

    metadata_path = _write_metadata(point_cloud, args=args, recipe=recipe)
    artifact = _write_artifact(point_cloud, args=args, recipe=recipe)
    return PreprocessOutput(metadata_path=metadata_path, artifact=artifact)


def preprocess_metadata(*, args: argparse.Namespace, recipe: PointCloudRecipe) -> dict[str, Any]:
    adc = recipe.detection.transform.decode.adc
    return {
        "shape_source": "ti_cfg" if args.ti_cfg is not None else "preset_or_args",
        "ti_cfg": str(args.ti_cfg) if args.ti_cfg is not None else None,
        "adc_spec": {
            "num_chirps": adc.num_chirps,
            "num_rx": adc.num_rx,
            "num_samples": adc.num_samples,
            "layout": adc.layout.value,
        },
    }


def _write_metadata(
    point_cloud: PointCloudFrame,
    *,
    args: argparse.Namespace,
    recipe: PointCloudRecipe,
) -> Path | None:
    metadata_path = args.metadata_output
    if metadata_path is None:
        return None
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(_metadata(point_cloud, args=args, recipe=recipe), indent=2),
        encoding="utf-8",
    )
    return metadata_path


def _write_artifact(
    point_cloud: PointCloudFrame,
    *,
    args: argparse.Namespace,
    recipe: PointCloudRecipe,
) -> PointCloudArtifact | None:
    if args.artifact_manifest is None:
        return None
    return write_point_cloud_artifact(
        args.artifact_manifest,
        sample_id=args.sample_id or args.frame_id or args.input.stem,
        point_cloud_path=args.output,
        point_cloud=point_cloud,
        raw_adc_path=args.input,
        metadata={"preprocess": preprocess_metadata(args=args, recipe=recipe)},
    )


def _metadata(
    point_cloud: PointCloudFrame,
    *,
    args: argparse.Namespace,
    recipe: PointCloudRecipe,
) -> dict[str, Any]:
    return {
        "channels": list(point_cloud.channels),
        "frame_id": point_cloud.frame_id,
        "timestamp": point_cloud.timestamp,
        "source": point_cloud.source,
        "coordinate_frame": point_cloud.coordinate_frame,
        "units": point_cloud.units,
        "num_points": point_cloud.num_points,
        "metadata": point_cloud.metadata,
        "preprocess": preprocess_metadata(args=args, recipe=recipe),
    }
