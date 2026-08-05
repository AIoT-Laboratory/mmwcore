from __future__ import annotations

import json

import numpy as np

from mmwcore.core import PointCloudFrame, RawADCFrame
from mmwcore.session import write_point_cloud_artifact, write_raw_adc_artifact


def test_write_point_cloud_artifact_writes_jsonl_record(tmp_path) -> None:
    point_cloud_path = tmp_path / "point_clouds" / "sample.npy"
    adc_path = tmp_path / "raw" / "sample.bin"
    point_cloud = PointCloudFrame(
        np.array([[0.0, 1.0, 0.0, 0.5]], dtype=np.float32),
        channels=("x", "y", "z", "velocity"),
        frame_id="sample",
        timestamp=12.5,
        source="fixture.bin",
        units={"x": "m", "y": "m", "z": "m", "velocity": "m/s"},
        metadata={"source": "unit"},
    )

    artifact = write_point_cloud_artifact(
        tmp_path / "artifacts" / "point_clouds.jsonl",
        sample_id="sample",
        point_cloud_path=point_cloud_path,
        point_cloud=point_cloud,
        root=tmp_path,
        raw_adc_path=adc_path,
    )

    assert artifact.sample_id == "sample"
    record = json.loads((tmp_path / "artifacts" / "point_clouds.jsonl").read_text())
    assert record["point_cloud"] == "point_clouds/sample.npy"
    assert record["point_channels"] == ["x", "y", "z", "velocity"]
    assert record["raw_sources"] == {"adc": "raw/sample.bin"}
    assert record["timestamp"] == 12.5
    assert record["source"] == "fixture.bin"
    assert record["units"]["velocity"] == "m/s"
    assert record["metadata"] == {"source": "unit"}


def test_write_point_cloud_artifact_merges_extra_metadata(tmp_path) -> None:
    point_cloud = PointCloudFrame(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        metadata={"source": "unit"},
    )

    write_point_cloud_artifact(
        tmp_path / "artifacts" / "point_clouds.jsonl",
        sample_id="sample",
        point_cloud_path=tmp_path / "point_clouds" / "sample.npy",
        point_cloud=point_cloud,
        root=tmp_path,
        metadata={"preprocess": {"shape_source": "args"}},
    )

    record = json.loads((tmp_path / "artifacts" / "point_clouds.jsonl").read_text())
    assert record["metadata"] == {
        "source": "unit",
        "preprocess": {"shape_source": "args"},
    }


def test_write_raw_adc_artifact_writes_jsonl_record(tmp_path) -> None:
    raw = RawADCFrame(
        np.array([1, 2, 3, 4], dtype=np.int16),
        frame_id="frame-0",
        source="dca1000://fixture",
        metadata={"packet_loss": 0},
    )

    artifact = write_raw_adc_artifact(
        tmp_path / "artifacts" / "raw_adc.jsonl",
        sample_id="sample",
        adc_path=tmp_path / "raw" / "sample.bin",
        raw=raw,
        root=tmp_path,
        metadata={"adc_spec": {"num_chirps": 2}},
    )

    assert artifact.sample_id == "sample"
    record = json.loads((tmp_path / "artifacts" / "raw_adc.jsonl").read_text())
    assert record["adc"] == "raw/sample.bin"
    assert record["dtype"] == "int16"
    assert record["num_values"] == 4
    assert record["frame_id"] == "frame-0"
    assert record["source"] == "dca1000://fixture"
    assert record["metadata"] == {
        "packet_loss": 0,
        "adc_spec": {"num_chirps": 2},
    }
