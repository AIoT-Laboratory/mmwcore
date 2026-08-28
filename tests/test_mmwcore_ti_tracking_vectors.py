from __future__ import annotations

import json
from struct import pack
from typing import Any, cast

import numpy as np
import pytest

from mmwcore.core import (
    DBSCANSpec,
    GatingSpec,
    LifecycleSpec,
    Tracker2DSpec,
)
from mmwcore.tracking import (
    TiGTrack2DBenchmarkSpec,
    benchmark_cluster_tracker_on_ti_vectors,
    benchmark_measurement_tracker_on_ti_vectors,
    benchmark_ti_people_counting_2d,
    read_ti_gtrack_2d_vectors,
    run_cluster_tracker_on_ti_vectors,
    run_ti_people_counting_2d,
    ti_people_counting_2d_benchmark_spec,
)


def test_read_ti_gtrack_2d_vectors_parses_measurements_and_targets(tmp_path) -> None:
    measurement = pack("<ffff", 2.0, np.pi / 2, -0.5, 12.0)
    target = pack("<Iffffff", 7, 2.0, 3.0, 0.1, 0.2, 0.0, 0.0)
    vector_path = tmp_path / "gtrack.bin"
    vector_path.write_bytes(
        pack("<II", 1, 0)
        + pack("<II", 2, 2)
        + pack("<II", 6, len(measurement))
        + measurement
        + pack("<II", 7, len(target))
        + target
    )

    frames = read_ti_gtrack_2d_vectors(vector_path)

    assert [frame.frame_number for frame in frames] == [1, 2]
    np.testing.assert_allclose(frames[1].measurements, [[2.0, np.pi / 2, -0.5, 12.0]])
    np.testing.assert_allclose(
        frames[1].cartesian_points(),
        [[2.0, 0.0, 0.0, -0.5, 12.0]],
        atol=1e-6,
    )
    assert frames[1].ground_truth.track_ids.tolist() == [7]
    np.testing.assert_allclose(frames[1].ground_truth.positions, [[2.0, 3.0, 0.0]])


def test_ti_people_counting_benchmark_spec_is_explicit() -> None:
    spec = ti_people_counting_2d_benchmark_spec()

    assert spec.tracker.frame_period_s == pytest.approx(0.05)
    assert spec.tracker.allocation.min_points == 6
    assert spec.tracker.allocation.max_new_tracks_per_frame == 1
    assert spec.tracker.max_tracks == 20
    assert spec.clustering.velocity_scale_s == pytest.approx(0.75)
    assert spec.match_distance_m == pytest.approx(1.0)


def test_read_ti_gtrack_2d_vectors_rejects_truncated_payload(tmp_path) -> None:
    vector_path = tmp_path / "gtrack.bin"
    vector_path.write_bytes(pack("<II", 1, 1) + pack("<II", 6, 16) + b"short")

    with pytest.raises(ValueError, match="truncated TLV"):
        read_ti_gtrack_2d_vectors(vector_path)


def test_read_ti_gtrack_2d_vectors_skips_unknown_tlv_payload(tmp_path) -> None:
    vector_path = tmp_path / "gtrack.bin"
    vector_path.write_bytes(pack("<II", 3, 1) + pack("<II", 99, 5) + b"other" + pack("<II", 4, 0))

    frames = read_ti_gtrack_2d_vectors(vector_path)

    assert [frame.frame_number for frame in frames] == [3, 4]
    assert frames[0].measurements.size == 0


def test_benchmark_cluster_tracker_on_ti_vectors_reports_both_status_views(tmp_path) -> None:
    measurement = pack("<ffff", 1.0, 0.0, 0.0, 10.0)
    target = pack("<Iffffff", 7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    frame = (
        pack("<II", 1, 2)
        + pack("<II", 6, len(measurement))
        + measurement
        + pack("<II", 7, len(target))
        + target
    )
    vector_path = tmp_path / "gtrack.bin"
    vector_path.write_bytes(frame + frame)
    frames = read_ti_gtrack_2d_vectors(vector_path)

    comparison = benchmark_cluster_tracker_on_ti_vectors(
        frames,
        DBSCANSpec(eps_m=0.5, min_samples=1, use_z=False),
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            lifecycle=LifecycleSpec(confirmation_hits=2),
        ),
        match_distance_m=0.5,
    )

    assert comparison.all_tracks.matched_observations == 2
    assert comparison.confirmed_tracks.matched_observations == 1

    run = run_cluster_tracker_on_ti_vectors(
        frames,
        DBSCANSpec(eps_m=0.5, min_samples=1, use_z=False),
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            lifecycle=LifecycleSpec(confirmation_hits=2),
        ),
        match_distance_m=0.5,
    )
    assert len(run.frames) == 2
    assert run.comparison == comparison


def test_benchmark_measurement_tracker_on_ti_vectors_reports_both_status_views(
    tmp_path,
) -> None:
    measurement = pack("<ffff", 1.0, 0.0, 0.0, 10.0)
    target = pack("<Iffffff", 7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    frame = (
        pack("<II", 1, 2)
        + pack("<II", 6, len(measurement))
        + measurement
        + pack("<II", 7, len(target))
        + target
    )
    vector_path = tmp_path / "gtrack.bin"
    vector_path.write_bytes(frame + frame)
    frames = read_ti_gtrack_2d_vectors(vector_path)

    comparison = benchmark_measurement_tracker_on_ti_vectors(
        frames,
        DBSCANSpec(eps_m=0.5, min_samples=1, use_z=False),
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            lifecycle=LifecycleSpec(confirmation_hits=2),
        ),
        match_distance_m=0.5,
    )

    assert comparison.all_tracks.matched_observations == 2
    assert comparison.confirmed_tracks.matched_observations == 1


def test_paired_ti_benchmark_report_is_json_serializable(tmp_path) -> None:
    measurement = pack("<ffff", 1.0, 0.0, 0.0, 10.0)
    target = pack("<Iffffff", 7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    frame = (
        pack("<II", 4, 2)
        + pack("<II", 6, len(measurement))
        + measurement
        + pack("<II", 7, len(target))
        + target
    )
    vector_path = tmp_path / "gtrack.bin"
    vector_path.write_bytes(frame + frame)
    frames = read_ti_gtrack_2d_vectors(vector_path)
    spec = TiGTrack2DBenchmarkSpec(
        clustering=DBSCANSpec(eps_m=0.5, min_samples=1, use_z=False),
        tracker=Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=0.5),
            lifecycle=LifecycleSpec(confirmation_hits=2),
        ),
        match_distance_m=0.5,
    )

    record = benchmark_ti_people_counting_2d(frames, spec).to_record()
    encoded = json.dumps(record, sort_keys=True)
    results = cast(dict[str, Any], record["results"])

    assert record["schema_version"] == 1
    assert record["first_frame_number"] == 4
    assert record["last_frame_number"] == 4
    assert results["cluster"]["all_tracks"]["matched_observations"] == 2
    assert '"measurement"' in encoded

    run = run_ti_people_counting_2d(frames, spec)
    assert run.report.to_record() == record
    assert len(run.cluster.frames) == len(frames)
    assert len(run.measurement.frames) == len(frames)
