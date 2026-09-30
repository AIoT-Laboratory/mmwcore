"""Built-in Rust tracker against independently executed original-TI fixtures.

No SDK, manifest or DLL is required for these regression tests.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mmwcore import _native
from mmwcore.core import Box3D, PointCloudFrame, TrackStatus
from mmwcore.tracking import (
    TiGTrack3D,
    TiGTrack3DSpec,
    TiGTrackAllocation,
    TiGTrackGating,
    TiGTrackLifecycle,
    TiGTrackScenery,
)

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/ti_gtrack_3da_oracle.json"


def _spec(tilt: int = 0) -> TiGTrack3DSpec:
    boxes = (Box3D(-10, 10, -10, 10, -10, 10),)
    return TiGTrack3DSpec(
        0.1,
        4,
        0.125,
        max_points=16,
        max_tracks=1,
        max_acceleration_mps2=(0.5, 0.5, 0.5),
        gating=TiGTrackGating(4, 2, 2, 2, 2),
        allocation=TiGTrackAllocation(1, 1, 0.05, 4, 0.8, 1),
        lifecycle=TiGTrackLifecycle(1, 1, 2, 4, 2, 6),
        scenery=TiGTrackScenery(
            elevation_tilt_deg=tilt,
            boundary_boxes=boxes,
            static_boxes=boxes,
            occupancy_boxes=boxes,
            presence_points_threshold=4,
            presence_velocity_threshold_mps=0.05,
            presence_on_to_off=3,
        ),
    )


def _group(range_m: float, velocity: float) -> np.ndarray:
    return np.asarray(
        [
            [
                range_m + 0.012 * i,
                0.2 + 0.005 * i,
                0.15 + 0.005 * i,
                velocity + 0.002 * i if velocity else 0,
                20 + i,
            ]
            for i in range(6)
        ],
        np.float32,
    )


def _frames() -> list[np.ndarray]:
    return (
        [_group(2 + 0.03 * k, 0.3) for k in range(6)]
        + [_group(2.2, 0) for _ in range(12)]
        + [np.empty((0, 5), np.float32) for _ in range(60)]
        + [_group(4 + 0.02 * k, 0.2) for k in range(6)]
        + [np.empty((0, 5), np.float32) for _ in range(15)]
    )


@pytest.mark.parametrize("tilt,variance", [(0, False), (0, True), (90, False), (90, True)])
def test_full_step_matches_independent_original_c_oracle(tilt, variance) -> None:
    fixture = json.loads(ORACLE.read_text())
    expected = next(c for c in fixture["cases"] if c["tilt"] == tilt and c["variance"] == variance)
    seen_ids = set()
    with TiGTrack3D(_spec(tilt)) as tracker:
        for points, reference in zip(_frames(), expected["frames"], strict=True):
            var = np.tile([0.01, 0.001, 0.002, 0.02], (len(points), 1)) if variance else None
            actual = tracker.step_spherical(points, var)
            assert len(actual["targets"]) == len(reference["targets"])
            for observed, target in zip(actual["targets"], reference["targets"], strict=True):
                for key, value in target.items():
                    # Oracle prints 9 significant digits: recover original float32 values.
                    np.testing.assert_allclose(
                        np.asarray(observed[key]).ravel(),
                        np.asarray(value).ravel(),
                        rtol=3e-6,
                        atol=2e-7,
                        err_msg=key,
                    )
                assert observed["snr_weighting"] == (tilt == 0)
                assert observed["height_ignore"] == (tilt == 90)
                seen_ids.add((observed["uid"], observed["tid"]))
            for key in ("point_uid", "point_unique", "point_static", "presence"):
                assert actual[key] == reference[key]
            for key in ("point_score", "updated_doppler"):
                np.testing.assert_allclose(actual[key], reference[key], rtol=3e-6, atol=2e-7)
            mapping = {t["uid"]: t["tid"] for t in actual["targets"]}
            assert actual["point_tid"] == [mapping.get(uid, -1) for uid in actual["point_uid"]]
    assert seen_ids == {(0, 1), (0, 2)}


def test_pinned_isk_defaults_and_axis_encoding() -> None:
    spec = TiGTrack3DSpec(0.1, 4, 0.125)
    cfg = spec.native_config()
    assert cfg["gating_limits"] == [2, 2, 2, 4]
    assert cfg["state_thresholds"] == [3, 3, 12, 500, 5, 6000]
    assert cfg["allocation_points"] == 20
    assert (cfg["max_points"], cfg["max_tracks"]) == (800, 30)
    spec = replace(
        spec,
        max_acceleration_mps2=(1, 2, 3),
        scenery=TiGTrackScenery(boundary_boxes=(Box3D(1, 6, -3, 4, 0, 2),)),
    )
    assert spec.native_config()["max_acceleration"] == [2, 1, 3]
    assert spec.native_config()["boundary_boxes"] == [-3, 4, 1, 6, 0, 2, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "changes",
    [
        {"frame_period_s": 0},
        {"max_points": 1001},
        {"max_tracks": 201},
        {"max_acceleration_mps2": (1, -1, 1)},
        {"max_radial_velocity_mps": 0},
        {"radial_velocity_resolution_mps": 0},
        {"scenery": TiGTrackScenery(sensor_position_m=(1, 0, 2))},
        {"scenery": TiGTrackScenery()},
    ],
)
def test_invalid_config_rejected_before_plugin_io(tmp_path, changes) -> None:
    config = replace(_spec(), **changes).native_config()
    with pytest.raises(ValueError) as exc:
        _native.NativeTiGTrack3D(str(tmp_path / "missing.json"), json.dumps(config))
    assert "os error" not in str(exc.value)


def test_invalid_input_does_not_advance_and_reset_restarts_ids() -> None:
    with TiGTrack3D(_spec()) as tracker:
        points = _group(2, 0.3)
        for bad in (np.zeros((6, 4)), np.full((6, 4), np.nan), np.ones((5, 4))):
            with pytest.raises(ValueError, match="variances"):
                tracker.step_spherical(points, bad)
        with pytest.raises(ValueError, match="truncated"):
            tracker.step_spherical(np.tile(points, (3, 1)))
        with pytest.raises(ValueError, match="measurements"):
            tracker.step_spherical(np.full((6, 5), np.nan))
        original = points.copy()
        raw = tracker.step_spherical(points)
        np.testing.assert_array_equal(points, original)
        assert raw["targets"][0]["age"] == 1
        tracker.close()
        with pytest.raises(ValueError, match="closed"):
            tracker.step_spherical(points)
        tracker.reset()
        assert tracker.step_spherical(points)["targets"][0]["tid"] == 1


def test_cartesian_adapter_preserves_right_doppler_and_nine_states() -> None:
    points = np.asarray(
        [
            [2 + 0.01 * i, 0.4 + 0.005 * i, 0.3 + 0.005 * i, -0.3 - 0.002 * i, 20 + i]
            for i in range(6)
        ],
        np.float32,
    )
    with TiGTrack3D(_spec()) as tracker:
        for frame in range(3):
            result = tracker.step(
                PointCloudFrame(
                    points,
                    channels=("x", "y", "z", "velocity", "snr"),
                    coordinate_frame="sensor_forward_lateral_up",
                    frame_id=frame,
                )
            )
        assert result.statuses == (TrackStatus.CONFIRMED,)
        raw = result.metadata["tracker"]["ti_report"]
        native = raw["targets"][0]
        assert len(native["state_vector"]) == 9
        assert result.positions[0, 1] > 0
        assert all(v < 0 for v in raw["updated_doppler"])
        np.testing.assert_allclose(
            result.positions[0], np.asarray(native["state_vector"])[[1, 0, 2]]
        )
        assert np.linalg.eigvalsh(result.extent_covariances[0]).min() > -1e-6
        assert result.frame_id == 2


def test_default_ignores_old_plugin_environment(monkeypatch) -> None:
    monkeypatch.setenv("MMWCORE_TI_GTRACK_MANIFEST", "missing/manifest.json")
    with TiGTrack3D(_spec()) as tracker:
        assert tracker.provenance["implementation"] == "mmwcore-rust-gtrack-3da-v1"
        assert tracker.step_spherical(_group(2, 0.3))["targets"]


def test_two_targets_keep_separate_point_membership() -> None:
    spec = TiGTrack3DSpec(
        0.1, 4, 0.125, scenery=TiGTrackScenery(boundary_boxes=(Box3D(0, 8, -4, 4, 0, 4),))
    )
    points = np.asarray(
        [
            [2.5 + 0.004 * i, side + 0.002 * i, 0.1 + 0.001 * i, 0.3, 1000]
            for side in (-1.0, 1.0)
            for i in range(24)
        ],
        np.float32,
    )
    with TiGTrack3D(spec) as tracker:
        for _ in range(8):
            result = tracker.step(
                PointCloudFrame(
                    points,
                    channels=("x", "y", "z", "velocity", "snr"),
                    coordinate_frame="sensor_forward_lateral_up",
                )
            )
        assert result.statuses == (TrackStatus.CONFIRMED, TrackStatus.CONFIRMED)
        labels = result.observation_track_ids
        assert len(set(labels[:24])) == len(set(labels[24:])) == 1
        assert labels[0] >= 0 and labels[24] >= 0 and labels[0] != labels[24]
        assert (result.positions[:, 1] < 0).sum() == 1


def test_native_nonfinite_result_requires_reset() -> None:
    # Regression for original TI zero velocity-gate limit producing a singular gC.
    spec = replace(_spec(), gating=TiGTrackGating(4, 2, 2, 2, 0))
    with TiGTrack3D(spec) as tracker:
        with pytest.raises(ValueError, match="non-finite"):
            tracker.step_spherical(_group(2, 0.3))
        with pytest.raises(ValueError, match="reset"):
            tracker.step_spherical(np.empty((0, 5), np.float32))


def _cartesian_cloud(spherical: np.ndarray) -> PointCloudFrame:
    r, a, e = spherical[:, 0], spherical[:, 1], spherical[:, 2]
    values = np.column_stack(
        (r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e), spherical[:, 3:])
    )
    return PointCloudFrame(
        values.astype(np.float32),
        channels=("x", "y", "z", "velocity", "snr"),
        coordinate_frame="sensor_forward_right_up",
    )


@pytest.mark.parametrize("point_count", [1, 6])
def test_static_support_keeps_existing_id_without_rpc_then_recovers_and_releases(point_count):
    empty = _cartesian_cloud(np.empty((0, 5), np.float32))
    static = _cartesian_cloud(_group(2.2, 0)).xyz()[:point_count]
    with TiGTrack3D(_spec()) as tracker:
        for k in range(6):
            tracked = tracker.step(_cartesian_cloud(_group(2 + 0.03 * k, 0.3)))
        tid = tracked.track_ids.tolist()
        assert tracked.statuses == (TrackStatus.CONFIRMED,)
        # Static observations alone exceed both normal miss and sleep lifetimes.
        for _ in range(80):
            tracked = tracker.step(empty, static_positions=static)
            assert tracked.track_ids.tolist() == tid
            assert tracked.observation_track_ids.size == 0
            assert tracker.last_report is not None
            assert tracker.last_report["static_support"]["assigned_count"] > 0
        assert tracker.last_report is not None
        assert tracker.last_report["targets"][0]["is_static"] == 1
        assert tracker.last_report["targets"][0]["counters"][2] == 0
        for k in range(6):
            tracked = tracker.step(
                _cartesian_cloud(_group(2.2 + 0.03 * k, 0.3)), static_positions=static
            )
            assert tracked.track_ids.tolist() == tid
            assert tracker.last_report is not None
            assert tracker.last_report["static_support"]["assigned_count"] == 0
        assert tracker.last_report is not None
        assert tracker.last_report["targets"][0]["is_static"] == 0
        for _ in range(80):
            tracked = tracker.step(empty, static_positions=np.empty((0, 3)))
        assert tracked.track_ids.size == 0


def test_static_support_cannot_birth_or_confirm_candidate():
    empty = _cartesian_cloud(np.empty((0, 5), np.float32))
    static = _cartesian_cloud(_group(2.2, 0)).xyz()
    with TiGTrack3D(_spec()) as tracker:
        for _ in range(20):
            assert tracker.step(empty, static_positions=static).track_ids.size == 0
        tracker.step(_cartesian_cloud(_group(2.2, 0.3)))
        for _ in range(5):
            result = tracker.step(empty, static_positions=static)
            assert TrackStatus.CONFIRMED not in result.statuses
        assert result.track_ids.size == 0


def test_static_support_preserves_rpc_reports_and_original_point_labels():
    static = _cartesian_cloud(_group(2.2, 0)).xyz()
    with (
        TiGTrack3D(_spec()) as baseline,
        TiGTrack3D(_spec()) as supported,
    ):
        for k in range(12):
            cloud = _cartesian_cloud(_group(2 + 0.03 * k, 0.3))
            expected = baseline.step(cloud)
            actual = supported.step(cloud, static_positions=static)
            np.testing.assert_array_equal(
                actual.observation_track_ids, expected.observation_track_ids
            )
            first, second = baseline.last_report, supported.last_report
            assert first is not None and second is not None
            for key in ("targets", "sensor_targets", "presence"):
                assert first[key] == second[key]
            for key in (
                "point_uid",
                "point_tid",
                "point_unique",
                "point_static",
                "point_score",
                "updated_doppler",
            ):
                assert first[key] == second[key][: len(cloud.points)]


def test_static_support_inside_roi_exit_zone_and_outside_release():
    spec = replace(
        _spec(), scenery=replace(_spec().scenery, static_boxes=(Box3D(0, 1, -1, 1, -1, 1),))
    )
    empty = _cartesian_cloud(np.empty((0, 5), np.float32))
    static = _cartesian_cloud(_group(2.2, 0)).xyz()
    with TiGTrack3D(spec) as tracker:
        for k in range(6):
            result = tracker.step(_cartesian_cloud(_group(2 + 0.03 * k, 0.3)))
        tid = result.track_ids.tolist()
        for _ in range(40):
            result = tracker.step(empty, static_positions=static)
            assert result.track_ids.tolist() == tid
        outside = static + [20, 0, 0]
        for _ in range(40):
            result = tracker.step(empty, static_positions=outside)
            assert tracker.last_report is not None
            assert tracker.last_report["static_support"]["assigned_count"] == 0
        assert result.track_ids.size == 0


def test_one_person_can_lose_rpc_while_another_keeps_moving():
    spec = replace(_spec(), max_tracks=2, max_points=32)

    def group(side, frame, velocity):
        points = _group(2 + 0.01 * frame, velocity)
        points[:, 1] += side
        return points

    with TiGTrack3D(spec) as tracker:
        for k in range(6):
            cloud = _cartesian_cloud(np.concatenate((group(-0.5, k, 0.1), group(0.5, k, 0.1))))
            result = tracker.step(cloud)
        assert result.statuses == (TrackStatus.CONFIRMED, TrackStatus.CONFIRMED)
        original_ids = set(result.track_ids.tolist())
        stopped_id = int(result.observation_track_ids[0])
        moving_id = int(result.observation_track_ids[6])
        static = _cartesian_cloud(group(-0.5, 6, 0)).xyz()[:1]
        for k in range(6, 36):
            result = tracker.step(_cartesian_cloud(group(0.5, k, 0.1)), static_positions=static)
            assert set(result.track_ids.tolist()) == original_ids
            assert set(result.observation_track_ids.tolist()) == {moving_id}
            assert tracker.last_report is not None
            assert tracker.last_report["point_tid"][-1] == stopped_id


def test_invalid_static_support_does_not_advance_or_leak_to_next_step():
    cloud = _cartesian_cloud(_group(2, 0.3))
    with TiGTrack3D(_spec()) as tracker:
        for invalid in (np.zeros((2, 2)), np.array([[np.nan, 0, 0]])):
            with pytest.raises(ValueError, match="finite.*sensor XYZ"):
                tracker.step(cloud, static_positions=invalid)
        with pytest.raises(ValueError, match="max_points"):
            tracker.step(cloud, static_positions=np.tile([2, 0.4, 0.3], (11, 1)))
        with pytest.raises(ValueError, match="zero Doppler"):
            tracker._tracker.step(_group(2, 0.3), static_start=0)
        tracker.step(cloud)
        assert tracker.last_report is not None
        assert tracker.last_report["targets"][0]["age"] == 1
