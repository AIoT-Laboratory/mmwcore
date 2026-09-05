"""Full plugin contracts and an independently executed original-TI oracle.

Plugin tests skip without a local TI SDK build; input-contract/default tests do not.
The synthetic oracle configuration exercises lifecycle, not pilot-data tuning.
"""

from __future__ import annotations

import json
import os
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


@pytest.fixture
def manifest() -> Path:
    path = Path(
        os.environ.get("MMWCORE_TI_GTRACK_MANIFEST", ROOT / "build/ti-gtrack/manifest.json")
    )
    if not path.is_file():
        pytest.skip("Build the local TI-device-only plugin to run native oracle tests")
    return path


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
def test_full_step_matches_independent_original_c_oracle(manifest, tilt, variance) -> None:
    fixture = json.loads(ORACLE.read_text())
    expected = next(c for c in fixture["cases"] if c["tilt"] == tilt and c["variance"] == variance)
    seen_ids = set()
    with TiGTrack3D(_spec(tilt), plugin_manifest=manifest) as tracker:
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


def test_invalid_input_does_not_advance_and_reset_restarts_ids(manifest) -> None:
    with TiGTrack3D(_spec(), plugin_manifest=manifest) as tracker:
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


def test_cartesian_adapter_preserves_right_doppler_and_nine_states(manifest) -> None:
    points = np.asarray(
        [
            [2 + 0.01 * i, 0.4 + 0.005 * i, 0.3 + 0.005 * i, -0.3 - 0.002 * i, 20 + i]
            for i in range(6)
        ],
        np.float32,
    )
    with TiGTrack3D(_spec(), plugin_manifest=manifest) as tracker:
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


def test_hash_tampering_fails_before_loading(tmp_path, manifest) -> None:
    record = json.loads(manifest.read_text())
    record["library_sha256"] = "0" * 64
    (tmp_path / record["library"]).write_bytes((manifest.parent / record["library"]).read_bytes())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash"):
        TiGTrack3D(_spec(), plugin_manifest=path)


def test_two_targets_keep_separate_point_membership(manifest) -> None:
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
    with TiGTrack3D(spec, plugin_manifest=manifest) as tracker:
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


def test_native_nonfinite_result_requires_reset(manifest) -> None:
    # Regression for original TI zero velocity-gate limit producing a singular gC.
    spec = replace(_spec(), gating=TiGTrackGating(4, 2, 2, 2, 0))
    with TiGTrack3D(spec, plugin_manifest=manifest) as tracker:
        with pytest.raises(ValueError, match="non-finite"):
            tracker.step_spherical(_group(2, 0.3))
        with pytest.raises(ValueError, match="reset"):
            tracker.step_spherical(np.empty((0, 5), np.float32))
