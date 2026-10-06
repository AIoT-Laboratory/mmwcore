"""Migration oracle: original Python versus Rust, with exact discrete decisions."""

from dataclasses import asdict

import numpy as np
import pytest
from _multiscale_reference import ScatterBodyTracker as ReferenceTracker

from mmwcore import _native
from mmwcore.tracking import ScatterBodyTracker


def _compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple, np.ndarray)):
        assert len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _compare(left, right, f"{path}[{i}]")
    elif isinstance(expected, (float, np.floating)):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), path
    else:
        assert actual == expected, path


def _frames():
    rng = np.random.default_rng(20261006)
    for frame in range(100):
        first = np.zeros((12, 5))
        first[:, :2] = [2 + 0.007 * frame, 0.1] + rng.normal(0, 0.015, (12, 2))
        first[:, 2] = 1
        first[:, 3] = 0.07
        first[:, 4] = 20
        child = np.zeros((3, 5))
        child[:, 0] = first[:, 0].mean() + (0.25 if frame < 8 else 0.5)
        child[:, 0] += [-0.01, 0, 0.01]
        child[:, 1:4] = [0.1, 1, 0.07]
        child[:, 4] = 8 if frame < 18 else 20
        second = first.copy()
        second[:, 1] += 2
        pieces = [first, child]
        if 25 <= frame < 65:
            pieces.append(second)
        if frame >= 90 or 40 <= frame < 44:
            yield np.empty((0, 5))
        else:
            yield np.concatenate(pieces)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_bodies": 1},
        {"max_components": 2, "max_bodies": 2},
        {"lineage": False},
        {"prefer_recent": False},
        {"split_cores": False},
        {"temporal": False},
        {"height_m": 2.0, "velocity_resolution_mps": 0.125},
    ],
)
def test_frame_outputs_and_full_state_match_original_python(options):
    native, reference = ScatterBodyTracker(**options), ReferenceTracker(**options)
    for frame, points in enumerate(_frames()):
        dt = 0.05 if frame % 3 == 0 else 0.1
        _compare(native.step_points(points, dt), reference.step_points(points, dt))
        for name in (
            "tracks",
            "next_id",
            "last_matches",
            "position_history",
            "time_s",
            "previous_clouds",
            "parents",
            "lineage_events",
            "component_support",
        ):
            _compare(getattr(native, name), getattr(reference, name), name)


def test_reference_is_frozen_default_contract():
    # Prevent changing the oracle's parameters merely to match the migrated code.
    from _multiscale_reference import DEFAULT_CONFIG

    assert asdict(DEFAULT_CONFIG) == {
        "fine_radius_m": 0.35,
        "outer_radius_m": 0.8,
        "min_points": 3,
        "velocity_scale_mps": 0.2710796965422454,
        "core_power_ratio": 4.0,
    }


def test_native_boundary_rejects_fortran_point_order():
    points = np.asfortranarray(np.arange(15, dtype=float).reshape(3, 5))
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.scatter_body_step(points, "{}", "{}", 0.1)
