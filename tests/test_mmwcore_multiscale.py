"""Body-count priors must not disable scatter-component lineage."""

import numpy as np
import pytest

from mmwcore.tracking import ScatterBodyTracker


def _split_cloud(frame: int) -> np.ndarray:
    points = np.zeros((15, 5))
    points[:12, 0] = 2 + np.linspace(-0.03, 0.03, 12)
    points[:12, 4] = 20
    points[12:, 0] = (2.25 if frame < 3 else 2.5) + np.array([-0.01, 0, 0.01])
    points[12:, 4] = 8 if frame < 7 else 20
    points[:, 2] = 1
    return points


@pytest.mark.parametrize("max_components", [None, 2])
def test_one_body_keeps_split_components_and_rejects_independent_child(max_components):
    tracker = ScatterBodyTracker(max_bodies=1, max_components=max_components)
    unlimited = ScatterBodyTracker()
    for frame in range(9):
        points = _split_cloud(frame)
        original = points.copy()
        *_, bodies = tracker.step_points(points)
        reference = unlimited.step_points(points)[-1]
        np.testing.assert_array_equal(points, original)
        if frame < 7:
            assert bodies == reference
        if frame in (5, 6):
            assert len(bodies) == 1 and bodies[0]["component_ids"] == [0, 1]
            assert sorted(bodies[0]["body_measurement_members"]) == list(range(15))
            assert len(tracker.tracks) == 2
        if frame >= 7:
            assert len(reference) == 2
            assert len(bodies) == 1 and bodies[0]["id"] == 0
            assert tracker.parents == {}
            assert set(tracker.tracks) == set(tracker.last_matches) == {0}
            assert set(tracker.previous_clouds) == set(tracker.component_support) == {0}
            assert set(tracker.position_history) == {0}
            assert bodies[0]["body_measurement_members"] == list(range(12))


def test_component_capacity_is_separate_from_body_limit():
    tracker = ScatterBodyTracker(max_components=1, max_bodies=2)
    for frame in range(7):
        *_, bodies = tracker.step_points(_split_cloud(frame))
        assert len(tracker.tracks) <= 1
    assert len(bodies) == 1 and bodies[0]["component_ids"] == [0]
    assert tracker.parents == {}


@pytest.mark.parametrize("lineage", [False, True])
def test_person_prior_does_not_attach_unrelated_return_and_reuses_expired_slot(lineage):
    tracker = ScatterBodyTracker(max_bodies=1, lineage=lineage)
    core = _split_cloud(0)[:12]
    unrelated = core.copy()
    unrelated[:, 0] += 3
    unrelated[:, 4] += 20  # A stronger arrival must not displace an admitted body.
    for _ in range(3):
        tracker.step_points(core)
    for _ in range(4):
        *_, bodies = tracker.step_points(np.concatenate((core, unrelated)))
        assert len(bodies) == 1 and bodies[0]["id"] == 0
        assert set(tracker.tracks) == {0}
        assert tracker.parents == {}
        assert tracker.last_matches[0]["members"] == list(range(12))
    for _ in range(3):
        *_, bodies = tracker.step_points(unrelated)
        assert len(bodies) == 1 and bodies[0]["id"] == 0 and bodies[0]["coasting"]
    for _ in range(3):
        *_, bodies = tracker.step_points(unrelated)
    assert len(bodies) == 1 and bodies[0]["id"] != 0
    assert bodies[0]["xy"] == pytest.approx(unrelated[:, :2].mean(axis=0))
    for _ in range(4):
        *_, bodies = tracker.step_points(np.empty((0, 5)))
    assert bodies == []
    assert tracker.tracks == tracker.parents == tracker.last_matches == {}
    assert tracker.previous_clouds == tracker.component_support == tracker.position_history == {}


def test_child_becomes_independent_when_parent_expires():
    tracker = ScatterBodyTracker(max_bodies=1)
    for frame in range(7):
        tracker.step_points(_split_cloud(frame))
    child_points = _split_cloud(6)[12:]
    for _ in range(4):
        *_, bodies = tracker.step_points(child_points)
        assert len(bodies) == 1
    assert set(tracker.tracks) == {1}
    assert bodies[0]["id"] == 1
    assert tracker.parents == {}


@pytest.mark.parametrize("option", ["max_components", "max_bodies"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_invalid_capacity(option, value):
    with pytest.raises(ValueError, match="positive or None"):
        ScatterBodyTracker(**{option: value})
