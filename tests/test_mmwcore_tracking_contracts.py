from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from sys import maxsize

import numpy as np
import pytest

from mmwcore.core import (
    AllocationSpec,
    Box2D,
    DBSCANSpec,
    GatingSpec,
    LifecycleSpec,
    ScenerySpec,
    Tracker2DSpec,
    TrackFrame,
    TrackStatus,
)

type IntegerTrackingSpec = DBSCANSpec | AllocationSpec | LifecycleSpec | ScenerySpec | Tracker2DSpec
type PhysicalSpecFactory = Callable[[float], object]


def _tracker_spec_with(**changes: object) -> Tracker2DSpec:
    return replace(
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=1.0),
        ),
        **changes,
    )


_INTEGER_SPEC_FIELDS: tuple[tuple[IntegerTrackingSpec, str], ...] = (
    (DBSCANSpec(eps_m=0.5, min_samples=2), "min_samples"),
    (AllocationSpec(), "min_points"),
    (AllocationSpec(), "max_new_tracks_per_frame"),
    (LifecycleSpec(), "confirmation_hits"),
    (LifecycleSpec(), "tentative_max_misses"),
    (LifecycleSpec(), "confirmed_max_misses"),
    (ScenerySpec(), "outside_max_frames"),
    (
        Tracker2DSpec(
            frame_period_s=0.1,
            gating=GatingSpec(max_distance_m=1.0),
        ),
        "max_tracks",
    ),
)

_PHYSICAL_SPEC_FIELDS: tuple[tuple[str, PhysicalSpecFactory], ...] = (
    (
        "DBSCANSpec.eps_m",
        lambda value: DBSCANSpec(eps_m=value, min_samples=2),
    ),
    (
        "DBSCANSpec.velocity_scale_s",
        lambda value: DBSCANSpec(
            eps_m=0.5,
            min_samples=2,
            velocity_scale_s=value,
        ),
    ),
    (
        "GatingSpec.max_distance_m",
        lambda value: GatingSpec(max_distance_m=value),
    ),
    (
        "GatingSpec.max_radial_velocity_difference_mps",
        lambda value: GatingSpec(
            max_distance_m=1.0,
            max_radial_velocity_difference_mps=value,
        ),
    ),
    (
        "GatingSpec.max_mahalanobis_distance",
        lambda value: GatingSpec(
            max_distance_m=1.0,
            max_mahalanobis_distance=value,
        ),
    ),
    (
        "AllocationSpec.min_abs_radial_velocity_mps",
        lambda value: AllocationSpec(min_abs_radial_velocity_mps=value),
    ),
    (
        "AllocationSpec.min_total_snr",
        lambda value: AllocationSpec(min_total_snr=value),
    ),
    (
        "Box2D.x_min_m",
        lambda value: Box2D(value, 1.0, -1.0, 1.0),
    ),
    (
        "Box2D.x_max_m",
        lambda value: Box2D(-1.0, value, -1.0, 1.0),
    ),
    (
        "Box2D.y_min_m",
        lambda value: Box2D(-1.0, 1.0, value, 1.0),
    ),
    (
        "Box2D.y_max_m",
        lambda value: Box2D(-1.0, 1.0, -1.0, value),
    ),
    (
        "Tracker2DSpec.frame_period_s",
        lambda value: Tracker2DSpec(
            frame_period_s=value,
            gating=GatingSpec(max_distance_m=1.0),
        ),
    ),
    (
        "Tracker2DSpec.measurement_noise_m",
        lambda value: _tracker_spec_with(measurement_noise_m=value),
    ),
    (
        "Tracker2DSpec.initial_velocity_std_mps",
        lambda value: _tracker_spec_with(initial_velocity_std_mps=value),
    ),
    (
        "Tracker2DSpec.extent_covariance_smoothing",
        lambda value: _tracker_spec_with(extent_covariance_smoothing=value),
    ),
    (
        "Tracker2DSpec.max_acceleration_mps2",
        lambda value: _tracker_spec_with(max_acceleration_mps2=(value, 2.0)),
    ),
    (
        "Tracker2DSpec.max_acceleration_mps2",
        lambda value: _tracker_spec_with(max_acceleration_mps2=(2.0, value)),
    ),
)


def _single_track_frame(**integer_fields: np.ndarray) -> TrackFrame:
    fields = {
        "track_ids": np.array([1], dtype=np.int64),
        "ages": np.array([1], dtype=np.int64),
        "missed_counts": np.array([0], dtype=np.int64),
        "observation_track_ids": np.array([1], dtype=np.int64),
    }
    fields.update(integer_fields)
    return TrackFrame(
        track_ids=fields["track_ids"],
        positions=np.zeros((1, 3)),
        velocities=np.zeros((1, 3)),
        position_covariances=np.zeros((1, 2, 2)),
        extent_covariances=np.zeros((1, 2, 2)),
        statuses=(TrackStatus.TENTATIVE,),
        ages=fields["ages"],
        missed_counts=fields["missed_counts"],
        observation_track_ids=fields["observation_track_ids"],
    )


def test_cluster_tracker_spec_keeps_explicit_timing_and_lifecycle() -> None:
    spec = Tracker2DSpec(
        frame_period_s=0.1,
        gating=GatingSpec(
            max_distance_m=0.8,
            max_radial_velocity_difference_mps=1.5,
        ),
        allocation=AllocationSpec(min_points=3),
        lifecycle=LifecycleSpec(
            confirmation_hits=4,
            tentative_max_misses=2,
            confirmed_max_misses=10,
        ),
    )

    assert spec.frame_period_s == pytest.approx(0.1)
    assert spec.gating.max_distance_m == pytest.approx(0.8)
    assert spec.allocation.min_points == 3
    assert spec.lifecycle.confirmation_hits == 4


@pytest.mark.parametrize(
    ("spec", "field_name"),
    _INTEGER_SPEC_FIELDS,
    ids=[f"{type(spec).__name__}.{field_name}" for spec, field_name in _INTEGER_SPEC_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [pytest.param(True, id="bool"), pytest.param(1.0, id="float")],
)
def test_tracking_specs_reject_non_integer_public_counts(
    spec: IntegerTrackingSpec,
    field_name: str,
    invalid: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{type(spec).__name__}\.{field_name} must be an integer",
    ):
        replace(spec, **{field_name: invalid})


@pytest.mark.parametrize(
    ("spec", "field_name"),
    _INTEGER_SPEC_FIELDS,
    ids=[f"{type(spec).__name__}.{field_name}" for spec, field_name in _INTEGER_SPEC_FIELDS],
)
def test_tracking_specs_normalize_numpy_integer_counts(
    spec: IntegerTrackingSpec,
    field_name: str,
) -> None:
    normalized = replace(spec, **{field_name: np.int64(3)})

    value = getattr(normalized, field_name)
    assert value == 3
    assert type(value) is int


@pytest.mark.parametrize(
    ("spec", "field_name"),
    _INTEGER_SPEC_FIELDS,
    ids=[f"{type(spec).__name__}.{field_name}" for spec, field_name in _INTEGER_SPEC_FIELDS],
)
def test_tracking_specs_preserve_positive_count_domains(
    spec: IntegerTrackingSpec,
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{type(spec).__name__}.{field_name} must be positive",
    ):
        replace(spec, **{field_name: 0})


@pytest.mark.parametrize(
    ("spec", "field_name"),
    _INTEGER_SPEC_FIELDS,
    ids=[f"{type(spec).__name__}.{field_name}" for spec, field_name in _INTEGER_SPEC_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(maxsize + 1, id="above-platform"),
        pytest.param(-maxsize - 2, id="below-platform"),
    ],
)
def test_tracking_specs_reject_counts_outside_platform_range(
    spec: IntegerTrackingSpec,
    field_name: str,
    invalid: int,
) -> None:
    with pytest.raises(
        OverflowError,
        match=rf"{type(spec).__name__}\.{field_name} must fit the platform index range",
    ):
        replace(spec, **{field_name: invalid})


@pytest.mark.parametrize(
    ("field_name", "factory"),
    _PHYSICAL_SPEC_FIELDS,
    ids=[field_name for field_name, _factory in _PHYSICAL_SPEC_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
    ],
)
def test_tracking_specs_reject_non_finite_physical_fields(
    field_name: str,
    factory: PhysicalSpecFactory,
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{re.escape(field_name)}.*finite"):
        factory(invalid)


@pytest.mark.parametrize(
    ("field_name", "factory"),
    _PHYSICAL_SPEC_FIELDS,
    ids=[field_name for field_name, _factory in _PHYSICAL_SPEC_FIELDS],
)
def test_tracking_specs_reject_bool_physical_fields(
    field_name: str,
    factory: PhysicalSpecFactory,
) -> None:
    with pytest.raises(TypeError, match=rf"{re.escape(field_name)}.*not bool"):
        factory(True)


def test_dbscan_spec_preserves_positive_and_non_negative_domains() -> None:
    with pytest.raises(ValueError, match="eps_m"):
        DBSCANSpec(eps_m=0.0, min_samples=2)
    with pytest.raises(ValueError, match="velocity_scale_s"):
        DBSCANSpec(eps_m=0.5, min_samples=2, velocity_scale_s=-0.1)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(1, id="integer"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param("true", id="string"),
    ],
)
def test_dbscan_spec_requires_builtin_bool_use_z(invalid: object) -> None:
    with pytest.raises(TypeError, match=r"DBSCANSpec\.use_z must be a bool"):
        replace(
            DBSCANSpec(eps_m=0.5, min_samples=2),
            **{"use_z": invalid},
        )


def test_track_allocation_spec_rejects_non_positive_snr_threshold() -> None:
    with pytest.raises(ValueError, match="min_total_snr"):
        AllocationSpec(min_total_snr=0.0)


def test_track_gating_spec_rejects_non_positive_mahalanobis_limit() -> None:
    with pytest.raises(ValueError, match="max_mahalanobis_distance"):
        GatingSpec(max_distance_m=1.0, max_mahalanobis_distance=0.0)


def test_tracking_box_preserves_ordered_bound_domain() -> None:
    with pytest.raises(ValueError, match="minimum bounds"):
        Box2D(1.0, 1.0, -1.0, 1.0)


@pytest.mark.parametrize("smoothing", [0.0, 1.1])
def test_tracker_spec_preserves_smoothing_domain(smoothing: float) -> None:
    with pytest.raises(ValueError, match="extent_covariance_smoothing"):
        _tracker_spec_with(extent_covariance_smoothing=smoothing)


def test_tracker_spec_preserves_positive_acceleration_domain() -> None:
    with pytest.raises(ValueError, match="max_acceleration_mps2"):
        _tracker_spec_with(max_acceleration_mps2=(2.0, 0.0))


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param("2.0", id="string"),
        pytest.param(2.0 + 0.0j, id="complex"),
    ],
)
def test_tracker_spec_rejects_non_real_acceleration_entries(
    index: int,
    invalid: object,
) -> None:
    acceleration: list[object] = [1.0, 2.0]
    acceleration[index] = invalid

    with pytest.raises(TypeError, match="max_acceleration_mps2.*real numbers"):
        _tracker_spec_with(max_acceleration_mps2=tuple(acceleration))


def test_tracker_spec_normalizes_real_acceleration_entries() -> None:
    spec = _tracker_spec_with(
        max_acceleration_mps2=(np.float32(1.25), np.int64(2)),
    )

    assert spec.max_acceleration_mps2 == (1.25, 2.0)
    assert all(type(value) is float for value in spec.max_acceleration_mps2)


def test_track_frame_normalizes_state_and_associations() -> None:
    frame = TrackFrame(
        track_ids=np.array([3, 8], dtype=np.uint8),
        positions=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
        velocities=np.array([[0.1, 0.2, 0.0], [0.0, -0.1, 0.0]]),
        position_covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        extent_covariances=np.zeros((2, 2, 2)),
        statuses=(TrackStatus.CONFIRMED, TrackStatus.COASTING),
        ages=np.array([10, 4], dtype=np.uint16),
        missed_counts=np.array([0, 1], dtype=np.uint32),
        observation_track_ids=np.array([3, -1, 8], dtype=np.int16),
        frame_id=5,
        timestamp=0.5,
    )

    assert frame.num_tracks == 2
    assert frame.statuses == (TrackStatus.CONFIRMED, TrackStatus.COASTING)
    assert frame.observation_track_ids.tolist() == [3, -1, 8]
    assert frame.track_ids.dtype == np.dtype(np.int64)
    assert frame.ages.dtype == np.dtype(np.int64)
    assert frame.missed_counts.dtype == np.dtype(np.int64)
    assert frame.observation_track_ids.dtype == np.dtype(np.int64)


@pytest.mark.parametrize(
    ("field_name", "values"),
    [
        (field_name, values)
        for field_name in (
            "track_ids",
            "ages",
            "missed_counts",
            "observation_track_ids",
        )
        for values in (np.array([1.0]), np.array([True]))
    ],
)
def test_track_frame_rejects_non_integer_semantic_fields(
    field_name: str,
    values: np.ndarray,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"TrackFrame\.{field_name} must contain integer values",
    ):
        _single_track_frame(**{field_name: values})


@pytest.mark.parametrize(
    "field_name",
    ["track_ids", "ages", "missed_counts", "observation_track_ids"],
)
def test_track_frame_rejects_integer_values_outside_int64(field_name: str) -> None:
    values = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)

    with pytest.raises(
        ValueError,
        match=rf"TrackFrame\.{field_name} contains values outside the int64 range",
    ):
        _single_track_frame(**{field_name: values})


def test_track_frame_rejects_unknown_associated_track() -> None:
    with pytest.raises(ValueError, match="unknown track"):
        TrackFrame(
            track_ids=np.array([1]),
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
            position_covariances=np.zeros((1, 2, 2)),
            extent_covariances=np.zeros((1, 2, 2)),
            statuses=(TrackStatus.TENTATIVE,),
            ages=np.array([1]),
            missed_counts=np.array([0]),
            observation_track_ids=np.array([2]),
        )


def test_track_frame_rejects_non_positive_semidefinite_covariance() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        TrackFrame(
            track_ids=np.array([1]),
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
            position_covariances=np.array([[[1.0, 0.0], [0.0, -1.0]]]),
            extent_covariances=np.zeros((1, 2, 2)),
            statuses=(TrackStatus.TENTATIVE,),
            ages=np.array([1]),
            missed_counts=np.array([0]),
            observation_track_ids=np.array([1]),
        )


def test_tracking_scenery_accepts_any_configured_boundary_box() -> None:
    scenery = ScenerySpec(
        boundary_boxes=(
            Box2D(-1.0, 1.0, 0.0, 2.0),
            Box2D(2.0, 3.0, 4.0, 5.0),
        ),
        outside_max_frames=3,
    )

    assert scenery.contains(0.0, 1.0)
    assert scenery.contains(2.5, 4.5)
    assert not scenery.contains(0.0, 3.0)
