from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from sys import maxsize

import numpy as np
import pytest

from mmwcore.core import (
    CFAR1DSpec,
    CFARDetectionSpec,
    CFARInputScale,
    CFARMode,
    DetectionQualitySpec,
    PeakDetectionSpec,
    PeakGroupingSpec,
    PointCloudProjectionSpec,
    RangeDopplerCFARSpec,
)

type DetectionSpec = (
    PeakDetectionSpec
    | CFARDetectionSpec
    | CFAR1DSpec
    | RangeDopplerCFARSpec
    | PeakGroupingSpec
    | DetectionQualitySpec
    | PointCloudProjectionSpec
)
type PhysicalSpecFactory = Callable[[float], object]

_CFAR_1D = CFAR1DSpec(
    training_cells=2,
    guard_cells=1,
    threshold_scale=1.0,
)

_INTEGER_SPEC_FIELDS: tuple[tuple[DetectionSpec, str, int], ...] = (
    (PeakDetectionSpec(threshold=0.0), "azimuth_peak_radius", 0),
    (
        CFARDetectionSpec(training_cells=2, guard_cells=1, threshold_scale=1.0),
        "training_cells",
        1,
    ),
    (
        CFARDetectionSpec(training_cells=2, guard_cells=1, threshold_scale=1.0),
        "guard_cells",
        0,
    ),
    (_CFAR_1D, "training_cells", 1),
    (_CFAR_1D, "guard_cells", 0),
    (_CFAR_1D, "left_skip", 0),
    (_CFAR_1D, "right_skip", 0),
    (PeakGroupingSpec(), "range_radius", 0),
    (PeakGroupingSpec(), "doppler_radius", 0),
    (PointCloudProjectionSpec(), "doppler_bins", 1),
)

_PHYSICAL_SPEC_FIELDS: tuple[tuple[str, PhysicalSpecFactory, float], ...] = (
    (
        "PeakDetectionSpec.threshold",
        lambda value: PeakDetectionSpec(threshold=value),
        -1.0,
    ),
    (
        "CFARDetectionSpec.threshold_scale",
        lambda value: CFARDetectionSpec(
            training_cells=2,
            guard_cells=1,
            threshold_scale=value,
        ),
        -1.0,
    ),
    (
        "CFAR1DSpec.threshold_scale",
        lambda value: CFAR1DSpec(
            training_cells=2,
            guard_cells=1,
            threshold_scale=value,
        ),
        -1.0,
    ),
    (
        "DetectionQualitySpec.min_snr",
        lambda value: DetectionQualitySpec(min_snr=value),
        0.0,
    ),
    (
        "PointCloudProjectionSpec.range_resolution_m",
        lambda value: PointCloudProjectionSpec(range_resolution_m=value),
        0.0,
    ),
    (
        "PointCloudProjectionSpec.doppler_resolution_mps",
        lambda value: PointCloudProjectionSpec(doppler_resolution_mps=value),
        0.0,
    ),
)

_BOOL_SPEC_FIELDS: tuple[tuple[DetectionSpec, str], ...] = (
    (PeakDetectionSpec(threshold=0.0), "azimuth_peak_strict"),
    (_CFAR_1D, "cyclic"),
    (PeakGroupingSpec(), "cyclic_doppler"),
    (PeakGroupingSpec(), "strict"),
    (PointCloudProjectionSpec(), "center_doppler"),
    (PointCloudProjectionSpec(), "doppler_fftshifted"),
)

_AGGREGATE_SPEC_FIELDS: tuple[tuple[DetectionSpec, str], ...] = (
    (PeakDetectionSpec(threshold=0.0), "aggregate_rx"),
    (
        CFARDetectionSpec(training_cells=2, guard_cells=1, threshold_scale=1.0),
        "aggregate_rx",
    ),
    (RangeDopplerCFARSpec(range=_CFAR_1D), "aggregate_rx"),
    (PeakGroupingSpec(), "aggregate_rx"),
)


@pytest.mark.parametrize(
    ("spec", "field_name", "_minimum"),
    _INTEGER_SPEC_FIELDS,
    ids=[
        f"{type(spec).__name__}.{field_name}" for spec, field_name, _minimum in _INTEGER_SPEC_FIELDS
    ],
)
@pytest.mark.parametrize(
    "invalid",
    [pytest.param(True, id="bool"), pytest.param(1.0, id="float")],
)
def test_detection_specs_reject_non_integer_fields(
    spec: DetectionSpec,
    field_name: str,
    _minimum: int,
    invalid: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{type(spec).__name__}.{field_name} must be an integer",
    ):
        replace(spec, **{field_name: invalid})


@pytest.mark.parametrize(
    ("spec", "field_name", "minimum"),
    _INTEGER_SPEC_FIELDS,
    ids=[
        f"{type(spec).__name__}.{field_name}" for spec, field_name, _minimum in _INTEGER_SPEC_FIELDS
    ],
)
def test_detection_specs_normalize_numpy_integer_fields(
    spec: DetectionSpec,
    field_name: str,
    minimum: int,
) -> None:
    normalized = replace(spec, **{field_name: np.int64(minimum + 2)})

    value = getattr(normalized, field_name)
    assert value == minimum + 2
    assert type(value) is int


@pytest.mark.parametrize(
    ("spec", "field_name", "_minimum"),
    _INTEGER_SPEC_FIELDS,
    ids=[
        f"{type(spec).__name__}.{field_name}" for spec, field_name, _minimum in _INTEGER_SPEC_FIELDS
    ],
)
def test_detection_specs_bound_integer_fields_to_platform_indices(
    spec: DetectionSpec,
    field_name: str,
    _minimum: int,
) -> None:
    with pytest.raises(
        OverflowError,
        match=rf"{type(spec).__name__}.{field_name}.*platform index",
    ):
        replace(spec, **{field_name: maxsize + 1})


@pytest.mark.parametrize(
    ("spec", "field_name", "minimum"),
    _INTEGER_SPEC_FIELDS,
    ids=[
        f"{type(spec).__name__}.{field_name}" for spec, field_name, _minimum in _INTEGER_SPEC_FIELDS
    ],
)
def test_detection_specs_preserve_integer_domains(
    spec: DetectionSpec,
    field_name: str,
    minimum: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{type(spec).__name__}.{field_name}",
    ):
        replace(spec, **{field_name: minimum - 1})


@pytest.mark.parametrize(
    ("field_name", "factory", "_domain_invalid"),
    _PHYSICAL_SPEC_FIELDS,
    ids=[field_name for field_name, _factory, _invalid in _PHYSICAL_SPEC_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
    ],
)
def test_detection_specs_reject_non_finite_physical_fields(
    field_name: str,
    factory: PhysicalSpecFactory,
    _domain_invalid: float,
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{re.escape(field_name)}.*finite"):
        factory(invalid)


@pytest.mark.parametrize(
    ("field_name", "factory", "_domain_invalid"),
    _PHYSICAL_SPEC_FIELDS,
    ids=[field_name for field_name, _factory, _invalid in _PHYSICAL_SPEC_FIELDS],
)
def test_detection_specs_reject_bool_physical_fields(
    field_name: str,
    factory: PhysicalSpecFactory,
    _domain_invalid: float,
) -> None:
    with pytest.raises(TypeError, match=rf"{re.escape(field_name)}.*not bool"):
        factory(True)


@pytest.mark.parametrize(
    ("field_name", "factory", "domain_invalid"),
    _PHYSICAL_SPEC_FIELDS,
    ids=[field_name for field_name, _factory, _invalid in _PHYSICAL_SPEC_FIELDS],
)
def test_detection_specs_preserve_physical_domains(
    field_name: str,
    factory: PhysicalSpecFactory,
    domain_invalid: float,
) -> None:
    with pytest.raises(ValueError, match=re.escape(field_name)):
        factory(domain_invalid)


@pytest.mark.parametrize(
    ("spec", "field_name"),
    _BOOL_SPEC_FIELDS,
    ids=[f"{type(spec).__name__}.{field_name}" for spec, field_name in _BOOL_SPEC_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(1, id="integer"),
        pytest.param("true", id="string"),
        pytest.param(np.bool_(True), id="numpy-bool"),
    ],
)
def test_detection_specs_require_explicit_bool_policies(
    spec: DetectionSpec,
    field_name: str,
    invalid: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{type(spec).__name__}.{field_name} must be a bool",
    ):
        replace(spec, **{field_name: invalid})


@pytest.mark.parametrize(
    ("spec", "field_name"),
    _AGGREGATE_SPEC_FIELDS,
    ids=[f"{type(spec).__name__}.{field_name}" for spec, field_name in _AGGREGATE_SPEC_FIELDS],
)
def test_detection_specs_require_string_aggregation(
    spec: DetectionSpec,
    field_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{type(spec).__name__}.{field_name} must be a string",
    ):
        replace(spec, **{field_name: 1})


@pytest.mark.parametrize(
    ("spec", "field_name"),
    _AGGREGATE_SPEC_FIELDS,
    ids=[f"{type(spec).__name__}.{field_name}" for spec, field_name in _AGGREGATE_SPEC_FIELDS],
)
def test_detection_specs_close_aggregation_choices(
    spec: DetectionSpec,
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{type(spec).__name__}.{field_name} must be one of",
    ):
        replace(spec, **{field_name: "median"})


@pytest.mark.parametrize(
    ("spec", "field_name"),
    _AGGREGATE_SPEC_FIELDS,
    ids=[f"{type(spec).__name__}.{field_name}" for spec, field_name in _AGGREGATE_SPEC_FIELDS],
)
@pytest.mark.parametrize("aggregate_rx", ["max", "sum", "mean"])
def test_detection_specs_accept_closed_aggregation_choices(
    spec: DetectionSpec,
    field_name: str,
    aggregate_rx: str,
) -> None:
    normalized = replace(spec, **{field_name: aggregate_rx})

    assert getattr(normalized, field_name) == aggregate_rx


def test_detection_enum_strings_remain_closed_and_normalized() -> None:
    cfar = replace(_CFAR_1D, mode="go")
    range_doppler = replace(
        RangeDopplerCFARSpec(range=_CFAR_1D),
        input_scale="magnitude",
    )

    assert cfar.mode is CFARMode.GO
    assert range_doppler.input_scale is CFARInputScale.MAGNITUDE
    with pytest.raises(ValueError):
        replace(_CFAR_1D, mode="median")
    with pytest.raises(ValueError):
        replace(RangeDopplerCFARSpec(range=_CFAR_1D), input_scale="amplitude")


def test_detection_specs_preserve_dependent_policy_constraints() -> None:
    with pytest.raises(ValueError, match="at least one non-zero radius"):
        PeakGroupingSpec(range_radius=0, doppler_radius=0)
    with pytest.raises(ValueError, match="doppler_bins is required"):
        PointCloudProjectionSpec(center_doppler=True)
