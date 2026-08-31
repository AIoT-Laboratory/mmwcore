from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from sys import maxsize

import numpy as np
import pytest

from mmwcore.core import PlanarApertureLayout, RadarCube, SparsifySpec
from mmwcore.dsp import CartesianProjector

type PhysicalProjectorFactory = Callable[[object], CartesianProjector]


def _projector(
    *,
    grid_origin_xyz_m: tuple[float, float, float] = (1.0, 0.0, 1.0),
    mount_pitch_deg: float = 0.0,
    target_velocity_mps: float = 0.0,
) -> CartesianProjector:
    return CartesianProjector(
        aperture_layout=PlanarApertureLayout(
            ((0, 0), (1, 0), (0, 1), (1, 1)),
            name="fixture",
        ),
        range_resolution_m=0.5,
        source_range_bins=4,
        source_doppler_bins=3,
        source_velocity_start_mps=-1.0,
        source_velocity_step_mps=1.0,
        target_doppler_bins=1,
        target_velocity_start_mps=target_velocity_mps,
        target_velocity_step_mps=1.0,
        grid_shape_zyx=(1, 1, 1),
        grid_origin_xyz_m=grid_origin_xyz_m,
        grid_voxel_size_xyz_m=(0.5, 0.5, 0.5),
        coordinate_frame="forward_lateral_up",
        mount_height_m=1.0,
        mount_pitch_deg=mount_pitch_deg,
        azimuth_n_fft=4,
        elevation_n_fft=4,
    )


def _projector_with(**changes: object) -> CartesianProjector:
    return replace(_projector(), **changes)


def _projector_with_triplet_component(
    field_name: str,
    index: int,
    value: object,
) -> CartesianProjector:
    projector = _projector()
    values = list(getattr(projector, field_name))
    values[index] = value
    return replace(projector, **{field_name: tuple(values)})


_PROJECTOR_INTEGER_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("source_range_bins", 5, 1),
    ("source_doppler_bins", 4, 2),
    ("target_doppler_bins", 2, 1),
    ("azimuth_n_fft", 5, 2),
    ("elevation_n_fft", 5, 2),
)

_PROJECTOR_PHYSICAL_FIELDS: tuple[
    tuple[str, PhysicalProjectorFactory, float, bool],
    ...,
] = (
    (
        "range_resolution_m",
        lambda value: _projector_with(range_resolution_m=value),
        0.75,
        True,
    ),
    (
        "source_velocity_start_mps",
        lambda value: _projector_with(source_velocity_start_mps=value),
        -1.25,
        False,
    ),
    (
        "source_velocity_step_mps",
        lambda value: _projector_with(source_velocity_step_mps=value),
        0.75,
        True,
    ),
    (
        "target_velocity_start_mps",
        lambda value: _projector_with(target_velocity_start_mps=value),
        0.25,
        False,
    ),
    (
        "target_velocity_step_mps",
        lambda value: _projector_with(target_velocity_step_mps=value),
        0.75,
        True,
    ),
    (
        "aperture_spacing_wavelengths",
        lambda value: _projector_with(aperture_spacing_wavelengths=value),
        0.75,
        True,
    ),
    (
        "mount_height_m",
        lambda value: _projector_with(mount_height_m=value),
        1.5,
        True,
    ),
    *tuple(
        (
            f"grid_origin_xyz_m[{index}]",
            lambda value, index=index: _projector_with_triplet_component(
                "grid_origin_xyz_m",
                index,
                value,
            ),
            (0.75, 0.25, 0.25)[index],
            False,
        )
        for index in range(3)
    ),
    *tuple(
        (
            f"grid_voxel_size_xyz_m[{index}]",
            lambda value, index=index: _projector_with_triplet_component(
                "grid_voxel_size_xyz_m",
                index,
                value,
            ),
            0.75,
            True,
        )
        for index in range(3)
    ),
)


@pytest.mark.parametrize(
    ("field_name", "_valid_value", "_minimum"),
    _PROJECTOR_INTEGER_FIELDS,
)
@pytest.mark.parametrize(
    "invalid",
    [pytest.param(True, id="bool"), pytest.param(1.5, id="float")],
)
def test_planar_cartesian_projector_rejects_nonintegral_scalar_dimensions(
    field_name: str,
    _valid_value: int,
    _minimum: int,
    invalid: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"CartesianProjector.{field_name} must be an integer",
    ):
        _projector_with(**{field_name: invalid})


@pytest.mark.parametrize(
    ("field_name", "valid_value", "_minimum"),
    _PROJECTOR_INTEGER_FIELDS,
)
def test_planar_cartesian_projector_normalizes_numpy_scalar_dimensions(
    field_name: str,
    valid_value: int,
    _minimum: int,
) -> None:
    projector = _projector_with(**{field_name: np.int64(valid_value)})

    normalized = getattr(projector, field_name)
    assert normalized == valid_value
    assert type(normalized) is int


@pytest.mark.parametrize(
    ("field_name", "_valid_value", "_minimum"),
    _PROJECTOR_INTEGER_FIELDS,
)
def test_planar_cartesian_projector_bounds_scalar_dimensions_to_platform(
    field_name: str,
    _valid_value: int,
    _minimum: int,
) -> None:
    with pytest.raises(
        OverflowError,
        match=rf"CartesianProjector.{field_name}.*platform index",
    ):
        _projector_with(**{field_name: maxsize + 1})


@pytest.mark.parametrize(
    ("field_name", "_valid_value", "minimum"),
    _PROJECTOR_INTEGER_FIELDS,
)
def test_planar_cartesian_projector_preserves_scalar_dimension_domains(
    field_name: str,
    _valid_value: int,
    minimum: int,
) -> None:
    with pytest.raises(ValueError, match=rf"CartesianProjector.{field_name}"):
        _projector_with(**{field_name: minimum - 1})


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize(
    "invalid",
    [pytest.param(True, id="bool"), pytest.param(1.5, id="float")],
)
def test_planar_cartesian_projector_rejects_nonintegral_grid_shape(
    index: int,
    invalid: object,
) -> None:
    field_name = f"CartesianProjector.grid_shape_zyx[{index}]"
    with pytest.raises(
        TypeError,
        match=rf"{re.escape(field_name)} must be an integer",
    ):
        _projector_with_triplet_component("grid_shape_zyx", index, invalid)


@pytest.mark.parametrize("index", range(3))
def test_planar_cartesian_projector_normalizes_numpy_grid_shape(index: int) -> None:
    projector = _projector_with_triplet_component("grid_shape_zyx", index, np.int64(2))

    assert projector.grid_shape_zyx[index] == 2
    assert type(projector.grid_shape_zyx[index]) is int
    assert type(projector.grid_shape_zyx) is tuple


@pytest.mark.parametrize("index", range(3))
def test_planar_cartesian_projector_bounds_grid_shape_to_platform(index: int) -> None:
    with pytest.raises(
        OverflowError,
        match=rf"{re.escape(f'CartesianProjector.grid_shape_zyx[{index}]')}.*platform index",
    ):
        _projector_with_triplet_component("grid_shape_zyx", index, maxsize + 1)


@pytest.mark.parametrize("index", range(3))
def test_planar_cartesian_projector_preserves_positive_grid_shape(index: int) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"CartesianProjector.grid_shape_zyx[{index}]"),
    ):
        _projector_with_triplet_component("grid_shape_zyx", index, 0)


@pytest.mark.parametrize(
    ("field_name", "factory", "_valid_value", "_positive"),
    _PROJECTOR_PHYSICAL_FIELDS,
    ids=[field_name for field_name, _factory, _value, _positive in _PROJECTOR_PHYSICAL_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param("1.0", id="string"),
    ],
)
def test_planar_cartesian_projector_rejects_nonreal_physical_values(
    field_name: str,
    factory: PhysicalProjectorFactory,
    _valid_value: float,
    _positive: bool,
    invalid: object,
) -> None:
    with pytest.raises(TypeError, match=re.escape(field_name)):
        factory(invalid)


@pytest.mark.parametrize(
    ("field_name", "factory", "_valid_value", "_positive"),
    _PROJECTOR_PHYSICAL_FIELDS,
    ids=[field_name for field_name, _factory, _value, _positive in _PROJECTOR_PHYSICAL_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
    ],
)
def test_planar_cartesian_projector_rejects_nonfinite_physical_values(
    field_name: str,
    factory: PhysicalProjectorFactory,
    _valid_value: float,
    _positive: bool,
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{re.escape(field_name)}.*finite"):
        factory(invalid)


@pytest.mark.parametrize(
    ("field_name", "factory", "valid_value", "_positive"),
    _PROJECTOR_PHYSICAL_FIELDS,
    ids=[field_name for field_name, _factory, _value, _positive in _PROJECTOR_PHYSICAL_FIELDS],
)
def test_planar_cartesian_projector_normalizes_numpy_physical_values(
    field_name: str,
    factory: PhysicalProjectorFactory,
    valid_value: float,
    _positive: bool,
) -> None:
    projector = factory(np.float32(valid_value))
    root_name, _, index_text = field_name.partition("[")
    normalized = getattr(projector, root_name)
    if index_text:
        normalized = normalized[int(index_text.removesuffix("]"))]

    assert normalized == pytest.approx(valid_value)
    assert type(normalized) is float


@pytest.mark.parametrize(
    ("field_name", "factory", "_valid_value", "_positive"),
    [entry for entry in _PROJECTOR_PHYSICAL_FIELDS if entry[3]],
    ids=[
        field_name
        for field_name, _factory, _value, positive in _PROJECTOR_PHYSICAL_FIELDS
        if positive
    ],
)
def test_planar_cartesian_projector_preserves_positive_physical_domains(
    field_name: str,
    factory: PhysicalProjectorFactory,
    _valid_value: float,
    _positive: bool,
) -> None:
    with pytest.raises(ValueError, match=re.escape(field_name)):
        factory(0.0)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("grid_shape_zyx", (1, 1)),
        ("grid_origin_xyz_m", (0.0, 0.0)),
        ("grid_voxel_size_xyz_m", (1.0, 1.0)),
    ],
)
def test_planar_cartesian_projector_requires_three_value_grid_tuples(
    field_name: str,
    invalid: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match=rf"CartesianProjector.{field_name}"):
        _projector_with(**{field_name: invalid})


def test_planar_cartesian_projector_requires_exact_aperture_layout_type() -> None:
    with pytest.raises(TypeError, match="CartesianProjector.aperture_layout"):
        _projector_with(aperture_layout=object())


@pytest.mark.parametrize("invalid", [None, 1, b"sensor"])
def test_planar_cartesian_projector_requires_string_coordinate_frame(invalid: object) -> None:
    with pytest.raises(TypeError, match="CartesianProjector.coordinate_frame"):
        _projector_with(coordinate_frame=invalid)


def test_planar_cartesian_projector_normalizes_nonempty_coordinate_frame() -> None:
    projector = _projector_with(coordinate_frame="  sensor_frame  ")

    assert projector.coordinate_frame == "sensor_frame"
    with pytest.raises(ValueError, match="CartesianProjector.coordinate_frame"):
        _projector_with(coordinate_frame=" 	 ")


@pytest.mark.parametrize(
    ("pitch_deg", "level_origin"),
    [(0.0, (1.0, 0.0, 1.0)), (30.0, (np.sqrt(3.0) / 2.0, 0.0, 0.5)), (90.0, (0.0, 0.0, 0.0))],
)
def test_planar_cartesian_projector_accepts_supported_mount_pitch(
    pitch_deg: float,
    level_origin: tuple[float, float, float],
) -> None:
    projector = _projector(grid_origin_xyz_m=level_origin, mount_pitch_deg=pitch_deg)

    assert projector.mount_pitch_deg == pitch_deg


@pytest.mark.parametrize("pitch_deg", [-1.0, 15.0, 91.0])
def test_planar_cartesian_projector_rejects_unsupported_mount_pitch(pitch_deg: float) -> None:
    with pytest.raises(ValueError, match="mount_pitch_deg must be 0, 30, or 90"):
        _projector_with(mount_pitch_deg=pitch_deg)


@pytest.mark.parametrize(
    ("pitch_deg", "level_forward_m", "level_up_m"),
    [(0.0, 1.0, 1.0), (30.0, np.sqrt(3.0) / 2.0, 0.5), (90.0, 0.0, 0.0)],
)
def test_planar_cartesian_projector_samples_level_grid_for_downward_mount(
    pitch_deg: float,
    level_forward_m: float,
    level_up_m: float,
) -> None:
    source = np.zeros((1, 3, 4, 4), dtype=np.complex64)
    source[0, 1, :, 2] = 1.0 + 0.0j
    projector = _projector(
        grid_origin_xyz_m=(level_forward_m, 0.0, level_up_m),
        mount_pitch_deg=pitch_deg,
    )

    projected = projector.project(
        RadarCube(source, axes=("frame", "doppler_bin", "virtual_rx", "range_bin"))
    )

    assert projected.magnitude_dzyx[0, 0, 0, 0] == pytest.approx(4.0, rel=1e-5)
    np.testing.assert_allclose(projected.x_m, [level_forward_m])
    np.testing.assert_allclose(projected.z_m, [level_up_m])
    metadata = projected.metadata["planar_cartesian_projection"]
    assert metadata["mount_height_m"] == 1.0
    assert metadata["mount_pitch_deg"] == pitch_deg


def test_planar_cartesian_projector_maps_broadside_target_to_metric_voxel() -> None:
    source = np.zeros((1, 3, 4, 4), dtype=np.complex64)
    source[0, 1, :, 2] = 1.0 + 0.0j

    projected = _projector().project(
        RadarCube(
            source,
            axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
            frame_id="frame-1",
        )
    )

    assert projected.magnitude_dzyx.dtype == np.float32
    assert projected.magnitude_dzyx.shape == (1, 1, 1, 1)
    assert projected.magnitude_dzyx[0, 0, 0, 0] == pytest.approx(4.0)
    np.testing.assert_allclose(projected.doppler_velocity_mps, [0.0])
    np.testing.assert_allclose(projected.x_m, [1.0])
    np.testing.assert_allclose(projected.y_m, [0.0])
    np.testing.assert_allclose(projected.z_m, [1.0])
    assert projected.coordinate_frame == "forward_lateral_up"
    assert projected.frame_id == "frame-1"
    projection = projected.metadata["planar_cartesian_projection"]
    assert projection["source_selection"] == {
        "doppler_start": 1,
        "doppler_stop": 2,
        "range_start": 2,
        "range_stop": 3,
    }
    assert projection["valid_spatial_voxel_fraction"] == pytest.approx(1.0)
    assert projection["valid_target_doppler_fraction"] == pytest.approx(1.0)


def test_planar_cartesian_projector_preserves_off_axis_direction_cosines() -> None:
    source = np.zeros((1, 3, 4, 4), dtype=np.complex64)
    aperture_indices = ((0, 0), (1, 0), (0, 1), (1, 1))
    source[0, 1, :, 2] = np.asarray(
        [
            np.exp(2j * np.pi * (azimuth + elevation) / 4.0)
            for azimuth, elevation in aperture_indices
        ],
        dtype=np.complex64,
    )
    projected = _projector(
        grid_origin_xyz_m=(np.sqrt(0.5), 0.5, 1.5),
    ).project(
        RadarCube(
            source,
            axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
        )
    )

    assert projected.magnitude_dzyx[0, 0, 0, 0] == pytest.approx(4.0, rel=1e-5)
    np.testing.assert_allclose(projected.x_m, [np.sqrt(0.5)])
    np.testing.assert_allclose(projected.y_m, [0.5])
    np.testing.assert_allclose(projected.z_m, [1.5])


def test_planar_cartesian_projector_interpolates_physical_doppler_magnitude() -> None:
    source = np.zeros((1, 3, 4, 4), dtype=np.complex64)
    source[0, 1, :, 2] = 1.0 + 0.0j
    source[0, 2, :, 2] = 3.0 + 0.0j

    projected = _projector(target_velocity_mps=0.5).project(
        RadarCube(
            source,
            axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
        )
    )

    assert projected.magnitude_dzyx[0, 0, 0, 0] == pytest.approx(8.0)
    np.testing.assert_allclose(projected.doppler_velocity_mps, [0.5])


def test_planar_cartesian_projector_rejects_real_source_contract() -> None:
    source = RadarCube(
        np.ones((1, 3, 4, 4), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    object.__setattr__(source, "data", np.ones(source.data.shape, dtype=np.float32))

    with pytest.raises(TypeError, match="complex antenna samples"):
        _projector().project(source)


@pytest.mark.parametrize(
    "field_name",
    [
        "max_points",
        "spatial_peak_radius",
        "doppler_peak_radius",
        "max_doppler_peaks_per_spatial",
        "boundary_margin_voxels",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5])
def test_cartesian_sparsification_rejects_nonintegral_integer_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        replace(SparsifySpec(), **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_points", 17),
        ("spatial_peak_radius", 2),
        ("doppler_peak_radius", 3),
        ("max_doppler_peaks_per_spatial", 4),
        ("boundary_margin_voxels", 5),
    ],
)
def test_cartesian_sparsification_normalizes_numpy_integer_fields(
    field_name: str,
    value: int,
) -> None:
    spec = replace(SparsifySpec(), **{field_name: np.int64(value)})
    normalized = getattr(spec, field_name)
    assert normalized == value
    assert type(normalized) is int


@pytest.mark.parametrize(
    "field_name",
    [
        "max_points",
        "spatial_peak_radius",
        "doppler_peak_radius",
        "max_doppler_peaks_per_spatial",
        "boundary_margin_voxels",
    ],
)
def test_cartesian_sparsification_rejects_platform_index_overflow(field_name: str) -> None:
    with pytest.raises(OverflowError, match=field_name):
        replace(SparsifySpec(), **{field_name: maxsize + 1})


@pytest.mark.parametrize(
    "field_name",
    [
        "min_snr_db",
        "noise_floor_scale",
        "static_point_capacity_fraction",
        "static_velocity_threshold_mps",
    ],
)
@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_cartesian_sparsification_rejects_invalid_physical_scalars(
    field_name: str,
    value: object,
) -> None:
    error = TypeError if value is True else ValueError
    with pytest.raises(error, match=field_name):
        replace(SparsifySpec(), **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("min_snr_db", -3.5),
        ("noise_floor_scale", 2.0),
        ("static_point_capacity_fraction", 0.5),
        ("static_velocity_threshold_mps", 0.25),
    ],
)
def test_cartesian_sparsification_normalizes_numpy_physical_scalars(
    field_name: str,
    value: float,
) -> None:
    spec = replace(SparsifySpec(), **{field_name: np.float32(value)})
    normalized = getattr(spec, field_name)
    assert normalized == pytest.approx(value)
    assert type(normalized) is float


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_points", 0),
        ("spatial_peak_radius", -1),
        ("doppler_peak_radius", -1),
        ("max_doppler_peaks_per_spatial", 0),
        ("boundary_margin_voxels", -1),
        ("noise_floor_scale", 0.0),
        ("static_point_capacity_fraction", 0.0),
        ("static_point_capacity_fraction", 1.1),
        ("static_velocity_threshold_mps", -0.1),
    ],
)
def test_cartesian_sparsification_rejects_values_outside_field_domains(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        replace(SparsifySpec(), **{field_name: value})


@pytest.mark.parametrize("value", [1, 0, "yes", np.bool_(True)])
def test_cartesian_sparsification_requires_exact_boolean_fallback(value: object) -> None:
    with pytest.raises(TypeError, match="strongest_point_fallback"):
        SparsifySpec(strongest_point_fallback=value)  # type: ignore[arg-type]
