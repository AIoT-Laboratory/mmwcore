"""Metric Cartesian projection for planar range-Doppler antenna tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from operator import index as integer_index
from sys import maxsize as _MAX_PLATFORM_INDEX

import numpy as np

from mmwcore.core import CartesianVolume, PlanarApertureLayout, RadarCube

from ._cartesian import NativeCartesianProjector, NativePlanarCartesianConfig


def _platform_index(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        normalized = int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if not -_MAX_PLATFORM_INDEX - 1 <= normalized <= _MAX_PLATFORM_INDEX:
        raise OverflowError(f"{name} must fit the platform index range.")
    return normalized


def _integer_at_least(value: int, *, name: str, minimum: int) -> int:
    normalized = _platform_index(value, name=name)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {normalized}.")
    return normalized


def _integer_triplet(
    values: tuple[int, int, int],
    *,
    name: str,
) -> tuple[int, int, int]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of three integers.") from exc
    if len(raw_values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    normalized = tuple(
        _integer_at_least(value, name=f"{name}[{index}]", minimum=1)
        for index, value in enumerate(raw_values)
    )
    return normalized[0], normalized[1], normalized[2]


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool or string.")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite; got {normalized}.")
    return normalized


def _positive_real(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive; got {normalized}.")
    return normalized


def _real_triplet(
    values: tuple[float, float, float],
    *,
    name: str,
    positive: bool,
) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of three real numbers.")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of three real numbers.") from exc
    if len(raw_values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    normalize = _positive_real if positive else _finite_real
    normalized = tuple(
        normalize(value, name=f"{name}[{index}]") for index, value in enumerate(raw_values)
    )
    return normalized[0], normalized[1], normalized[2]


@dataclass(frozen=True)
class CartesianProjector:
    """Project one planar virtual array onto a metric DZYX magnitude grid.

    Source and target Doppler coordinates are physical radial velocities. The
    native kernel scatters the declared aperture, applies zero-padded planar
    FFTs, and interpolates complex angle samples in direction-cosine space.
    """

    aperture_layout: PlanarApertureLayout
    range_resolution_m: float
    source_range_bins: int
    source_doppler_bins: int
    source_velocity_start_mps: float
    source_velocity_step_mps: float
    target_doppler_bins: int
    target_velocity_start_mps: float
    target_velocity_step_mps: float
    grid_shape_zyx: tuple[int, int, int]
    grid_origin_xyz_m: tuple[float, float, float]
    grid_voxel_size_xyz_m: tuple[float, float, float]
    coordinate_frame: str
    azimuth_n_fft: int = 128
    elevation_n_fft: int = 32
    aperture_spacing_wavelengths: float = 0.5
    _native_projector: NativeCartesianProjector = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.aperture_layout) is not PlanarApertureLayout:
            raise TypeError("CartesianProjector.aperture_layout must be a PlanarApertureLayout.")

        range_resolution_m = _positive_real(
            self.range_resolution_m,
            name="CartesianProjector.range_resolution_m",
        )
        source_range_bins = _integer_at_least(
            self.source_range_bins,
            name="CartesianProjector.source_range_bins",
            minimum=1,
        )
        source_doppler_bins = _integer_at_least(
            self.source_doppler_bins,
            name="CartesianProjector.source_doppler_bins",
            minimum=2,
        )
        source_velocity_start_mps = _finite_real(
            self.source_velocity_start_mps,
            name="CartesianProjector.source_velocity_start_mps",
        )
        source_velocity_step_mps = _positive_real(
            self.source_velocity_step_mps,
            name="CartesianProjector.source_velocity_step_mps",
        )
        target_doppler_bins = _integer_at_least(
            self.target_doppler_bins,
            name="CartesianProjector.target_doppler_bins",
            minimum=1,
        )
        target_velocity_start_mps = _finite_real(
            self.target_velocity_start_mps,
            name="CartesianProjector.target_velocity_start_mps",
        )
        target_velocity_step_mps = _positive_real(
            self.target_velocity_step_mps,
            name="CartesianProjector.target_velocity_step_mps",
        )
        azimuth_n_fft = _integer_at_least(
            self.azimuth_n_fft,
            name="CartesianProjector.azimuth_n_fft",
            minimum=2,
        )
        elevation_n_fft = _integer_at_least(
            self.elevation_n_fft,
            name="CartesianProjector.elevation_n_fft",
            minimum=2,
        )
        aperture_spacing_wavelengths = _positive_real(
            self.aperture_spacing_wavelengths,
            name="CartesianProjector.aperture_spacing_wavelengths",
        )
        shape = _integer_triplet(
            self.grid_shape_zyx,
            name="CartesianProjector.grid_shape_zyx",
        )
        origin = _real_triplet(
            self.grid_origin_xyz_m,
            name="CartesianProjector.grid_origin_xyz_m",
            positive=False,
        )
        voxel_size = _real_triplet(
            self.grid_voxel_size_xyz_m,
            name="CartesianProjector.grid_voxel_size_xyz_m",
            positive=True,
        )
        if not isinstance(self.coordinate_frame, str):
            raise TypeError("CartesianProjector.coordinate_frame must be a string.")
        frame = self.coordinate_frame.strip()
        if not frame:
            raise ValueError("CartesianProjector.coordinate_frame must not be empty.")

        object.__setattr__(self, "range_resolution_m", range_resolution_m)
        object.__setattr__(self, "source_range_bins", source_range_bins)
        object.__setattr__(self, "source_doppler_bins", source_doppler_bins)
        object.__setattr__(self, "source_velocity_start_mps", source_velocity_start_mps)
        object.__setattr__(self, "source_velocity_step_mps", source_velocity_step_mps)
        object.__setattr__(self, "target_doppler_bins", target_doppler_bins)
        object.__setattr__(self, "target_velocity_start_mps", target_velocity_start_mps)
        object.__setattr__(self, "target_velocity_step_mps", target_velocity_step_mps)
        object.__setattr__(self, "azimuth_n_fft", azimuth_n_fft)
        object.__setattr__(self, "elevation_n_fft", elevation_n_fft)
        object.__setattr__(
            self,
            "aperture_spacing_wavelengths",
            aperture_spacing_wavelengths,
        )
        object.__setattr__(self, "grid_shape_zyx", shape)
        object.__setattr__(self, "grid_origin_xyz_m", origin)
        object.__setattr__(self, "grid_voxel_size_xyz_m", voxel_size)
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(
            self,
            "_native_projector",
            NativeCartesianProjector(
                source_range_bins=self.source_range_bins,
                grid_indices=self.aperture_layout.grid_indices,
                config=self._native_config(),
            ),
        )

    @property
    def target_velocities_mps(self) -> np.ndarray:
        """Return the physical target Doppler coordinates."""

        return self.target_velocity_start_mps + self.target_velocity_step_mps * np.arange(
            self.target_doppler_bins, dtype=np.float32
        )

    def coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return increasing z, y, and x grid coordinates in meters."""

        z_size, y_size, x_size = self.grid_shape_zyx
        x_step, y_step, z_step = self.grid_voxel_size_xyz_m
        x_origin, y_origin, z_origin = self.grid_origin_xyz_m
        return (
            z_origin + np.arange(z_size, dtype=np.float32) * z_step,
            y_origin + np.arange(y_size, dtype=np.float32) * y_step,
            x_origin + np.arange(x_size, dtype=np.float32) * x_step,
        )

    def project(self, range_doppler: RadarCube) -> CartesianVolume:
        """Return target-velocity Cartesian magnitude with axes DZYX."""

        expected_axes = ("frame", "doppler_bin", "virtual_rx", "range_bin")
        if range_doppler.axes != expected_axes:
            raise ValueError(
                f"Planar Cartesian source axes must be {expected_axes}; got {range_doppler.axes}."
            )
        if range_doppler.data.shape[0] != 1:
            raise ValueError("Planar Cartesian projection requires exactly one radar frame.")
        if range_doppler.data.shape[1] != self.source_doppler_bins:
            raise ValueError("Planar Cartesian source Doppler size does not match the projector.")
        if range_doppler.data.shape[2] != self.aperture_layout.num_antennas:
            raise ValueError(
                "Planar Cartesian source virtual-array size does not match the aperture."
            )
        if range_doppler.data.shape[3] != self.source_range_bins:
            raise ValueError("Planar Cartesian source range size does not match the projector.")
        if not np.iscomplexobj(range_doppler.data):
            raise TypeError("Planar Cartesian source must retain complex antenna samples.")

        (
            magnitude,
            doppler_start,
            doppler_stop,
            range_start,
            range_stop,
            spatial_valid_count,
            doppler_valid_count,
        ) = self._native_projector.project(range_doppler.data)
        z_m, y_m, x_m = self.coordinates()
        spatial_voxel_count = (
            self.grid_shape_zyx[0] * self.grid_shape_zyx[1] * self.grid_shape_zyx[2]
        )
        return CartesianVolume(
            magnitude_dzyx=magnitude,
            doppler_velocity_mps=self.target_velocities_mps,
            z_m=z_m,
            y_m=y_m,
            x_m=x_m,
            frame_id=range_doppler.frame_id,
            timestamp=range_doppler.timestamp,
            source=range_doppler.source,
            coordinate_frame=self.coordinate_frame,
            metadata={
                **range_doppler.metadata,
                "planar_cartesian_projection": {
                    "schema": "mmwcore.planar_cartesian_projection.v1",
                    "coordinate_frame": self.coordinate_frame,
                    "aperture": self.aperture_layout.as_metadata(),
                    "aperture_spacing_wavelengths": self.aperture_spacing_wavelengths,
                    "angle_fft": [self.azimuth_n_fft, self.elevation_n_fft],
                    "range_resolution_m": self.range_resolution_m,
                    "source_range_bins": self.source_range_bins,
                    "source_doppler_bins": self.source_doppler_bins,
                    "source_velocity_start_mps": self.source_velocity_start_mps,
                    "source_velocity_step_mps": self.source_velocity_step_mps,
                    "target_velocity_start_mps": self.target_velocity_start_mps,
                    "target_velocity_step_mps": self.target_velocity_step_mps,
                    "target_doppler_bins": self.target_doppler_bins,
                    "grid_shape_zyx": list(self.grid_shape_zyx),
                    "grid_origin_xyz_m": list(self.grid_origin_xyz_m),
                    "grid_voxel_size_xyz_m": list(self.grid_voxel_size_xyz_m),
                    "valid_spatial_voxel_fraction": spatial_valid_count / spatial_voxel_count,
                    "valid_target_doppler_fraction": doppler_valid_count / self.target_doppler_bins,
                    "source_selection": {
                        "doppler_start": doppler_start,
                        "doppler_stop": doppler_stop,
                        "range_start": range_start,
                        "range_stop": range_stop,
                    },
                    "interpolation": (
                        "complex_trilinear_range_angle_then_linear_magnitude_doppler"
                    ),
                    "doppler_axis_resolution_ratio": (
                        self.source_velocity_step_mps / self.target_velocity_step_mps
                    ),
                },
            },
        )

    def artifact_metadata(self) -> dict[str, object]:
        """Serialize the target-independent projection contract."""

        return {
            "type": type(self).__name__,
            "coordinate_frame": self.coordinate_frame,
            "aperture": self.aperture_layout.as_metadata(),
            "range_resolution_m": self.range_resolution_m,
            "source_range_bins": self.source_range_bins,
            "source_doppler": {
                "bins": self.source_doppler_bins,
                "velocity_start_mps": self.source_velocity_start_mps,
                "velocity_step_mps": self.source_velocity_step_mps,
            },
            "target_doppler": {
                "bins": self.target_doppler_bins,
                "velocity_start_mps": self.target_velocity_start_mps,
                "velocity_step_mps": self.target_velocity_step_mps,
            },
            "grid_shape_zyx": list(self.grid_shape_zyx),
            "grid_origin_xyz_m": list(self.grid_origin_xyz_m),
            "grid_voxel_size_xyz_m": list(self.grid_voxel_size_xyz_m),
            "angle_fft": [self.azimuth_n_fft, self.elevation_n_fft],
        }

    def _native_config(self) -> NativePlanarCartesianConfig:
        return (
            self.range_resolution_m,
            self.source_doppler_bins,
            self.source_velocity_start_mps,
            self.source_velocity_step_mps,
            self.target_doppler_bins,
            self.target_velocity_start_mps,
            self.target_velocity_step_mps,
            self.grid_shape_zyx,
            self.grid_origin_xyz_m,
            self.grid_voxel_size_xyz_m,
            self.azimuth_n_fft,
            self.elevation_n_fft,
            self.aperture_spacing_wavelengths,
        )


__all__ = ["CartesianProjector"]
