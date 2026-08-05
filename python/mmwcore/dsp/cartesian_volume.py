"""Metric Cartesian projection for planar range-Doppler antenna tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from mmwcore.core import CartesianRadarVolume, PlanarApertureLayout, RadarCube

from ._cartesian import NativePlanarCartesianConfig, NativePlanarCartesianProjector


@dataclass(frozen=True)
class PlanarCartesianProjector:
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
    _native_projector: NativePlanarCartesianProjector = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        positive_scalars = (
            self.range_resolution_m,
            self.source_velocity_step_mps,
            self.target_velocity_step_mps,
            self.aperture_spacing_wavelengths,
        )
        if any(not isfinite(value) or value <= 0.0 for value in positive_scalars):
            raise ValueError("Cartesian projection resolutions must be finite and positive.")
        if (
            self.source_range_bins <= 0
            or self.source_doppler_bins <= 1
            or self.target_doppler_bins <= 0
            or self.azimuth_n_fft <= 1
            or self.elevation_n_fft <= 1
        ):
            raise ValueError("Cartesian projection FFT and Doppler sizes are invalid.")
        if not isfinite(self.source_velocity_start_mps) or not isfinite(
            self.target_velocity_start_mps
        ):
            raise ValueError("Cartesian projection velocity origins must be finite.")
        shape = tuple(int(value) for value in self.grid_shape_zyx)
        origin = tuple(float(value) for value in self.grid_origin_xyz_m)
        voxel_size = tuple(float(value) for value in self.grid_voxel_size_xyz_m)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("Cartesian projection grid_shape_zyx must be positive.")
        if len(origin) != 3 or not all(isfinite(value) for value in origin):
            raise ValueError("Cartesian projection grid origin must be finite.")
        if len(voxel_size) != 3 or not all(isfinite(value) and value > 0.0 for value in voxel_size):
            raise ValueError("Cartesian projection voxel sizes must be positive.")
        frame = self.coordinate_frame.strip()
        if not frame:
            raise ValueError("Cartesian projection coordinate_frame must not be empty.")
        object.__setattr__(self, "grid_shape_zyx", shape)
        object.__setattr__(self, "grid_origin_xyz_m", origin)
        object.__setattr__(self, "grid_voxel_size_xyz_m", voxel_size)
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(
            self,
            "_native_projector",
            NativePlanarCartesianProjector(
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

    def project(self, range_doppler: RadarCube) -> CartesianRadarVolume:
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
        return CartesianRadarVolume(
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


__all__ = ["PlanarCartesianProjector"]
