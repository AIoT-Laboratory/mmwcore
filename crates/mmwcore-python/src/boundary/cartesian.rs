//! Cartesian boundary contracts.

use super::contiguous_axis;
use mmwcore::{
    CartesianProjectionError, CartesianSparsificationConfig, CartesianSparsificationError,
    DetectionPointCloudConfig, DetectionPointCloudError, PlanarCartesianProjectionConfig,
};
use numpy::{PyArray2, PyArrayDyn, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn detection_point_cloud_error(error: DetectionPointCloudError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn cartesian_error(error: CartesianProjectionError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn sparsification_error(error: CartesianSparsificationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn cartesian_axes(axes: NativeCartesianAxes<'_>) -> PyResult<NativeCartesianAxisValues> {
    let (doppler_velocity_mps, z_m, y_m, x_m) = axes;
    Ok((
        contiguous_axis(doppler_velocity_mps, "doppler_velocity_mps")?,
        contiguous_axis(z_m, "z_m")?,
        contiguous_axis(y_m, "y_m")?,
        contiguous_axis(x_m, "x_m")?,
    ))
}

pub(crate) type NativeDopplerAxis = (usize, f32, f32);

pub(crate) type NativeGridShape = (usize, usize, usize);

pub(crate) type NativeGridCoordinates = (f32, f32, f32);

pub(crate) type NativePlanarAngleConfig = (usize, usize, f32);

pub(crate) type NativePlanarCartesianConfig = (
    f32,
    NativeDopplerAxis,
    NativeDopplerAxis,
    NativeGridShape,
    NativeGridCoordinates,
    NativeGridCoordinates,
    (f32, f32),
    NativePlanarAngleConfig,
);

pub(crate) type NativePlanarCartesianResult<'py> = (
    Bound<'py, PyArrayDyn<f32>>,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);

pub(crate) type NativeCartesianAxes<'py> = (
    PyReadonlyArray1<'py, f32>,
    PyReadonlyArray1<'py, f32>,
    PyReadonlyArray1<'py, f32>,
    PyReadonlyArray1<'py, f32>,
);

pub(crate) type NativeCartesianAxisValues = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

pub(crate) type NativeCartesianSparsificationThresholdConfig = (f32, usize);

pub(crate) type NativeCartesianSparsificationPeakConfig = (usize, usize, Option<usize>, usize);

pub(crate) type NativeCartesianSparsificationBackgroundConfig = (f32, f32, f32, bool);

pub(crate) type NativeCartesianSparsificationConfig = (
    NativeCartesianSparsificationThresholdConfig,
    NativeCartesianSparsificationPeakConfig,
    NativeCartesianSparsificationBackgroundConfig,
);

pub(crate) type NativeCartesianSparsificationResult<'py> = (
    Bound<'py, PyArray2<f32>>,
    (f32, f32, f32),
    (usize, usize, usize, usize, usize, usize, usize),
    (bool, usize),
);

pub(crate) fn planar_cartesian_config(
    config: NativePlanarCartesianConfig,
) -> PlanarCartesianProjectionConfig {
    let (
        range_resolution_m,
        source_doppler,
        target_doppler,
        grid_shape_zyx,
        grid_origin_xyz_m,
        grid_voxel_size_xyz_m,
        mount,
        angle,
    ) = config;
    let (source_doppler_bins, source_velocity_start_mps, source_velocity_step_mps) = source_doppler;
    let (target_doppler_bins, target_velocity_start_mps, target_velocity_step_mps) = target_doppler;
    let (azimuth_n_fft, elevation_n_fft, aperture_spacing_wavelengths) = angle;
    let (mount_height_m, mount_pitch_deg) = mount;
    PlanarCartesianProjectionConfig {
        range_resolution_m,
        source_doppler_bins,
        source_velocity_start_mps,
        source_velocity_step_mps,
        target_doppler_bins,
        target_velocity_start_mps,
        target_velocity_step_mps,
        grid_shape_zyx: [grid_shape_zyx.0, grid_shape_zyx.1, grid_shape_zyx.2],
        grid_origin_xyz_m: [
            grid_origin_xyz_m.0,
            grid_origin_xyz_m.1,
            grid_origin_xyz_m.2,
        ],
        grid_voxel_size_xyz_m: [
            grid_voxel_size_xyz_m.0,
            grid_voxel_size_xyz_m.1,
            grid_voxel_size_xyz_m.2,
        ],
        mount_height_m,
        mount_pitch_deg,
        azimuth_n_fft,
        elevation_n_fft,
        aperture_spacing_wavelengths,
    }
}

pub(crate) fn cartesian_sparsification_config(
    config: NativeCartesianSparsificationConfig,
) -> CartesianSparsificationConfig {
    let (threshold, peaks, background) = config;
    let (min_snr_db, max_points) = threshold;
    let (
        spatial_peak_radius,
        doppler_peak_radius,
        max_doppler_peaks_per_spatial,
        boundary_margin_voxels,
    ) = peaks;
    let (
        noise_floor_scale,
        static_point_capacity_fraction,
        static_velocity_threshold_mps,
        strongest_point_fallback,
    ) = background;
    CartesianSparsificationConfig {
        min_snr_db,
        max_points,
        spatial_peak_radius,
        doppler_peak_radius,
        max_doppler_peaks_per_spatial,
        boundary_margin_voxels,
        noise_floor_scale,
        static_point_capacity_fraction,
        static_velocity_threshold_mps,
        strongest_point_fallback,
    }
}

pub(crate) fn detection_point_cloud_config(
    config: NativeDetectionPointCloudConfig,
) -> DetectionPointCloudConfig {
    let (
        range_resolution_m,
        doppler_resolution_mps,
        doppler_sign,
        center_doppler,
        doppler_bins,
        doppler_fftshifted,
    ) = config;
    DetectionPointCloudConfig {
        range_resolution_m,
        doppler_resolution_mps,
        doppler_sign,
        center_doppler,
        doppler_bins,
        doppler_fftshifted,
    }
}

pub(crate) type NativeDetectionPointCloudColumns = (
    usize,
    usize,
    usize,
    usize,
    usize,
    Option<(usize, usize)>,
    Vec<usize>,
);

pub(crate) type NativeDetectionPointCloudConfig = (f32, f32, i8, bool, Option<usize>, bool);
