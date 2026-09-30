from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mmwcore.config import (
    RadarProfile,
    iwr6843_isk_3d_point_cloud_pipeline,
    iwr6843_isk_antenna_geometry,
    iwr6843_isk_point_cloud_pipeline,
    iwr6843_isk_range_doppler_pipeline,
)
from mmwcore.core import ADCFrame, FFTWindow, VirtualChannelCalibration
from mmwcore.dsp import (
    detect,
    point_cloud,
    process_range_doppler_to_calibrated_point_cloud,
    range_doppler,
)


def _small_isk_profile() -> RadarProfile:
    return RadarProfile(
        start_frequency_hz=60e9,
        frequency_slope_hz_per_s=60e12,
        adc_sample_rate_hz=4e6,
        adc_start_time_s=1e-6,
        ramp_end_time_s=10e-6,
        idle_time_s=10e-6,
        num_adc_samples=8,
        num_chirps_per_tx=8,
        num_tx=3,
        num_rx=4,
        speed_of_light_mps=300e6,
    )


def _synthesize_isk_target(
    profile: RadarProfile,
    *,
    range_bin: int,
    doppler_bin: int,
    azimuth_rad: float,
    elevation_rad: float = 0.0,
    amplitude: float = 1000.0,
    channel_errors: tuple[complex, ...] | None = None,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> ADCFrame:
    geometry = iwr6843_isk_antenna_geometry()
    lateral_direction = np.cos(elevation_rad) * np.sin(azimuth_rad)
    vertical_direction = np.sin(elevation_rad)
    cube = np.empty(
        (profile.chirps_per_frame, profile.num_rx, profile.num_adc_samples),
        dtype=np.complex64,
    )
    for chirp in range(profile.chirps_per_frame):
        loop, tx_slot = divmod(chirp, profile.num_tx)
        tx_position = geometry.tx_positions_wavelengths[tx_order[tx_slot]]
        slow_time = loop + tx_slot / profile.num_tx
        for rx_index, rx_position in enumerate(geometry.rx_positions_wavelengths):
            virtual_index = tx_slot * profile.num_rx + rx_index
            channel_error = channel_errors[virtual_index] if channel_errors else 1 + 0j
            virtual_x = tx_position[0] + rx_position[0]
            virtual_z = tx_position[2] + rx_position[2]
            for sample in range(profile.num_adc_samples):
                phase = (
                    2
                    * np.pi
                    * (
                        range_bin * sample / profile.num_adc_samples
                        + doppler_bin * slow_time / profile.num_chirps_per_tx
                        + virtual_x * lateral_direction
                        + virtual_z * vertical_direction
                    )
                )
                cube[chirp, rx_index, sample] = amplitude * np.exp(1j * phase) * channel_error

    iq = np.empty((*cube.shape, 2), dtype=np.int16)
    iq[..., 0] = np.rint(cube.real).astype(np.int16)
    iq[..., 1] = np.rint(cube.imag).astype(np.int16)
    return ADCFrame(iq.reshape(-1), frame_id="synthetic-isk", source="synthetic")


def test_iwr6843_recipe_maps_tdm_adc_to_virtual_range_doppler_cube() -> None:
    profile = _small_isk_profile()
    raw = _synthesize_isk_target(profile, range_bin=2, doppler_bin=1, azimuth_rad=np.pi / 6)
    recipe = iwr6843_isk_point_cloud_pipeline(
        100_000.0,
        profile,
        range_window=FFTWindow.NONE,
        doppler_window=FFTWindow.NONE,
        angle_window=FFTWindow.NONE,
        angle_n_fft=8,
    )

    cube = range_doppler(raw, recipe.detection.transform)

    assert cube.axes == ("frame", "doppler_bin", "virtual_rx", "range_bin")
    assert cube.data.shape == (1, 8, 12, 5)
    assert cube.metadata["tdm_doppler_compensation"]["tx_order"] == [0, 2, 1]


@pytest.mark.parametrize("range_factor,doppler_factor", [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("doppler_bin", [-1, 1])
def test_zero_padding_preserves_target_range_and_velocity(
    range_factor: int,
    doppler_factor: int,
    shift: bool,
    doppler_bin: int,
) -> None:
    profile = _small_isk_profile()
    raw = _synthesize_isk_target(
        profile,
        range_bin=2,
        doppler_bin=doppler_bin,
        azimuth_rad=np.pi / 6,
    )
    base = iwr6843_isk_point_cloud_pipeline(
        100_000,
        profile,
        range_window=FFTWindow.NONE,
        doppler_window=FFTWindow.NONE,
        angle_window=FFTWindow.NONE,
        angle_n_fft=8,
    )
    transform = base.detection.transform
    range_size = profile.num_adc_samples * range_factor
    doppler_size = profile.num_chirps_per_tx * doppler_factor
    recipe = replace(
        base,
        detection=replace(
            base.detection,
            transform=replace(
                transform,
                range_fft=replace(transform.range_fft, n_fft=range_size),
                doppler_fft=replace(transform.doppler_fft, n_fft=doppler_size, fftshift=shift),
            ),
        ),
        projection=profile.to_point_cloud_projection_spec(
            range_n_fft=range_size,
            doppler_bins=doppler_size,
            doppler_fftshifted=shift,
        ),
    )
    cloud = point_cloud(raw, recipe)
    reused = process_range_doppler_to_calibrated_point_cloud(
        range_doppler(raw, recipe.detection.transform),
        recipe,
    )
    np.testing.assert_array_equal(reused.points, cloud.points)
    peak = cloud.points[np.argmax(cloud.points[:, cloud.channels.index("magnitude")])]
    assert np.linalg.norm(peak[:3]) == pytest.approx(2 * profile.range_resolution_m, abs=1e-5)
    assert peak[3] == pytest.approx(doppler_bin * profile.velocity_resolution_mps, abs=1e-5)


@pytest.mark.parametrize(
    "mismatch", ["shift", "length", "tx_order", "calibration", "metadata", "axes"]
)
def test_precomputed_rd_rejects_incompatible_recipe(mismatch: str) -> None:
    profile = _small_isk_profile()
    raw = _synthesize_isk_target(profile, range_bin=2, doppler_bin=1, azimuth_rad=0.0)
    recipe = iwr6843_isk_point_cloud_pipeline(100_000, profile)
    transform = recipe.detection.transform
    if mismatch == "shift":
        transform = replace(transform, doppler_fft=replace(transform.doppler_fft, fftshift=False))
    elif mismatch == "length":
        transform = replace(transform, doppler_fft=replace(transform.doppler_fft, n_fft=16))
    elif mismatch == "tx_order":
        transform = iwr6843_isk_range_doppler_pipeline(profile, tx_order=(0, 1, 2))
    elif mismatch == "calibration":
        transform = replace(transform, channel_calibration=VirtualChannelCalibration((1j,) * 12))
    cube = range_doppler(raw, transform)
    if mismatch == "metadata":
        cube = replace(cube, metadata={})
    elif mismatch == "axes":
        cube = replace(cube, axes=("frame", "virtual_rx", "doppler_bin", "range_bin"))
    with pytest.raises(ValueError, match="Precomputed RD"):
        process_range_doppler_to_calibrated_point_cloud(cube, recipe)


@pytest.mark.parametrize("field,value", [("doppler_fftshifted", False), ("doppler_bins", 16)])
def test_point_cloud_recipe_rejects_inconsistent_projection(field: str, value: object) -> None:
    recipe = iwr6843_isk_point_cloud_pipeline(100_000, _small_isk_profile())
    with pytest.raises(ValueError, match="projection.*must match Doppler FFT"):
        replace(recipe, projection=replace(recipe.projection, **{field: value}))


def test_iwr6843_recipe_maps_active_tx_subset_to_virtual_range_doppler_cube() -> None:
    profile = replace(_small_isk_profile(), num_tx=2)
    raw = _synthesize_isk_target(
        profile,
        range_bin=2,
        doppler_bin=1,
        azimuth_rad=np.pi / 6,
        tx_order=(0, 2),
    )
    recipe = iwr6843_isk_range_doppler_pipeline(
        profile,
        range_window=FFTWindow.NONE,
        doppler_window=FFTWindow.NONE,
        tx_order=(0, 2),
    )

    cube = range_doppler(raw, recipe)

    assert cube.axes == ("frame", "doppler_bin", "virtual_rx", "range_bin")
    assert cube.data.shape == (1, 8, 8, 5)
    assert cube.metadata["tdm_doppler_compensation"]["tx_order"] == [0, 2]
    assert cube.metadata["tdm_doppler_compensation"]["num_tx"] == 2


def test_iwr6843_static_clutter_removal_preserves_moving_target() -> None:
    profile = _small_isk_profile()
    raw = _synthesize_isk_target(
        profile,
        range_bin=2,
        doppler_bin=1,
        azimuth_rad=np.pi / 6,
    )
    recipe = iwr6843_isk_point_cloud_pipeline(
        100_000.0,
        profile,
        range_window=FFTWindow.NONE,
        doppler_window=FFTWindow.NONE,
        angle_window=FFTWindow.NONE,
        angle_n_fft=8,
    )

    full = range_doppler(raw, recipe.detection.transform)
    filtered = range_doppler(
        raw,
        replace(recipe.detection.transform, remove_static_clutter=True),
    )
    full_points = process_range_doppler_to_calibrated_point_cloud(
        full,
        recipe,
    )
    filtered_points = process_range_doppler_to_calibrated_point_cloud(
        filtered,
        recipe,
    )

    np.testing.assert_allclose(filtered.data, full.data, atol=1e-3)
    np.testing.assert_allclose(filtered_points.points, full_points.points, atol=1e-5)
    assert full.metadata.get("static_clutter_removal") is None
    assert filtered.metadata["static_clutter_removal"] == {"axis": "loop"}


def test_iwr6843_static_clutter_removal_removes_stationary_target() -> None:
    profile = _small_isk_profile()
    raw = _synthesize_isk_target(
        profile,
        range_bin=2,
        doppler_bin=0,
        azimuth_rad=np.pi / 6,
    )
    recipe = iwr6843_isk_point_cloud_pipeline(
        100_000.0,
        profile,
        range_window=FFTWindow.NONE,
        doppler_window=FFTWindow.NONE,
        angle_window=FFTWindow.NONE,
        angle_n_fft=8,
    )

    full = range_doppler(raw, recipe.detection.transform)
    filtered = range_doppler(
        raw,
        replace(recipe.detection.transform, remove_static_clutter=True),
    )
    full_points = process_range_doppler_to_calibrated_point_cloud(
        full,
        recipe,
    )
    filtered_points = process_range_doppler_to_calibrated_point_cloud(
        filtered,
        recipe,
    )

    assert full_points.num_points == 1
    assert filtered_points.num_points == 0


@pytest.mark.parametrize("tx_order", [(0, 2, 1), (0, 1, 2)])
@pytest.mark.parametrize(
    ("doppler_bin", "shifted_doppler_bin", "physical_velocity_sign"),
    [(1, 5, 1), (-1, 3, -1)],
)
def test_iwr6843_point_cloud_recipe_recovers_synthetic_target_coordinates(
    tx_order: tuple[int, ...],
    doppler_bin: int,
    shifted_doppler_bin: int,
    physical_velocity_sign: int,
) -> None:
    profile = _small_isk_profile()
    azimuth = np.pi / 6
    raw = _synthesize_isk_target(
        profile,
        range_bin=2,
        doppler_bin=doppler_bin,
        azimuth_rad=azimuth,
        tx_order=tx_order,
    )
    recipe = iwr6843_isk_point_cloud_pipeline(
        100_000.0,
        profile,
        range_window=FFTWindow.NONE,
        doppler_window=FFTWindow.NONE,
        angle_window=FFTWindow.NONE,
        angle_n_fft=8,
        tx_order=tx_order,
    )

    detections = detect(raw, recipe.detection)
    cloud = point_cloud(raw, recipe)

    assert detections.detections.shape == (1, 6)
    detection = dict(zip(detections.channels, detections.detections[0], strict=True))
    assert detection["range_bin"] == pytest.approx(2)
    assert detection["doppler_bin"] == pytest.approx(shifted_doppler_bin)
    assert detection["azimuth_rad"] == pytest.approx(azimuth, abs=1e-6)

    expected_range = 2 * profile.range_resolution_m
    assert cloud.num_points == 1
    assert cloud.points[0, 0] == pytest.approx(expected_range * np.sin(azimuth))
    assert cloud.points[0, 1] == pytest.approx(expected_range * np.cos(azimuth))
    assert cloud.points[0, 3] == pytest.approx(
        physical_velocity_sign * profile.velocity_resolution_mps
    )
    assert cloud.frame_id == "synthetic-isk"
    assert cloud.source == "synthetic"


def test_iwr6843_channel_calibration_restores_azimuth() -> None:
    profile = _small_isk_profile()
    azimuth = np.pi / 6
    errors = tuple(
        np.exp(2j * np.pi * virtual_index / 8) if virtual_index < 8 else 1 + 0j
        for virtual_index in range(12)
    )
    raw = _synthesize_isk_target(
        profile,
        range_bin=2,
        doppler_bin=1,
        azimuth_rad=azimuth,
        channel_errors=errors,
    )
    options = {
        "range_window": FFTWindow.NONE,
        "doppler_window": FFTWindow.NONE,
        "angle_window": FFTWindow.NONE,
        "angle_n_fft": 8,
    }
    uncalibrated = iwr6843_isk_point_cloud_pipeline(100_000.0, profile, **options)
    calibration = VirtualChannelCalibration(
        tuple(1 / error for error in errors),
        source="synthetic",
        version="known-error-v1",
    )
    calibrated = iwr6843_isk_point_cloud_pipeline(
        100_000.0,
        profile,
        channel_calibration=calibration,
        **options,
    )

    uncalibrated_detections = detect(raw, uncalibrated.detection)
    calibrated_detections = detect(raw, calibrated.detection)
    uncalibrated_angle = dict(
        zip(
            uncalibrated_detections.channels,
            uncalibrated_detections.detections[0],
            strict=True,
        )
    )["azimuth_rad"]
    calibrated_angle = dict(
        zip(
            calibrated_detections.channels,
            calibrated_detections.detections[0],
            strict=True,
        )
    )["azimuth_rad"]

    assert uncalibrated_angle != pytest.approx(azimuth, abs=1e-6)
    assert calibrated_angle == pytest.approx(azimuth, abs=1e-6)
    assert (
        calibrated_detections.metadata["virtual_channel_calibration"]["version"] == "known-error-v1"
    )


@pytest.mark.parametrize("tx_order", [(0, 2, 1), (0, 1, 2)])
def test_iwr6843_3d_point_cloud_recovers_synthetic_target(
    tx_order: tuple[int, ...],
) -> None:
    profile = _small_isk_profile()
    vertical_direction = 0.25
    lateral_direction = 0.25
    elevation = float(np.arcsin(vertical_direction))
    azimuth = float(np.arcsin(lateral_direction / np.cos(elevation)))
    raw = _synthesize_isk_target(
        profile,
        range_bin=2,
        doppler_bin=1,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        tx_order=tx_order,
    )
    recipe = iwr6843_isk_3d_point_cloud_pipeline(
        100_000.0,
        profile,
        range_window=FFTWindow.NONE,
        doppler_window=FFTWindow.NONE,
        angle_window=FFTWindow.NONE,
        angle_n_fft=8,
        tx_order=tx_order,
    )

    detections = detect(raw, recipe.detection)
    cloud = point_cloud(raw, recipe)

    detection = dict(zip(detections.channels, detections.detections[0], strict=True))
    assert detection["elevation_rad"] == pytest.approx(elevation, abs=1e-4)
    expected_range = 2 * profile.range_resolution_m
    expected = expected_range * np.array(
        [
            lateral_direction,
            np.sqrt(1.0 - lateral_direction**2 - vertical_direction**2),
            vertical_direction,
        ]
    )
    np.testing.assert_allclose(cloud.points[0, :3], expected, atol=1e-4)
    assert cloud.channels[9:11] == ("elevation_rad", "elevation_magnitude")
    assert cloud.metadata["pointcloud_projection"]["spatial_dimensions"] == 3
