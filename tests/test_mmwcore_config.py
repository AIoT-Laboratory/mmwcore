from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

import mmwcore.config
from mmwcore.config import (
    RadarCaptureSpec,
    RadarProfile,
    awr1843_aop_antenna_geometry,
    iwr6843_aop_antenna_geometry,
    iwr6843_isk_3d_cfar_point_cloud_recipe,
    iwr6843_isk_3d_point_cloud_recipe,
    iwr6843_isk_antenna_geometry,
    iwr6843_isk_azimuth_subarray,
    iwr6843_isk_cfar_point_cloud_recipe,
    iwr6843_isk_elevation_subarray,
    iwr6843_isk_planar_aperture_layout,
    iwr6843_isk_tdm_virtual_array,
    iwr6843_profile,
    parse_ti_cli_capture_spec,
    xwr1642_antenna_geometry,
    xwr1843_evm_antenna_geometry,
)
from mmwcore.core import (
    ADCComplexLayout,
    ADCFrameSpec,
    AntennaArrayGeometry,
    CFAR1DSpec,
    DetectionMethod,
    DetectionQualitySpec,
    PeakGroupingSpec,
    PointCloudProjectionSpec,
    RangeDopplerCFARSpec,
)


def test_mmwcore_config_exports_radar_profile() -> None:
    assert mmwcore.config.RadarProfile is RadarProfile


def test_radar_profile_derives_physical_dimensions() -> None:
    profile = RadarProfile(
        start_frequency_hz=60e9,
        frequency_slope_hz_per_s=60e12,
        adc_sample_rate_hz=4e6,
        adc_start_time_s=5e-6,
        ramp_end_time_s=65e-6,
        idle_time_s=35e-6,
        num_adc_samples=4,
        num_chirps_per_tx=8,
        num_tx=2,
        num_rx=4,
        speed_of_light_mps=300e6,
    )

    assert profile.wavelength_m == pytest.approx(0.005)
    assert profile.bandwidth_hz == pytest.approx(60e6)
    assert profile.range_resolution_m == pytest.approx(2.5)
    assert profile.max_range_m == pytest.approx(10.0)
    assert profile.chirps_per_frame == 16
    assert profile.virtual_antennas == 8
    assert profile.max_velocity_mps == pytest.approx(6.25)
    assert profile.velocity_resolution_mps == pytest.approx(1.5625)


def test_radar_profile_builds_adc_frame_spec() -> None:
    profile = RadarProfile(num_adc_samples=32, num_chirps_per_tx=16, num_tx=3, num_rx=4)

    spec = profile.to_adc_frame_spec(layout=ADCComplexLayout.GROUP2_I_THEN_Q)

    assert spec.num_chirps == 48
    assert spec.num_rx == 4
    assert spec.num_samples == 32
    assert spec.layout is ADCComplexLayout.GROUP2_I_THEN_Q


def test_radar_profile_builds_point_cloud_projection_spec() -> None:
    profile = RadarProfile(num_chirps_per_tx=16)

    spec = profile.to_point_cloud_projection_spec()

    assert isinstance(spec, PointCloudProjectionSpec)
    assert spec.range_resolution_m == profile.range_resolution_m
    assert spec.doppler_resolution_mps == profile.velocity_resolution_mps
    assert spec.center_doppler is True
    assert spec.doppler_bins == 16


def test_radar_profile_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_rx"):
        RadarProfile(num_rx=0)

    with pytest.raises(ValueError, match="before ramp"):
        RadarProfile(adc_start_time_s=65e-6, ramp_end_time_s=65e-6)


@pytest.mark.parametrize(
    "field_name",
    [
        "start_frequency_hz",
        "frequency_slope_hz_per_s",
        "adc_sample_rate_hz",
        "adc_start_time_s",
        "ramp_end_time_s",
        "idle_time_s",
        "speed_of_light_mps",
    ],
)
def test_radar_profile_rejects_boolean_physical_values(field_name: str) -> None:
    with pytest.raises(TypeError, match=rf"RadarProfile\.{field_name} must be a real number"):
        replace(RadarProfile(), **{field_name: True})


@pytest.mark.parametrize(
    "field_name",
    ["num_adc_samples", "num_chirps_per_tx", "num_tx", "num_rx"],
)
@pytest.mark.parametrize("value", [True, 1.5])
def test_radar_profile_rejects_nonintegral_dimensions(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=rf"RadarProfile\.{field_name} must be an integer"):
        replace(RadarProfile(), **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("num_adc_samples", 17),
        ("num_chirps_per_tx", 19),
        ("num_tx", 2),
        ("num_rx", 4),
    ],
)
def test_radar_profile_normalizes_python_and_numpy_integral_dimensions(
    field_name: str,
    value: int,
) -> None:
    for supplied in (value, np.int64(value)):
        profile = replace(RadarProfile(), **{field_name: supplied})
        normalized = getattr(profile, field_name)
        assert normalized == value
        assert type(normalized) is int


@pytest.mark.parametrize(
    ("field_name", "factory"),
    [
        ("start_frequency_hz", lambda value: RadarProfile(start_frequency_hz=value)),
        (
            "frequency_slope_hz_per_s",
            lambda value: RadarProfile(frequency_slope_hz_per_s=value),
        ),
        ("adc_sample_rate_hz", lambda value: RadarProfile(adc_sample_rate_hz=value)),
        ("adc_start_time_s", lambda value: RadarProfile(adc_start_time_s=value)),
        ("ramp_end_time_s", lambda value: RadarProfile(ramp_end_time_s=value)),
        ("idle_time_s", lambda value: RadarProfile(idle_time_s=value)),
        ("speed_of_light_mps", lambda value: RadarProfile(speed_of_light_mps=value)),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_radar_profile_rejects_nonfinite_physical_values(
    field_name: str,
    factory: Callable[[float], RadarProfile],
    value: float,
) -> None:
    message = rf"RadarProfile\.{field_name} must be finite and positive"
    with pytest.raises(ValueError, match=message):
        factory(value)


def test_radar_capture_spec_joins_explicit_capture_contract() -> None:
    profile = RadarProfile(
        idle_time_s=7e-6,
        num_tx=3,
        num_rx=4,
        num_adc_samples=256,
        num_chirps_per_tx=128,
    )
    capture = RadarCaptureSpec(
        profile=profile,
        adc=ADCFrameSpec(
            num_chirps=384,
            num_rx=4,
            num_samples=256,
            layout=ADCComplexLayout.GROUP2_I_THEN_Q,
        ),
        tx_order=(0, 1, 2),
        frame_periodicity_s=0.1,
        num_frames=600,
    )

    assert capture.expected_size_bytes == 943_718_400
    assert capture.tx_order == (0, 1, 2)
    assert RadarCaptureSpec.from_record(capture.to_record()) == capture

    tampered = capture.to_record()
    tampered["expected_size_bytes"] = 1
    with pytest.raises(ValueError, match="expected_size_bytes"):
        RadarCaptureSpec.from_record(tampered)


@pytest.mark.parametrize(
    ("adc", "tx_order", "match"),
    [
        (ADCFrameSpec(384, 2, 256), (0, 1, 2), "Rx mismatch"),
        (ADCFrameSpec(384, 4, 128), (0, 1, 2), "sample mismatch"),
        (ADCFrameSpec(256, 4, 256), (0, 1, 2), "chirp mismatch"),
        (ADCFrameSpec(384, 4, 256), (0, 0, 2), "duplicates"),
        (ADCFrameSpec(256, 4, 256), (0, 1), "one physical identifier"),
        (ADCFrameSpec(384, 4, 256), (0, 1, -1), "non-negative"),
    ],
)
def test_radar_capture_spec_rejects_inconsistent_contract(
    adc: ADCFrameSpec,
    tx_order: tuple[int, ...],
    match: str,
) -> None:
    profile = RadarProfile(
        num_tx=3,
        num_rx=4,
        num_adc_samples=256,
        num_chirps_per_tx=128,
    )

    with pytest.raises(ValueError, match=match):
        RadarCaptureSpec(profile=profile, adc=adc, tx_order=tx_order)


def test_radar_capture_spec_preserves_sparse_physical_tx_identifiers() -> None:
    profile = RadarProfile(
        num_tx=2,
        num_rx=4,
        num_adc_samples=256,
        num_chirps_per_tx=32,
    )

    capture = RadarCaptureSpec(
        profile=profile,
        adc=profile.to_adc_frame_spec(layout=ADCComplexLayout.GROUP2_I_THEN_Q),
        tx_order=(0, 2),
        frame_periodicity_s=0.1,
        num_frames=100,
    )

    assert capture.tx_order == (0, 2)
    assert capture.expected_size_bytes == 26_214_400


def test_radar_capture_spec_rejects_impossible_frame_period() -> None:
    profile = RadarProfile(
        idle_time_s=7e-6,
        ramp_end_time_s=65e-6,
        num_adc_samples=256,
        num_chirps_per_tx=128,
        num_tx=3,
        num_rx=4,
    )

    with pytest.raises(ValueError, match="active chirp time"):
        RadarCaptureSpec(
            profile=profile,
            adc=profile.to_adc_frame_spec(),
            tx_order=(0, 1, 2),
            frame_periodicity_s=0.02,
            num_frames=1,
        )


def test_iwr6843_profile_preset_matches_default_capture_shape() -> None:
    profile = iwr6843_profile()

    assert profile.start_frequency_hz == 60e9
    assert profile.num_tx == 3
    assert profile.num_rx == 4
    assert profile.chirps_per_frame == 192
    assert profile.to_adc_frame_spec().raw_values_per_frame == 393216


def test_iwr6843_profile_preset_supports_overrides() -> None:
    profile = iwr6843_profile(num_adc_samples=64, num_chirps_per_tx=16)

    assert profile.num_adc_samples == 64
    assert profile.num_chirps_per_tx == 16
    assert profile.chirps_per_frame == 48


def test_iwr6843_isk_geometry_uses_standard_evm_phase_centers() -> None:
    geometry = iwr6843_isk_antenna_geometry()

    assert geometry.tx_positions_wavelengths == (
        (0.0, 0.0, 0.5),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.5),
    )
    assert geometry.rx_positions_wavelengths == (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
    )


@pytest.mark.parametrize(
    ("factory", "name", "tx_positions", "rx_positions"),
    [
        (
            xwr1642_antenna_geometry,
            "xwr1642",
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.5, 0.0, 0.0),
            ),
        ),
        (
            xwr1843_evm_antenna_geometry,
            "xwr1843_evm",
            ((0.0, 0.0, 0.5), (1.0, 0.0, 0.0), (2.0, 0.0, 0.5)),
            (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.5, 0.0, 0.0),
            ),
        ),
        (
            iwr6843_aop_antenna_geometry,
            "iwr6843_aop",
            ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            (
                (0.5, 0.0, 0.5),
                (0.5, 0.0, 0.0),
                (0.0, 0.0, 0.5),
                (0.0, 0.0, 0.0),
            ),
        ),
        (
            awr1843_aop_antenna_geometry,
            "awr1843_aop",
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)),
            (
                (1.5, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        ),
    ],
)
def test_ti_sdk_antenna_geometry_presets_match_half_wavelength_offsets(
    factory: Callable[[], AntennaArrayGeometry],
    name: str,
    tx_positions: tuple[tuple[float, float, float], ...],
    rx_positions: tuple[tuple[float, float, float], ...],
) -> None:
    geometry = factory()

    assert geometry.name == name
    assert geometry.tx_positions_wavelengths == tx_positions
    assert geometry.rx_positions_wavelengths == rx_positions


def test_iwr6843_isk_tdm_array_keeps_configured_tx_order() -> None:
    spec = iwr6843_isk_tdm_virtual_array()

    assert spec.tx_order == (0, 2, 1)
    assert spec.num_virtual_antennas == 12


def test_iwr6843_range_doppler_recipe_can_match_range_dc_policy() -> None:
    recipe = mmwcore.config.iwr6843_isk_range_doppler_recipe(
        remove_range_dc=True,
    )

    assert recipe.range_fft.remove_dc is True


def test_iwr6843_range_doppler_recipe_supports_active_tx_subset() -> None:
    profile = iwr6843_profile(num_tx=2, num_chirps_per_tx=32)

    recipe = mmwcore.config.iwr6843_isk_range_doppler_recipe(
        profile,
        adc_layout=ADCComplexLayout.GROUP2_I_THEN_Q,
        tx_order=(0, 2),
    )

    assert recipe.decode.adc == profile.to_adc_frame_spec(layout=ADCComplexLayout.GROUP2_I_THEN_Q)
    assert recipe.tdm_virtual_array is not None
    assert recipe.tdm_virtual_array.tx_order == (0, 2)
    assert recipe.tdm_virtual_array.geometry.num_tx == 3


def test_iwr6843_range_doppler_recipe_rejects_active_tx_count_mismatch() -> None:
    profile = iwr6843_profile(num_tx=2)

    with pytest.raises(ValueError, match="active-Tx count"):
        mmwcore.config.iwr6843_isk_range_doppler_recipe(
            profile,
            tx_order=(0, 2, 1),
        )


def test_iwr6843_isk_planar_aperture_tracks_virtual_channel_order() -> None:
    layout = iwr6843_isk_planar_aperture_layout()

    assert layout.aperture_shape == (8, 2)
    assert layout.num_antennas == 12
    assert layout.num_unique_positions == 12
    assert layout.grid_indices[:4] == tuple((index, 1) for index in range(4))
    assert layout.grid_indices[4:8] == tuple((index, 1) for index in range(4, 8))
    assert layout.grid_indices[8:] == tuple((index, 0) for index in range(2, 6))


def test_iwr6843_isk_azimuth_subarray_is_eight_element_ula() -> None:
    subarray = iwr6843_isk_azimuth_subarray()

    assert subarray.antenna_indices == tuple(range(8))
    assert subarray.layout.positions_wavelengths == tuple(
        (index * 0.5, 0.0, 0.5) for index in range(8)
    )


def test_iwr6843_isk_azimuth_subarray_tracks_tx_slots() -> None:
    subarray = iwr6843_isk_azimuth_subarray(tx_order=(0, 1, 2))

    assert subarray.antenna_indices == (0, 1, 2, 3, 8, 9, 10, 11)
    assert subarray.layout.positions_wavelengths == tuple(
        (index * 0.5, 0.0, 0.5) for index in range(8)
    )


def test_iwr6843_isk_elevation_subarray_tracks_displaced_row() -> None:
    default = iwr6843_isk_elevation_subarray()
    alternate = iwr6843_isk_elevation_subarray(tx_order=(0, 1, 2))

    assert default.antenna_indices == (8, 9, 10, 11)
    assert alternate.antenna_indices == (4, 5, 6, 7)
    assert default.layout.positions_wavelengths == tuple(
        (1.0 + index * 0.5, 0.0, 0.0) for index in range(4)
    )


def test_iwr6843_3d_recipe_declares_paired_elevation_row() -> None:
    recipe = iwr6843_isk_3d_point_cloud_recipe(100.0, tx_order=(0, 1, 2))

    assert recipe.detection.virtual_subarray == iwr6843_isk_azimuth_subarray(tx_order=(0, 1, 2))
    assert recipe.detection.elevation_subarray == iwr6843_isk_elevation_subarray(tx_order=(0, 1, 2))


def test_iwr6843_cfar_recipe_uses_candidate_level_aoa() -> None:
    recipe = iwr6843_isk_cfar_point_cloud_recipe(
        RangeDopplerCFARSpec(
            range=CFAR1DSpec(training_cells=2, guard_cells=1, threshold_scale=4.0)
        ),
        PeakGroupingSpec(),
        quality_filter=DetectionQualitySpec(25.0),
    )

    assert recipe.detection.detection_method is DetectionMethod.CFAR
    assert recipe.detection.peak_detection is None
    assert recipe.detection.peak_grouping is not None
    assert recipe.detection.quality_filter == DetectionQualitySpec(25.0)
    assert recipe.detection.angle_fft is not None
    assert recipe.detection.angle_fft.input_axis == "virtual_rx"


def test_iwr6843_3d_cfar_recipe_declares_paired_elevation_row() -> None:
    recipe = iwr6843_isk_3d_cfar_point_cloud_recipe(
        RangeDopplerCFARSpec(
            range=CFAR1DSpec(training_cells=2, guard_cells=1, threshold_scale=4.0)
        ),
        PeakGroupingSpec(),
        tx_order=(0, 1, 2),
    )

    assert recipe.detection.detection_method is DetectionMethod.CFAR
    assert recipe.detection.virtual_subarray == iwr6843_isk_azimuth_subarray(tx_order=(0, 1, 2))
    assert recipe.detection.elevation_subarray == iwr6843_isk_elevation_subarray(tx_order=(0, 1, 2))


def test_parse_ti_cli_capture_spec_preserves_physical_capture_contract() -> None:
    text = "\n".join(
        [
            "dfeDataOutputMode 1",
            "channelCfg 15 7 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 60 7 3 24 0 0 166 1 256 12500 0 0 158",
            "chirpCfg 1 1 0 0 0 0 0 4",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "frameCfg 0 1 32 100 100 1 0",
            "lvdsStreamCfg -1 0 1 0",
        ]
    )

    capture = parse_ti_cli_capture_spec(
        text,
        layout=ADCComplexLayout.GROUP2_I_THEN_Q,
        family="xwr68xx",
    )

    assert capture.tx_order == (0, 2)
    assert capture.profile.num_tx == 2
    assert capture.profile.num_rx == 4
    assert capture.profile.num_adc_samples == 256
    assert capture.profile.num_chirps_per_tx == 32
    assert capture.profile.start_frequency_hz == pytest.approx(60e9)
    assert capture.profile.frequency_slope_hz_per_s == pytest.approx(166e12)
    assert capture.profile.adc_sample_rate_hz == pytest.approx(12.5e6)
    assert capture.adc == ADCFrameSpec(
        num_chirps=64,
        num_rx=4,
        num_samples=256,
        layout=ADCComplexLayout.GROUP2_I_THEN_Q,
    )
    assert capture.frame_periodicity_s == pytest.approx(0.1)
    assert capture.num_frames == 100
    assert capture.expected_size_bytes == 26_214_400


@pytest.mark.parametrize(
    ("family", "channel_tx_mask", "chirp_tx_masks", "expected_tx_order"),
    [
        ("xwr16xx", 3, (2, 1), (1, 0)),
        ("xwr18xx", 7, (4, 1, 2), (2, 0, 1)),
    ],
)
def test_parse_ti_cli_capture_spec_accepts_77_ghz_family_tx_contracts(
    family: str,
    channel_tx_mask: int,
    chirp_tx_masks: tuple[int, ...],
    expected_tx_order: tuple[int, ...],
) -> None:
    text = _ti_family_capture_config(
        start_frequency_ghz=77,
        channel_tx_mask=channel_tx_mask,
        chirp_tx_masks=chirp_tx_masks,
    )

    capture = parse_ti_cli_capture_spec(
        text,
        layout=ADCComplexLayout.GROUP2_I_THEN_Q,
        family=family,
    )

    assert capture.profile.start_frequency_hz == pytest.approx(77e9)
    assert capture.profile.num_tx == len(chirp_tx_masks)
    assert capture.tx_order == expected_tx_order


@pytest.mark.parametrize(
    ("family", "start_frequency_ghz", "channel_tx_mask", "chirp_tx_masks"),
    [
        ("xwr16xx", 60, 3, (1, 2)),
        ("xwr18xx", 60, 7, (1, 4, 2)),
        ("xwr68xx", 77, 7, (1, 4, 2)),
        ("xwr16xx", 77, 7, (1, 4)),
    ],
)
def test_parse_ti_cli_capture_spec_rejects_family_config_mismatch(
    family: str,
    start_frequency_ghz: int,
    channel_tx_mask: int,
    chirp_tx_masks: tuple[int, ...],
) -> None:
    text = _ti_family_capture_config(
        start_frequency_ghz=start_frequency_ghz,
        channel_tx_mask=channel_tx_mask,
        chirp_tx_masks=chirp_tx_masks,
    )

    with pytest.raises(ValueError, match="start frequency|TX mask"):
        parse_ti_cli_capture_spec(
            text,
            layout=ADCComplexLayout.GROUP2_I_THEN_Q,
            family=family,
        )


def test_parse_ti_cli_capture_spec_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="xwr16xx, xwr18xx, xwr68xx"):
        parse_ti_cli_capture_spec(
            "",
            layout=ADCComplexLayout.GROUP2_I_THEN_Q,
            family="xwr14xx",
        )


def test_parse_ti_cli_capture_spec_supports_continuous_frames() -> None:
    text = "\n".join(
        [
            "dfeDataOutputMode 1",
            "channelCfg 15 7 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 60 7 3 24 0 0 166 1 256 12500 0 0 158",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "chirpCfg 1 1 0 0 0 0 0 4",
            "chirpCfg 2 2 0 0 0 0 0 2",
            "frameCfg 0 2 64 0 100 1 0",
            "lvdsStreamCfg -1 0 1 0",
        ]
    )
    capture = parse_ti_cli_capture_spec(
        text,
        layout=ADCComplexLayout.IQ_INTERLEAVED,
        family="xwr68xx",
    )

    assert capture.tx_order == (0, 2, 1)
    assert capture.num_frames is None
    assert capture.expected_size_bytes is None
    assert capture.adc.layout is ADCComplexLayout.IQ_INTERLEAVED


def test_parse_ti_cli_capture_spec_rejects_late_flush() -> None:
    with pytest.raises(ValueError, match="flushCfg must precede"):
        parse_ti_cli_capture_spec(
            "\n".join(
                [
                    "dfeDataOutputMode 1",
                    "channelCfg 15 1 0",
                    "flushCfg",
                ]
            ),
            layout=ADCComplexLayout.IQ_INTERLEAVED,
            family="xwr68xx",
        )


@pytest.mark.parametrize(
    ("chirp_lines", "match"),
    [
        (["chirpCfg 0 0 0 0 0 0 0 3"], "exactly one TX"),
        (
            [
                "chirpCfg 0 0 0 0 0 0 0 1",
                "chirpCfg 1 1 0 0 0 0 0 1",
            ],
            "each active TX exactly once",
        ),
    ],
)
def test_parse_ti_cli_capture_spec_rejects_non_tdm_tx_sequences(
    chirp_lines: list[str],
    match: str,
) -> None:
    chirp_end = len(chirp_lines) - 1
    text = "\n".join(
        [
            "dfeDataOutputMode 1",
            "channelCfg 15 7 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 60 7 3 24 0 0 166 1 256 12500 0 0 158",
            *chirp_lines,
            f"frameCfg 0 {chirp_end} 32 100 100 1 0",
            "lvdsStreamCfg -1 0 1 0",
        ]
    )

    with pytest.raises(ValueError, match=match):
        parse_ti_cli_capture_spec(
            text,
            layout=ADCComplexLayout.GROUP2_I_THEN_Q,
            family="xwr68xx",
        )


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ("adcCfg 2 0", "16-bit complex"),
        ("adcbufCfg -1 0 1 0 1", "exact adcbufCfg"),
        ("chirpCfg 0 0 0 0 1 0 0 1", "chirpCfg variations"),
        ("frameCfg 0 1 32 100 100 2 0", "software-triggered"),
        ("lvdsStreamCfg -1 1 1 0", "no header"),
        ("channelCfg 15 7", "expected exactly 3"),
        ("channelCfg 16 7 0", "RX mask"),
        ("channelCfg 5 7 0", "Sparse"),
        ("channelCfg 15 8 0", "TX mask"),
        ("dfeDataOutputMode 1 0", "expected exactly 1"),
        ("adcCfg 2 1 0", "expected exactly 2"),
        ("adcbufCfg -1 0 1 1 1 0", "expected exactly 5"),
        ("profileCfg 0 60 7 3 24 0 0 166 1 256 12500 0 0 158 0", "expected exactly 14"),
        ("profileCfg 4 60 7 3 24 0 0 166 1 256 12500 0 0 158", "ID must be"),
        ("chirpCfg 0 0 0 0 0 0 0 1 0", "expected exactly 8"),
        ("chirpCfg 0 512 0 0 0 0 0 1", "indices must be"),
        ("frameCfg 0 1 32 100 100 1 0 0", "expected exactly 7"),
        ("frameCfg 0 512 32 100 100 1 0", "chirp indices must be"),
        ("frameCfg 0 1 256 100 100 1 0", "num loops"),
        ("frameCfg 0 1 32 65536 100 1 0", "num frames"),
        ("lvdsStreamCfg -1 0 1 0 0", "expected exactly 4"),
    ],
)
def test_parse_ti_cli_capture_spec_rejects_ambiguous_physical_contracts(
    replacement: str,
    match: str,
) -> None:
    lines = [
        "dfeDataOutputMode 1",
        "channelCfg 15 7 0",
        "adcCfg 2 1",
        "adcbufCfg -1 0 1 1 1",
        "profileCfg 0 60 7 3 24 0 0 166 1 256 12500 0 0 158",
        "chirpCfg 0 0 0 0 0 0 0 1",
        "chirpCfg 1 1 0 0 0 0 0 4",
        "frameCfg 0 1 32 100 100 1 0",
        "lvdsStreamCfg -1 0 1 0",
    ]
    command = replacement.split(maxsplit=1)[0]
    lines[next(index for index, line in enumerate(lines) if line.startswith(command))] = replacement

    with pytest.raises(ValueError, match=match):
        parse_ti_cli_capture_spec(
            "\n".join(lines),
            layout=ADCComplexLayout.GROUP2_I_THEN_Q,
            family="xwr68xx",
        )


@pytest.mark.parametrize(
    ("contract_lines", "match"),
    [
        (["adcCfg 2 1"], "dfeDataOutputMode"),
        (["dfeDataOutputMode 1"], "adcCfg"),
        (["dfeDataOutputMode 2", "adcCfg 2 1"], "dfeDataOutputMode 1"),
        (["dfeDataOutputMode 1", "adcCfg 2 0"], "16-bit complex"),
    ],
)
def test_ti_cli_capture_spec_rejects_missing_or_invalid_capture_commands(
    contract_lines: list[str],
    match: str,
) -> None:
    text = "\n".join(
        [
            *contract_lines,
            "channelCfg 15 1 0",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 60 7 3 24 0 0 166 1 256 12500 0 0 158",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "frameCfg 0 0 32 100 100 1 0",
            "lvdsStreamCfg -1 0 1 0",
        ]
    )

    with pytest.raises(ValueError, match=match):
        parse_ti_cli_capture_spec(
            text,
            layout=ADCComplexLayout.GROUP2_I_THEN_Q,
            family="xwr68xx",
        )


@pytest.mark.parametrize("command", ["adcbufCfg", "lvdsStreamCfg"])
def test_ti_cli_capture_spec_requires_raw_hardware_stream_commands(command: str) -> None:
    lines = [
        "dfeDataOutputMode 1",
        "channelCfg 15 1 0",
        "adcCfg 2 1",
        "adcbufCfg -1 0 1 1 1",
        "profileCfg 0 60 7 3 24 0 0 166 1 256 12500 0 0 158",
        "chirpCfg 0 0 0 0 0 0 0 1",
        "frameCfg 0 0 64 0 100 1 0",
        "lvdsStreamCfg -1 0 1 0",
    ]
    lines = [line for line in lines if not line.startswith(command)]

    with pytest.raises(ValueError, match=command):
        parse_ti_cli_capture_spec(
            "\n".join(lines),
            layout=ADCComplexLayout.IQ_INTERLEAVED,
            family="xwr68xx",
        )


@pytest.mark.parametrize(
    "layout",
    [
        ADCComplexLayout.GROUP2_I_THEN_Q,
        cast(ADCComplexLayout, ADCComplexLayout.GROUP2_I_THEN_Q.value),
    ],
)
def test_ti_cli_capture_spec_rejects_odd_group2_sample_count(
    layout: ADCComplexLayout,
) -> None:
    text = "\n".join(
        [
            "dfeDataOutputMode 1",
            "channelCfg 15 1 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 60 7 3 24 0 0 166 1 255 12500 0 0 158",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "frameCfg 0 0 64 0 100 1 0",
            "lvdsStreamCfg -1 0 1 0",
        ]
    )
    with pytest.raises(ValueError, match="even numAdcSamples"):
        parse_ti_cli_capture_spec(
            text,
            layout=layout,
            family="xwr68xx",
        )


def _ti_family_capture_config(
    *,
    start_frequency_ghz: int,
    channel_tx_mask: int,
    chirp_tx_masks: tuple[int, ...],
) -> str:
    chirps = [
        f"chirpCfg {index} {index} 0 0 0 0 0 {tx_mask}"
        for index, tx_mask in enumerate(chirp_tx_masks)
    ]
    return "\n".join(
        [
            "flushCfg",
            "dfeDataOutputMode 1",
            f"channelCfg 15 {channel_tx_mask} 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            f"profileCfg 0 {start_frequency_ghz} 7 3 24 0 0 166 1 4 12500 0 0 158",
            *chirps,
            f"frameCfg 0 {len(chirps) - 1} 1 1 10 1 0",
            "lvdsStreamCfg -1 0 1 0",
        ]
    )
