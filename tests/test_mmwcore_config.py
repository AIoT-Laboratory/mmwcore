from __future__ import annotations

import json

import pytest

import mmwcore.config
from mmwcore.config import (
    DCA1000ConfigSpec,
    RadarCaptureSpec,
    RadarProfile,
    TiCliConfigSpec,
    iwr6843_isk_3d_cfar_point_cloud_recipe,
    iwr6843_isk_3d_point_cloud_recipe,
    iwr6843_isk_antenna_geometry,
    iwr6843_isk_azimuth_subarray,
    iwr6843_isk_cfar_point_cloud_recipe,
    iwr6843_isk_elevation_subarray,
    iwr6843_isk_planar_aperture_layout,
    iwr6843_isk_tdm_virtual_array,
    iwr6843_profile,
    parse_ti_cli_config,
    render_dca1000_config,
    render_ti_cli_config,
    write_dca1000_config,
    write_ti_cli_config,
)
from mmwcore.core import (
    ADCComplexLayout,
    ADCFrameSpec,
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


def test_iwr6843_isk_tdm_array_keeps_configured_tx_order() -> None:
    spec = iwr6843_isk_tdm_virtual_array()

    assert spec.tx_order == (0, 2, 1)
    assert spec.num_virtual_antennas == 12


def test_iwr6843_range_doppler_recipe_can_match_range_dc_policy() -> None:
    recipe = mmwcore.config.iwr6843_isk_range_doppler_recipe(
        remove_range_dc=True,
    )

    assert recipe.range_fft.remove_dc is True


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


def test_render_ti_cli_config_uses_profile_geometry() -> None:
    profile = RadarProfile(
        start_frequency_hz=60e9,
        frequency_slope_hz_per_s=60e12,
        adc_sample_rate_hz=4.4e6,
        idle_time_s=300e-6,
        adc_start_time_s=6e-6,
        ramp_end_time_s=65e-6,
        num_adc_samples=256,
        num_chirps_per_tx=64,
        num_tx=3,
        num_rx=4,
    )

    text = render_ti_cli_config(profile)

    assert text.endswith("\n")
    assert "channelCfg 15 7 0" in text
    assert "profileCfg 0 60.000 300.00 6.00 65.00 0 0 60.000 6.00 256 4400 0 0 30" in text
    assert "chirpCfg 0 0 0 0 0 0 0 1" in text
    assert "chirpCfg 1 1 0 0 0 0 0 4" in text
    assert "chirpCfg 2 2 0 0 0 0 0 2" in text
    assert "frameCfg 0 2 64 0 100.0 1 0" in text
    assert "sensorStart" not in text


def test_parse_ti_cli_config_recovers_capture_shape() -> None:
    text = render_ti_cli_config(
        RadarProfile(num_tx=3, num_rx=4, num_adc_samples=256, num_chirps_per_tx=64)
    )

    summary = parse_ti_cli_config(text)

    assert summary.num_tx == 3
    assert summary.num_rx == 4
    assert summary.num_adc_samples == 256
    assert summary.num_loops == 64
    assert summary.num_chirps_per_loop == 3
    assert summary.num_chirps_per_tx == 64
    assert summary.chirps_per_frame == 192
    assert summary.to_adc_frame_spec().raw_values_per_frame == 393216


def test_parse_ti_cli_config_supports_single_tx_shape() -> None:
    text = "\n".join(
        [
            "channelCfg 15 1 0",
            "adcCfg 2 1",
            "profileCfg 0 60 30 7 57.14 0 0 60 1 128 5209 0 0 158",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "frameCfg 0 0 2 0 10 1 0",
        ]
    )

    summary = parse_ti_cli_config(text)

    assert summary.num_tx == 1
    assert summary.num_rx == 4
    assert summary.num_adc_samples == 128
    assert summary.num_loops == 2
    assert summary.num_chirps_per_loop == 1
    assert summary.num_chirps_per_tx == 2
    assert summary.chirps_per_frame == 2
    assert summary.frame_periodicity_s == pytest.approx(0.01)


def test_write_ti_cli_config_supports_custom_export_spec(tmp_path) -> None:
    profile = RadarProfile(num_tx=2, num_rx=2, num_chirps_per_tx=8)
    spec = TiCliConfigSpec(
        profile_id=1,
        frame_periodicity_s=0.05,
        tx_enable_order=(1, 0),
        include_sensor_start=True,
    )

    output = write_ti_cli_config(tmp_path / "profiles" / "radar.cfg", profile, spec)

    text = output.read_text(encoding="utf-8")
    assert "channelCfg 3 3 0" in text
    assert "profileCfg 1" in text
    assert "chirpCfg 0 0 1 0 0 0 0 2" in text
    assert "chirpCfg 1 1 1 0 0 0 0 1" in text
    assert "frameCfg 0 1 8 0 50.0 1 0" in text
    assert text.rstrip().endswith("sensorStart")


def test_ti_cli_config_rejects_invalid_export_spec() -> None:
    with pytest.raises(ValueError, match="adc_bits"):
        TiCliConfigSpec(adc_bits=10)

    profile = RadarProfile(num_tx=3)
    with pytest.raises(ValueError, match="one entry per TX"):
        render_ti_cli_config(profile, TiCliConfigSpec(tx_enable_order=(0, 1)))


def test_render_dca1000_config_uses_capture_settings() -> None:
    profile = RadarProfile(num_rx=4)
    spec = DCA1000ConfigSpec(
        capture_path=r"dataset\subject\radar",
        file_prefix="adc",
        system_ip="192.168.1.10",
        dca_ip="192.168.1.20",
        dca_mac="00.11.22.33.44.55",
        config_port=5000,
        data_port=5002,
        packet_delay_us=30,
        frames_to_capture=4,
    )

    payload = render_dca1000_config(profile, spec)
    config = payload["DCA1000Config"]

    assert config["lvdsMode"] == 2
    assert config["dataFormatMode"] == 3
    assert config["packetDelay_us"] == 30
    assert config["ethernetConfig"]["DCA1000IPAddress"] == "192.168.1.20"
    assert config["ethernetConfigUpdate"]["systemIPAddress"] == "192.168.1.10"
    assert config["captureConfig"]["fileBasePath"] == "dataset/subject/radar"
    assert config["captureConfig"]["filePrefix"] == "adc"
    assert config["captureConfig"]["framesToCapture"] == 4
    assert config["dataFormatConfig"]["reorderEnable"] == 1
    assert len(config["dataFormatConfig"]["dataPortConfig"]) == 4
    assert config["dataFormatConfig"]["dataPortConfig"][0] == {
        "portIdx": 0,
        "dataType": "complex",
    }


def test_write_dca1000_config_writes_json(tmp_path) -> None:
    profile = RadarProfile(num_rx=2)
    spec = DCA1000ConfigSpec(
        dca_mac="00.11.22.33.44.55",
        capture_path="radar",
        adc_bits=14,
        lvds_lanes=4,
        sequence_number_enable=False,
        reorder_enable=False,
        data_type="real",
    )

    output = write_dca1000_config(tmp_path / "dca" / "config.json", profile, spec)

    payload = json.loads(output.read_text(encoding="utf-8"))
    config = payload["DCA1000Config"]
    assert config["dataFormatMode"] == 2
    assert config["lvdsMode"] == 4
    assert config["captureConfig"]["sequenceNumberEnable"] == 0
    assert config["dataFormatConfig"]["reorderEnable"] == 0
    assert config["dataFormatConfig"]["dataPortConfig"][-1] == {
        "portIdx": 3,
        "dataType": "real",
    }


def test_dca1000_config_rejects_invalid_export_spec() -> None:
    with pytest.raises(ValueError, match="adc_bits"):
        DCA1000ConfigSpec(dca_mac="00.11.22.33.44.55", adc_bits=10)

    with pytest.raises(ValueError, match="data_type"):
        DCA1000ConfigSpec(dca_mac="00.11.22.33.44.55", data_type="iq")

    with pytest.raises(ValueError, match="dca_mac"):
        DCA1000ConfigSpec(dca_mac="")
