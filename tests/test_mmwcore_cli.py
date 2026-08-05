from __future__ import annotations

import json

import numpy as np
import pytest

from mmwcore.cli.export_config import main as export_config_main
from mmwcore.cli.inspect import inspect_adc_file
from mmwcore.cli.inspect import main as inspect_main
from mmwcore.cli.preprocess_adc import main


def test_preprocess_adc_cli_writes_point_cloud_and_metadata(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    output_path = tmp_path / "point_cloud.npy"
    metadata_path = tmp_path / "point_cloud.json"
    manifest_path = tmp_path / "artifacts" / "point_clouds.jsonl"
    np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16).tofile(adc_path)

    result = main(
        [
            str(adc_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_path),
            "--artifact-manifest",
            str(manifest_path),
            "--sample-id",
            "sample-001",
            "--num-chirps",
            "1",
            "--num-rx",
            "4",
            "--num-samples",
            "1",
            "--range-one-sided",
            "--no-doppler-shift",
            "--threshold",
            "1.0",
            "--angle-fft",
            "--angle-no-shift",
            "--azimuth-peak-radius",
            "0",
            "--virtual-antennas",
            "4",
            "--range-resolution-m",
            "0.5",
            "--frame-id",
            "cli-fixture",
        ]
    )

    assert result == 0
    points = np.load(output_path)
    assert points.shape == (4, 9)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["frame_id"] == "cli-fixture"
    assert metadata["num_points"] == 4
    assert metadata["channels"][:3] == ["x", "y", "z"]
    assert metadata["preprocess"]["adc_spec"] == {
        "num_chirps": 1,
        "num_rx": 4,
        "num_samples": 1,
        "layout": "iq_interleaved",
    }

    artifact = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert artifact["sample_id"] == "sample-001"
    assert artifact["point_channels"][:3] == ["x", "y", "z"]
    assert artifact["raw_sources"]["adc"].endswith("adc.bin")
    assert artifact["metadata"]["preprocess"]["adc_spec"] == {
        "num_chirps": 1,
        "num_rx": 4,
        "num_samples": 1,
        "layout": "iq_interleaved",
    }


def test_preprocess_adc_cli_rejects_multi_tx_preset(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    output_path = tmp_path / "point_cloud.npy"
    np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16).tofile(adc_path)

    with pytest.raises(SystemExit, match="explicit TDM recipe"):
        main(
            [
                str(adc_path),
                "--output",
                str(output_path),
                "--preset",
                "iwr6843",
                "--threshold",
                "1.0",
                "--angle-fft",
                "--virtual-antennas",
                "4",
            ]
        )


def test_preprocess_adc_cli_can_use_ti_cfg_shape(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    cfg_path = tmp_path / "radar.cfg"
    output_path = tmp_path / "point_cloud.npy"
    metadata_path = tmp_path / "point_cloud.json"
    np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16).tofile(adc_path)
    cfg_path.write_text(
        "\n".join(
            [
                "channelCfg 15 1 0",
                "adcCfg 2 1",
                "profileCfg 0 60 30 7 57.14 0 0 60 1 1 5209 0 0 158",
                "chirpCfg 0 0 0 0 0 0 0 1",
                "frameCfg 0 0 1 0 10 1 0",
            ]
        ),
        encoding="utf-8",
    )

    result = main(
        [
            str(adc_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_path),
            "--ti-cfg",
            str(cfg_path),
            "--range-one-sided",
            "--no-doppler-shift",
            "--threshold",
            "1.0",
            "--angle-fft",
            "--angle-no-shift",
            "--azimuth-peak-radius",
            "0",
            "--virtual-antennas",
            "4",
        ]
    )

    assert result == 0
    points = np.load(output_path)
    assert points.shape == (4, 9)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["preprocess"] == {
        "shape_source": "ti_cfg",
        "ti_cfg": str(cfg_path),
        "adc_spec": {
            "num_chirps": 1,
            "num_rx": 4,
            "num_samples": 1,
            "layout": "iq_interleaved",
        },
    }


def test_preprocess_adc_cli_rejects_ti_cfg_with_other_shape_sources(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    cfg_path = tmp_path / "radar.cfg"
    output_path = tmp_path / "point_cloud.npy"
    np.array([1, 0, 0, 0], dtype=np.int16).tofile(adc_path)
    cfg_path.write_text(
        "\n".join(
            [
                "channelCfg 1 1 0",
                "profileCfg 0 60 30 7 57.14 0 0 60 1 2 5209 0 0 158",
                "chirpCfg 0 0 0 0 0 0 0 1",
                "frameCfg 0 0 1 0 10 1 0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="cannot be combined"):
        main(
            [
                str(adc_path),
                "--output",
                str(output_path),
                "--ti-cfg",
                str(cfg_path),
                "--preset",
                "iwr6843",
                "--threshold",
                "1.0",
            ]
        )

    with pytest.raises(SystemExit, match="cannot be combined"):
        main(
            [
                str(adc_path),
                "--output",
                str(output_path),
                "--ti-cfg",
                str(cfg_path),
                "--num-chirps",
                "1",
                "--num-rx",
                "1",
                "--num-samples",
                "2",
                "--threshold",
                "1.0",
            ]
        )


def test_preprocess_adc_cli_rejects_invalid_numeric_args(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    output_path = tmp_path / "point_cloud.npy"
    np.array([1, 0, 0, 0], dtype=np.int16).tofile(adc_path)

    with pytest.raises(SystemExit):
        main(
            [
                str(adc_path),
                "--output",
                str(output_path),
                "--num-chirps",
                "0",
                "--num-rx",
                "1",
                "--num-samples",
                "2",
                "--threshold",
                "1.0",
            ]
        )

    with pytest.raises(SystemExit):
        main(
            [
                str(adc_path),
                "--output",
                str(output_path),
                "--num-chirps",
                "1",
                "--num-rx",
                "1",
                "--num-samples",
                "2",
                "--threshold",
                "-1.0",
            ]
        )

    with pytest.raises(SystemExit):
        main(
            [
                str(adc_path),
                "--output",
                str(output_path),
                "--num-chirps",
                "1",
                "--num-rx",
                "1",
                "--num-samples",
                "2",
                "--range-resolution-m",
                "0",
                "--threshold",
                "1.0",
            ]
        )


def test_preprocess_adc_cli_rejects_cfar_without_point_cloud_aoa(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    output_path = tmp_path / "point_cloud.npy"
    metadata_path = tmp_path / "point_cloud.json"
    np.zeros(50, dtype=np.int16).tofile(adc_path)

    with pytest.raises(SystemExit, match="requires --angle-fft"):
        main(
            [
                str(adc_path),
                "--output",
                str(output_path),
                "--metadata-output",
                str(metadata_path),
                "--num-chirps",
                "5",
                "--num-rx",
                "1",
                "--num-samples",
                "5",
                "--detector",
                "cfar",
                "--cfar-training-cells",
                "1",
                "--cfar-guard-cells",
                "0",
                "--cfar-threshold-scale",
                "2.0",
            ]
        )


def test_preprocess_adc_cli_supports_angle_fft(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    output_path = tmp_path / "point_cloud.npy"
    metadata_path = tmp_path / "point_cloud.json"
    np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16).tofile(adc_path)

    result = main(
        [
            str(adc_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_path),
            "--num-chirps",
            "1",
            "--num-rx",
            "4",
            "--num-samples",
            "1",
            "--range-one-sided",
            "--no-doppler-shift",
            "--threshold",
            "1.0",
            "--angle-fft",
            "--angle-no-shift",
            "--azimuth-peak-radius",
            "0",
            "--virtual-antennas",
            "4",
            "--virtual-spacing-wavelengths",
            "0.5",
        ]
    )

    assert result == 0
    points = np.load(output_path)
    assert points.shape == (4, 9)
    np.testing.assert_array_equal(points[:, 7], np.array([0, 1, 2, 3], dtype=np.float32))
    np.testing.assert_allclose(
        points[:, 8],
        np.array([0.0, np.pi / 6, -np.pi / 2, -np.pi / 6], dtype=np.float32),
        atol=1e-6,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["channels"][-2:] == ["azimuth_bin", "azimuth_rad"]
    assert metadata["metadata"]["angle_fft"]["virtual_layout"]["num_antennas"] == 4


def test_preprocess_adc_cli_supports_candidate_aoa_with_cfar(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    output_path = tmp_path / "point_cloud.npy"
    np.zeros(100, dtype=np.int16).tofile(adc_path)

    result = main(
        [
            str(adc_path),
            "--output",
            str(output_path),
            "--num-chirps",
            "5",
            "--num-rx",
            "2",
            "--num-samples",
            "5",
            "--detector",
            "cfar",
            "--cfar-training-cells",
            "1",
            "--cfar-guard-cells",
            "0",
            "--cfar-threshold-scale",
            "2.0",
            "--angle-fft",
            "--virtual-antennas",
            "2",
        ]
    )

    assert result == 0
    assert np.load(output_path).shape == (0, 12)


def test_preprocess_adc_cli_requires_cfar_parameters(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    output_path = tmp_path / "point_cloud.npy"
    np.zeros(50, dtype=np.int16).tofile(adc_path)

    with pytest.raises(SystemExit, match="required when --detector=cfar"):
        main(
            [
                str(adc_path),
                "--output",
                str(output_path),
                "--num-chirps",
                "5",
                "--num-rx",
                "1",
                "--num-samples",
                "5",
                "--detector",
                "cfar",
            ]
        )


def test_preprocess_adc_cli_requires_shape_without_preset(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    output_path = tmp_path / "point_cloud.npy"
    np.array([1, 0, 0, 0], dtype=np.int16).tofile(adc_path)

    with pytest.raises(SystemExit, match="required without a preset"):
        main([str(adc_path), "--output", str(output_path), "--threshold", "1.0"])


def test_export_config_cli_writes_ti_and_dca_configs(tmp_path) -> None:
    ti_path = tmp_path / "radar.cfg"
    dca_path = tmp_path / "dca" / "config.json"

    result = export_config_main(
        [
            "--ti-cfg",
            str(ti_path),
            "--dca-json",
            str(dca_path),
            "--num-tx",
            "2",
            "--num-rx",
            "2",
            "--num-adc-samples",
            "32",
            "--num-chirps-per-tx",
            "8",
            "--frame-periodicity-s",
            "0.05",
            "--capture-path",
            r"dataset\radar",
            "--dca-mac",
            "00.11.22.33.44.55",
            "--frames-to-capture",
            "3",
            "--include-sensor-start",
        ]
    )

    assert result == 0
    ti_text = ti_path.read_text(encoding="utf-8")
    assert "channelCfg 3 5 0" in ti_text
    assert "frameCfg 0 1 8 0 50.0 1 0" in ti_text
    assert ti_text.rstrip().endswith("sensorStart")

    dca_config = json.loads(dca_path.read_text(encoding="utf-8"))["DCA1000Config"]
    assert dca_config["captureConfig"]["fileBasePath"] == "dataset/radar"
    assert dca_config["captureConfig"]["framesToCapture"] == 3
    assert dca_config["ethernetConfigUpdate"]["DCA1000MACAddress"] == "00.11.22.33.44.55"
    assert len(dca_config["dataFormatConfig"]["dataPortConfig"]) == 2


def test_export_config_cli_requires_dca_mac(tmp_path) -> None:
    with pytest.raises(SystemExit, match="--dca-mac"):
        export_config_main(
            [
                "--dca-json",
                str(tmp_path / "dca.json"),
                "--preset",
                "iwr6843",
            ]
        )


def test_export_config_cli_uses_iwr6843_preset(tmp_path) -> None:
    ti_path = tmp_path / "radar.cfg"

    result = export_config_main(["--preset", "iwr6843", "--ti-cfg", str(ti_path)])

    assert result == 0
    ti_text = ti_path.read_text(encoding="utf-8")
    assert "channelCfg 15 7 0" in ti_text
    assert "frameCfg 0 2 64 0 100.0 1 0" in ti_text


def test_export_config_cli_preset_allows_profile_overrides(tmp_path) -> None:
    ti_path = tmp_path / "radar.cfg"

    result = export_config_main(
        [
            "--preset",
            "iwr6843",
            "--ti-cfg",
            str(ti_path),
            "--num-tx",
            "2",
            "--num-rx",
            "2",
            "--num-chirps-per-tx",
            "8",
        ]
    )

    assert result == 0
    ti_text = ti_path.read_text(encoding="utf-8")
    assert "channelCfg 3 5 0" in ti_text
    assert "frameCfg 0 1 8 0 100.0 1 0" in ti_text


def test_export_config_cli_requires_an_output() -> None:
    with pytest.raises(SystemExit, match="at least one"):
        export_config_main([])


def test_export_config_cli_rejects_invalid_numeric_args(tmp_path) -> None:
    ti_path = tmp_path / "radar.cfg"

    with pytest.raises(SystemExit):
        export_config_main(
            [
                "--ti-cfg",
                str(ti_path),
                "--num-adc-samples",
                "0",
            ]
        )

    with pytest.raises(SystemExit, match="before ramp"):
        export_config_main(
            [
                "--ti-cfg",
                str(ti_path),
                "--adc-start-time-s",
                "1",
                "--ramp-end-time-s",
                "0.5",
            ]
        )


def test_inspect_adc_file_counts_frames_without_loading(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.arange(10, dtype=np.int16).tofile(adc_path)

    inspection = inspect_adc_file(adc_path)

    assert inspection.size_bytes == 20
    assert inspection.int16_values == 10
    assert inspection.trailing_bytes == 0


def test_inspect_adc_cli_prints_json_frame_counts(tmp_path, capsys) -> None:
    adc_path = tmp_path / "adc.bin"
    np.arange(10, dtype=np.int16).tofile(adc_path)

    result = inspect_main(
        [
            "adc",
            str(adc_path),
            "--num-chirps",
            "1",
            "--num-rx",
            "1",
            "--num-samples",
            "2",
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["int16_values"] == 10
    assert payload["raw_values_per_frame"] == 4
    assert payload["complete_frames"] == 2
    assert payload["leftover_values"] == 2
    assert payload["adc_spec"] == {
        "num_chirps": 1,
        "num_rx": 1,
        "num_samples": 2,
        "layout": "iq_interleaved",
    }
    assert payload["shape_candidates"] == []


def test_inspect_adc_cli_can_use_ti_cfg_shape(tmp_path, capsys) -> None:
    adc_path = tmp_path / "adc.bin"
    cfg_path = tmp_path / "radar.cfg"
    np.zeros(4096 * 3 // 2, dtype=np.int16).tofile(adc_path)
    cfg_path.write_text(
        "\n".join(
            [
                "channelCfg 15 1 0",
                "adcCfg 2 1",
                "profileCfg 0 60 30 7 57.14 0 0 60 1 128 5209 0 0 158",
                "chirpCfg 0 0 0 0 0 0 0 1",
                "frameCfg 0 0 2 0 10 1 0",
            ]
        ),
        encoding="utf-8",
    )

    result = inspect_main(["adc", str(adc_path), "--ti-cfg", str(cfg_path), "--json"])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["raw_values_per_frame"] == 2048
    assert payload["complete_frames"] == 3
    assert payload["leftover_values"] == 0
    assert payload["adc_spec"] == {
        "num_chirps": 2,
        "num_rx": 4,
        "num_samples": 128,
        "layout": "iq_interleaved",
    }


def test_inspect_adc_cli_infers_candidate_shapes(tmp_path, capsys) -> None:
    adc_path = tmp_path / "adc.bin"
    np.zeros(4096 * 3 // 2, dtype=np.int16).tofile(adc_path)

    result = inspect_main(
        [
            "adc",
            str(adc_path),
            "--infer-shapes",
            "--candidate-num-chirps",
            "2",
            "3",
            "--candidate-num-rx",
            "4",
            "--candidate-num-samples",
            "128",
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape_candidates"][0] == {
        "num_chirps": 2,
        "num_rx": 4,
        "num_samples": 128,
        "raw_values_per_frame": 2048,
        "complete_frames": 3,
        "leftover_values": 0,
        "is_complete": True,
    }


def test_inspect_adc_cli_rejects_non_positive_infer_args(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.zeros(4, dtype=np.int16).tofile(adc_path)

    with pytest.raises(SystemExit):
        inspect_main(
            [
                "adc",
                str(adc_path),
                "--infer-shapes",
                "--max-candidates",
                "0",
            ]
        )

    with pytest.raises(SystemExit):
        inspect_main(
            [
                "adc",
                str(adc_path),
                "--infer-shapes",
                "--candidate-num-chirps",
                "0",
            ]
        )


def test_inspect_adc_cli_requires_complete_adc_shape(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.arange(4, dtype=np.int16).tofile(adc_path)

    with pytest.raises(SystemExit, match="must be provided together"):
        inspect_main(["adc", str(adc_path), "--num-chirps", "1"])


def test_inspect_adc_cli_rejects_ti_cfg_with_explicit_shape(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    cfg_path = tmp_path / "radar.cfg"
    np.arange(4, dtype=np.int16).tofile(adc_path)
    cfg_path.write_text(
        "\n".join(
            [
                "channelCfg 1 1 0",
                "profileCfg 0 60 30 7 57.14 0 0 60 1 1 5209 0 0 158",
                "chirpCfg 0 0 0 0 0 0 0 1",
                "frameCfg 0 0 1 0 10 1 0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="cannot be combined"):
        inspect_main(
            [
                "adc",
                str(adc_path),
                "--ti-cfg",
                str(cfg_path),
                "--num-chirps",
                "1",
                "--num-rx",
                "1",
                "--num-samples",
                "1",
            ]
        )


def test_inspect_adc_cli_reports_invalid_ti_cfg(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    cfg_path = tmp_path / "radar.cfg"
    np.arange(4, dtype=np.int16).tofile(adc_path)
    cfg_path.write_text("channelCfg 1 1 0\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="--ti-cfg"):
        inspect_main(["adc", str(adc_path), "--ti-cfg", str(cfg_path)])
