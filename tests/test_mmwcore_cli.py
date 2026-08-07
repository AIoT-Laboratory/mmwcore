from __future__ import annotations

import json

import numpy as np
import pytest

from mmwcore.cli.inspect import inspect_adc_file
from mmwcore.cli.inspect import main as inspect_main


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
