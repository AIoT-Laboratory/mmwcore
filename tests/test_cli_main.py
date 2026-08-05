from __future__ import annotations

from mmwcore.cli import main as main_module


def test_mmwcore_help_exposes_low_level_command_groups(capsys) -> None:
    result = main_module.main(["--help"])

    assert result == 0
    output = capsys.readouterr().out
    assert "inspect" in output
    assert "preprocess-adc" in output
    assert "export-config" in output


def test_mmwcore_rejects_unknown_command(capsys) -> None:
    result = main_module.main(["unknown-command"])

    assert result == 2
    assert "unknown command 'unknown-command'" in capsys.readouterr().err
