# Runnable examples

All examples read caller-supplied completed files. They do not open hardware or start mmwcli.

| Use | Command |
| --- | --- |
| Create an ADC archive | `python examples/adc_archive.py adc.bin radar.mmwa --capture-spec capture.json` |
| Read archive windows | `python examples/adc_archive_windows.py radar.mmwa 100 104 --window-frames 4` |

Use `ADCArchiveReader` directly in datasets and inference code. Keep hardware acquisition in
mmwcli and model-specific preprocessing in OpenMMW.
