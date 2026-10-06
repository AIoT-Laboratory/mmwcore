# Runnable examples

All examples read caller-supplied completed files. They do not open hardware or start mmwcli.

| Use | Command |
| --- | --- |
| Compress raw ADC | `python examples/compress_adc.py adc.bin radar.mmwa --capture-spec capture.json` |
| Decompress raw ADC | `python examples/decompress_adc.py radar.mmwa restored.bin` |
| Decompress ADC windows | `python examples/decompress_adc_windows.py radar.mmwa 100 104 --window-frames 4` |

Use `CompressedADCReader` directly in datasets and inference code. The two older script filenames
remain runnable for compatibility; they use the explicit compression API. See
[ADC compression](../docs/adc-compression.md). Keep hardware acquisition in
mmwcli and model-specific preprocessing in OpenMMW.
