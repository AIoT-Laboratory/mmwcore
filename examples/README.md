# Runnable examples

These examples consume caller-supplied files or binary stdin. They do not open hardware, launch
mmwcli, or restore mmwcore.session.

| Use | Command |
| --- | --- |
| Strict capture directory | python examples/capture_or_raw.py capture CAPTURE |
| Headerless ADC frames | python examples/capture_or_raw.py raw adc.bin --chirps 32 --rx 4 --samples 256 --layout group2_i_then_q |
| Archive completed ADC | python examples/adc_archive.py adc.bin adc.mmwa --capture-spec capture.json |
| Read ordered archive windows | python examples/adc_archive_windows.py adc.mmwa 100 104 100 --window-frames 4 |
| Self-describing archive reader | use `ADCArchiveFrameReader("adc.mmwa")` |
| Explicit XWR1843 EVM recipe | python examples/xwr18_range_doppler.py XWR18_CAPTURE |
| Radar live stream | mmwcli ... capture ... --stream \| python examples/radar_live_stream.py |
| Offline multi-sensor training | python examples/multisensor_offline_training.py SESSION |
| Multi-sensor live inference | mmwcli ... capture ... --multisensor-plan PLAN --stream \| python examples/multisensor_live_inference.py |

capture_or_raw.py validates either a published mmwcli capture directory or an explicit raw-file
contract. xwr18_range_doppler.py deliberately selects the standard XWR1843 EVM geometry and must
only be used when it matches the actual board.

Live results remain provisional until COMMIT and physical EOF. The aggregate example uses
commit.accepts to discard items from failed or omitted optional sources. Offline pairs use
conservative mapped-time intervals, never equal indices or arrival order.
