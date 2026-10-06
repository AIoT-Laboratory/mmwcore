# ADC compression and decompression

mmwcore defines a byte-exact compression format for complete raw ADC frames. Standardization
means one explicit frame/capture contract, bounded restart groups, deterministic compression,
and integrity-checked decompression. It does not rescale amplitudes, reorganize I/Q, alter
sample order, or discard samples. The output restores the original raw bytes exactly.

## File API

```python
from mmwcore.io import (
    compress_adc_file, decompress_adc_file, open_compressed_adc, CompressedADCReader,
)

# capture is the RadarCaptureSpec for the completed raw file.
compressed = compress_adc_file("adc.bin", "radar.mmwa", capture)
raw_window = compressed.decompress_frames(0, 4)  # bytes, interval [0, 4)
restored_capture = decompress_adc_file("radar.mmwa", "restored.bin")

compressed = open_compressed_adc("radar.mmwa")
reader = CompressedADCReader("radar.mmwa")
frame = reader.read_frame(0)  # ADCFrame containing the decompressed int16 samples
```

`compress_adc_file` returns `CompressedADC` and embeds the finalized capture specification.
`decompress_adc_file` restores the entire raw file and returns that `RadarCaptureSpec`.
Both refuse to overwrite an existing destination. Decompression holds one bounded restart
group at a time and checks every chunk plus the complete ADC SHA-256 before publishing output.
A failure leaves no partial destination. Ordinary filesystem failures retain their Python
I/O exception types; invalid compressed data raises `ADCCompressionError`.

`CompressedADC.decompress_windows(starts, window_frames)` restores ordered, equal-sized windows,
sharing chunk decompression for repeated or overlapping requests. `verify_all()` checks the
complete logical stream without creating a raw output file. `compressed_size_bytes` includes
payload, capture metadata, index, and integrity records.

## Frame-group codec

`compress_adc_frames(data, frame_bytes, block_samples=512)` and
`decompress_adc_frames(payload, frame_bytes, frame_count, block_samples=512)` operate on bytes
in memory. Inputs are complete little-endian int16 frames. The result is a codec payload,
without a file header, capture specification, index, or checksums. The caller must preserve
the three decoding parameters. Use the file API when those parameters must travel with data.

The codec uses a restart frame, homologous-frame deltas, adaptive Rice coding, and exact raw
blocks when Rice is not shorter. See the [.mmwa v3 wire specification](adc-archive-format.md).
The existing `.mmwa` extension, version/magic, and historical format identity
`mmwcore.adc_archive.v3` remain unchanged; existing compressed data needs no conversion.

## Code ownership

```text
crates/mmwcore/src/io/adc_compression/
  codec.rs + codec/rice.rs    frame-group compression and decompression
  container/                 .mmwa contract, wire layout, file compression and decompression
python/mmwcore/io/
  adc_compression.py         checked Python file/codec API
  compressed_adc_reader.py   ADCFrame adapter for DSP and datasets
```

Rust is the authoritative codec and file implementation. Take/session packaging stays in
`take.py`; it calls the compression API rather than defining another ADC compressor.

## Compatibility names

The old names are compatibility imports, not a second implementation:

| Previous name | Canonical name |
| --- | --- |
| `write_adc_archive` | `compress_adc_file` |
| `open_adc_archive` | `open_compressed_adc` |
| `ADCArchive` | `CompressedADC` |
| `ADCArchiveError` | `ADCCompressionError` |
| `ADCArchiveReader` | `CompressedADCReader` |
| `ADCArchive.read_frames/read_windows` | `CompressedADC.decompress_frames/decompress_windows` |
| `encode_adc_archive_chunk` / `decode_adc_archive_chunk` in Rust/native | `compress_adc_frames` / `decompress_adc_frames` |

Historical Python modules, native names, Rust imports, and `Take.archive` remain compatible.
New code should use the canonical compression names above.
