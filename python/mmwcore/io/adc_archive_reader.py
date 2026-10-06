"""Compatibility import; use CompressedADCReader for decompressed ADC frames."""

from .compressed_adc_reader import CompressedADCReader as ADCArchiveReader

__all__ = ["ADCArchiveReader"]
