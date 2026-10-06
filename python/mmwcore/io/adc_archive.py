"""Compatibility imports; use mmwcore.io.adc_compression for ADC compression."""

from .adc_compression import (
    ADCCompressionError as ADCArchiveError,
)
from .adc_compression import (
    CompressedADC as ADCArchive,
)
from .adc_compression import (
    compress_adc_file as write_adc_archive,
)
from .adc_compression import (
    open_compressed_adc as open_adc_archive,
)

__all__ = ["ADCArchive", "ADCArchiveError", "write_adc_archive", "open_adc_archive"]
