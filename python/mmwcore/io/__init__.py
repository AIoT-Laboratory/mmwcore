"""Finite ADC and take storage."""

from __future__ import annotations

from .adc_archive import ADCArchive, ADCArchiveError, open_adc_archive, write_adc_archive
from .adc_archive_reader import ADCArchiveReader
from .adc_file import ADCFileReader, load_adc_cube, load_adc_file
from .adc_reader import ADCReader
from .capture import Capture, SceneROI, SetupSnapshot, read_capture
from .take import (
    TAKE_SCHEMA,
    Camera,
    CameraFrame,
    HostTimeRange,
    SampleKey,
    Take,
    open_take,
    write_take,
)

__all__ = [
    "ADCArchive",
    "ADCArchiveError",
    "ADCArchiveReader",
    "ADCFileReader",
    "ADCReader",
    "Camera",
    "CameraFrame",
    "Capture",
    "HostTimeRange",
    "SampleKey",
    "SceneROI",
    "SetupSnapshot",
    "TAKE_SCHEMA",
    "Take",
    "load_adc_cube",
    "load_adc_file",
    "open_adc_archive",
    "open_take",
    "read_capture",
    "write_adc_archive",
    "write_take",
]
