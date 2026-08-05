"""Native threshold-detection boundary for range-Doppler radar cubes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native

type NativeThresholdDetections = tuple[NDArray[np.int64], NDArray[np.float32]]


def range_doppler_magnitude(
    data: NDArray[np.complex64],
    *,
    frame_axis: int,
    doppler_axis: int,
    receiver_axis: int,
    range_axis: int,
    aggregate_rx: str,
) -> NDArray[np.float32]:
    """Aggregate complex RX channels into a canonical F-D-R magnitude map."""

    return _native.range_doppler_magnitude_complex(
        np.ascontiguousarray(data, dtype=np.complex64),
        (frame_axis, doppler_axis, receiver_axis, range_axis),
        _aggregation_code(aggregate_rx),
    )


def threshold_range_doppler(
    data: NDArray[np.complex64],
    *,
    frame_axis: int,
    doppler_axis: int,
    receiver_axis: int,
    range_axis: int,
    aggregate_rx: str,
    threshold: float,
) -> NativeThresholdDetections:
    """Return canonical F-D-R threshold detections from a complex radar cube."""

    return _native.threshold_range_doppler_complex(
        np.ascontiguousarray(data, dtype=np.complex64),
        (frame_axis, doppler_axis, receiver_axis, range_axis),
        _aggregation_code(aggregate_rx),
        threshold,
    )


def threshold_range_doppler_azimuth(
    data: NDArray[np.complex64],
    *,
    frame_axis: int,
    doppler_axis: int,
    azimuth_axis: int,
    range_axis: int,
    threshold: float,
    azimuth_peak_radius: int,
    azimuth_peak_strict: bool,
) -> NativeThresholdDetections:
    """Return canonical F-D-A-R threshold detections with azimuth peak selection."""

    return _native.threshold_range_doppler_azimuth_complex(
        np.ascontiguousarray(data, dtype=np.complex64),
        (frame_axis, doppler_axis, azimuth_axis, range_axis),
        threshold,
        azimuth_peak_radius,
        azimuth_peak_strict,
    )


def _aggregation_code(aggregate_rx: str) -> int:
    if aggregate_rx == "max":
        return 0
    if aggregate_rx == "sum":
        return 1
    if aggregate_rx == "mean":
        return 2
    raise ValueError(f"Unsupported RX aggregation: {aggregate_rx}")
