"""Native CFAR boundary for canonical range-Doppler radar cubes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import CFAR1DSpec, CFARInputScale, CFARMode

type NativeCfar1DResult = tuple[NDArray[np.int64], NDArray[np.float32]]
type NativeCfarDetections = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]


def detect_cfar_1d(power: NDArray[np.float32], spec: CFAR1DSpec) -> NativeCfar1DResult:
    """Return native one-dimensional CFAR hits and noise estimates."""

    return _native.detect_cfar_1d(
        np.ascontiguousarray(power, dtype=np.float32),
        _cfar_1d_config(spec),
    )


def detect_range_doppler_cfar(
    data: NDArray[np.complex64],
    *,
    frame_axis: int,
    doppler_axis: int,
    receiver_axis: int,
    range_axis: int,
    aggregate_rx: str,
    range_spec: CFAR1DSpec,
    doppler_spec: CFAR1DSpec | None,
    input_scale: CFARInputScale,
) -> NativeCfarDetections:
    """Return native composed range-Doppler CFAR candidates."""

    return _native.detect_range_doppler_cfar_complex(
        np.ascontiguousarray(data, dtype=np.complex64),
        (frame_axis, doppler_axis, receiver_axis, range_axis),
        _aggregation_code(aggregate_rx),
        _cfar_1d_config(range_spec),
        _cfar_1d_config(doppler_spec) if doppler_spec is not None else None,
        _input_scale_code(input_scale),
    )


def detect_cfar_2d(
    data: NDArray[np.complex64],
    *,
    frame_axis: int,
    doppler_axis: int,
    receiver_axis: int,
    range_axis: int,
    aggregate_rx: str,
    training_cells: int,
    guard_cells: int,
    threshold_scale: float,
) -> NativeCfarDetections:
    """Return native symmetric two-dimensional CFAR candidates."""

    return _native.detect_cfar_2d_complex(
        np.ascontiguousarray(data, dtype=np.complex64),
        (frame_axis, doppler_axis, receiver_axis, range_axis),
        _aggregation_code(aggregate_rx),
        (training_cells, guard_cells, threshold_scale),
    )


def _cfar_1d_config(spec: CFAR1DSpec) -> tuple[int, int, float, int, bool, int, int]:
    return (
        spec.training_cells,
        spec.guard_cells,
        spec.threshold_scale,
        _mode_code(spec.mode),
        spec.cyclic,
        spec.left_skip,
        spec.right_skip,
    )


def _mode_code(mode: CFARMode) -> int:
    if mode is CFARMode.CA:
        return 0
    if mode is CFARMode.GO:
        return 1
    if mode is CFARMode.SO:
        return 2
    if mode is CFARMode.CACC:
        return 3
    raise ValueError(f"Unsupported CFAR mode: {mode}")


def _input_scale_code(input_scale: CFARInputScale) -> int:
    if input_scale is CFARInputScale.MAGNITUDE:
        return 0
    if input_scale is CFARInputScale.POWER:
        return 1
    raise ValueError(f"Unsupported CFAR input scale: {input_scale}")


def _aggregation_code(aggregate_rx: str) -> int:
    if aggregate_rx == "max":
        return 0
    if aggregate_rx == "sum":
        return 1
    if aggregate_rx == "mean":
        return 2
    raise ValueError(f"Unsupported RX aggregation: {aggregate_rx}")
