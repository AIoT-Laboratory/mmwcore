"""Cell-averaging CFAR helpers for offline mmwcore processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmwcore.core import (
    CFAR1DSpec,
    CFARDetectionSpec,
    DetectionFrame,
    RadarCube,
    RangeDopplerCFARSpec,
)
from mmwcore.dsp._cfar import (
    detect_cfar_1d as native_detect_cfar_1d,
)
from mmwcore.dsp._cfar import (
    detect_cfar_2d as native_detect_cfar_2d,
)
from mmwcore.dsp._cfar import (
    detect_range_doppler_cfar as native_detect_range_doppler_cfar,
)
from mmwcore.dsp.detection import _detections_from_hits, _range_doppler_axes


@dataclass(frozen=True)
class CFAR1DResult:
    """Detected CUT indices and their noise estimates."""

    indices: np.ndarray
    noise: np.ndarray


def detect_cfar_1d(power: np.ndarray, spec: CFAR1DSpec) -> CFAR1DResult:
    """Apply CA/GO/SO/CACC CFAR to a one-dimensional power array."""

    values = np.asarray(power, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"CFAR input must be one-dimensional; got {values.shape}.")
    indices, noise = native_detect_cfar_1d(values, spec)
    return CFAR1DResult(indices, noise)


def detect_range_doppler_cfar(
    cube: RadarCube,
    spec: RangeDopplerCFARSpec,
) -> DetectionFrame:
    """Detect bins by composing floating-point range and Doppler CFAR passes."""

    frame_axis, doppler_axis, receiver_axis, range_axis = _range_doppler_axes(cube)
    hit_indices, magnitudes, noise, snr = native_detect_range_doppler_cfar(
        cube.data,
        frame_axis=frame_axis,
        doppler_axis=doppler_axis,
        receiver_axis=receiver_axis,
        range_axis=range_axis,
        aggregate_rx=spec.aggregate_rx,
        range_spec=spec.range,
        doppler_spec=spec.doppler,
        input_scale=spec.input_scale,
    )
    return _detections_from_hits(
        hit_indices,
        magnitudes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "range_doppler_cfar": {
                "range": _cfar_spec_metadata(spec.range),
                "doppler": (
                    _cfar_spec_metadata(spec.doppler) if spec.doppler is not None else None
                ),
                "input_scale": spec.input_scale.value,
                "aggregate_rx": spec.aggregate_rx,
                "output_candidates": int(hit_indices.shape[0]),
            },
        },
        extra_channels=("noise", "snr"),
        extra_values=np.column_stack((noise, snr)).astype(np.float32, copy=False),
        extra_units={"noise": spec.input_scale.value, "snr": "linear_ratio"},
    )


def _cfar_spec_metadata(spec: CFAR1DSpec) -> dict[str, object]:
    return {
        "training_cells": spec.training_cells,
        "guard_cells": spec.guard_cells,
        "threshold_scale": spec.threshold_scale,
        "mode": spec.mode.value,
        "cyclic": spec.cyclic,
        "left_skip": spec.left_skip,
        "right_skip": spec.right_skip,
    }


def detect_cfar(cube: RadarCube, spec: CFARDetectionSpec) -> DetectionFrame:
    """Detect range-Doppler bins with symmetric cell-averaging CFAR.

    The input cube is interpreted as `(frame, doppler_bin, rx, range_bin)` after
    axis normalization. Edges that cannot fit the full training and guard window
    are skipped.
    """

    frame_axis, doppler_axis, receiver_axis, range_axis = _range_doppler_axes(cube)
    hit_indices, magnitudes, noise, snr = native_detect_cfar_2d(
        cube.data,
        frame_axis=frame_axis,
        doppler_axis=doppler_axis,
        receiver_axis=receiver_axis,
        range_axis=range_axis,
        aggregate_rx=spec.aggregate_rx,
        training_cells=spec.training_cells,
        guard_cells=spec.guard_cells,
        threshold_scale=spec.threshold_scale,
    )
    return _detections_from_hits(
        hit_indices,
        magnitudes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "cfar_detection": {
                "training_cells": spec.training_cells,
                "guard_cells": spec.guard_cells,
                "threshold_scale": spec.threshold_scale,
                "aggregate_rx": spec.aggregate_rx,
                "output_candidates": int(hit_indices.shape[0]),
            },
        },
        extra_channels=("noise", "snr"),
        extra_values=np.column_stack((noise, snr)).astype(np.float32, copy=False),
        extra_units={"noise": "magnitude", "snr": "linear_ratio"},
    )
