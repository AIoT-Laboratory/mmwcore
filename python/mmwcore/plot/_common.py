"""Shared validation and display scaling for radar maps."""

from __future__ import annotations

from typing import Literal

import numpy as np

MapScale = Literal["magnitude", "power", "db"]


def display_values(values: np.ndarray, scale: MapScale) -> np.ndarray:
    magnitude = np.abs(values).astype(np.float64, copy=False)
    if scale == "magnitude":
        return magnitude
    if scale == "power":
        return np.square(magnitude)
    if scale == "db":
        floor = np.finfo(np.float64).tiny
        return 20.0 * np.log10(np.maximum(magnitude, floor))
    raise ValueError("scale must be 'magnitude', 'power', or 'db'.")


def coordinate_axis(
    values: np.ndarray | None,
    length: int,
    *,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.arange(length, dtype=np.float64)
    axis = np.asarray(values, dtype=np.float64)
    if axis.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},).")
    if not np.isfinite(axis).all() or (length > 1 and np.any(np.diff(axis) <= 0)):
        raise ValueError(f"{name} must contain finite, strictly increasing values.")
    return axis


def image_extent(x_axis: np.ndarray, y_axis: np.ndarray) -> tuple[float, float, float, float]:
    return (
        *_axis_edges(x_axis),
        *_axis_edges(y_axis),
    )


def _axis_edges(axis: np.ndarray) -> tuple[float, float]:
    if axis.size == 1:
        return float(axis[0] - 0.5), float(axis[0] + 0.5)
    return (
        float(axis[0] - (axis[1] - axis[0]) / 2.0),
        float(axis[-1] + (axis[-1] - axis[-2]) / 2.0),
    )
