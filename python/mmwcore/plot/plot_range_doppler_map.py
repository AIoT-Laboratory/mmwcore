"""Range-Doppler map plotting."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mmwcore.core import RadarCube
from mmwcore.plot._common import MapScale, coordinate_axis, display_values, image_extent


def plot_range_doppler_map(
    data: RadarCube | np.ndarray,
    *,
    range_axis_m: np.ndarray | None = None,
    doppler_axis_mps: np.ndarray | None = None,
    scale: MapScale = "db",
    title: str | None = None,
) -> Any:
    """Plot a `(range, doppler)` map, aggregating other named cube axes."""

    values = _range_doppler_values(data)
    ranges = coordinate_axis(range_axis_m, values.shape[0], name="range_axis_m")
    dopplers = coordinate_axis(
        doppler_axis_mps,
        values.shape[1],
        name="doppler_axis_mps",
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(
        display_values(values, scale),
        origin="lower",
        aspect="auto",
        extent=image_extent(dopplers, ranges),
        cmap="viridis",
    )
    axis.set_xlabel("radial velocity (m/s)" if doppler_axis_mps is not None else "doppler bin")
    axis.set_ylabel("range (m)" if range_axis_m is not None else "range bin")
    if title is not None:
        axis.set_title(title)
    figure.colorbar(image, ax=axis, label=scale)
    figure.tight_layout()
    return figure


def _range_doppler_values(data: RadarCube | np.ndarray) -> np.ndarray:
    if not isinstance(data, RadarCube):
        values = np.asarray(data)
        if values.ndim != 2:
            raise ValueError(f"range-Doppler data must have shape (R, D); got {values.shape}.")
        return values
    try:
        range_index = data.axes.index("range")
        doppler_index = data.axes.index("doppler")
    except ValueError:
        raise ValueError("RadarCube must contain 'range' and 'doppler' axes.") from None
    moved = np.moveaxis(data.data, (range_index, doppler_index), (-2, -1))
    aggregate_axes = tuple(range(moved.ndim - 2))
    return np.abs(moved).mean(axis=aggregate_axes) if aggregate_axes else moved
