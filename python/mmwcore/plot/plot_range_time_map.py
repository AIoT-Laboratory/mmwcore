"""Range-time map plotting."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mmwcore.core import RadarCube
from mmwcore.plot._common import MapScale, coordinate_axis, display_values, image_extent


def plot_range_time_map(
    data: RadarCube | np.ndarray,
    *,
    range_axis_m: np.ndarray | None = None,
    time_axis_s: np.ndarray | None = None,
    scale: MapScale = "db",
    title: str | None = None,
) -> Any:
    """Plot a `(time, range)` sequence, aggregating other named cube axes."""

    values = _range_time_values(data)
    times = coordinate_axis(time_axis_s, values.shape[0], name="time_axis_s")
    ranges = coordinate_axis(range_axis_m, values.shape[1], name="range_axis_m")
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(
        display_values(values, scale).T,
        origin="lower",
        aspect="auto",
        extent=image_extent(times, ranges),
        cmap="viridis",
    )
    axis.set_xlabel("time (s)" if time_axis_s is not None else "time bin")
    axis.set_ylabel("range (m)" if range_axis_m is not None else "range bin")
    if title is not None:
        axis.set_title(title)
    figure.colorbar(image, ax=axis, label=scale)
    figure.tight_layout()
    return figure


def _range_time_values(data: RadarCube | np.ndarray) -> np.ndarray:
    if not isinstance(data, RadarCube):
        values = np.asarray(data)
        if values.ndim != 2:
            raise ValueError(f"range-time data must have shape (T, R); got {values.shape}.")
        return values
    try:
        time_index = data.axes.index("time")
    except ValueError:
        try:
            time_index = data.axes.index("frame")
        except ValueError:
            raise ValueError("RadarCube must contain a 'time' or 'frame' axis.") from None
    try:
        range_index = data.axes.index("range")
    except ValueError:
        raise ValueError("RadarCube must contain a 'range' axis.") from None
    moved = np.moveaxis(data.data, (time_index, range_index), (-2, -1))
    aggregate_axes = tuple(range(moved.ndim - 2))
    return np.abs(moved).mean(axis=aggregate_axes) if aggregate_axes else moved
