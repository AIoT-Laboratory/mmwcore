"""Cartesian point-cloud plotting."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mmwcore.core import PointCloudFrame


def plot_point_cloud(
    point_cloud: PointCloudFrame | np.ndarray,
    *,
    color_channel: str | int | None = None,
    title: str | None = None,
) -> Any:
    """Plot a Cartesian point cloud with a viridis scalar channel."""

    if isinstance(point_cloud, PointCloudFrame):
        points = point_cloud.points
        channels = point_cloud.channels
    else:
        points = np.asarray(point_cloud, dtype=np.float32)
        channels = ()
        if points.ndim == 2:
            extra_channels = tuple(f"channel_{index}" for index in range(3, points.shape[1]))
            channels = ("x", "y", "z", *extra_channels)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"point_cloud must have shape (N, C>=3); got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("point_cloud contains NaN or Inf values.")

    color_index = _color_index(color_channel, channels, points.shape[1])
    color_label = channels[color_index] if channels else f"channel_{color_index}"
    figure = plt.figure(figsize=(7, 6))
    axis: Any = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=points[:, color_index],
        cmap="viridis",
        s=10,
        alpha=0.75,
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    if title is not None:
        axis.set_title(title)
    figure.colorbar(scatter, ax=axis, label=color_label, pad=0.1)
    figure.tight_layout()
    return figure


def _color_index(
    color_channel: str | int | None,
    channels: tuple[str, ...],
    num_channels: int,
) -> int:
    if color_channel is None:
        return 2
    if isinstance(color_channel, str):
        try:
            return channels.index(color_channel)
        except ValueError:
            raise ValueError(f"Unknown point-cloud color channel: {color_channel!r}.") from None
    if color_channel < 0 or color_channel >= num_channels:
        raise ValueError(f"color_channel index must be in [0, {num_channels}).")
    return color_channel
