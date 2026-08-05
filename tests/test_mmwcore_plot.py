from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from mmwcore.core import PointCloudFrame, RadarCube
from mmwcore.plot import plot_point_cloud, plot_range_doppler_map, plot_range_time_map

matplotlib.use("Agg")


def test_plot_point_cloud_uses_selected_channel_and_viridis() -> None:
    frame = PointCloudFrame(
        np.array(
            [[0.0, 1.0, 0.0, -0.2], [0.5, 1.5, 0.1, 0.3]],
            dtype=np.float32,
        ),
        channels=("x", "y", "z", "velocity"),
    )

    figure = plot_point_cloud(frame, color_channel="velocity")

    assert figure.axes[0].collections[0].get_cmap().name == "viridis"
    assert figure.axes[1].get_ylabel() == "velocity"


def test_plot_range_doppler_map_aggregates_named_cube_axes() -> None:
    data = np.ones((2, 3, 4), dtype=np.complex64)
    data[:, 1, 2] = 10.0
    cube = RadarCube(data, axes=("rx", "doppler", "range"))

    figure = plot_range_doppler_map(
        cube,
        range_axis_m=np.array([0.0, 0.5, 1.0, 1.5]),
        doppler_axis_mps=np.array([-1.0, 0.0, 1.0]),
        scale="magnitude",
    )
    image = figure.axes[0].images[0]

    assert image.get_cmap().name == "viridis"
    assert image.get_array().shape == (4, 3)
    assert image.get_array()[2, 1] == pytest.approx(10.0)
    assert figure.axes[0].get_xlabel() == "radial velocity (m/s)"


def test_plot_range_time_map_accepts_frame_axis_and_db_scale() -> None:
    cube = RadarCube(
        np.array([[[1.0, 10.0]], [[10.0, 100.0]]], dtype=np.complex64),
        axes=("frame", "rx", "range"),
    )

    figure = plot_range_time_map(
        cube,
        range_axis_m=np.array([0.0, 0.5]),
        time_axis_s=np.array([0.0, 0.1]),
        scale="db",
    )
    displayed = figure.axes[0].images[0].get_array()

    np.testing.assert_allclose(displayed, [[0.0, 20.0], [20.0, 40.0]])
    assert figure.axes[0].get_xlabel() == "time (s)"
    assert figure.axes[0].images[0].get_cmap().name == "viridis"


def test_radar_map_plots_reject_invalid_shapes_and_axes() -> None:
    with pytest.raises(ValueError, match="shape"):
        plot_range_doppler_map(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="strictly increasing"):
        plot_range_time_map(
            np.zeros((2, 2)),
            time_axis_s=np.array([0.1, 0.0]),
        )
    with pytest.raises(ValueError, match="range.*doppler"):
        plot_range_doppler_map(RadarCube(np.ones((2, 2)), axes=("frame", "range")))
