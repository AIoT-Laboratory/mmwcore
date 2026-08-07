"""Native clustering boundary for Cartesian radar point clouds."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import DBSCANClusteringSpec

type NativeClusterResult = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int64],
]


def cluster_points(
    points: NDArray[np.float32],
    *,
    velocity_index: int | None,
    spec: DBSCANClusteringSpec,
) -> NativeClusterResult:
    """Return native DBSCAN labels and per-cluster Cartesian summaries."""

    return _native.cluster_points(
        np.ascontiguousarray(points, dtype=np.float32),
        (0, 1, 2, velocity_index),
        (spec.eps_m, spec.min_samples, spec.velocity_scale_s, spec.use_z),
    )
