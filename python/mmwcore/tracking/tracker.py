"""Native cluster-level two-dimensional tracking adapter."""

from __future__ import annotations

import numpy as np

from mmwcore import _native
from mmwcore.core import ClusterFrame, Tracker2DSpec, TrackFrame
from mmwcore.tracking._native_adapter import native_tracker_config, track_frame


class ClusterTracker2D:
    """Advance one Rust-owned constant-velocity cluster tracker per frame."""

    def __init__(self, spec: Tracker2DSpec) -> None:
        if spec.allocation.min_total_snr is not None:
            raise ValueError(
                "ClusterTracker2D cannot use TrackAllocationSpec.min_total_snr; "
                "use MeasurementTracker2D with an SNR channel."
            )
        self.spec = spec
        self._tracker = _native.NativeClusterTracker2D(native_tracker_config(spec))

    def step(self, clusters: ClusterFrame) -> TrackFrame:
        """Advance state once and assemble the typed Python report."""

        result = self._tracker.step(
            np.ascontiguousarray(clusters.centers, dtype=np.float32),
            np.ascontiguousarray(clusters.extents, dtype=np.float32),
            np.ascontiguousarray(clusters.mean_velocities, dtype=np.float32),
            np.ascontiguousarray(clusters.point_counts, dtype=np.int64),
        )
        return track_frame(
            result,
            frame_id=clusters.frame_id,
            timestamp=clusters.timestamp,
            source=clusters.source,
            coordinate_frame=clusters.coordinate_frame,
            metadata=clusters.metadata,
            model="constant_velocity_2d_cluster",
        )
