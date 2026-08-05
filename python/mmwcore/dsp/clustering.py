"""Point-cloud clustering for radar tracking inputs."""

from __future__ import annotations

from mmwcore.core import ClusterFrame, DBSCANClusteringSpec, PointCloudFrame
from mmwcore.dsp._clustering import cluster_points as native_cluster_points


def cluster_point_cloud(
    point_cloud: PointCloudFrame,
    spec: DBSCANClusteringSpec,
) -> ClusterFrame:
    """Cluster Cartesian radar points with spatial and radial-velocity distance."""

    velocity_index = _velocity_index(point_cloud, required=spec.velocity_scale_s > 0)
    labels, centers, extents, mean_velocities, point_counts = native_cluster_points(
        point_cloud.points,
        velocity_index=velocity_index,
        spec=spec,
    )

    return ClusterFrame(
        centers=centers,
        extents=extents,
        mean_velocities=mean_velocities,
        point_counts=point_counts,
        point_labels=labels,
        frame_id=point_cloud.frame_id,
        timestamp=point_cloud.timestamp,
        source=point_cloud.source,
        coordinate_frame=point_cloud.coordinate_frame,
        metadata={
            **point_cloud.metadata,
            "dbscan": {
                "eps_m": spec.eps_m,
                "min_samples": spec.min_samples,
                "velocity_scale_s": spec.velocity_scale_s,
                "use_z": spec.use_z,
                "input_points": int(point_cloud.num_points),
                "output_clusters": int(centers.shape[0]),
                "noise_points": int((labels < 0).sum()),
            },
        },
    )


def _velocity_index(point_cloud: PointCloudFrame, *, required: bool) -> int | None:
    try:
        return point_cloud.channels.index("velocity")
    except ValueError:
        if required:
            raise ValueError(
                'PointCloudFrame must include a "velocity" channel when velocity_scale_s > 0.'
            ) from None
        return None
