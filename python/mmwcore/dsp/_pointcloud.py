"""Native calibrated detection-to-point-cloud projection boundary."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import PointCloudProjectionSpec


def project_detection_point_cloud(
    detections: NDArray[np.float32],
    *,
    range_bin_column: int,
    doppler_bin_column: int,
    magnitude_column: int,
    azimuth_bin_column: int,
    azimuth_rad_column: int,
    elevation_columns: tuple[int, int] | None,
    passthrough_columns: tuple[int, ...],
    spec: PointCloudProjectionSpec,
) -> NDArray[np.float32]:
    """Project one calibrated detection matrix through the native core."""

    return _native.project_detection_point_cloud(
        np.ascontiguousarray(detections, dtype=np.float32),
        (
            range_bin_column,
            doppler_bin_column,
            magnitude_column,
            azimuth_bin_column,
            azimuth_rad_column,
            elevation_columns,
            list(passthrough_columns),
        ),
        (
            spec.range_resolution_m,
            spec.doppler_resolution_mps,
            spec.center_doppler,
            spec.doppler_bins,
            spec.doppler_fftshifted,
        ),
    )
