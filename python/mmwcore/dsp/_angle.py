"""Native uniform-linear-array angle calibration boundary."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import VirtualAntennaLayout

_ANGLE_AXIS_CODES = {"azimuth": 0, "elevation": 1}


def calibrate_angle_bins(
    num_bins: int,
    layout: VirtualAntennaLayout,
    *,
    fftshift: bool,
) -> NDArray[np.float32]:
    """Calibrate one physical angle axis through the native core."""

    return _native.calibrate_angle_bins(
        np.ascontiguousarray(layout.positions_wavelengths, dtype=np.float32),
        num_bins,
        _ANGLE_AXIS_CODES[layout.angle_axis],
        fftshift,
    )
