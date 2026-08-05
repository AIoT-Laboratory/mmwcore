"""Static-clutter suppression for named-axis radar cubes."""

from __future__ import annotations

import numpy as np

from mmwcore import _native
from mmwcore.core import RadarCube


def remove_static_clutter(cube: RadarCube, *, axis: str = "chirp") -> RadarCube:
    """Subtract the coherent mean along a named slow-time axis."""

    try:
        axis_index = cube.axes.index(axis)
    except ValueError as exc:
        raise ValueError(f"RadarCube axes must include {axis!r}; got {cube.axes}.") from exc

    data = np.ascontiguousarray(cube.data, dtype=np.complex64)
    filtered_data = _native.remove_static_clutter_complex(data, axis_index)
    return RadarCube(
        filtered_data,
        axes=cube.axes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "static_clutter_removal": {"axis": axis},
        },
    )
