"""Native detection post-processing boundaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.dsp._detection import _aggregation_code


def group_range_doppler_candidates(
    data: NDArray[np.complex64],
    *,
    axes: tuple[int, int, int, int],
    aggregate_rx: str,
    candidates: NDArray[np.float32],
    columns: tuple[int, int, int],
    range_radius: int,
    doppler_radius: int,
    cyclic_doppler: bool,
    strict: bool,
) -> NDArray[np.int64]:
    """Return candidate rows that satisfy the native local-peak policy."""

    return _native.group_range_doppler_candidates(
        np.ascontiguousarray(data, dtype=np.complex64),
        axes,
        _aggregation_code(aggregate_rx),
        np.ascontiguousarray(candidates, dtype=np.float32),
        columns,
        (range_radius, doppler_radius, cyclic_doppler, strict),
    )


def quality_filter_indices(
    candidates: NDArray[np.float32],
    *,
    snr_column: int,
    min_snr: float,
) -> NDArray[np.int64]:
    """Return candidate rows meeting the native inclusive SNR threshold."""

    return _native.filter_detection_quality_rows(
        np.ascontiguousarray(candidates, dtype=np.float32),
        snr_column,
        min_snr,
    )
