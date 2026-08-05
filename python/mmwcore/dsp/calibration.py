"""Virtual-channel calibration primitives."""

from __future__ import annotations

import numpy as np

from mmwcore import _native
from mmwcore.core import RadarCube, TimeDomainChannelCalibration, VirtualChannelCalibration


def apply_time_domain_channel_calibration(
    cube: RadarCube,
    calibration: TimeDomainChannelCalibration,
) -> RadarCube:
    """Apply per-Tx/Rx phase ramps and complex corrections on ADC samples."""

    try:
        tx_axis = cube.axes.index("tx")
        rx_axis = cube.axes.index("rx")
        sample_axis = cube.axes.index("sample")
    except ValueError as exc:
        raise ValueError(
            f'RadarCube axes must include "tx", "rx", and "sample"; got {cube.axes}.'
        ) from exc
    num_tx = cube.data.shape[tx_axis]
    num_rx = cube.data.shape[rx_axis]
    if (num_tx, num_rx) != (calibration.num_tx, calibration.num_rx):
        raise ValueError(
            "Calibration shape must match tx/rx axes; "
            f"got {(calibration.num_tx, calibration.num_rx)} for {(num_tx, num_rx)}."
        )

    corrected = _native.apply_time_domain_channel_calibration_complex(
        np.ascontiguousarray(cube.data, dtype=np.complex64),
        tx_axis,
        rx_axis,
        sample_axis,
        np.ascontiguousarray(
            np.asarray(calibration.frequency_rad_per_sample, dtype=np.float32).reshape(-1)
        ),
        np.ascontiguousarray(
            np.asarray(calibration.complex_corrections, dtype=np.complex64).reshape(-1)
        ),
    )
    return RadarCube(
        corrected,
        axes=cube.axes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "calibration_applied": True,
            "time_domain_channel_calibration": calibration.as_metadata(),
        },
    )


def apply_virtual_channel_calibration(
    cube: RadarCube,
    calibration: VirtualChannelCalibration,
) -> RadarCube:
    """Apply ordered complex correction coefficients on the virtual-RX axis."""

    try:
        virtual_axis = cube.axes.index("virtual_rx")
    except ValueError as exc:
        raise ValueError(f'RadarCube axes must include "virtual_rx"; got {cube.axes}.') from exc
    num_virtual = cube.data.shape[virtual_axis]
    if num_virtual != calibration.num_channels:
        raise ValueError(
            "Calibration channel count must match virtual_rx axis; "
            f"got {calibration.num_channels} coefficients for {num_virtual} channels."
        )

    coefficients = np.ascontiguousarray(np.asarray(calibration.coefficients, dtype=np.complex64))
    return RadarCube(
        _native.apply_virtual_channel_calibration_complex(
            np.ascontiguousarray(cube.data, dtype=np.complex64),
            virtual_axis,
            coefficients,
        ),
        axes=cube.axes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "virtual_channel_calibration": calibration.as_metadata(),
        },
    )
