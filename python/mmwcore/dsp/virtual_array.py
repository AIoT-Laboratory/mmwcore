"""TDM-MIMO virtual-array construction primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.core import (
    PlanarApertureLayout,
    RadarCube,
    TDMVirtualArraySpec,
    VirtualSubarraySpec,
)


def map_tdm_virtual_array(cube: RadarCube, spec: TDMVirtualArraySpec) -> RadarCube:
    """Map interleaved TDM chirps and receivers to loop and virtual-RX axes."""

    try:
        chirp_axis = cube.axes.index("chirp")
        rx_axis = cube.axes.index("rx")
    except ValueError as exc:
        raise ValueError(f'RadarCube axes must include "chirp" and "rx"; got {cube.axes}.') from exc

    num_chirps = cube.data.shape[chirp_axis]
    num_rx = cube.data.shape[rx_axis]
    if num_rx != spec.geometry.num_rx:
        raise ValueError(
            "RadarCube rx axis must match antenna geometry; "
            f"got {num_rx} samples for {spec.geometry.num_rx} receivers."
        )
    if num_chirps % spec.num_tx:
        raise ValueError(
            "RadarCube chirp count must contain complete TDM loops; "
            f"got {num_chirps} chirps for {spec.num_tx} transmitters."
        )

    loops = num_chirps // spec.num_tx
    mapped = _native.map_tdm_virtual_array_complex(
        _contiguous_cube_data(cube),
        chirp_axis,
        rx_axis,
        spec.num_tx,
    )

    axes = list(cube.axes)
    axes[chirp_axis] = "loop"
    axes[rx_axis] = "virtual_rx"
    return RadarCube(
        mapped,
        axes=tuple(axes),
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "tdm_virtual_array": {
                "tx_order": list(spec.tx_order),
                "num_loops": loops,
                "layout": spec.virtual_layout().as_metadata(),
            },
        },
    )


def compensate_tdm_doppler_phase(
    cube: RadarCube,
    spec: TDMVirtualArraySpec,
    *,
    fftshift: bool = True,
) -> RadarCube:
    """Remove inter-Tx motion phase from a Doppler-domain virtual array."""

    try:
        doppler_axis = cube.axes.index("doppler_bin")
        virtual_axis = cube.axes.index("virtual_rx")
    except ValueError as exc:
        raise ValueError(
            f'RadarCube axes must include "doppler_bin" and "virtual_rx"; got {cube.axes}.'
        ) from exc

    num_virtual = cube.data.shape[virtual_axis]
    if num_virtual != spec.num_virtual_antennas:
        raise ValueError(
            "RadarCube virtual_rx axis must match TDM geometry; "
            f"got {num_virtual} channels for {spec.num_virtual_antennas} virtual antennas."
        )

    return RadarCube(
        _native.compensate_tdm_doppler_phase_complex(
            _contiguous_cube_data(cube),
            doppler_axis,
            virtual_axis,
            spec.num_tx,
            spec.geometry.num_rx,
            fftshift,
        ),
        axes=cube.axes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "tdm_doppler_compensation": {
                "num_tx": spec.num_tx,
                "num_rx": spec.geometry.num_rx,
                "fftshift": fftshift,
                "tx_order": list(spec.tx_order),
            },
        },
    )


def map_planar_aperture(
    cube: RadarCube,
    layout: PlanarApertureLayout,
    *,
    input_axis: str = "virtual_rx",
) -> RadarCube:
    """Scatter virtual channels onto a sparse azimuth/elevation FFT aperture.

    When channels share one physical phase center, the first channel is kept.
    This matches common cascade-array processing and makes the policy explicit.
    """

    try:
        virtual_axis = cube.axes.index(input_axis)
    except ValueError as exc:
        raise ValueError(f'RadarCube axes must include "{input_axis}"; got {cube.axes}.') from exc
    num_virtual = cube.data.shape[virtual_axis]
    if num_virtual != layout.num_antennas:
        raise ValueError(
            "Planar aperture channel count must match the input axis; "
            f"got {layout.num_antennas} positions for {num_virtual} channels."
        )

    planar = _native.map_planar_aperture_complex(
        _contiguous_cube_data(cube),
        virtual_axis,
        layout.grid_indices,
    )
    axes = list(cube.axes)
    axes[virtual_axis : virtual_axis + 1] = ["azimuth_aperture", "elevation_aperture"]
    return RadarCube(
        planar,
        axes=tuple(axes),
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "planar_aperture": layout.as_metadata(),
        },
    )


def select_virtual_subarray(cube: RadarCube, spec: VirtualSubarraySpec) -> RadarCube:
    """Select and order physical virtual channels for one angle-estimation aperture."""

    try:
        virtual_axis = cube.axes.index("virtual_rx")
    except ValueError as exc:
        raise ValueError(f'RadarCube axes must include "virtual_rx"; got {cube.axes}.') from exc
    num_virtual = cube.data.shape[virtual_axis]
    if max(spec.antenna_indices) >= num_virtual:
        raise ValueError(
            "VirtualSubarraySpec index exceeds virtual_rx axis; "
            f"got max index {max(spec.antenna_indices)} for {num_virtual} channels."
        )

    return RadarCube(
        _native.select_virtual_subarray_complex(
            _contiguous_cube_data(cube),
            virtual_axis,
            spec.antenna_indices,
        ),
        axes=cube.axes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "virtual_subarray": {
                "antenna_indices": list(spec.antenna_indices),
                "layout": spec.layout.as_metadata(),
            },
        },
    )


def _contiguous_cube_data(cube: RadarCube) -> NDArray[np.complex64]:
    return np.ascontiguousarray(cube.data, dtype=np.complex64)
