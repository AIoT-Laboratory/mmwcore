"""Slow-time phase primitives for exploratory vital-sign sensing."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mmwcore import _native
from mmwcore.core import RadarCube, VitalSignQuantity, VitalSignWaveform


def extract_vital_sign_phase(
    cube: RadarCube,
    *,
    range_bin: int,
    sample_rate_hz: float,
    selectors: Mapping[str, int] | None = None,
    slow_time_axis: str = "frame",
    range_axis: str = "range_bin",
    remove_mean: bool = True,
) -> VitalSignWaveform:
    """Extract one explicit range/channel slow-time phase sequence.

    Every non-singleton axis other than slow time and range requires an index
    in ``selectors``. This prevents accidental coherent averaging across
    antennas, chirps, or views.
    """

    selected = dict(selectors or {})
    _validate_axes(cube, slow_time_axis=slow_time_axis, range_axis=range_axis)
    time_index = cube.axes.index(slow_time_axis)
    range_index = cube.axes.index(range_axis)
    if not 0 <= range_bin < cube.data.shape[range_index]:
        raise ValueError(f"range_bin is outside the {range_axis!r} axis.")
    unknown = sorted(set(selected) - set(cube.axes))
    if unknown:
        raise ValueError(f"selectors contains unknown RadarCube axes: {unknown}.")
    if slow_time_axis in selected or range_axis in selected:
        raise ValueError("selectors must not override the slow-time or range axis.")

    indices: list[int | slice] = []
    resolved_selectors: dict[str, int] = {}
    for axis_index, (axis, size) in enumerate(zip(cube.axes, cube.data.shape, strict=True)):
        if axis_index == time_index:
            indices.append(slice(None))
        elif axis_index == range_index:
            indices.append(range_bin)
        else:
            index = _selected_axis_index(axis, size=size, selectors=selected)
            indices.append(index)
            resolved_selectors[axis] = index

    samples = np.asarray(cube.data[tuple(indices)])
    if samples.ndim != 1:
        raise ValueError("Vital-sign selection must resolve to one slow-time sequence.")
    phase = _native.unwrap_vital_phase(
        np.ascontiguousarray(samples, dtype=np.complex64),
        remove_mean,
    )
    return VitalSignWaveform(
        phase,
        sample_rate_hz=sample_rate_hz,
        quantity=VitalSignQuantity.PHASE_RAD,
        start_time_s=cube.timestamp if cube.timestamp is not None else 0.0,
        range_bin=range_bin,
        source=cube.source,
        metadata={
            "source_frame_id": cube.frame_id,
            "slow_time_axis": slow_time_axis,
            "range_axis": range_axis,
            "selectors": resolved_selectors,
            "source_cube_units": cube.units,
            "phase_unwrapped": True,
            "mean_removed": remove_mean,
        },
    )


def phase_to_displacement(
    waveform: VitalSignWaveform,
    *,
    wavelength_m: float,
) -> VitalSignWaveform:
    """Convert monostatic round-trip phase into relative displacement."""

    if waveform.quantity is not VitalSignQuantity.PHASE_RAD:
        raise ValueError("phase_to_displacement requires a phase_rad waveform.")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be finite and positive.")
    displacement = _native.vital_phase_to_displacement(
        np.ascontiguousarray(waveform.values, dtype=np.float32),
        float(wavelength_m),
    )
    return VitalSignWaveform(
        displacement,
        sample_rate_hz=waveform.sample_rate_hz,
        quantity=VitalSignQuantity.DISPLACEMENT_M,
        start_time_s=waveform.start_time_s,
        range_bin=waveform.range_bin,
        source=waveform.source,
        metadata={
            **waveform.metadata,
            "phase_to_displacement": {
                "wavelength_m": float(wavelength_m),
                "geometry": "monostatic_round_trip",
                "formula": "displacement = phase * wavelength / (4*pi)",
            },
        },
    )


def _validate_axes(cube: RadarCube, *, slow_time_axis: str, range_axis: str) -> None:
    if slow_time_axis == range_axis:
        raise ValueError("slow_time_axis and range_axis must be different.")
    if len(set(cube.axes)) != len(cube.axes):
        raise ValueError("Vital-sign extraction requires unique RadarCube axis names.")
    missing = [axis for axis in (slow_time_axis, range_axis) if axis not in cube.axes]
    if missing:
        raise ValueError(f"RadarCube is missing vital-sign axes: {missing}.")


def _selected_axis_index(
    axis: str,
    *,
    size: int,
    selectors: Mapping[str, int],
) -> int:
    if axis not in selectors:
        if size == 1:
            return 0
        raise ValueError(f"selectors requires an explicit index for non-singleton axis {axis!r}.")
    index = selectors[axis]
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"selector for axis {axis!r} must be an integer.")
    if not 0 <= index < size:
        raise ValueError(f"selector for axis {axis!r} is outside its axis length {size}.")
    return index


__all__ = ["extract_vital_sign_phase", "phase_to_displacement"]
