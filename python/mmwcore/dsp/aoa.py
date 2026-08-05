"""Angle-domain transforms for offline mmwcore processing."""

from __future__ import annotations

import numpy as np

from mmwcore.core import (
    AngleFFTSpec,
    DetectionFrame,
    PlanarAngleFFTSpec,
    RadarCube,
    VirtualAntennaLayout,
    VirtualSubarraySpec,
)
from mmwcore.dsp._angle import calibrate_angle_bins
from mmwcore.dsp._candidate_aoa import candidate_azimuth_peaks, candidate_elevations
from mmwcore.dsp._fft import fft_complex_axis


def angle_fft(cube: RadarCube, spec: AngleFFTSpec | None = None) -> RadarCube:
    """Run an FFT over a virtual antenna axis and return an angle-bin cube.

    This is a standalone shape primitive. It estimates angle bins from an input
    antenna axis, but it does not convert bins into physical azimuth or
    elevation angles.
    """

    fft_spec = spec or AngleFFTSpec()
    try:
        antenna_axis = cube.axes.index(fft_spec.input_axis)
    except ValueError as exc:
        raise ValueError(
            f'RadarCube axes must include "{fft_spec.input_axis}"; got {cube.axes}.'
        ) from exc

    data = cube.data
    if (
        fft_spec.virtual_layout is not None
        and fft_spec.virtual_layout.num_antennas != data.shape[antenna_axis]
    ):
        raise ValueError(
            "AngleFFTSpec.virtual_layout antenna count must match the input axis length; "
            f"got {fft_spec.virtual_layout.num_antennas} antennas for axis length "
            f"{data.shape[antenna_axis]}."
        )

    n_fft = fft_spec.n_fft or data.shape[antenna_axis]
    transformed = fft_complex_axis(
        data,
        axis=antenna_axis,
        n_fft=n_fft,
        window=fft_spec.window,
        remove_dc=False,
        fftshift=fft_spec.fftshift,
        one_sided=False,
    )

    axes = list(cube.axes)
    axes[antenna_axis] = fft_spec.output_axis
    return RadarCube(
        transformed,
        axes=tuple(axes),
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units="angle_fft",
        metadata={
            **cube.metadata,
            "angle_fft": {
                "n_fft": n_fft,
                "window": fft_spec.window.value,
                "fftshift": fft_spec.fftshift,
                "input_axis": fft_spec.input_axis,
                "output_axis": fft_spec.output_axis,
                "virtual_layout": (
                    fft_spec.virtual_layout.as_metadata()
                    if fft_spec.virtual_layout is not None
                    else None
                ),
            },
        },
    )


def estimate_candidate_azimuths(
    cube: RadarCube,
    detections: DetectionFrame,
    spec: AngleFFTSpec,
) -> DetectionFrame:
    """Estimate one strongest physical azimuth for each range-Doppler candidate."""

    layout = spec.virtual_layout
    if layout is None:
        raise ValueError("Candidate azimuth estimation requires a calibrated virtual layout.")
    cube_axes = _candidate_cube_axes(cube, input_axis=spec.input_axis)
    channel_indices = _candidate_channel_indices(detections)
    _reject_existing_angle_channels(detections)
    peak_bins, peak_angles, peak_magnitudes = candidate_azimuth_peaks(
        cube.data,
        cube_axes=cube_axes,
        candidates=detections.detections,
        candidate_columns=(
            channel_indices["frame"],
            channel_indices["range_bin"],
            channel_indices["doppler_bin"],
        ),
        layout=layout,
        spec=spec,
    )
    output, output_channels = _candidate_output(
        detections,
        channel_indices=channel_indices,
        peak_bins=peak_bins,
        peak_angles=peak_angles,
        peak_magnitudes=peak_magnitudes,
    )
    n_fft = spec.n_fft or layout.num_antennas
    return DetectionFrame(
        output,
        channels=output_channels,
        frame_id=detections.frame_id,
        timestamp=detections.timestamp,
        source=detections.source,
        units={**detections.units, "azimuth_rad": "rad", "angle_magnitude": "angle_fft"},
        metadata={
            **detections.metadata,
            "candidate_azimuth": {
                "n_fft": n_fft,
                "window": spec.window.value,
                "fftshift": spec.fftshift,
                "input_axis": spec.input_axis,
                "virtual_layout": layout.as_metadata(),
                "input_candidates": int(detections.detections.shape[0]),
                "output_detections": int(output.shape[0]),
            },
        },
    )


def planar_angle_fft(
    cube: RadarCube,
    spec: PlanarAngleFFTSpec | None = None,
) -> RadarCube:
    """Run separable azimuth and elevation FFTs over a planar aperture."""

    fft_spec = spec or PlanarAngleFFTSpec()
    try:
        azimuth_axis = cube.axes.index(fft_spec.azimuth_input_axis)
        elevation_axis = cube.axes.index(fft_spec.elevation_input_axis)
    except ValueError as exc:
        raise ValueError(
            f"RadarCube axes must include the configured planar aperture axes; got {cube.axes}."
        ) from exc

    azimuth_n_fft = fft_spec.azimuth_n_fft or cube.data.shape[azimuth_axis]
    elevation_n_fft = fft_spec.elevation_n_fft or cube.data.shape[elevation_axis]
    transformed = fft_complex_axis(
        cube.data,
        axis=azimuth_axis,
        n_fft=azimuth_n_fft,
        window=fft_spec.window,
        remove_dc=False,
        fftshift=fft_spec.fftshift,
        one_sided=False,
    )
    transformed = fft_complex_axis(
        transformed,
        axis=elevation_axis,
        n_fft=elevation_n_fft,
        window=fft_spec.window,
        remove_dc=False,
        fftshift=fft_spec.fftshift,
        one_sided=False,
    )

    axes = list(cube.axes)
    axes[azimuth_axis] = fft_spec.azimuth_output_axis
    axes[elevation_axis] = fft_spec.elevation_output_axis
    return RadarCube(
        transformed.astype(np.complex64, copy=False),
        axes=tuple(axes),
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units="angle_fft",
        metadata={
            **cube.metadata,
            "planar_angle_fft": {
                "azimuth_n_fft": azimuth_n_fft,
                "elevation_n_fft": elevation_n_fft,
                "window": fft_spec.window.value,
                "fftshift": fft_spec.fftshift,
                "azimuth_input_axis": fft_spec.azimuth_input_axis,
                "elevation_input_axis": fft_spec.elevation_input_axis,
                "azimuth_output_axis": fft_spec.azimuth_output_axis,
                "elevation_output_axis": fft_spec.elevation_output_axis,
            },
        },
    )


def estimate_candidate_elevations(
    cube: RadarCube,
    detections: DetectionFrame,
    spec: AngleFFTSpec,
    *,
    azimuth_subarray: VirtualSubarraySpec,
    elevation_subarray: VirtualSubarraySpec,
) -> DetectionFrame:
    """Recover elevation from two horizontally aligned, vertically displaced rows.

    The estimator follows the IWR6843/TI AoA geometry: the azimuth FFT supplies
    the lateral direction cosine, while the phase difference between the two
    row spectra at that same FFT bin supplies the vertical direction cosine.
    """

    _reject_existing_elevation_channels(detections)
    required = {"azimuth_bin", "azimuth_rad"}
    missing = sorted(required.difference(detections.channels))
    if missing:
        raise ValueError(f"Elevation estimation requires detection channels: {missing}.")

    cube_axes = _candidate_cube_axes(cube, input_axis=spec.input_axis)
    channel_indices = _candidate_channel_indices(detections)
    valid_indices, elevation_angles, elevation_magnitudes, (x_offset, z_offset) = (
        candidate_elevations(
            cube.data,
            cube_axes=cube_axes,
            candidates=detections.detections,
            candidate_columns=(
                channel_indices["frame"],
                channel_indices["range_bin"],
                channel_indices["doppler_bin"],
                detections.channels.index("azimuth_bin"),
                detections.channels.index("azimuth_rad"),
            ),
            azimuth_subarray=azimuth_subarray,
            elevation_subarray=elevation_subarray,
            spec=spec,
        )
    )
    output = np.concatenate(
        (
            detections.detections[valid_indices],
            elevation_angles[:, None],
            elevation_magnitudes[:, None],
        ),
        axis=1,
    )
    channels = (*detections.channels, "elevation_rad", "elevation_magnitude")
    return DetectionFrame(
        output,
        channels=channels,
        frame_id=detections.frame_id,
        timestamp=detections.timestamp,
        source=detections.source,
        units={
            **detections.units,
            "elevation_rad": "rad",
            "elevation_magnitude": "angle_fft",
        },
        metadata={
            **detections.metadata,
            "candidate_elevation": {
                "n_fft": spec.n_fft or azimuth_subarray.layout.num_antennas,
                "window": spec.window.value,
                "fftshift": spec.fftshift,
                "azimuth_subarray": azimuth_subarray.layout.as_metadata(),
                "elevation_subarray": elevation_subarray.layout.as_metadata(),
                "row_offset_wavelengths": [x_offset, 0.0, z_offset],
                "input_candidates": int(detections.detections.shape[0]),
                "output_detections": int(output.shape[0]),
                "rejected_direction_cosines": int(detections.detections.shape[0] - output.shape[0]),
            },
        },
    )


def _candidate_cube_axes(
    cube: RadarCube,
    *,
    input_axis: str,
) -> tuple[int, int, int, int]:
    required_axes = ("frame", "doppler_bin", input_axis, "range_bin")
    try:
        return (
            cube.axes.index("frame"),
            cube.axes.index("doppler_bin"),
            cube.axes.index(input_axis),
            cube.axes.index("range_bin"),
        )
    except ValueError as exc:
        raise ValueError(f"RadarCube axes must include {required_axes}; got {cube.axes}.") from exc


def _reject_existing_elevation_channels(detections: DetectionFrame) -> None:
    existing = {"elevation_rad", "elevation_magnitude"} & set(detections.channels)
    if existing:
        raise ValueError(f"Detections already contain elevation channels: {sorted(existing)}.")


def _reject_existing_angle_channels(detections: DetectionFrame) -> None:
    existing_angle_channels = {
        "azimuth_bin",
        "azimuth_rad",
        "angle_magnitude",
    } & set(detections.channels)
    if existing_angle_channels:
        raise ValueError(
            "Candidate azimuth estimation requires range-Doppler candidates without "
            f"existing angle channels; got {sorted(existing_angle_channels)}."
        )


def _candidate_output(
    detections: DetectionFrame,
    *,
    channel_indices: dict[str, int],
    peak_bins: np.ndarray,
    peak_angles: np.ndarray,
    peak_magnitudes: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    index_channels = {"frame", "range_bin", "doppler_bin"}
    passthrough_channels = tuple(
        channel for channel in detections.channels if channel not in index_channels
    )
    output_channels = (
        "frame",
        "range_bin",
        "doppler_bin",
        "azimuth_bin",
        "azimuth_rad",
        *passthrough_channels,
        "angle_magnitude",
    )
    output = np.empty((detections.detections.shape[0], len(output_channels)), dtype=np.float32)
    if output.size:
        output[:, 0] = detections.detections[:, channel_indices["frame"]]
        output[:, 1] = detections.detections[:, channel_indices["range_bin"]]
        output[:, 2] = detections.detections[:, channel_indices["doppler_bin"]]
        output[:, 3] = peak_bins
        output[:, 4] = peak_angles
        for output_index, channel in enumerate(passthrough_channels, start=5):
            output[:, output_index] = detections.detections[:, detections.channels.index(channel)]
        output[:, -1] = peak_magnitudes
    return output, output_channels


def _candidate_channel_indices(detections: DetectionFrame) -> dict[str, int]:
    required = ("frame", "range_bin", "doppler_bin")
    missing = [name for name in required if name not in detections.channels]
    if missing:
        raise ValueError(f"DetectionFrame is missing candidate channels: {missing}.")
    return {name: detections.channels.index(name) for name in required}


def angle_bin_angles(
    num_bins: int,
    layout: VirtualAntennaLayout,
    *,
    fftshift: bool = True,
) -> np.ndarray:
    """Return calibrated physical angles for a uniform linear array.

    The layout axis selects an azimuth ULA along x or elevation ULA along z.
    Bins outside visible sine-space are rejected.
    """

    return calibrate_angle_bins(num_bins, layout, fftshift=fftshift)
