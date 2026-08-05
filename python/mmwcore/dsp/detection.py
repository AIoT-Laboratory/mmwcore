"""Threshold detection helpers backed by the native mmwcore kernel."""

from __future__ import annotations

import numpy as np

from mmwcore.core import DetectionFrame, PeakDetectionSpec, RadarCube, VirtualAntennaLayout
from mmwcore.dsp._detection import (
    range_doppler_magnitude as native_range_doppler_magnitude,
)
from mmwcore.dsp._detection import (
    threshold_range_doppler,
    threshold_range_doppler_azimuth,
)
from mmwcore.dsp.aoa import angle_bin_angles

_DETECTION_CHANNELS = ("frame", "range_bin", "doppler_bin", "magnitude")
_AZIMUTH_DETECTION_CHANNELS = (
    "frame",
    "range_bin",
    "doppler_bin",
    "azimuth_bin",
    "magnitude",
)
_CALIBRATED_AZIMUTH_DETECTION_CHANNELS = (
    "frame",
    "range_bin",
    "doppler_bin",
    "azimuth_bin",
    "azimuth_rad",
    "magnitude",
)


def _range_doppler_magnitude(
    cube: RadarCube,
    aggregate_rx: str,
) -> np.ndarray:
    frame_axis, doppler_axis, rx_axis, range_axis = _range_doppler_axes(cube)
    return native_range_doppler_magnitude(
        cube.data,
        frame_axis=frame_axis,
        doppler_axis=doppler_axis,
        receiver_axis=rx_axis,
        range_axis=range_axis,
        aggregate_rx=aggregate_rx,
    )


def _range_doppler_axes(cube: RadarCube) -> tuple[int, int, int, int]:
    axes = cube.axes
    try:
        frame_axis = axes.index("frame")
        doppler_axis = axes.index("doppler_bin")
        rx_axis = axes.index("rx") if "rx" in axes else axes.index("virtual_rx")
        range_axis = axes.index("range_bin")
    except ValueError as exc:
        raise ValueError(
            'RadarCube axes must include "frame", "doppler_bin", "rx" or "virtual_rx", '
            'and "range_bin"; '
            f"got {axes}."
        ) from exc
    return frame_axis, doppler_axis, rx_axis, range_axis


def _range_doppler_azimuth_axes(cube: RadarCube) -> tuple[int, int, int, int]:
    try:
        frame_axis = cube.axes.index("frame")
        doppler_axis = cube.axes.index("doppler_bin")
        azimuth_axis = cube.axes.index("azimuth_bin")
        range_axis = cube.axes.index("range_bin")
    except ValueError as exc:
        raise ValueError(
            'RadarCube axes must include "frame", "doppler_bin", "azimuth_bin", '
            f'and "range_bin"; got {cube.axes}.'
        ) from exc
    return frame_axis, doppler_axis, azimuth_axis, range_axis


def _detections_from_indices(
    rd_map: np.ndarray,
    hit_indices: np.ndarray,
    *,
    frame_id: str | int | None,
    timestamp: float | None,
    source: str | None,
    units: str | dict[str, str] | None,
    metadata: dict[str, object],
    extra_channels: tuple[str, ...] = (),
    extra_values: np.ndarray | None = None,
    extra_units: dict[str, str] | None = None,
) -> DetectionFrame:
    values = (
        rd_map[tuple(hit_indices.T)].astype(np.float32, copy=False)
        if hit_indices.size
        else np.empty(0, dtype=np.float32)
    )
    return _detections_from_hits(
        hit_indices,
        values,
        frame_id=frame_id,
        timestamp=timestamp,
        source=source,
        units=units,
        metadata=metadata,
        extra_channels=extra_channels,
        extra_values=extra_values,
        extra_units=extra_units,
    )


def _detections_from_hits(
    hit_indices: np.ndarray,
    magnitudes: np.ndarray,
    *,
    frame_id: str | int | None,
    timestamp: float | None,
    source: str | None,
    units: str | dict[str, str] | None,
    metadata: dict[str, object],
    extra_channels: tuple[str, ...] = (),
    extra_values: np.ndarray | None = None,
    extra_units: dict[str, str] | None = None,
) -> DetectionFrame:
    hit_indices = np.asarray(hit_indices, dtype=np.int64)
    magnitudes = np.asarray(magnitudes, dtype=np.float32)
    if hit_indices.ndim != 2 or hit_indices.shape[1] != 3:
        raise ValueError(
            f"Range-Doppler detection indices must have shape (N, 3); got {hit_indices.shape}."
        )
    if magnitudes.shape != (hit_indices.shape[0],):
        raise ValueError(
            "Range-Doppler detection magnitudes must have one value per hit; "
            f"got {magnitudes.shape} for {hit_indices.shape[0]} hit(s)."
        )
    if extra_values is None:
        extra_values = np.empty((hit_indices.shape[0], 0), dtype=np.float32)
    else:
        extra_values = np.asarray(extra_values, dtype=np.float32)
    expected_extra_shape = (hit_indices.shape[0], len(extra_channels))
    if extra_values.shape != expected_extra_shape:
        raise ValueError(
            "Detection extra values must match hit count and extra channels; "
            f"expected {expected_extra_shape}, got {extra_values.shape}."
        )
    channels = (*_DETECTION_CHANNELS, *extra_channels)
    detections = np.zeros((hit_indices.shape[0], len(channels)), dtype=np.float32)
    if hit_indices.size:
        frame_idx = hit_indices[:, 0]
        doppler_idx = hit_indices[:, 1]
        range_idx = hit_indices[:, 2]
        detections[:, 0] = frame_idx
        detections[:, 1] = range_idx
        detections[:, 2] = doppler_idx
        detections[:, 3] = magnitudes
        detections[:, len(_DETECTION_CHANNELS) :] = extra_values

    return DetectionFrame(
        detections,
        channels=channels,
        frame_id=frame_id,
        timestamp=timestamp,
        source=source,
        units={
            **({"magnitude": units} if isinstance(units, str) else (units or {})),
            **(extra_units or {}),
        },
        metadata=metadata,
    )


def _azimuth_detections_from_hits(
    hit_indices: np.ndarray,
    magnitudes: np.ndarray,
    *,
    num_azimuth_bins: int,
    frame_id: str | int | None,
    timestamp: float | None,
    source: str | None,
    units: str | dict[str, str] | None,
    metadata: dict[str, object],
) -> DetectionFrame:
    hit_indices = np.asarray(hit_indices, dtype=np.int64)
    magnitudes = np.asarray(magnitudes, dtype=np.float32)
    if hit_indices.ndim != 2 or hit_indices.shape[1] != 4:
        raise ValueError(
            "Range-Doppler-azimuth detection indices must have shape (N, 4); "
            f"got {hit_indices.shape}."
        )
    if magnitudes.shape != (hit_indices.shape[0],):
        raise ValueError(
            "Range-Doppler-azimuth detection magnitudes must have one value per hit; "
            f"got {magnitudes.shape} for {hit_indices.shape[0]} hit(s)."
        )
    azimuth_angles = _azimuth_angles_from_metadata(num_azimuth_bins, metadata)
    channels = (
        _CALIBRATED_AZIMUTH_DETECTION_CHANNELS
        if azimuth_angles is not None
        else _AZIMUTH_DETECTION_CHANNELS
    )
    detections = np.zeros(
        (hit_indices.shape[0], len(channels)),
        dtype=np.float32,
    )
    if hit_indices.size:
        frame_idx = hit_indices[:, 0]
        doppler_idx = hit_indices[:, 1]
        azimuth_idx = hit_indices[:, 2]
        range_idx = hit_indices[:, 3]
        detections[:, 0] = frame_idx
        detections[:, 1] = range_idx
        detections[:, 2] = doppler_idx
        detections[:, 3] = azimuth_idx
        if azimuth_angles is None:
            detections[:, 4] = magnitudes
        else:
            detections[:, 4] = azimuth_angles[azimuth_idx]
            detections[:, 5] = magnitudes

    return DetectionFrame(
        detections,
        channels=channels,
        frame_id=frame_id,
        timestamp=timestamp,
        source=source,
        units={"magnitude": units} if isinstance(units, str) else (units or {}),
        metadata=metadata,
    )


def _azimuth_angles_from_metadata(
    num_bins: int,
    metadata: dict[str, object],
) -> np.ndarray | None:
    angle_fft_metadata = metadata.get("angle_fft")
    if not isinstance(angle_fft_metadata, dict):
        return None
    layout_metadata = angle_fft_metadata.get("virtual_layout")
    if not isinstance(layout_metadata, dict):
        return None

    positions = layout_metadata.get("positions_wavelengths")
    if not isinstance(positions, list):
        return None

    # Validate that positions are 3D tuples
    positions_tuples: list[tuple[float, float, float]] = []
    for position in positions:
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            return None
        positions_tuples.append((float(position[0]), float(position[1]), float(position[2])))

    layout = VirtualAntennaLayout(
        tuple(positions_tuples),
        name=str(layout_metadata.get("name", "virtual_array")),
        angle_axis=str(layout_metadata.get("angle_axis", "azimuth")),
    )
    return angle_bin_angles(
        num_bins,
        layout,
        fftshift=bool(angle_fft_metadata.get("fftshift", True)),
    )


def detect_peaks(cube: RadarCube, spec: PeakDetectionSpec) -> DetectionFrame:
    """Threshold detector with optional angle-domain local-maximum selection."""

    if "azimuth_bin" in cube.axes:
        frame_axis, doppler_axis, azimuth_axis, range_axis = _range_doppler_azimuth_axes(cube)
        hit_indices, magnitudes = threshold_range_doppler_azimuth(
            cube.data,
            frame_axis=frame_axis,
            doppler_axis=doppler_axis,
            azimuth_axis=azimuth_axis,
            range_axis=range_axis,
            threshold=spec.threshold,
            azimuth_peak_radius=spec.azimuth_peak_radius,
            azimuth_peak_strict=spec.azimuth_peak_strict,
        )
        return _azimuth_detections_from_hits(
            hit_indices,
            magnitudes,
            num_azimuth_bins=cube.data.shape[azimuth_axis],
            frame_id=cube.frame_id,
            timestamp=cube.timestamp,
            source=cube.source,
            units=cube.units,
            metadata={
                **cube.metadata,
                "peak_detection": {
                    "threshold": spec.threshold,
                    "aggregate_rx": None,
                    "azimuth_peak_radius": spec.azimuth_peak_radius,
                    "azimuth_peak_strict": spec.azimuth_peak_strict,
                    "output_detections": int(hit_indices.shape[0]),
                },
            },
        )

    frame_axis, doppler_axis, rx_axis, range_axis = _range_doppler_axes(cube)
    hit_indices, magnitudes = threshold_range_doppler(
        cube.data,
        frame_axis=frame_axis,
        doppler_axis=doppler_axis,
        receiver_axis=rx_axis,
        range_axis=range_axis,
        aggregate_rx=spec.aggregate_rx,
        threshold=spec.threshold,
    )

    return _detections_from_hits(
        hit_indices,
        magnitudes,
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        units=cube.units,
        metadata={
            **cube.metadata,
            "peak_detection": {
                "threshold": spec.threshold,
                "aggregate_rx": spec.aggregate_rx,
                "output_detections": int(hit_indices.shape[0]),
            },
        },
    )
