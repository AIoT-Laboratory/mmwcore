"""Calibrated detection-point-cloud assembly."""

from __future__ import annotations

from mmwcore.core import DetectionFrame, PointCloudFrame, PointCloudProjectionSpec

from ._pointcloud import project_detection_point_cloud as native_project_detection_point_cloud

_CALIBRATED_AZIMUTH_POINT_CHANNELS = (
    "x",
    "y",
    "z",
    "velocity",
    "magnitude",
    "range_bin",
    "doppler_bin",
    "azimuth_bin",
    "azimuth_rad",
)


def detections_to_point_cloud(
    detections: DetectionFrame,
    spec: PointCloudProjectionSpec | None = None,
) -> PointCloudFrame:
    """Project calibrated range-Doppler-angle detections into radar Cartesian space."""

    projection = spec or PointCloudProjectionSpec()
    range_idx = _channel(detections, "range_bin")
    doppler_idx = _channel(detections, "doppler_bin")
    magnitude_idx = _channel(detections, "magnitude")
    azimuth_idx = _channel(detections, "azimuth_bin")
    azimuth_rad_idx = _channel(detections, "azimuth_rad")
    has_elevation = "elevation_rad" in detections.channels
    elevation_channels = ("elevation_rad", "elevation_magnitude") if has_elevation else ()
    elevation_indices = (
        (_channel(detections, "elevation_rad"), _channel(detections, "elevation_magnitude"))
        if has_elevation
        else None
    )
    passthrough_channels = tuple(
        channel for channel in ("noise", "snr", "angle_magnitude") if channel in detections.channels
    )
    passthrough_indices = tuple(_channel(detections, channel) for channel in passthrough_channels)
    channels = (
        *_CALIBRATED_AZIMUTH_POINT_CHANNELS,
        *elevation_channels,
        *passthrough_channels,
    )
    points = native_project_detection_point_cloud(
        detections.detections,
        range_bin_column=range_idx,
        doppler_bin_column=doppler_idx,
        magnitude_column=magnitude_idx,
        azimuth_bin_column=azimuth_idx,
        azimuth_rad_column=azimuth_rad_idx,
        elevation_columns=elevation_indices,
        passthrough_columns=passthrough_indices,
        spec=projection,
    )

    return PointCloudFrame(
        points,
        channels=channels,
        frame_id=detections.frame_id,
        timestamp=detections.timestamp,
        source=detections.source,
        coordinate_frame="radar",
        units={
            "x": "m",
            "y": "m",
            "z": "m",
            "velocity": "m/s",
            "magnitude": detections.units.get("magnitude", "arbitrary"),
            "range_bin": "bin",
            "doppler_bin": "bin",
            "azimuth_bin": "bin",
            "azimuth_rad": "rad",
            **(
                {"elevation_rad": "rad", "elevation_magnitude": "angle_fft"}
                if has_elevation
                else {}
            ),
            **{
                channel: detections.units.get(channel, "arbitrary")
                for channel in passthrough_channels
            },
        },
        metadata={
            **detections.metadata,
            "pointcloud_projection": {
                "range_resolution_m": projection.range_resolution_m,
                "doppler_resolution_mps": projection.doppler_resolution_mps,
                "doppler_sign": projection.doppler_sign,
                "center_doppler": projection.center_doppler,
                "doppler_bins": projection.doppler_bins,
                "doppler_fftshifted": projection.doppler_fftshifted,
                "input_detections": int(detections.detections.shape[0]),
                "output_points": int(points.shape[0]),
                "spatial_dimensions": 3 if has_elevation else 2,
            },
        },
    )


def _channel(detections: DetectionFrame, name: str) -> int:
    try:
        return detections.channels.index(name)
    except ValueError as exc:
        raise ValueError(f'DetectionFrame channels must include "{name}".') from exc


__all__ = ["detections_to_point_cloud"]
