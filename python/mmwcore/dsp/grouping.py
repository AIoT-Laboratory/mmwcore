"""Detector-independent range-Doppler peak grouping."""

from __future__ import annotations

from mmwcore.core import DetectionFrame, PeakGroupingSpec, RadarCube
from mmwcore.dsp._postprocess import group_range_doppler_candidates
from mmwcore.dsp.detection import _range_doppler_axes


def group_detection_peaks(
    cube: RadarCube,
    detections: DetectionFrame,
    spec: PeakGroupingSpec | None = None,
) -> DetectionFrame:
    """Keep detection candidates that are local range-Doppler maxima."""

    grouping = spec or PeakGroupingSpec()
    channel_indices = _required_channel_indices(detections)
    axes = _range_doppler_axes(cube)
    keep = group_range_doppler_candidates(
        cube.data,
        axes=axes,
        aggregate_rx=grouping.aggregate_rx,
        candidates=detections.detections,
        columns=(
            channel_indices["frame"],
            channel_indices["range_bin"],
            channel_indices["doppler_bin"],
        ),
        range_radius=grouping.range_radius,
        doppler_radius=grouping.doppler_radius,
        cyclic_doppler=grouping.cyclic_doppler,
        strict=grouping.strict,
    )

    return DetectionFrame(
        detections.detections[keep],
        channels=detections.channels,
        frame_id=detections.frame_id,
        timestamp=detections.timestamp,
        source=detections.source,
        units=detections.units,
        metadata={
            **detections.metadata,
            "peak_grouping": {
                "range_radius": grouping.range_radius,
                "doppler_radius": grouping.doppler_radius,
                "cyclic_doppler": grouping.cyclic_doppler,
                "strict": grouping.strict,
                "aggregate_rx": grouping.aggregate_rx,
                "input_candidates": int(detections.detections.shape[0]),
                "output_peaks": int(keep.size),
            },
        },
    )


def _required_channel_indices(detections: DetectionFrame) -> dict[str, int]:
    required = ("frame", "range_bin", "doppler_bin")
    missing = [name for name in required if name not in detections.channels]
    if missing:
        raise ValueError(f"DetectionFrame is missing grouping channels: {missing}.")
    return {name: detections.channels.index(name) for name in required}
