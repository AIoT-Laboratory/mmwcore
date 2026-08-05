"""Detector-independent quality filtering for radar detections."""

from __future__ import annotations

from mmwcore.core import DetectionFrame, DetectionQualitySpec
from mmwcore.dsp._postprocess import quality_filter_indices


def filter_detection_quality(
    detections: DetectionFrame,
    spec: DetectionQualitySpec,
) -> DetectionFrame:
    """Keep detections that satisfy an explicit linear-SNR threshold."""

    try:
        snr_index = detections.channels.index("snr")
    except ValueError:
        raise ValueError('Detection quality filtering requires an "snr" channel.') from None
    snr_unit = detections.units.get("snr")
    if snr_unit != "linear_ratio":
        raise ValueError(
            'Detection quality filtering requires units["snr"] == "linear_ratio"; '
            f"got {snr_unit!r}."
        )
    keep = quality_filter_indices(
        detections.detections,
        snr_column=snr_index,
        min_snr=spec.min_snr,
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
            "quality_filter": {
                "min_snr": spec.min_snr,
                "snr_unit": "linear_ratio",
                "input_detections": int(detections.detections.shape[0]),
                "output_detections": int(keep.size),
            },
        },
    )
