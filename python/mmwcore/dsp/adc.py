"""Python contract boundary for native ADC decoding."""

from __future__ import annotations

import numpy as np

from mmwcore import _native
from mmwcore.core import ADCComplexLayout, ADCFrameSpec, RadarCube, RawADCFrame


def organize_adc_samples(
    raw: RawADCFrame | np.ndarray,
    spec: ADCFrameSpec,
    *,
    drop_incomplete: bool = False,
) -> RadarCube:
    """Decode raw int16 ADC values through the native mmwcore core.

    The output shape is always ``(num_frames, num_chirps, num_rx, num_samples)``
    with axes ``("frame", "chirp", "rx", "sample")``.
    """

    raw_frame = raw if isinstance(raw, RawADCFrame) else RawADCFrame(raw)
    samples = np.ascontiguousarray(raw_frame.samples, dtype=np.int16)
    cube_data = _native.decode_adc_i16(
        samples,
        spec.num_chirps,
        spec.num_rx,
        spec.num_samples,
        _layout_code(spec.layout),
        drop_incomplete,
    )
    return RadarCube(
        cube_data,
        frame_id=raw_frame.frame_id,
        timestamp=raw_frame.timestamp,
        source=raw_frame.source,
        metadata={
            "profile": raw_frame.profile,
            **raw_frame.metadata,
        },
    )


def _layout_code(layout: ADCComplexLayout) -> int:
    if layout is ADCComplexLayout.IQ_INTERLEAVED:
        return 0
    if layout is ADCComplexLayout.SAMPLE_I_THEN_Q:
        return 1
    if layout is ADCComplexLayout.GROUP2_I_THEN_Q:
        return 2
    if layout is ADCComplexLayout.GROUP4_I_THEN_Q:
        return 3
    raise ValueError(f"Unsupported ADC layout: {layout}")  # pragma: no cover
