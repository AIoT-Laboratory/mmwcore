"""Composable offline processing runners for mmwcore."""

from __future__ import annotations

import numpy as np

from mmwcore.core import (
    ADCFrame,
    DetectionFrame,
    DetectionMethod,
    DetectionPipeline,
    PointCloudFrame,
    PointCloudPipeline,
    RadarCube,
    RangeDopplerCFARSpec,
    RangeDopplerPipeline,
)
from mmwcore.dsp.adc import organize_adc_samples
from mmwcore.dsp.aoa import (
    angle_fft,
    estimate_candidate_azimuths,
    estimate_candidate_elevations,
)
from mmwcore.dsp.calibration import apply_virtual_channel_calibration
from mmwcore.dsp.cfar import detect_cfar, detect_range_doppler_cfar
from mmwcore.dsp.clutter import remove_static_clutter
from mmwcore.dsp.detection import detect_peaks
from mmwcore.dsp.doppler import doppler_fft
from mmwcore.dsp.grouping import group_detection_peaks
from mmwcore.dsp.pointcloud import detections_to_point_cloud
from mmwcore.dsp.quality import filter_detection_quality
from mmwcore.dsp.range import range_fft
from mmwcore.dsp.virtual_array import (
    compensate_tdm_doppler_phase,
    map_tdm_virtual_array,
    select_virtual_subarray,
)


def detect(
    raw: ADCFrame | np.ndarray,
    recipe: DetectionPipeline,
) -> DetectionFrame:
    """Run an explicit ADC-to-detection recipe."""

    range_doppler_cube = range_doppler(raw, recipe.transform)
    return process_range_doppler_to_detections(range_doppler_cube, recipe)


def process_range_doppler_to_detections(
    range_doppler_cube: RadarCube,
    recipe: DetectionPipeline,
) -> DetectionFrame:
    """Detect targets in a range-Doppler cube produced by the same recipe."""

    _validate_range_doppler(range_doppler_cube, recipe.transform)
    detection_cube = _detection_cube(range_doppler_cube, recipe)
    detections = _detect(detection_cube, recipe)
    detections = _filter_detections(detections, recipe)
    return _estimate_detection_angles(
        range_doppler_cube,
        detection_cube,
        detections,
        recipe,
    )


def _validate_range_doppler(cube: RadarCube, recipe: RangeDopplerPipeline) -> None:
    """Check physical interpretation when reusing an already transformed cube.

    Clutter subtraction and unrelated provenance may differ between consumers
    sharing an FFT. Axis ordering, FFT conventions and channel corrections may not.
    """
    adc = recipe.decode.adc
    tdm = recipe.tdm_virtual_array
    range_size = recipe.range_fft.n_fft or adc.num_samples
    slow_samples = adc.num_chirps // (tdm.num_tx if tdm else 1)
    axes = ["frame", "loop" if tdm else "chirp", "virtual_rx" if tdm else "rx", "range_bin"]
    shape = [
        cube.data.shape[0],
        slow_samples,
        adc.num_rx * (tdm.num_tx if tdm else 1),
        range_size // 2 + 1 if recipe.range_fft.one_sided else range_size,
    ]
    doppler_axis = axes.index(recipe.doppler_fft.input_axis)
    doppler_size = recipe.doppler_fft.n_fft or shape[doppler_axis]
    axes[doppler_axis] = "doppler_bin"
    shape[doppler_axis] = doppler_size
    if cube.axes != tuple(axes) or cube.data.shape != tuple(shape):
        raise ValueError("Precomputed RD axes/shape do not match the transform recipe.")
    _check_rd_metadata(
        cube,
        "range_fft",
        {
            "n_fft": range_size,
            "window": recipe.range_fft.window.value,
            "one_sided": recipe.range_fft.one_sided,
            "remove_dc": recipe.range_fft.remove_dc,
        },
    )
    _check_rd_metadata(
        cube,
        "doppler_fft",
        {
            "n_fft": doppler_size,
            "window": recipe.doppler_fft.window.value,
            "fftshift": recipe.doppler_fft.fftshift,
            "input_axis": recipe.doppler_fft.input_axis,
        },
    )
    _check_rd_metadata(
        cube,
        "tdm_virtual_array",
        None
        if tdm is None
        else {
            "tx_order": list(tdm.tx_order),
            "num_loops": slow_samples,
            "layout": tdm.virtual_layout().as_metadata(),
        },
    )
    _check_rd_metadata(
        cube,
        "tdm_doppler_compensation",
        None
        if tdm is None
        else {
            "tx_order": list(tdm.tx_order),
            "num_tx": tdm.num_tx,
            "num_rx": tdm.geometry.num_rx,
            "fftshift": recipe.doppler_fft.fftshift,
        },
    )
    calibration = recipe.channel_calibration
    _check_rd_metadata(
        cube,
        "virtual_channel_calibration",
        None
        if calibration is None
        else {
            "coefficients": calibration.as_metadata()["coefficients"],
        },
    )


def _check_rd_metadata(cube: RadarCube, stage: str, expected: dict[str, object] | None) -> None:
    actual = cube.metadata.get(stage)
    if expected is None:
        if actual is not None:
            raise ValueError(f"Precomputed RD has unexpected {stage}.")
        return
    if not isinstance(actual, dict):
        raise ValueError(f"Precomputed RD is missing {stage} metadata.")
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Precomputed RD {stage}.{key} does not match the transform recipe.")


def _detection_cube(range_doppler_cube: RadarCube, recipe: DetectionPipeline) -> RadarCube:
    if recipe.virtual_subarray is None:
        return range_doppler_cube
    return select_virtual_subarray(range_doppler_cube, recipe.virtual_subarray)


def _detect(detection_cube: RadarCube, recipe: DetectionPipeline) -> DetectionFrame:
    if recipe.detection_method is DetectionMethod.CFAR:
        if recipe.cfar_detection is None:  # pragma: no cover - recipe validation covers this.
            raise ValueError("CFAR detection requires DetectionPipeline.cfar_detection.")
        if isinstance(recipe.cfar_detection, RangeDopplerCFARSpec):
            detections = detect_range_doppler_cfar(detection_cube, recipe.cfar_detection)
        else:
            detections = detect_cfar(detection_cube, recipe.cfar_detection)
        if recipe.peak_grouping is not None:
            detections = group_detection_peaks(detection_cube, detections, recipe.peak_grouping)
    else:
        if recipe.angle_fft is not None:
            detection_cube = angle_fft(detection_cube, recipe.angle_fft)
        if recipe.peak_detection is None:  # pragma: no cover - recipe validation covers this.
            raise ValueError("Threshold detection requires PeakDetectionSpec.")
        detections = detect_peaks(detection_cube, recipe.peak_detection)
    return detections


def _filter_detections(
    detections: DetectionFrame,
    recipe: DetectionPipeline,
) -> DetectionFrame:
    if recipe.quality_filter is not None:
        return filter_detection_quality(detections, recipe.quality_filter)
    return detections


def _estimate_detection_angles(
    range_doppler_cube: RadarCube,
    detection_cube: RadarCube,
    detections: DetectionFrame,
    recipe: DetectionPipeline,
) -> DetectionFrame:
    if recipe.detection_method is DetectionMethod.CFAR and recipe.angle_fft is not None:
        detections = estimate_candidate_azimuths(detection_cube, detections, recipe.angle_fft)
    if recipe.elevation_subarray is not None:
        if recipe.angle_fft is None or recipe.virtual_subarray is None:
            raise ValueError("Elevation estimation requires calibrated azimuth AoA.")
        detections = estimate_candidate_elevations(
            range_doppler_cube,
            detections,
            recipe.angle_fft,
            azimuth_subarray=recipe.virtual_subarray,
            elevation_subarray=recipe.elevation_subarray,
        )
    return detections


def process_detections_to_point_cloud(
    detections: DetectionFrame,
    recipe: PointCloudPipeline,
) -> PointCloudFrame:
    """Project calibrated detections into Cartesian radar coordinates."""

    return detections_to_point_cloud(detections, recipe.projection)


def process_range_doppler_to_calibrated_point_cloud(
    range_doppler_cube: RadarCube,
    recipe: PointCloudPipeline,
) -> PointCloudFrame:
    """Detect and project one precomputed range-Doppler product."""

    detections = process_range_doppler_to_detections(
        range_doppler_cube,
        recipe.detection,
    )
    return process_detections_to_point_cloud(detections, recipe)


def point_cloud(
    raw: ADCFrame | np.ndarray,
    recipe: PointCloudPipeline,
) -> PointCloudFrame:
    """Run detection and calibrated Cartesian projection recipes."""

    detections = detect(raw, recipe.detection)
    return process_detections_to_point_cloud(detections, recipe)


def range_doppler(
    raw: ADCFrame | np.ndarray,
    recipe: RangeDopplerPipeline,
) -> RadarCube:
    """Decode ADC words and return the complex range-Doppler radar cube."""

    range_cube = _process_adc_to_range_cube(raw, recipe)
    if recipe.remove_static_clutter:
        range_cube = remove_static_clutter(
            range_cube,
            axis=recipe.doppler_fft.input_axis,
        )
    return _process_range_cube_to_range_doppler(range_cube, recipe)


def _process_adc_to_range_cube(
    raw: ADCFrame | np.ndarray,
    recipe: RangeDopplerPipeline,
) -> RadarCube:
    adc_cube = organize_adc_samples(
        raw,
        recipe.decode.adc,
        drop_incomplete=recipe.decode.drop_incomplete,
    )
    range_cube = range_fft(adc_cube, recipe.range_fft)
    if recipe.tdm_virtual_array is not None:
        range_cube = map_tdm_virtual_array(range_cube, recipe.tdm_virtual_array)
    return range_cube


def _process_range_cube_to_range_doppler(
    range_cube: RadarCube,
    recipe: RangeDopplerPipeline,
) -> RadarCube:
    doppler_cube = doppler_fft(range_cube, recipe.doppler_fft)
    if recipe.channel_calibration is not None:
        doppler_cube = apply_virtual_channel_calibration(
            doppler_cube,
            recipe.channel_calibration,
        )
    if recipe.tdm_virtual_array is not None:
        doppler_cube = compensate_tdm_doppler_phase(
            doppler_cube,
            recipe.tdm_virtual_array,
            fftshift=recipe.doppler_fft.fftshift,
        )
    return doppler_cube
