"""Build ADC preprocessing pipeline specs from CLI arguments."""

from __future__ import annotations

import argparse

from mmwcore.config import iwr6843_profile, parse_ti_cli_config_file
from mmwcore.core import (
    ADCComplexLayout,
    ADCDecodeRecipe,
    ADCFrameSpec,
    AngleFFTSpec,
    CFARDetectionSpec,
    DetectionMethod,
    DetectionRecipe,
    DopplerFFTSpec,
    FFTWindow,
    PeakDetectionSpec,
    PointCloudProjectionSpec,
    PointCloudRecipe,
    RangeDopplerRecipe,
    RangeFFTSpec,
    VirtualAntennaLayout,
)


def build_point_cloud_recipe(args: argparse.Namespace) -> PointCloudRecipe:
    window = FFTWindow(args.window)
    detection_method = DetectionMethod(args.detector)
    profile = _profile_from_preset(args.preset)
    adc = _adc_spec_from_args(args, profile=profile)
    point_cloud = _point_cloud_spec_from_args(args, profile=profile)
    angle_fft = _angle_fft_spec_from_args(args, window)
    peak_detection = _peak_detection_spec_from_args(args, detection_method)
    cfar_detection = _cfar_detection_spec_from_args(args, detection_method)
    if angle_fft is None:
        raise SystemExit("point-cloud preprocessing requires --angle-fft")
    if angle_fft.virtual_layout is None:
        raise SystemExit("point-cloud preprocessing requires --virtual-antennas")
    detection = DetectionRecipe(
        transform=RangeDopplerRecipe(
            decode=ADCDecodeRecipe(adc, drop_incomplete=args.drop_incomplete_adc),
            range_fft=RangeFFTSpec(
                n_fft=args.range_n_fft,
                window=window,
                one_sided=args.range_one_sided,
            ),
            doppler_fft=DopplerFFTSpec(
                n_fft=args.doppler_n_fft,
                window=window,
                fftshift=not args.no_doppler_shift,
            ),
        ),
        detection_method=detection_method,
        angle_fft=angle_fft,
        peak_detection=peak_detection,
        cfar_detection=cfar_detection,
    )
    return PointCloudRecipe(detection=detection, projection=point_cloud)


def _profile_from_preset(preset: str | None):
    if preset == "iwr6843":
        return iwr6843_profile()
    return None


def _adc_spec_from_args(args: argparse.Namespace, *, profile) -> ADCFrameSpec:
    num_chirps = args.num_chirps
    num_rx = args.num_rx
    num_samples = args.num_samples
    if args.ti_cfg is not None:
        if profile is not None:
            raise SystemExit("--ti-cfg cannot be combined with --preset")
        if any(value is not None for value in (num_chirps, num_rx, num_samples)):
            raise SystemExit("--ti-cfg cannot be combined with explicit ADC shape arguments")
        try:
            summary = parse_ti_cli_config_file(args.ti_cfg)
            if summary.num_tx != 1:
                raise SystemExit(
                    "multi-Tx point-cloud preprocessing requires an explicit TDM recipe"
                )
            return summary.to_adc_frame_spec(
                layout=ADCComplexLayout(args.adc_layout),
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--ti-cfg: {exc}") from exc

    if profile is not None:
        if profile.num_tx != 1:
            raise SystemExit("multi-Tx point-cloud preprocessing requires an explicit TDM recipe")
        num_chirps = num_chirps if num_chirps is not None else profile.chirps_per_frame
        num_rx = num_rx if num_rx is not None else profile.num_rx
        num_samples = num_samples if num_samples is not None else profile.num_adc_samples

    values = (num_chirps, num_rx, num_samples)
    if any(value is None for value in values):
        raise SystemExit("--num-chirps, --num-rx, and --num-samples are required without a preset")

    return ADCFrameSpec(
        num_chirps=num_chirps,
        num_rx=num_rx,
        num_samples=num_samples,
        layout=ADCComplexLayout(args.adc_layout),
    )


def _point_cloud_spec_from_args(args: argparse.Namespace, *, profile) -> PointCloudProjectionSpec:
    range_resolution_m = args.range_resolution_m
    doppler_resolution_mps = args.doppler_resolution_mps
    doppler_bins = args.doppler_bins
    if profile is not None:
        range_resolution_m = (
            range_resolution_m if range_resolution_m is not None else profile.range_resolution_m
        )
        doppler_resolution_mps = (
            doppler_resolution_mps
            if doppler_resolution_mps is not None
            else profile.velocity_resolution_mps
        )
        doppler_bins = doppler_bins if doppler_bins is not None else profile.num_chirps_per_tx

    return PointCloudProjectionSpec(
        range_resolution_m=range_resolution_m or 1.0,
        doppler_resolution_mps=doppler_resolution_mps or 1.0,
        center_doppler=args.center_doppler,
        doppler_bins=doppler_bins,
    )


def _angle_fft_spec_from_args(
    args: argparse.Namespace,
    window: FFTWindow,
) -> AngleFFTSpec | None:
    if not args.angle_fft:
        return None

    virtual_layout = None
    if args.virtual_antennas is not None:
        virtual_layout = VirtualAntennaLayout.uniform_linear(
            args.virtual_antennas,
            spacing_wavelengths=args.virtual_spacing_wavelengths,
        )

    return AngleFFTSpec(
        n_fft=args.angle_n_fft,
        window=window,
        fftshift=not args.angle_no_shift,
        input_axis=args.angle_input_axis,
        output_axis=args.angle_output_axis,
        virtual_layout=virtual_layout,
    )


def _peak_detection_spec_from_args(
    args: argparse.Namespace,
    detection_method: DetectionMethod,
) -> PeakDetectionSpec | None:
    if detection_method is DetectionMethod.CFAR:
        return None
    if args.threshold is None:
        raise SystemExit("--threshold is required when --detector=threshold")
    return PeakDetectionSpec(
        threshold=args.threshold,
        aggregate_rx=args.aggregate_rx,
        azimuth_peak_radius=args.azimuth_peak_radius,
        azimuth_peak_strict=not args.non_strict_azimuth_peaks,
    )


def _cfar_detection_spec_from_args(
    args: argparse.Namespace,
    detection_method: DetectionMethod,
) -> CFARDetectionSpec | None:
    if detection_method is not DetectionMethod.CFAR:
        return None

    missing = [
        flag
        for flag, value in (
            ("--cfar-training-cells", args.cfar_training_cells),
            ("--cfar-guard-cells", args.cfar_guard_cells),
            ("--cfar-threshold-scale", args.cfar_threshold_scale),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"{', '.join(missing)} required when --detector=cfar")

    return CFARDetectionSpec(
        training_cells=args.cfar_training_cells,
        guard_cells=args.cfar_guard_cells,
        threshold_scale=args.cfar_threshold_scale,
        aggregate_rx=args.aggregate_rx,
    )
