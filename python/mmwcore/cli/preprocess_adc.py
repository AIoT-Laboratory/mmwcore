"""CLI for preprocessing ADC binary files into point-cloud arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmwcore.cli._args import (
    non_negative_float,
    non_negative_int,
    positive_float,
    positive_int,
)
from mmwcore.cli._preprocess_adc_output import write_preprocess_outputs
from mmwcore.cli._preprocess_adc_spec import build_point_cloud_recipe
from mmwcore.core import ADCComplexLayout, DetectionMethod, FFTWindow
from mmwcore.dsp import process_adc_file_to_calibrated_point_cloud


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess an ADC .bin file into a point cloud.")
    parser.add_argument("input", type=Path, help="Input int16 ADC binary file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy point-cloud path.")
    parser.add_argument("--metadata-output", type=Path, help="Optional output JSON metadata path.")
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        help="Optional JSONL artifact manifest path.",
    )
    parser.add_argument(
        "--sample-id",
        help="Sample id for artifact metadata. Defaults to frame id or input stem.",
    )
    parser.add_argument("--preset", choices=["iwr6843"], help="Optional radar profile preset.")
    parser.add_argument(
        "--ti-cfg",
        type=Path,
        help="Use a TI mmWave SDK CLI config to derive the ADC frame shape.",
    )
    parser.add_argument("--num-chirps", type=positive_int)
    parser.add_argument("--num-rx", type=positive_int)
    parser.add_argument("--num-samples", type=positive_int)
    parser.add_argument(
        "--adc-layout",
        choices=[layout.value for layout in ADCComplexLayout],
        default=ADCComplexLayout.IQ_INTERLEAVED.value,
    )
    parser.add_argument("--range-n-fft", type=positive_int)
    parser.add_argument("--range-one-sided", action="store_true")
    parser.add_argument("--doppler-n-fft", type=positive_int)
    parser.add_argument("--no-doppler-shift", action="store_true")
    parser.add_argument("--angle-fft", action="store_true")
    parser.add_argument("--angle-n-fft", type=positive_int)
    parser.add_argument("--angle-no-shift", action="store_true")
    parser.add_argument("--angle-input-axis", default="rx")
    parser.add_argument("--angle-output-axis", default="azimuth_bin")
    parser.add_argument(
        "--azimuth-peak-radius",
        type=non_negative_int,
        default=1,
        help="Angle-bin local-maximum radius; use 0 for a threshold mask.",
    )
    parser.add_argument(
        "--non-strict-azimuth-peaks",
        action="store_true",
        help="Keep equal-valued angle plateaus instead of requiring strict maxima.",
    )
    parser.add_argument("--virtual-antennas", type=positive_int)
    parser.add_argument("--virtual-spacing-wavelengths", type=positive_float, default=0.5)
    parser.add_argument(
        "--window",
        choices=[window.value for window in FFTWindow],
        default=FFTWindow.NONE.value,
    )
    parser.add_argument(
        "--detector",
        choices=[method.value for method in DetectionMethod],
        default=DetectionMethod.THRESHOLD.value,
    )
    parser.add_argument("--threshold", type=non_negative_float)
    parser.add_argument("--cfar-training-cells", type=positive_int)
    parser.add_argument("--cfar-guard-cells", type=non_negative_int)
    parser.add_argument("--cfar-threshold-scale", type=non_negative_float)
    parser.add_argument("--aggregate-rx", choices=["max", "sum", "mean"], default="max")
    parser.add_argument("--range-resolution-m", type=positive_float)
    parser.add_argument("--doppler-resolution-mps", type=positive_float)
    parser.add_argument("--center-doppler", action="store_true")
    parser.add_argument("--doppler-bins", type=positive_int)
    parser.add_argument("--drop-incomplete-adc", action="store_true")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--frame-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recipe = build_point_cloud_recipe(args)
    point_cloud = process_adc_file_to_calibrated_point_cloud(
        args.input,
        recipe,
        frame_id=args.frame_id,
        mmap=args.mmap,
    )
    output = write_preprocess_outputs(point_cloud, args=args, recipe=recipe)
    if output.artifact is not None:
        print(f"saved_artifact_manifest={args.artifact_manifest}")
        print(f"sample_id={output.artifact.sample_id}")
    print(f"saved_point_cloud={args.output}")
    if output.metadata_path is not None:
        print(f"saved_metadata={output.metadata_path}")
    print(f"num_points={point_cloud.num_points}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
