"""Render the real ADC figures embedded in the project README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmwcore.core import ADCComplexLayout, ADCFrameSpec, FFTWindow, RangeFFTSpec, RawADCFrame
from mmwcore.dsp import organize_adc_samples, range_fft

FRAME_PERIOD_S = 0.01
ADC_SPEC = ADCFrameSpec(
    num_chirps=2,
    num_rx=4,
    num_samples=128,
    layout=ADCComplexLayout.GROUP2_I_THEN_Q,
)
RANGE_SPEC = RangeFFTSpec(
    n_fft=128,
    window=FFTWindow.HANN,
    one_sided=True,
    remove_dc=True,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source int16 ADC capture.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Destination for the two README PNG files.",
    )
    return parser.parse_args()


def _to_db(values: np.ndarray) -> np.ndarray:
    floor = np.finfo(np.float32).tiny
    return 20.0 * np.log10(np.maximum(values, floor))


def _display_limits(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, (1.0, 99.5))
    return float(low), float(high)


def _decode_range_cube(input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.memmap(input_path, dtype=np.int16, mode="r")
    adc_cube = organize_adc_samples(
        RawADCFrame(raw, source=str(input_path)),
        ADC_SPEC,
    )
    ranges = range_fft(adc_cube, RANGE_SPEC).data
    return adc_cube.data, ranges


def _render_range_time(range_values: np.ndarray, output_path: Path) -> np.ndarray:
    raw_map = _to_db(np.abs(range_values).mean(axis=(1, 2))).T
    background = range_values.mean(axis=0, keepdims=True)
    dynamic_values = range_values - background
    dynamic_map = _to_db(np.abs(dynamic_values).mean(axis=(1, 2))).T
    duration_s = range_values.shape[0] * FRAME_PERIOD_S
    extent = (0.0, duration_s, 0, range_values.shape[-1])

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True, constrained_layout=True)
    panels = (
        (axes[0], raw_map, "Range-Time magnitude"),
        (axes[1], dynamic_map, "Temporal-background-suppressed magnitude"),
    )
    for axis, values, title in panels:
        vmin, vmax = _display_limits(values)
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_ylabel("Range bin")
        figure.colorbar(image, ax=axis, label="Magnitude (dB)", pad=0.015)
    axes[-1].set_xlabel("Time (s)")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return dynamic_values


def _representative_frame(dynamic_values: np.ndarray) -> int:
    energy = np.square(np.abs(dynamic_values)).mean(axis=(1, 2, 3))
    ordered = np.argsort(energy)
    return int(ordered[round(0.98 * (ordered.size - 1))])


def _render_frame_diagnostics(
    adc: np.ndarray,
    range_values: np.ndarray,
    dynamic_values: np.ndarray,
    output_path: Path,
) -> None:
    frame_index = _representative_frame(dynamic_values)
    raw_frame = adc[frame_index, 0, 0]
    spectra = _to_db(np.abs(range_values[frame_index]).mean(axis=0))

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)
    axes[0].plot(raw_frame.real, label="I", linewidth=1.1)
    axes[0].plot(raw_frame.imag, label="Q", linewidth=1.1)
    axes[0].set_title("Raw ADC, chirp 0 / RX 0", loc="left", fontweight="semibold")
    axes[0].set_xlabel("ADC sample")
    axes[0].set_ylabel("Amplitude (LSB)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, spectra.shape[0]))
    for receiver, (spectrum, color) in enumerate(zip(spectra, colors, strict=True)):
        axes[1].plot(spectrum, color=color, linewidth=1.2, label=f"RX {receiver}")
    axes[1].set_title(
        f"Range spectrum, frame {frame_index}",
        loc="left",
        fontweight="semibold",
    )
    axes[1].set_xlabel("Range bin")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].legend(frameon=False, ncols=2)
    axes[1].grid(alpha=0.2)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "font.size": 9,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adc, ranges = _decode_range_cube(args.input)
    dynamic = _render_range_time(ranges, args.output_dir / "adc-range-time.png")
    _render_frame_diagnostics(
        adc,
        ranges,
        dynamic,
        args.output_dir / "adc-frame-diagnostics.png",
    )
    print(f"frames={adc.shape[0]}")
    print(f"duration_s={adc.shape[0] * FRAME_PERIOD_S:.2f}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
