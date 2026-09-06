"""Native ISK default dynamic Capon chain, separate from the RPC baseline.

The supported contract is Toolbox 4.00.00.05 method 1, unity calibration,
TX order 0/1/2, full Blackman range FFT, no dynamic TDM compensation. It uses
host floating arithmetic, not HWA/C674 quantization. Detector parameters are
deliberately frozen; this is not a generic all-mode TI frontend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.config import iwr6843_isk_tdm_virtual_array
from mmwcore.core import (
    ADCDecodeSpec,
    ADCFrame,
    DopplerFFTSpec,
    FFTWindow,
    PointCloudFrame,
    RadarCube,
    RangeDopplerPipeline,
    RangeFFTSpec,
)
from mmwcore.core.spec_adc import _positive_dimension, _positive_real
from mmwcore.dsp.runners import _process_adc_to_range_cube


@dataclass(frozen=True)
class IskCaponSpec:
    """Physical FFT bin spacings; source algorithm defaults remain fixed.

    Velocity is approaching negative, receding positive. The aliased Nyquist
    bin follows TI's positive-end convention. With padding, velocity spacing
    must describe the padded FFT. Range FFT size equals ADC sample count.
    """

    range_resolution_m: float
    velocity_resolution_mps: float
    doppler_bins: int

    def __post_init__(self) -> None:
        for name in ("range_resolution_m", "velocity_resolution_mps"):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name=name))
        n = _positive_dimension(self.doppler_bins, name="doppler_bins")
        if n < 2 or n > 65536 or n & (n - 1):
            raise ValueError("doppler_bins must be a power of two in [2, 65536].")
        object.__setattr__(self, "doppler_bins", n)


@dataclass(frozen=True)
class CaponDetection:
    range_bin: int
    azimuth_bin: int
    elevation_bin: int
    elevation_interpolated_bin: float
    doppler_bin: int
    xyz_m: tuple[float, float, float]
    velocity_mps: float
    snr: float
    range_noise: float
    ra_power: float
    elevation_power: float


@dataclass(frozen=True)
class CaponFrame:
    """All accepted points plus the unthresholded RA power and drop counters.

    RA axes are (TI nu, range), with nu=-sin(70°)+i*sin(70°)*.75/70;
    positive nu is project-left. XYZ uses project forward/right/up.
    No ROI, point deduplication, tracker filtering or SNR packing is applied.
    """

    points: PointCloudFrame
    ra_power: NDArray[np.float32]
    detections: tuple[CaponDetection, ...]
    diagnostics: dict[str, int]


def isk_capon_from_range_cube(cube: RadarCube, spec: IskCaponSpec) -> CaponFrame:
    """Process one uncompensated (frame, loop, virtual_rx, range_bin) cube.

    Caller owns TX0/1/2 and unity-calibration provenance. Data must precede
    Doppler processing; this function removes the slow-time mean itself.
    Use :func:`isk_capon_point_cloud` for the checked ADC-to-points path.
    """
    if not isinstance(spec, IskCaponSpec):
        raise TypeError("spec must be IskCaponSpec.")
    if cube.axes != ("frame", "loop", "virtual_rx", "range_bin") or cube.data.shape[:1] != (1,):
        raise ValueError("Capon requires one (frame, loop, virtual_rx, range_bin) cube.")
    power, encoded = _native.isk_capon_complex(
        np.ascontiguousarray(cube.data[0], dtype=np.complex64),
        spec.range_resolution_m,
        spec.velocity_resolution_mps,
        spec.doppler_bins,
    )
    report = json.loads(encoded)
    detections = tuple(
        CaponDetection(**{**row, "xyz_m": tuple(row["xyz_m"])}) for row in report["detections"]
    )
    data = np.asarray(
        [(*d.xyz_m, d.velocity_mps, d.snr) for d in detections], dtype=np.float32
    ).reshape(-1, 5)
    if not np.isfinite(data).all():
        raise ValueError("Capon points exceed finite float32 range.")
    points = PointCloudFrame(
        data,
        channels=("x", "y", "z", "velocity", "snr"),
        frame_id=cube.frame_id,
        timestamp=cube.timestamp,
        source=cube.source,
        coordinate_frame="sensor_forward_lateral_up",
        units={"x": "m", "y": "m", "z": "m", "velocity": "m/s", "snr": "linear_ratio"},
        metadata={
            **cube.metadata,
            "frontend": "isk_capon_dynamic_v1",
            "source_contract": "toolbox_4_00_00_05_method1_raCAAll_default",
            "numeric_contract": "host_float_unquantized_snr",
            "tdm_doppler_compensation": False,
            "nyquist_velocity_endpoint": "positive",
        },
    )
    return CaponFrame(points, power, detections, report["diagnostics"])


def isk_capon_point_cloud(
    raw: ADCFrame | NDArray[np.int16],
    decode: ADCDecodeSpec,
    spec: IskCaponSpec,
    *,
    tx_order: tuple[int, ...],
) -> CaponFrame:
    """ADC → full Blackman range FFT → dynamic Capon → CFAR → angle/Doppler.

    ADC IQ layout is taken from ``decode``. Explicit TX order prevents using
    the historical preset order by accident. No hardware or tracker is owned
    by this function. StaticRetention/BPM/nonunity calibration are unsupported.
    """
    if not isinstance(decode, ADCDecodeSpec) or not isinstance(spec, IskCaponSpec):
        raise TypeError("Expected ADCDecodeSpec and IskCaponSpec.")
    adc = decode.adc
    if tx_order != (0, 1, 2) or adc.num_rx != 4 or adc.num_chirps % 3:
        raise ValueError("ISK Capon requires TX order (0, 1, 2), four RX and complete TDM loops.")
    if adc.num_samples < 64 or adc.num_samples & (adc.num_samples - 1):
        raise ValueError("ISK Capon requires a power-of-two ADC sample count >= 64.")
    if decode.drop_incomplete:
        raise ValueError("ISK Capon requires complete ADC frames.")
    recipe = RangeDopplerPipeline(
        decode=decode,
        range_fft=RangeFFTSpec(window=FFTWindow.BLACKMAN),
        doppler_fft=DopplerFFTSpec(n_fft=spec.doppler_bins, input_axis="loop"),
        tdm_virtual_array=iwr6843_isk_tdm_virtual_array(tx_order=tx_order),
    )
    return isk_capon_from_range_cube(_process_adc_to_range_cube(raw, recipe), spec)
