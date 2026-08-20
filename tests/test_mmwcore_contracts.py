from __future__ import annotations

from dataclasses import replace
from sys import maxsize
from typing import cast

import numpy as np
import pytest

import mmwcore
from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import (
    ADCComplexLayout,
    ADCDecodeRecipe,
    ADCFrameSpec,
    AngleFFTSpec,
    CascadeADCFrameSpec,
    CFARDetectionSpec,
    DetectionFrame,
    DetectionMethod,
    DetectionQualitySpec,
    DetectionRecipe,
    DopplerFFTSpec,
    FFTWindow,
    PeakDetectionSpec,
    PeakGroupingSpec,
    PlanarAngleFFTSpec,
    PlanarApertureLayout,
    PointCloudFrame,
    PointCloudProjectionSpec,
    PointCloudRecipe,
    RadarCube,
    RangeDopplerRecipe,
    RangeFFTSpec,
    RawADCFrame,
    VirtualAntennaLayout,
)
from mmwcore.dsp import (
    detect_peaks,
    detections_to_point_cloud,
    doppler_fft,
    organize_adc_samples,
    process_adc_file_to_calibrated_point_cloud,
    process_adc_file_to_range_doppler,
    process_adc_to_calibrated_point_cloud,
    process_adc_to_detections,
    process_adc_to_range_doppler,
    range_fft,
)
from mmwcore.io import ADCFileFrameReader, load_adc_cube, load_adc_file

type FFTSpec = RangeFFTSpec | DopplerFFTSpec | AngleFFTSpec | PlanarAngleFFTSpec

_FFT_SIZE_FIELDS: tuple[tuple[FFTSpec, str], ...] = (
    (RangeFFTSpec(), "n_fft"),
    (DopplerFFTSpec(), "n_fft"),
    (AngleFFTSpec(), "n_fft"),
    (PlanarAngleFFTSpec(), "azimuth_n_fft"),
    (PlanarAngleFFTSpec(), "elevation_n_fft"),
)

_FFT_BOOL_FIELDS: tuple[tuple[FFTSpec, str], ...] = (
    (RangeFFTSpec(), "one_sided"),
    (RangeFFTSpec(), "remove_dc"),
    (DopplerFFTSpec(), "fftshift"),
    (AngleFFTSpec(), "fftshift"),
    (PlanarAngleFFTSpec(), "fftshift"),
)

_FFT_AXIS_FIELDS: tuple[tuple[FFTSpec, str], ...] = (
    (DopplerFFTSpec(), "input_axis"),
    (AngleFFTSpec(), "input_axis"),
    (AngleFFTSpec(), "output_axis"),
    (PlanarAngleFFTSpec(), "azimuth_input_axis"),
    (PlanarAngleFFTSpec(), "elevation_input_axis"),
    (PlanarAngleFFTSpec(), "azimuth_output_axis"),
    (PlanarAngleFFTSpec(), "elevation_output_axis"),
)


def test_mmwcore_import_is_lightweight() -> None:
    assert mmwcore.__version__ == "0.6.0"


def test_raw_adc_frame_normalizes_representable_integers_to_int16() -> None:
    frame = RawADCFrame(np.arange(8, dtype=np.int32), frame_id="f0")

    assert frame.samples.dtype == np.int16
    assert frame.samples.shape == (8,)
    assert frame.frame_id == "f0"


@pytest.mark.parametrize(
    ("samples", "error", "message"),
    [
        (np.array([1.0, 2.0]), TypeError, "integer ADC values"),
        (np.array([np.iinfo(np.int16).min - 1]), ValueError, "outside the int16 range"),
        (np.array([np.iinfo(np.int16).max + 1]), ValueError, "outside the int16 range"),
    ],
)
def test_raw_adc_frame_rejects_lossy_int16_conversion(
    samples: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        RawADCFrame(samples)


def test_raw_adc_frame_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RawADCFrame(np.array([], dtype=np.int16))


def test_radar_cube_requires_axes_to_match_dimensions() -> None:
    cube = RadarCube(np.ones((1, 2, 3, 4), dtype=np.float32))

    assert cube.data.dtype == np.complex64
    assert cube.axes == ("frame", "chirp", "rx", "sample")

    with pytest.raises(ValueError, match="dimensions"):
        RadarCube(np.ones((2, 3), dtype=np.complex64))


@pytest.mark.parametrize("axes", [("sample", "sample"), ("sample", "")])
def test_radar_cube_requires_named_unique_axes(axes: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="unique|non-empty"):
        RadarCube(np.ones((2, 3), dtype=np.complex64), axes=axes)


def test_detection_frame_validates_channels() -> None:
    detections = DetectionFrame(
        np.zeros((2, 5), dtype=np.float64),
        channels=("range", "doppler", "azimuth", "elevation", "snr"),
    )

    assert detections.detections.dtype == np.float32
    assert detections.channels[-1] == "snr"

    with pytest.raises(ValueError, match="channels length"):
        DetectionFrame(np.zeros((2, 3)), channels=("range", "doppler"))


def test_point_cloud_frame_rejects_non_xyz_prefix() -> None:
    with pytest.raises(ValueError, match="must start"):
        PointCloudFrame(np.zeros((1, 3)), channels=("range", "azimuth", "elevation"))


@pytest.mark.parametrize("invalid", [True, 1.5])
@pytest.mark.parametrize("field", ["num_chirps", "num_rx", "num_samples"])
def test_adc_frame_spec_rejects_non_integral_dimensions(
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(TypeError, match=rf"ADCFrameSpec\.{field} must be an integer"):
        replace(ADCFrameSpec(1, 1, 2), **{field: invalid})


@pytest.mark.parametrize("invalid", [True, 1.5])
@pytest.mark.parametrize(
    "field",
    ["num_samples", "num_loops", "num_tx", "num_rx_per_device"],
)
def test_cascade_adc_frame_spec_rejects_non_integral_dimensions(
    field: str,
    invalid: object,
) -> None:
    spec = CascadeADCFrameSpec(2, 1, 1, 4, ("device-0",))
    with pytest.raises(TypeError, match=rf"CascadeADCFrameSpec\.{field} must be an integer"):
        replace(spec, **{field: invalid})


def test_adc_integer_contracts_normalize_safe_numpy_integers() -> None:
    frame = ADCFrameSpec(
        cast(int, np.int64(2)),
        cast(int, np.uint64(3)),
        cast(int, np.int32(4)),
    )
    cascade = CascadeADCFrameSpec(
        cast(int, np.uint16(4)),
        cast(int, np.int64(3)),
        cast(int, np.uint8(2)),
        cast(int, np.int32(4)),
        ("device-0", "device-1"),
    )
    aperture = PlanarApertureLayout(
        ((cast(int, np.int64(0)), cast(int, np.uint64(1))),),
    )
    linear = VirtualAntennaLayout.uniform_linear(cast(int, np.int64(2)))

    assert (frame.num_chirps, frame.num_rx, frame.num_samples) == (2, 3, 4)
    assert all(type(value) is int for value in (frame.num_chirps, frame.num_rx, frame.num_samples))
    assert (
        cascade.num_samples,
        cascade.num_loops,
        cascade.num_tx,
        cascade.num_rx_per_device,
    ) == (4, 3, 2, 4)
    assert all(
        type(value) is int
        for value in (
            cascade.num_samples,
            cascade.num_loops,
            cascade.num_tx,
            cascade.num_rx_per_device,
        )
    )
    assert aperture.grid_indices == ((0, 1),)
    assert all(type(value) is int for value in aperture.grid_indices[0])
    assert linear.num_antennas == 2


def test_adc_integer_contracts_reject_platform_overflow_before_allocation() -> None:
    too_large = maxsize + 1

    with pytest.raises(OverflowError, match="num_chirps.*platform index"):
        ADCFrameSpec(too_large, 1, 1)
    with pytest.raises(OverflowError, match="raw frame size.*platform index"):
        ADCFrameSpec(maxsize, 1, 1)
    with pytest.raises(OverflowError, match="num_samples.*platform index"):
        CascadeADCFrameSpec(too_large, 1, 1, 1, ("device-0",))
    with pytest.raises(OverflowError, match="per-device frame size.*platform index"):
        CascadeADCFrameSpec(maxsize, 1, 1, 1, ("device-0",))
    with pytest.raises(OverflowError, match="num_antennas.*platform index"):
        VirtualAntennaLayout.uniform_linear(too_large)
    with pytest.raises(ValueError, match="integers within the platform range"):
        PlanarApertureLayout(((too_large, 0),))


@pytest.mark.parametrize("invalid", [True, 1.5])
def test_uniform_linear_rejects_non_integral_antenna_counts(invalid: object) -> None:
    with pytest.raises(TypeError, match="num_antennas must be an integer"):
        VirtualAntennaLayout.uniform_linear(cast(int, invalid))


@pytest.mark.parametrize(
    ("spec", "field"),
    _FFT_SIZE_FIELDS,
    ids=[f"{type(spec).__name__}.{field}" for spec, field in _FFT_SIZE_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [pytest.param(True, id="bool"), pytest.param(1.5, id="float")],
)
def test_fft_specs_reject_non_integral_sizes(
    spec: FFTSpec,
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{type(spec).__name__}\.{field} must be an integer",
    ):
        replace(spec, **{field: invalid})


@pytest.mark.parametrize(
    ("spec", "field"),
    _FFT_SIZE_FIELDS,
    ids=[f"{type(spec).__name__}.{field}" for spec, field in _FFT_SIZE_FIELDS],
)
def test_fft_specs_normalize_numpy_integer_sizes(spec: FFTSpec, field: str) -> None:
    normalized = replace(spec, **{field: np.int64(8)})

    value = getattr(normalized, field)
    assert value == 8
    assert type(value) is int


@pytest.mark.parametrize(
    ("spec", "field"),
    _FFT_SIZE_FIELDS,
    ids=[f"{type(spec).__name__}.{field}" for spec, field in _FFT_SIZE_FIELDS],
)
def test_fft_specs_preserve_positive_size_domains(spec: FFTSpec, field: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{type(spec).__name__}\.{field} must be positive",
    ):
        replace(spec, **{field: 0})


@pytest.mark.parametrize(
    ("spec", "field"),
    _FFT_SIZE_FIELDS,
    ids=[f"{type(spec).__name__}.{field}" for spec, field in _FFT_SIZE_FIELDS],
)
def test_fft_specs_reject_platform_size_overflow(spec: FFTSpec, field: str) -> None:
    with pytest.raises(
        OverflowError,
        match=rf"{type(spec).__name__}\.{field} must fit the platform index range",
    ):
        replace(spec, **{field: maxsize + 1})


@pytest.mark.parametrize(
    ("spec", "field"),
    _FFT_BOOL_FIELDS,
    ids=[f"{type(spec).__name__}.{field}" for spec, field in _FFT_BOOL_FIELDS],
)
@pytest.mark.parametrize(
    "invalid",
    [pytest.param(1, id="integer"), pytest.param(np.bool_(True), id="numpy-bool")],
)
def test_fft_specs_require_boolean_policy_fields(
    spec: FFTSpec,
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{type(spec).__name__}\.{field} must be a bool",
    ):
        replace(spec, **{field: invalid})


@pytest.mark.parametrize(
    ("spec", "field"),
    _FFT_AXIS_FIELDS,
    ids=[f"{type(spec).__name__}.{field}" for spec, field in _FFT_AXIS_FIELDS],
)
@pytest.mark.parametrize(
    ("invalid", "error", "message"),
    [
        pytest.param(1, TypeError, "must be a string", id="truthy-integer"),
        pytest.param("", ValueError, "must not be empty", id="empty"),
    ],
)
def test_fft_specs_require_nonempty_string_axes(
    spec: FFTSpec,
    field: str,
    invalid: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(
        error,
        match=rf"{type(spec).__name__}\.{field} {message}",
    ):
        replace(spec, **{field: invalid})


def test_organize_adc_samples_interleaved_iq_layout() -> None:
    spec = ADCFrameSpec(num_chirps=1, num_rx=2, num_samples=2)
    raw = RawADCFrame(
        np.array([1, 10, 2, 20, 3, 30, 4, 40], dtype=np.int16),
        frame_id="frame-0",
        source="fixture.bin",
    )

    cube = organize_adc_samples(raw, spec)

    assert cube.data.shape == (1, 1, 2, 2)
    assert cube.frame_id == "frame-0"
    assert cube.source == "fixture.bin"
    np.testing.assert_array_equal(
        cube.data,
        np.array([[[[1 + 10j, 2 + 20j], [3 + 30j, 4 + 40j]]]], dtype=np.complex64),
    )


def test_organize_adc_samples_canonicalizes_non_contiguous_raw_input() -> None:
    raw = np.array([1, 999, 10, 999, 2, 999, 20, 999], dtype=np.int16)[::2]
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    cube = organize_adc_samples(raw, spec)

    np.testing.assert_array_equal(
        cube.data,
        np.array([[[[1 + 10j, 2 + 20j]]]], dtype=np.complex64),
    )


def test_organize_adc_samples_sample_i_then_q_layout() -> None:
    spec = ADCFrameSpec(
        num_chirps=1,
        num_rx=2,
        num_samples=2,
        layout=ADCComplexLayout.SAMPLE_I_THEN_Q,
    )
    raw = np.array([1, 3, 10, 30, 2, 4, 20, 40], dtype=np.int16)

    cube = organize_adc_samples(raw, spec)

    np.testing.assert_array_equal(
        cube.data,
        np.array([[[[1 + 10j, 2 + 20j], [3 + 30j, 4 + 40j]]]], dtype=np.complex64),
    )


def test_organize_adc_samples_group2_i_then_q_layout() -> None:
    spec = ADCFrameSpec(
        num_chirps=1,
        num_rx=2,
        num_samples=4,
        layout=ADCComplexLayout.GROUP2_I_THEN_Q,
    )
    raw = np.array(
        [1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60, 7, 8, 70, 80],
        dtype=np.int16,
    )

    cube = organize_adc_samples(raw, spec)

    np.testing.assert_array_equal(
        cube.data,
        np.array(
            [[[[1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j], [5 + 50j, 6 + 60j, 7 + 70j, 8 + 80j]]]],
            dtype=np.complex64,
        ),
    )


def test_organize_adc_samples_group2_requires_even_sample_count() -> None:
    spec = ADCFrameSpec(
        num_chirps=1,
        num_rx=1,
        num_samples=3,
        layout=ADCComplexLayout.GROUP2_I_THEN_Q,
    )

    with pytest.raises(ValueError, match="even num_samples"):
        organize_adc_samples(np.arange(6, dtype=np.int16), spec)


def test_organize_adc_samples_group4_i_then_q_layout() -> None:
    spec = ADCFrameSpec(
        num_chirps=1,
        num_rx=2,
        num_samples=4,
        layout=ADCComplexLayout.GROUP4_I_THEN_Q,
    )
    raw = np.array(
        [1, 5, 2, 6, 10, 50, 20, 60, 3, 7, 4, 8, 30, 70, 40, 80],
        dtype=np.int16,
    )

    cube = organize_adc_samples(raw, spec)

    np.testing.assert_array_equal(
        cube.data,
        np.array(
            [[[[1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j], [5 + 50j, 6 + 60j, 7 + 70j, 8 + 80j]]]],
            dtype=np.complex64,
        ),
    )


def test_organize_adc_samples_group4_can_span_chirp_boundaries() -> None:
    spec = ADCFrameSpec(
        num_chirps=2,
        num_rx=1,
        num_samples=2,
        layout=ADCComplexLayout.GROUP4_I_THEN_Q,
    )

    cube = organize_adc_samples(np.array([1, 2, 3, 4, 10, 20, 30, 40]), spec)

    np.testing.assert_array_equal(
        cube.data,
        np.array([[[[1 + 10j, 2 + 20j]], [[3 + 30j, 4 + 40j]]]], dtype=np.complex64),
    )


def test_organize_adc_samples_group4_requires_four_complex_value_frame_alignment() -> None:
    spec = ADCFrameSpec(
        num_chirps=1,
        num_rx=2,
        num_samples=1,
        layout=ADCComplexLayout.GROUP4_I_THEN_Q,
    )

    with pytest.raises(
        ValueError,
        match="num_chirps \\* num_rx \\* num_samples to be divisible by 4",
    ):
        organize_adc_samples(np.arange(4, dtype=np.int16), spec)


def test_organize_adc_samples_rejects_incomplete_frames_by_default() -> None:
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match="whole number of frames"):
        organize_adc_samples(np.array([1, 2, 3, 4, 5], dtype=np.int16), spec)


def test_organize_adc_samples_can_drop_incomplete_tail() -> None:
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    cube = organize_adc_samples(
        np.array([1, 10, 2, 20, 999], dtype=np.int16),
        spec,
        drop_incomplete=True,
    )

    np.testing.assert_array_equal(
        cube.data,
        np.array([[[[1 + 10j, 2 + 20j]]]], dtype=np.complex64),
    )


def test_load_adc_file_returns_raw_adc_frame(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.array([1, 2, 3, 4], dtype=np.int16).tofile(adc_path)

    raw = load_adc_file(adc_path, frame_id="file-0", profile={"device": "fixture"})

    assert raw.frame_id == "file-0"
    assert raw.source == str(adc_path)
    assert raw.profile == {"device": "fixture"}
    np.testing.assert_array_equal(raw.samples, np.array([1, 2, 3, 4], dtype=np.int16))


def test_load_adc_file_supports_memmap(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.array([1, 2, 3, 4], dtype=np.int16).tofile(adc_path)

    raw = load_adc_file(adc_path, mmap=True)

    assert isinstance(raw.samples, np.memmap)
    np.testing.assert_array_equal(raw.samples, np.array([1, 2, 3, 4], dtype=np.int16))


def test_adc_file_frame_reader_maps_requested_frame_and_timestamp(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.arange(12, dtype=np.int16).tofile(adc_path)
    reader = ADCFileFrameReader(
        adc_path,
        ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2),
        frame_periodicity_s=0.1,
        profile={"device": "fixture"},
    )

    frame = reader.read_frame(2)

    assert reader.num_frames == 3
    assert isinstance(frame.samples, np.memmap)
    np.testing.assert_array_equal(frame.samples, np.array([8, 9, 10, 11], dtype=np.int16))
    assert frame.frame_id == 2
    assert frame.timestamp == pytest.approx(0.2)
    assert frame.profile == {"device": "fixture"}
    assert frame.metadata == {"frame_index": 2, "num_frames": 3}


def test_adc_file_frame_reader_rejects_invalid_files_and_indices(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.arange(5, dtype=np.int16).tofile(adc_path)
    spec = ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)

    with pytest.raises(ValueError, match="whole number"):
        ADCFileFrameReader(adc_path, spec)

    np.arange(8, dtype=np.int16).tofile(adc_path)
    reader = ADCFileFrameReader(adc_path, spec)
    with pytest.raises(IndexError, match="outside"):
        reader.read_frame(2)
    with pytest.raises(TypeError, match="not bool"):
        reader.read_frame(True)
    with pytest.raises(ValueError, match="finite and positive"):
        ADCFileFrameReader(adc_path, spec, frame_periodicity_s=float("nan"))


def test_adc_file_frame_reader_validates_explicit_capture(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.arange(16, dtype=np.int16).tofile(adc_path)
    profile = RadarProfile(
        num_tx=1,
        num_rx=1,
        num_adc_samples=2,
        num_chirps_per_tx=1,
    )
    capture = RadarCaptureSpec(
        profile=profile,
        adc=ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2),
        tx_order=(0,),
        frame_periodicity_s=0.05,
        num_frames=4,
    )

    reader = ADCFileFrameReader.from_capture(adc_path, capture, metadata={"session": "test"})
    frame = reader.read_frame(3)

    assert reader.num_frames == 4
    assert frame.timestamp == pytest.approx(0.15)
    assert frame.profile["num_tx"] == 1
    assert frame.metadata["tx_order"] == [0]
    assert frame.metadata["session"] == "test"

    with pytest.raises(ValueError, match="must not override capture tx_order"):
        ADCFileFrameReader.from_capture(adc_path, capture, metadata={"tx_order": [2]})

    invalid_capture = RadarCaptureSpec(
        profile=profile,
        adc=capture.adc,
        tx_order=(0,),
        num_frames=3,
    )
    with pytest.raises(ValueError, match="frame count"):
        ADCFileFrameReader.from_capture(adc_path, invalid_capture)


def test_load_adc_cube_loads_and_organizes_file(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.array([1, 10, 2, 20], dtype=np.int16).tofile(adc_path)

    cube = load_adc_cube(adc_path, ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2))

    assert cube.source == str(adc_path)
    np.testing.assert_array_equal(
        cube.data,
        np.array([[[[1 + 10j, 2 + 20j]]]], dtype=np.complex64),
    )


def test_range_fft_transforms_sample_axis_to_range_bin() -> None:
    cube = RadarCube(
        np.array([[[[1, 0, 0, 0]]]], dtype=np.complex64),
        frame_id="cube-0",
        metadata={"source": "unit"},
    )

    transformed = range_fft(cube)

    assert transformed.axes == ("frame", "chirp", "rx", "range_bin")
    assert transformed.frame_id == "cube-0"
    assert transformed.units == "range_fft"
    assert transformed.metadata["source"] == "unit"
    assert transformed.metadata["range_fft"] == {
        "n_fft": 4,
        "window": "none",
        "one_sided": False,
        "remove_dc": False,
    }
    np.testing.assert_allclose(
        transformed.data,
        np.array([[[[1, 1, 1, 1]]]], dtype=np.complex64),
    )


def test_range_fft_supports_window_nfft_and_one_sided_output() -> None:
    cube = RadarCube(np.ones((1, 1, 1, 4), dtype=np.complex64))

    transformed = range_fft(
        cube,
        RangeFFTSpec(n_fft=8, window=FFTWindow.HANN, one_sided=True),
    )

    assert transformed.data.shape == (1, 1, 1, 5)
    assert transformed.metadata["range_fft"] == {
        "n_fft": 8,
        "window": "hann",
        "one_sided": True,
        "remove_dc": False,
    }


def test_range_fft_can_remove_per_chirp_sample_dc() -> None:
    cube = RadarCube(np.ones((1, 1, 1, 4), dtype=np.complex64))

    transformed = range_fft(cube, RangeFFTSpec(remove_dc=True))

    np.testing.assert_array_equal(transformed.data, np.zeros_like(transformed.data))
    assert transformed.metadata["range_fft"]["remove_dc"] is True


def test_range_fft_requires_sample_axis() -> None:
    cube = RadarCube(np.ones((1, 1, 1, 4), dtype=np.complex64), axes=("a", "b", "c", "d"))

    with pytest.raises(ValueError, match="sample"):
        range_fft(cube)


def test_doppler_fft_transforms_chirp_axis_to_doppler_bin() -> None:
    cube = RadarCube(
        np.array([[[[1]], [[0]], [[0]], [[0]]]], dtype=np.complex64),
        axes=("frame", "chirp", "rx", "range_bin"),
        frame_id="range-0",
        metadata={"source": "unit"},
    )

    transformed = doppler_fft(cube, DopplerFFTSpec(fftshift=False))

    assert transformed.axes == ("frame", "doppler_bin", "rx", "range_bin")
    assert transformed.frame_id == "range-0"
    assert transformed.units == "doppler_fft"
    assert transformed.metadata["source"] == "unit"
    assert transformed.metadata["doppler_fft"] == {
        "n_fft": 4,
        "window": "none",
        "fftshift": False,
        "input_axis": "chirp",
    }
    np.testing.assert_allclose(
        transformed.data,
        np.array([[[[1]], [[1]], [[1]], [[1]]]], dtype=np.complex64),
    )


def test_doppler_fft_supports_window_nfft_and_shift() -> None:
    cube = RadarCube(
        np.ones((1, 4, 1, 2), dtype=np.complex64),
        axes=("frame", "chirp", "rx", "range_bin"),
    )

    transformed = doppler_fft(
        cube,
        DopplerFFTSpec(n_fft=8, window=FFTWindow.HAMMING, fftshift=True),
    )

    assert transformed.data.shape == (1, 8, 1, 2)
    assert transformed.axes == ("frame", "doppler_bin", "rx", "range_bin")
    assert transformed.metadata["doppler_fft"] == {
        "n_fft": 8,
        "window": "hamming",
        "fftshift": True,
        "input_axis": "chirp",
    }


def test_doppler_fft_requires_chirp_axis() -> None:
    cube = RadarCube(
        np.ones((1, 4, 1, 2), dtype=np.complex64),
        axes=("frame", "time", "rx", "range_bin"),
    )

    with pytest.raises(ValueError, match="chirp"):
        doppler_fft(cube)


def test_detect_peaks_returns_detection_frame() -> None:
    data = np.zeros((1, 4, 2, 3), dtype=np.complex64)
    data[0, 2, 0, 1] = 3 + 4j
    data[0, 1, 1, 2] = 2 + 0j
    cube = RadarCube(
        data,
        axes=("frame", "doppler_bin", "rx", "range_bin"),
        frame_id="det-0",
        timestamp=42.0,
        source="fixture.bin",
        units="doppler_fft",
        metadata={"fixture": "unit"},
    )

    detections = detect_peaks(cube, PeakDetectionSpec(threshold=5.0))

    assert detections.channels == ("frame", "range_bin", "doppler_bin", "magnitude")
    assert detections.frame_id == "det-0"
    assert detections.timestamp == 42.0
    assert detections.source == "fixture.bin"
    assert detections.units == {"magnitude": "doppler_fft"}
    assert detections.metadata["fixture"] == "unit"
    assert detections.metadata["peak_detection"] == {
        "threshold": 5.0,
        "aggregate_rx": "max",
        "output_detections": 1,
    }
    np.testing.assert_array_equal(
        detections.detections,
        np.array([[0, 1, 2, 5]], dtype=np.float32),
    )


def test_detect_peaks_supports_sum_rx_aggregation() -> None:
    data = np.zeros((1, 1, 2, 1), dtype=np.complex64)
    data[0, 0, 0, 0] = 2
    data[0, 0, 1, 0] = 3
    cube = RadarCube(data, axes=("frame", "doppler_bin", "rx", "range_bin"))

    detections = detect_peaks(cube, PeakDetectionSpec(threshold=5.0, aggregate_rx="sum"))

    np.testing.assert_array_equal(
        detections.detections,
        np.array([[0, 0, 0, 5]], dtype=np.float32),
    )


def test_detect_peaks_returns_empty_detection_frame_when_no_hits() -> None:
    cube = RadarCube(
        np.zeros((1, 2, 1, 3), dtype=np.complex64),
        axes=("frame", "doppler_bin", "rx", "range_bin"),
    )

    detections = detect_peaks(cube, PeakDetectionSpec(threshold=1.0))

    assert detections.detections.shape == (0, 4)


def test_detect_peaks_requires_range_doppler_axes() -> None:
    cube = RadarCube(np.ones((1, 2, 1, 3), dtype=np.complex64))

    with pytest.raises(ValueError, match="doppler_bin"):
        detect_peaks(cube, PeakDetectionSpec(threshold=1.0))


def test_detect_peaks_preserves_azimuth_bins_after_angle_fft() -> None:
    data = np.zeros((1, 2, 3, 4), dtype=np.complex64)
    data[0, 1, 2, 3] = 6
    cube = RadarCube(
        data,
        axes=("frame", "doppler_bin", "azimuth_bin", "range_bin"),
        frame_id="angle-det-0",
        units="angle_fft",
        metadata={"source": "unit"},
    )

    detections = detect_peaks(cube, PeakDetectionSpec(threshold=5.0))

    assert detections.channels == (
        "frame",
        "range_bin",
        "doppler_bin",
        "azimuth_bin",
        "magnitude",
    )
    assert detections.metadata["peak_detection"] == {
        "threshold": 5.0,
        "aggregate_rx": None,
        "azimuth_peak_radius": 1,
        "azimuth_peak_strict": True,
        "output_detections": 1,
    }
    np.testing.assert_array_equal(
        detections.detections,
        np.array([[0, 3, 1, 2, 6]], dtype=np.float32),
    )


def test_detect_peaks_adds_azimuth_angles_when_layout_is_available() -> None:
    layout = VirtualAntennaLayout.uniform_linear(4, spacing_wavelengths=0.5)
    data = np.zeros((1, 1, 4, 1), dtype=np.complex64)
    data[0, 0, 2, 0] = 7
    cube = RadarCube(
        data,
        axes=("frame", "doppler_bin", "azimuth_bin", "range_bin"),
        units="angle_fft",
        metadata={
            "angle_fft": {
                "n_fft": 4,
                "window": "none",
                "fftshift": True,
                "input_axis": "rx",
                "output_axis": "azimuth_bin",
                "virtual_layout": layout.as_metadata(),
            },
        },
    )

    detections = detect_peaks(cube, PeakDetectionSpec(threshold=5.0))

    assert detections.channels == (
        "frame",
        "range_bin",
        "doppler_bin",
        "azimuth_bin",
        "azimuth_rad",
        "magnitude",
    )
    np.testing.assert_allclose(
        detections.detections,
        np.array([[0, 0, 0, 2, 0.0, 7]], dtype=np.float32),
        atol=1e-6,
    )


def test_detections_to_point_cloud_rejects_uncalibrated_detections() -> None:
    detections = DetectionFrame(
        np.array([[0, 3, 5, 12]], dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "magnitude"),
        frame_id="det-0",
        metadata={"source": "unit"},
    )

    with pytest.raises(ValueError, match="azimuth_bin"):
        detections_to_point_cloud(detections)


def test_detections_to_point_cloud_can_center_doppler_bins() -> None:
    detections = DetectionFrame(
        np.array([[0, 1, 7, 0, 0, 3]], dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "azimuth_bin", "azimuth_rad", "magnitude"),
    )

    point_cloud = detections_to_point_cloud(
        detections,
        PointCloudProjectionSpec(
            range_resolution_m=1.0,
            doppler_resolution_mps=0.5,
            center_doppler=True,
            doppler_bins=8,
        ),
    )

    assert point_cloud.points[0, 3] == pytest.approx(-0.5)


def test_detections_to_point_cloud_centers_fftshifted_doppler_bins() -> None:
    detections = DetectionFrame(
        np.array([[0, 1, 5, 0, 0, 3]], dtype=np.float32),
        channels=(
            "frame",
            "range_bin",
            "doppler_bin",
            "azimuth_bin",
            "azimuth_rad",
            "magnitude",
        ),
    )

    point_cloud = detections_to_point_cloud(
        detections,
        PointCloudProjectionSpec(
            doppler_resolution_mps=0.5,
            center_doppler=True,
            doppler_bins=8,
            doppler_fftshifted=True,
        ),
    )

    assert point_cloud.points[0, 3] == pytest.approx(0.5)


def test_detect_peaks_keeps_only_angle_local_maxima() -> None:
    data = np.zeros((1, 1, 5, 1), dtype=np.complex64)
    data[0, 0, :, 0] = [2, 6, 10, 7, 3]
    cube = RadarCube(
        data,
        axes=("frame", "doppler_bin", "azimuth_bin", "range_bin"),
    )

    peaks = detect_peaks(cube, PeakDetectionSpec(threshold=5.0))
    threshold_mask = detect_peaks(cube, PeakDetectionSpec(threshold=5.0, azimuth_peak_radius=0))

    np.testing.assert_array_equal(peaks.detections[:, 3], [2])
    np.testing.assert_array_equal(threshold_mask.detections[:, 3], [1, 2, 3])


def test_detect_peaks_angle_plateau_policy_is_explicit() -> None:
    data = np.zeros((1, 1, 3, 1), dtype=np.complex64)
    data[0, 0, :, 0] = [2, 6, 6]
    cube = RadarCube(
        data,
        axes=("frame", "doppler_bin", "azimuth_bin", "range_bin"),
    )

    strict = detect_peaks(cube, PeakDetectionSpec(threshold=5.0))
    non_strict = detect_peaks(cube, PeakDetectionSpec(threshold=5.0, azimuth_peak_strict=False))

    assert strict.detections.shape[0] == 0
    np.testing.assert_array_equal(non_strict.detections[:, 3], [1, 2])


def test_detections_to_point_cloud_rejects_uncalibrated_azimuth_bin() -> None:
    detections = DetectionFrame(
        np.array([[0, 3, 5, 2, 12]], dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "azimuth_bin", "magnitude"),
        frame_id="det-az-0",
    )

    with pytest.raises(ValueError, match="azimuth_rad"):
        detections_to_point_cloud(detections)


def test_detections_to_point_cloud_projects_calibrated_azimuth() -> None:
    detections = DetectionFrame(
        np.array([[0, 3, 5, 2, 0.25, 12]], dtype=np.float32),
        channels=(
            "frame",
            "range_bin",
            "doppler_bin",
            "azimuth_bin",
            "azimuth_rad",
            "magnitude",
        ),
    )

    point_cloud = detections_to_point_cloud(detections)

    assert point_cloud.channels == (
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
    np.testing.assert_allclose(
        point_cloud.points,
        np.array([[3 * np.sin(0.25), 3 * np.cos(0.25), 0, 5, 12, 3, 5, 2, 0.25]], dtype=np.float32),
        atol=1e-6,
    )


def test_detections_to_point_cloud_preserves_cfar_quality() -> None:
    detections = DetectionFrame(
        np.array([[0, 3, 5, 2, 0.25, 12, 2, 6, 20]], dtype=np.float32),
        channels=(
            "frame",
            "range_bin",
            "doppler_bin",
            "azimuth_bin",
            "azimuth_rad",
            "magnitude",
            "noise",
            "snr",
            "angle_magnitude",
        ),
        units={"noise": "power", "snr": "linear_ratio", "angle_magnitude": "angle_fft"},
    )

    point_cloud = detections_to_point_cloud(detections)

    assert point_cloud.channels[-3:] == ("noise", "snr", "angle_magnitude")
    np.testing.assert_allclose(point_cloud.points[0, -3:], [2, 6, 20])
    assert point_cloud.units["noise"] == "power"
    assert point_cloud.units["snr"] == "linear_ratio"


def test_detections_to_point_cloud_can_project_azimuth_angle() -> None:
    detections = DetectionFrame(
        np.array([[0, 4, 0, 1, np.pi / 6, 10]], dtype=np.float32),
        channels=(
            "frame",
            "range_bin",
            "doppler_bin",
            "azimuth_bin",
            "azimuth_rad",
            "magnitude",
        ),
        timestamp=42.0,
        source="fixture.bin",
        units={"magnitude": "angle_fft"},
    )

    point_cloud = detections_to_point_cloud(
        detections,
        PointCloudProjectionSpec(range_resolution_m=0.5),
    )

    np.testing.assert_allclose(
        point_cloud.points[0, :3],
        np.array([1.0, np.sqrt(3), 0.0], dtype=np.float32),
        atol=1e-6,
    )
    assert point_cloud.units["velocity"] == "m/s"
    assert point_cloud.units["magnitude"] == "angle_fft"
    assert point_cloud.units["azimuth_rad"] == "rad"
    assert point_cloud.timestamp == 42.0
    assert point_cloud.source == "fixture.bin"


def test_detections_to_point_cloud_requires_angle_channel_for_azimuth_projection() -> None:
    detections = DetectionFrame(
        np.array([[0, 4, 0, 10]], dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "magnitude"),
    )

    with pytest.raises(ValueError, match="azimuth_bin"):
        detections_to_point_cloud(detections)


def test_detections_to_point_cloud_preserves_empty_detections() -> None:
    detections = DetectionFrame(
        np.zeros((0, 6), dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "azimuth_bin", "azimuth_rad", "magnitude"),
    )

    point_cloud = detections_to_point_cloud(detections)

    assert point_cloud.points.shape == (0, 9)


def test_detections_to_point_cloud_requires_detection_channels() -> None:
    detections = DetectionFrame(
        np.zeros((1, 3), dtype=np.float32),
        channels=("frame", "range_bin", "magnitude"),
    )

    with pytest.raises(ValueError, match="doppler_bin"):
        detections_to_point_cloud(detections)


def test_point_cloud_recipe_rejects_detection_without_calibrated_aoa() -> None:
    detection = DetectionRecipe(
        transform=RangeDopplerRecipe(
            decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2))
        ),
        peak_detection=PeakDetectionSpec(threshold=1.0),
    )

    with pytest.raises(ValueError, match="calibrated virtual antenna"):
        PointCloudRecipe(detection)


def test_process_adc_to_range_doppler_runs_explicit_recipe() -> None:
    raw = RawADCFrame(
        np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int16),
        frame_id="frame-7",
        timestamp=12.5,
        source="fixture.bin",
    )
    recipe = RangeDopplerRecipe(
        decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=2, num_rx=1, num_samples=2)),
        range_fft=RangeFFTSpec(one_sided=True),
        doppler_fft=DopplerFFTSpec(fftshift=False),
    )

    cube = process_adc_to_range_doppler(raw, recipe)

    assert cube.axes == ("frame", "doppler_bin", "rx", "range_bin")
    assert cube.data.shape == (1, 2, 1, 2)
    assert cube.frame_id == "frame-7"
    assert cube.timestamp == 12.5
    assert cube.source == "fixture.bin"
    assert cube.metadata["range_fft"]["one_sided"] is True
    assert cube.metadata["doppler_fft"]["fftshift"] is False
    np.testing.assert_array_equal(cube.data[:, 0], np.array([[[2 + 0j, 2 + 0j]]]))
    np.testing.assert_array_equal(cube.data[:, 1], np.zeros((1, 1, 2), dtype=np.complex64))


def test_process_adc_to_detections_supports_cfar_without_point_cloud_projection() -> None:
    raw = np.zeros(50, dtype=np.int16)
    recipe = DetectionRecipe(
        transform=RangeDopplerRecipe(
            decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=5, num_rx=1, num_samples=5)),
            range_fft=RangeFFTSpec(one_sided=True),
            doppler_fft=DopplerFFTSpec(fftshift=False),
        ),
        peak_detection=PeakDetectionSpec(threshold=999.0),
        detection_method=DetectionMethod.CFAR,
        cfar_detection=CFARDetectionSpec(
            training_cells=1,
            guard_cells=0,
            threshold_scale=2.0,
        ),
        quality_filter=DetectionQualitySpec(2.0),
    )

    detections = process_adc_to_detections(raw, recipe)
    assert detections.channels == (
        "frame",
        "range_bin",
        "doppler_bin",
        "magnitude",
        "noise",
        "snr",
    )
    assert detections.metadata["quality_filter"]["output_detections"] == 0


def test_threshold_recipe_rejects_quality_filter_without_snr_at_runtime() -> None:
    recipe = DetectionRecipe(
        transform=RangeDopplerRecipe(
            decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2))
        ),
        peak_detection=PeakDetectionSpec(threshold=0.0),
        quality_filter=DetectionQualitySpec(2.0),
    )

    with pytest.raises(ValueError, match='requires an "snr" channel'):
        process_adc_to_detections(np.zeros(4, dtype=np.int16), recipe)


def test_process_adc_to_calibrated_point_cloud_composes_recipes() -> None:
    raw = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16)
    layout = VirtualAntennaLayout.uniform_linear(4)
    recipe = PointCloudRecipe(
        detection=DetectionRecipe(
            transform=RangeDopplerRecipe(
                decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=4, num_samples=1)),
                range_fft=RangeFFTSpec(one_sided=True),
                doppler_fft=DopplerFFTSpec(fftshift=False),
            ),
            angle_fft=AngleFFTSpec(fftshift=False, virtual_layout=layout),
            peak_detection=PeakDetectionSpec(threshold=1.0, azimuth_peak_radius=0),
        ),
    )

    point_cloud = process_adc_to_calibrated_point_cloud(raw, recipe)
    assert isinstance(point_cloud, PointCloudFrame)

    assert point_cloud.channels == (
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
    assert point_cloud.metadata["angle_fft"]["virtual_layout"] == layout.as_metadata()
    np.testing.assert_array_equal(
        point_cloud.points[:, 7],
        np.array([0, 1, 2, 3], dtype=np.float32),
    )
    np.testing.assert_allclose(
        point_cloud.points[:, 8],
        np.array([0.0, np.pi / 6, -np.pi / 2, -np.pi / 6], dtype=np.float32),
        atol=1e-6,
    )


def test_detection_recipe_requires_cfar_config_for_cfar_method() -> None:
    with pytest.raises(ValueError, match="cfar_detection"):
        DetectionRecipe(
            transform=RangeDopplerRecipe(
                decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2))
            ),
            peak_detection=PeakDetectionSpec(threshold=1.0),
            detection_method=DetectionMethod.CFAR,
        )


def test_detection_recipe_composes_cfar_grouping_and_candidate_aoa() -> None:
    raw = np.zeros(200, dtype=np.int16)
    recipe = PointCloudRecipe(
        detection=DetectionRecipe(
            transform=RangeDopplerRecipe(
                decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=5, num_rx=4, num_samples=5)),
                range_fft=RangeFFTSpec(one_sided=True),
                doppler_fft=DopplerFFTSpec(fftshift=False),
            ),
            detection_method=DetectionMethod.CFAR,
            cfar_detection=CFARDetectionSpec(1, 0, 2.0),
            peak_grouping=PeakGroupingSpec(),
            angle_fft=AngleFFTSpec(
                input_axis="rx",
                virtual_layout=VirtualAntennaLayout.uniform_linear(4),
            ),
        ),
    )

    point_cloud = process_adc_to_calibrated_point_cloud(raw, recipe)

    assert point_cloud.points.shape == (0, 12)
    assert point_cloud.channels[-3:] == ("noise", "snr", "angle_magnitude")
    assert point_cloud.channels[7:9] == ("azimuth_bin", "azimuth_rad")
    assert point_cloud.metadata["candidate_azimuth"]["input_axis"] == "rx"


def test_process_adc_file_to_calibrated_point_cloud_loads_file(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16).tofile(adc_path)
    recipe = PointCloudRecipe(
        detection=DetectionRecipe(
            transform=RangeDopplerRecipe(
                decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=4, num_samples=1)),
                range_fft=RangeFFTSpec(one_sided=True),
                doppler_fft=DopplerFFTSpec(fftshift=False),
            ),
            angle_fft=AngleFFTSpec(
                fftshift=False,
                virtual_layout=VirtualAntennaLayout.uniform_linear(4),
            ),
            peak_detection=PeakDetectionSpec(threshold=1.0, azimuth_peak_radius=0),
        ),
    )

    point_cloud = process_adc_file_to_calibrated_point_cloud(adc_path, recipe, frame_id="file-0")
    assert isinstance(point_cloud, PointCloudFrame)

    assert point_cloud.frame_id == "file-0"
    assert point_cloud.source == str(adc_path)
    assert point_cloud.points.shape == (4, 9)


def test_process_adc_file_to_range_doppler_loads_file(tmp_path) -> None:
    adc_path = tmp_path / "adc.bin"
    np.array([1, 0, 0, 0], dtype=np.int16).tofile(adc_path)
    recipe = RangeDopplerRecipe(
        decode=ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2)),
        doppler_fft=DopplerFFTSpec(fftshift=False),
    )

    cube = process_adc_file_to_range_doppler(
        adc_path,
        recipe,
        frame_id="capture-rd",
        mmap=True,
    )

    assert cube.frame_id == "capture-rd"
    assert cube.axes == ("frame", "doppler_bin", "rx", "range_bin")


@pytest.mark.parametrize(
    ("recipe_name", "field_name"),
    [
        ("decode", "drop_incomplete"),
        ("transform", "remove_static_clutter"),
    ],
)
@pytest.mark.parametrize("value", [1, "yes", np.bool_(True)])
def test_recipe_boolean_policies_require_builtin_bool(
    recipe_name: str,
    field_name: str,
    value: object,
) -> None:
    decode = ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2))
    recipes: dict[str, ADCDecodeRecipe | RangeDopplerRecipe] = {
        "decode": decode,
        "transform": RangeDopplerRecipe(decode=decode),
    }

    with pytest.raises(TypeError, match=field_name):
        replace(recipes[recipe_name], **{field_name: value})


@pytest.mark.parametrize(
    ("recipe_name", "field_name"),
    [
        ("decode", "drop_incomplete"),
        ("transform", "remove_static_clutter"),
    ],
)
@pytest.mark.parametrize("value", [False, True])
def test_recipe_boolean_policies_accept_builtin_bool(
    recipe_name: str,
    field_name: str,
    value: bool,
) -> None:
    decode = ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=1, num_samples=2))
    recipes: dict[str, ADCDecodeRecipe | RangeDopplerRecipe] = {
        "decode": decode,
        "transform": RangeDopplerRecipe(decode=decode),
    }

    updated = replace(recipes[recipe_name], **{field_name: value})
    assert getattr(updated, field_name) is value


@pytest.mark.parametrize(
    ("recipe_name", "field_name"),
    [
        ("decode", "adc"),
        ("transform", "decode"),
        ("transform", "range_fft"),
        ("transform", "doppler_fft"),
        ("transform", "tdm_virtual_array"),
        ("transform", "channel_calibration"),
        ("detection", "transform"),
        ("detection", "peak_detection"),
        ("detection", "cfar_detection"),
        ("detection", "peak_grouping"),
        ("detection", "quality_filter"),
        ("detection", "angle_fft"),
        ("detection", "virtual_subarray"),
        ("detection", "elevation_subarray"),
        ("point_cloud", "detection"),
        ("point_cloud", "projection"),
    ],
)
def test_recipe_nested_contracts_require_declared_types(
    recipe_name: str,
    field_name: str,
) -> None:
    layout = VirtualAntennaLayout.uniform_linear(4)
    decode = ADCDecodeRecipe(ADCFrameSpec(num_chirps=1, num_rx=4, num_samples=2))
    transform = RangeDopplerRecipe(decode=decode)
    detection = DetectionRecipe(
        transform=transform,
        peak_detection=PeakDetectionSpec(threshold=1.0),
        angle_fft=AngleFFTSpec(virtual_layout=layout),
    )
    recipes: dict[
        str,
        ADCDecodeRecipe | RangeDopplerRecipe | DetectionRecipe | PointCloudRecipe,
    ] = {
        "decode": decode,
        "transform": transform,
        "detection": detection,
        "point_cloud": PointCloudRecipe(detection=detection),
    }

    with pytest.raises(TypeError, match=field_name):
        replace(recipes[recipe_name], **{field_name: object()})
