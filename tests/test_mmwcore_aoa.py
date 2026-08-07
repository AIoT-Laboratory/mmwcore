from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import (
    AngleFFTSpec,
    AntennaArrayGeometry,
    DetectionFrame,
    FFTWindow,
    PlanarAngleFFTSpec,
    PlanarApertureLayout,
    RadarCube,
    TDMVirtualArraySpec,
    VirtualAntennaLayout,
    VirtualChannelCalibration,
    VirtualSubarraySpec,
)
from mmwcore.dsp import (
    angle_bin_angles,
    angle_fft,
    apply_virtual_channel_calibration,
    compensate_tdm_doppler_phase,
    estimate_candidate_azimuths,
    map_planar_aperture,
    map_tdm_virtual_array,
    planar_angle_fft,
    select_virtual_subarray,
)


def test_angle_fft_spec_validates_parameters() -> None:
    with pytest.raises(ValueError, match="n_fft"):
        AngleFFTSpec(n_fft=0)

    with pytest.raises(ValueError, match="input_axis"):
        AngleFFTSpec(input_axis="")

    with pytest.raises(ValueError, match="output_axis"):
        AngleFFTSpec(output_axis="")


def test_virtual_antenna_layout_builds_uniform_linear_array() -> None:
    layout = VirtualAntennaLayout.uniform_linear(4, spacing_wavelengths=0.5)

    assert layout.num_antennas == 4
    assert layout.positions_wavelengths == (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
    )
    assert layout.as_metadata() == {
        "name": "uniform_linear",
        "angle_axis": "azimuth",
        "num_antennas": 4,
        "positions_wavelengths": [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
    }


def test_virtual_antenna_layout_validates_positions() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        VirtualAntennaLayout(())

    with pytest.raises(ValueError, match="3D"):
        VirtualAntennaLayout(((0.0, 0.0),))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported angle axis"):
        VirtualAntennaLayout(((0.0, 0.0, 0.0),), angle_axis="roll")


def test_planar_aperture_layout_reports_sparse_duplicate_positions() -> None:
    layout = PlanarApertureLayout(((0, 0), (1, 0), (1, 0), (2, 1)), name="fixture")

    assert layout.num_antennas == 4
    assert layout.num_unique_positions == 3
    assert layout.aperture_shape == (3, 2)
    assert layout.as_metadata()["duplicate_policy"] == "first"


def test_planar_aperture_layout_validates_indices() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        PlanarApertureLayout(())
    with pytest.raises(ValueError, match="pairs"):
        PlanarApertureLayout(((0, 1, 2),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        PlanarApertureLayout(((-1, 0),))
    for indices in [((0.0, 0),), ((0.5, 0),), ((True, 0),)]:
        with pytest.raises(ValueError, match="must contain integers"):
            PlanarApertureLayout(indices)  # type: ignore[arg-type]


def test_tdm_geometry_builds_ordered_virtual_phase_centers() -> None:
    geometry = AntennaArrayGeometry(
        tx_positions_wavelengths=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        rx_positions_wavelengths=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        name="fixture",
    )
    spec = TDMVirtualArraySpec(geometry, tx_order=(1, 0))

    assert spec.num_virtual_antennas == 4
    assert spec.virtual_layout().positions_wavelengths == (
        (2.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
    )


def test_tdm_geometry_validates_tx_order() -> None:
    geometry = AntennaArrayGeometry(
        tx_positions_wavelengths=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        rx_positions_wavelengths=((0.0, 0.0, 0.0),),
    )

    with pytest.raises(ValueError, match="duplicates"):
        TDMVirtualArraySpec(geometry, tx_order=(0, 0))
    with pytest.raises(ValueError, match="within"):
        TDMVirtualArraySpec(geometry, tx_order=(0, 2))


@pytest.mark.parametrize("tx_order", [(0.0,), (0.5,), (True,)])
def test_tdm_geometry_rejects_non_integer_tx_order(tx_order: tuple[object, ...]) -> None:
    geometry = AntennaArrayGeometry(
        tx_positions_wavelengths=((0.0, 0.0, 0.0),),
        rx_positions_wavelengths=((0.0, 0.0, 0.0),),
    )

    with pytest.raises(ValueError, match="must contain integers"):
        geometry.virtual_layout(tx_order)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must contain integers"):
        TDMVirtualArraySpec(geometry, tx_order=tx_order)  # type: ignore[arg-type]


def test_map_tdm_virtual_array_preserves_loop_tx_rx_order() -> None:
    geometry = AntennaArrayGeometry(
        tx_positions_wavelengths=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        rx_positions_wavelengths=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
    )
    cube = RadarCube(
        np.arange(8, dtype=np.float32).reshape(1, 4, 2, 1).astype(np.complex64),
        frame_id="tdm-0",
        timestamp=4.0,
        source="fixture.bin",
    )

    mapped = map_tdm_virtual_array(cube, TDMVirtualArraySpec(geometry, (0, 1)))

    assert mapped.axes == ("frame", "loop", "virtual_rx", "sample")
    assert mapped.data.shape == (1, 2, 4, 1)
    np.testing.assert_array_equal(
        mapped.data[..., 0],
        np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=np.complex64),
    )
    assert mapped.frame_id == "tdm-0"
    assert mapped.timestamp == 4.0
    assert mapped.source == "fixture.bin"
    assert mapped.metadata["tdm_virtual_array"]["tx_order"] == [0, 1]


def test_map_tdm_virtual_array_rejects_incomplete_loops() -> None:
    geometry = AntennaArrayGeometry(
        tx_positions_wavelengths=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        rx_positions_wavelengths=((0.0, 0.0, 0.0),),
    )
    cube = RadarCube(np.ones((1, 3, 1, 1), dtype=np.complex64))

    with pytest.raises(ValueError, match="complete TDM loops"):
        map_tdm_virtual_array(cube, TDMVirtualArraySpec(geometry, (0, 1)))


@pytest.mark.parametrize("fftshift", [False, True])
def test_compensate_tdm_doppler_phase_removes_tx_time_offset(fftshift: bool) -> None:
    geometry = AntennaArrayGeometry(
        tx_positions_wavelengths=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        rx_positions_wavelengths=((0.0, 0.0, 0.0),),
    )
    spec = TDMVirtualArraySpec(geometry, (0, 1))
    signed_bins = np.fft.fftfreq(4) * 4
    if fftshift:
        signed_bins = np.fft.fftshift(signed_bins)
    motion_phase = np.exp(2j * np.pi * signed_bins / 8).astype(np.complex64)
    data = np.ones((1, 4, 2, 1), dtype=np.complex64)
    data[0, :, 1, 0] = motion_phase
    cube = RadarCube(
        data,
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )

    compensated = compensate_tdm_doppler_phase(cube, spec, fftshift=fftshift)

    np.testing.assert_allclose(compensated.data, np.ones_like(data), atol=1e-6)
    assert compensated.metadata["tdm_doppler_compensation"] == {
        "num_tx": 2,
        "num_rx": 1,
        "fftshift": fftshift,
        "tx_order": [0, 1],
    }


def test_select_virtual_subarray_preserves_declared_channel_order() -> None:
    cube = RadarCube(
        np.arange(12, dtype=np.float32).reshape(1, 1, 12, 1).astype(np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    layout = VirtualAntennaLayout.uniform_linear(3)
    spec = VirtualSubarraySpec((5, 2, 9), layout)

    selected = select_virtual_subarray(cube, spec)

    np.testing.assert_array_equal(selected.data[0, 0, :, 0], np.array([5, 2, 9]))
    assert selected.metadata["virtual_subarray"]["antenna_indices"] == [5, 2, 9]


@pytest.mark.parametrize("antenna_indices", [(0.0,), (0.5,), (True,)])
def test_virtual_subarray_rejects_non_integer_indices(
    antenna_indices: tuple[object, ...],
) -> None:
    layout = VirtualAntennaLayout.uniform_linear(1)

    with pytest.raises(ValueError, match="must contain integers"):
        VirtualSubarraySpec(antenna_indices, layout)  # type: ignore[arg-type]


def test_select_virtual_subarray_rejects_out_of_range_index() -> None:
    cube = RadarCube(
        np.ones((1, 1, 4, 1), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    spec = VirtualSubarraySpec((0, 4), VirtualAntennaLayout.uniform_linear(2))

    with pytest.raises(ValueError, match="exceeds"):
        select_virtual_subarray(cube, spec)


def test_map_planar_aperture_keeps_first_duplicate_channel() -> None:
    cube = RadarCube(
        np.array([[[[1], [2], [99], [4]]]], dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
        metadata={"stage": "rd"},
    )
    layout = PlanarApertureLayout(((0, 0), (1, 0), (1, 0), (2, 1)))

    mapped = map_planar_aperture(cube, layout)

    assert mapped.axes == (
        "frame",
        "doppler_bin",
        "azimuth_aperture",
        "elevation_aperture",
        "range_bin",
    )
    assert mapped.data.shape == (1, 1, 3, 2, 1)
    np.testing.assert_array_equal(mapped.data[0, 0, :, :, 0], [[1, 0], [2, 0], [0, 4]])
    assert mapped.metadata["stage"] == "rd"
    assert mapped.metadata["planar_aperture"]["num_unique_positions"] == 3


def test_map_planar_aperture_validates_channel_count() -> None:
    cube = RadarCube(
        np.ones((1, 1, 2, 1), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )

    with pytest.raises(ValueError, match="channel count"):
        map_planar_aperture(cube, PlanarApertureLayout(((0, 0),)))


def test_apply_virtual_channel_calibration_corrects_complex_response() -> None:
    cube = RadarCube(
        np.array([[[[2 + 0j], [0 + 0.5j]]]], dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    calibration = VirtualChannelCalibration(
        (0.5 + 0j, 0 - 2j),
        source="corner-reflector",
        version="fixture-v1",
    )

    corrected = apply_virtual_channel_calibration(cube, calibration)

    np.testing.assert_allclose(corrected.data, np.ones_like(corrected.data))
    assert corrected.metadata["virtual_channel_calibration"]["source"] == "corner-reflector"
    assert corrected.metadata["virtual_channel_calibration"]["version"] == "fixture-v1"


def test_apply_virtual_channel_calibration_validates_channel_count() -> None:
    cube = RadarCube(
        np.ones((1, 1, 2, 1), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )

    with pytest.raises(ValueError, match="channel count"):
        apply_virtual_channel_calibration(cube, VirtualChannelCalibration((1 + 0j,)))


def test_angle_fft_transforms_virtual_rx_axis_to_angle_bin() -> None:
    cube = RadarCube(
        np.array([[[[1], [0], [0], [0]]]], dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
        frame_id="angle-0",
        units="doppler_fft",
        metadata={"source": "unit"},
    )

    transformed = angle_fft(
        cube,
        AngleFFTSpec(input_axis="virtual_rx", output_axis="azimuth_bin", fftshift=False),
    )

    assert transformed.axes == ("frame", "doppler_bin", "azimuth_bin", "range_bin")
    assert transformed.frame_id == "angle-0"
    assert transformed.units == "angle_fft"
    assert transformed.metadata["source"] == "unit"
    assert transformed.metadata["angle_fft"] == {
        "n_fft": 4,
        "window": "none",
        "fftshift": False,
        "input_axis": "virtual_rx",
        "output_axis": "azimuth_bin",
        "virtual_layout": None,
    }
    np.testing.assert_allclose(
        transformed.data,
        np.ones((1, 1, 4, 1), dtype=np.complex64),
    )


def test_angle_fft_supports_window_nfft_and_shift() -> None:
    data = np.ones((1, 1, 4, 1), dtype=np.complex64)
    cube = RadarCube(
        data,
        axes=("frame", "doppler_bin", "rx", "range_bin"),
    )

    layout = VirtualAntennaLayout.uniform_linear(4)
    transformed = angle_fft(
        cube,
        AngleFFTSpec(n_fft=8, window=FFTWindow.HANN, virtual_layout=layout),
    )

    assert transformed.data.shape == (1, 1, 8, 1)
    assert transformed.axes == ("frame", "doppler_bin", "azimuth_bin", "range_bin")
    assert transformed.metadata["angle_fft"] == {
        "n_fft": 8,
        "window": "hann",
        "fftshift": True,
        "input_axis": "rx",
        "output_axis": "azimuth_bin",
        "virtual_layout": layout.as_metadata(),
    }
    expected = data * np.hanning(4).astype(np.float32).reshape(1, 1, 4, 1)
    expected = np.fft.fft(expected, n=8, axis=2)
    expected = np.fft.fftshift(expected, axes=2)
    np.testing.assert_allclose(transformed.data, expected, rtol=2e-5, atol=2e-5)


def test_angle_fft_validates_virtual_layout_size() -> None:
    cube = RadarCube(
        np.ones((1, 1, 4, 1), dtype=np.complex64),
        axes=("frame", "doppler_bin", "rx", "range_bin"),
    )

    with pytest.raises(ValueError, match="antenna count"):
        angle_fft(cube, AngleFFTSpec(virtual_layout=VirtualAntennaLayout.uniform_linear(3)))


def test_planar_angle_fft_matches_separable_numpy_reference() -> None:
    data = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 2, 2).astype(np.complex64)
    cube = RadarCube(
        data,
        axes=(
            "frame",
            "doppler_bin",
            "azimuth_aperture",
            "elevation_aperture",
            "range_bin",
        ),
    )
    spec = PlanarAngleFFTSpec(azimuth_n_fft=4, elevation_n_fft=3, fftshift=True)

    transformed = planar_angle_fft(cube, spec)

    expected = np.fft.fft(data, n=4, axis=2)
    expected = np.fft.fft(expected, n=3, axis=3)
    expected = np.fft.fftshift(expected, axes=(2, 3))
    np.testing.assert_allclose(transformed.data, expected, rtol=1e-6, atol=1e-6)
    assert transformed.axes == (
        "frame",
        "doppler_bin",
        "azimuth_bin",
        "elevation_bin",
        "range_bin",
    )
    assert transformed.metadata["planar_angle_fft"]["azimuth_n_fft"] == 4
    assert transformed.metadata["planar_angle_fft"]["elevation_n_fft"] == 3


def test_planar_angle_fft_validates_spec_and_axes() -> None:
    with pytest.raises(ValueError, match="azimuth_n_fft"):
        PlanarAngleFFTSpec(azimuth_n_fft=0)
    with pytest.raises(ValueError, match="distinct"):
        PlanarAngleFFTSpec(azimuth_input_axis="aperture", elevation_input_axis="aperture")

    cube = RadarCube(np.ones((1, 2), dtype=np.complex64), axes=("frame", "virtual_rx"))
    with pytest.raises(ValueError, match="planar aperture axes"):
        planar_angle_fft(cube)


def test_angle_bin_angles_calibrates_uniform_linear_bins() -> None:
    layout = VirtualAntennaLayout.uniform_linear(4, spacing_wavelengths=0.5)

    angles = angle_bin_angles(4, layout, fftshift=True)

    np.testing.assert_allclose(
        angles,
        np.array([-np.pi / 2, -np.pi / 6, 0.0, np.pi / 6], dtype=np.float32),
        atol=1e-6,
    )


def test_angle_bins_support_constant_offset_and_elevation_axis() -> None:
    azimuth = VirtualAntennaLayout(
        ((0.0, 0.0, 0.5), (0.5, 0.0, 0.5)),
        angle_axis="azimuth",
    )
    elevation = VirtualAntennaLayout(
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.5)),
        angle_axis="elevation",
    )

    np.testing.assert_allclose(angle_bin_angles(2, azimuth), [-np.pi / 2, 0.0])
    np.testing.assert_allclose(angle_bin_angles(2, elevation), [-np.pi / 2, 0.0])


def test_angle_bin_angles_validates_layout_assumptions() -> None:
    with pytest.raises(ValueError, match="positive"):
        angle_bin_angles(0, VirtualAntennaLayout.uniform_linear(2))
    with pytest.raises(ValueError, match="positive"):
        angle_bin_angles(-1, VirtualAntennaLayout.uniform_linear(2))

    with pytest.raises(ValueError, match="z-axis"):
        angle_bin_angles(
            4,
            VirtualAntennaLayout.uniform_linear(2, angle_axis="elevation"),
        )

    with pytest.raises(ValueError, match="x-axis"):
        angle_bin_angles(4, VirtualAntennaLayout(((0.0, 0.0, 0.0), (0.5, 0.1, 0.0))))

    with pytest.raises(ValueError, match="uniformly"):
        angle_bin_angles(
            4,
            VirtualAntennaLayout(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.2, 0.0, 0.0))),
        )

    with pytest.raises(ValueError, match="visible"):
        angle_bin_angles(4, VirtualAntennaLayout.uniform_linear(2, spacing_wavelengths=0.25))


def test_angle_fft_requires_input_axis() -> None:
    cube = RadarCube(
        np.ones((1, 1, 4, 1), dtype=np.complex64),
        axes=("frame", "doppler_bin", "antenna", "range_bin"),
    )

    with pytest.raises(ValueError, match="virtual_rx"):
        angle_fft(cube, AngleFFTSpec(input_axis="virtual_rx"))


def test_estimate_candidate_azimuths_selects_one_physical_peak() -> None:
    layout = VirtualAntennaLayout.uniform_linear(4, spacing_wavelengths=0.5)
    cube = RadarCube(
        np.ones((1, 1, 4, 2), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
        frame_id="rd-0",
    )
    candidates = DetectionFrame(
        np.array([[0, 1, 0, 4]], dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "magnitude"),
        frame_id="rd-0",
    )

    detections = estimate_candidate_azimuths(
        cube,
        candidates,
        AngleFFTSpec(
            n_fft=4,
            fftshift=True,
            input_axis="virtual_rx",
            virtual_layout=layout,
        ),
    )

    assert detections.channels == (
        "frame",
        "range_bin",
        "doppler_bin",
        "azimuth_bin",
        "azimuth_rad",
        "magnitude",
        "angle_magnitude",
    )
    np.testing.assert_allclose(detections.detections, [[0, 1, 0, 2, 0.0, 4.0, 4.0]])
    assert detections.metadata["candidate_azimuth"]["n_fft"] == 4
    assert detections.metadata["candidate_azimuth"]["input_candidates"] == 1
    assert detections.metadata["candidate_azimuth"]["output_detections"] == 1


@pytest.mark.parametrize("range_bin", [1.9, -0.5])
def test_estimate_candidate_azimuths_rejects_inexact_candidate_indices(
    range_bin: float,
) -> None:
    layout = VirtualAntennaLayout.uniform_linear(4, spacing_wavelengths=0.5)
    cube = RadarCube(
        np.ones((1, 1, 4, 2), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    candidates = DetectionFrame(
        np.array([[0, range_bin, 0, 4]], dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "magnitude"),
    )

    with pytest.raises(ValueError, match="outside the angle-estimation cube"):
        estimate_candidate_azimuths(
            cube,
            candidates,
            AngleFFTSpec(input_axis="virtual_rx", virtual_layout=layout),
        )


def test_estimate_candidate_azimuths_preserves_empty_candidate_frame() -> None:
    layout = VirtualAntennaLayout.uniform_linear(4, spacing_wavelengths=0.5)
    cube = RadarCube(
        np.ones((1, 1, 4, 2), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    candidates = DetectionFrame(
        np.empty((0, 4), dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "magnitude"),
    )

    detections = estimate_candidate_azimuths(
        cube,
        candidates,
        AngleFFTSpec(input_axis="virtual_rx", virtual_layout=layout),
    )

    assert detections.detections.shape == (0, 7)
    assert detections.metadata["candidate_azimuth"]["input_candidates"] == 0
    assert detections.metadata["candidate_azimuth"]["output_detections"] == 0


def test_estimate_candidate_azimuths_preserves_cfar_quality_channels() -> None:
    layout = VirtualAntennaLayout.uniform_linear(4, spacing_wavelengths=0.5)
    cube = RadarCube(
        np.ones((1, 1, 4, 1), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    candidates = DetectionFrame(
        np.array([[0, 0, 0, 8, 2, 4]], dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "magnitude", "noise", "snr"),
        units={"magnitude": "doppler_fft", "noise": "power", "snr": "linear_ratio"},
    )

    detections = estimate_candidate_azimuths(
        cube,
        candidates,
        AngleFFTSpec(input_axis="virtual_rx", virtual_layout=layout),
    )

    assert detections.channels[-4:] == ("magnitude", "noise", "snr", "angle_magnitude")
    np.testing.assert_allclose(detections.detections[0, -4:], [8, 2, 4, 4])
    assert detections.units["snr"] == "linear_ratio"
    assert detections.units["angle_magnitude"] == "angle_fft"


def test_estimate_candidate_azimuths_rejects_existing_angle_channels() -> None:
    layout = VirtualAntennaLayout.uniform_linear(2)
    cube = RadarCube(
        np.ones((1, 1, 2, 1), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    candidates = DetectionFrame(
        np.zeros((1, 4), dtype=np.float32),
        channels=("frame", "range_bin", "doppler_bin", "azimuth_bin"),
    )

    with pytest.raises(ValueError, match="without existing angle channels"):
        estimate_candidate_azimuths(
            cube,
            candidates,
            AngleFFTSpec(input_axis="virtual_rx", virtual_layout=layout),
        )


def test_estimate_candidate_azimuths_rejects_missing_layout() -> None:
    cube = RadarCube(
        np.ones((1, 1, 2, 1), dtype=np.complex64),
        axes=("frame", "doppler_bin", "virtual_rx", "range_bin"),
    )
    candidates = DetectionFrame(
        np.empty((0, 4)),
        channels=("frame", "range_bin", "doppler_bin", "magnitude"),
    )

    with pytest.raises(ValueError, match="calibrated virtual layout"):
        estimate_candidate_azimuths(cube, candidates, AngleFFTSpec(input_axis="virtual_rx"))
