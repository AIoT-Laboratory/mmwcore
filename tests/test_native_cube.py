from __future__ import annotations

import numpy as np
import pytest

from mmwcore import _native


def test_native_static_clutter_and_channel_calibrations() -> None:
    clutter = _native.remove_static_clutter_complex(
        np.array([[[[1 + 1j]], [[3 + 3j]]]], dtype=np.complex64),
        1,
    )
    np.testing.assert_allclose(
        clutter,
        np.array([[[[-1 - 1j]], [[1 + 1j]]]], dtype=np.complex64),
    )

    time_calibrated = _native.apply_time_domain_channel_calibration_complex(
        np.ones((1, 1, 2, 2, 3), dtype=np.complex64),
        2,
        3,
        4,
        np.array([0.0, np.pi / 2, np.pi, 0.0], dtype=np.float32),
        np.array([1 + 0j, 0 + 1j, 0.5 + 0j, 0 - 1j], dtype=np.complex64),
    )
    sample = np.arange(3, dtype=np.float32)
    expected = np.empty((2, 2, 3), dtype=np.complex64)
    expected[0, 0] = 1.0
    expected[0, 1] = 1j * np.exp(1j * np.pi / 2 * sample)
    expected[1, 0] = 0.5 * np.exp(1j * np.pi * sample)
    expected[1, 1] = -1j
    np.testing.assert_allclose(time_calibrated[0, 0], expected, atol=1e-6)

    virtual_calibrated = _native.apply_virtual_channel_calibration_complex(
        np.array([[[[2 + 0j], [0 + 0.5j]]]], dtype=np.complex64),
        2,
        np.array([0.5 + 0j, 0 - 2j], dtype=np.complex64),
    )
    np.testing.assert_allclose(virtual_calibrated, np.ones_like(virtual_calibrated))


@pytest.mark.parametrize("fftshift", [False, True])
def test_native_tdm_mapping_and_phase_compensation(fftshift: bool) -> None:
    mapped = _native.map_tdm_virtual_array_complex(
        np.arange(8, dtype=np.float32).reshape(1, 4, 2, 1).astype(np.complex64),
        1,
        2,
        2,
    )
    assert mapped.shape == (1, 2, 4, 1)
    np.testing.assert_array_equal(
        mapped[..., 0],
        np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=np.complex64),
    )

    signed_bins = np.fft.fftfreq(4) * 4
    if fftshift:
        signed_bins = np.fft.fftshift(signed_bins)
    data = np.ones((1, 4, 2, 1), dtype=np.complex64)
    data[0, :, 1, 0] = np.exp(2j * np.pi * signed_bins / 8).astype(np.complex64)
    compensated = _native.compensate_tdm_doppler_phase_complex(
        data,
        1,
        2,
        2,
        1,
        fftshift,
    )
    np.testing.assert_allclose(compensated, np.ones_like(data), atol=1e-6)


def test_native_planar_scatter_and_virtual_selection() -> None:
    data = np.array([[[[1], [2], [99], [4]]]], dtype=np.complex64)
    planar = _native.map_planar_aperture_complex(
        data,
        2,
        ((0, 0), (1, 0), (1, 0), (2, 1)),
    )
    assert planar.shape == (1, 1, 3, 2, 1)
    np.testing.assert_array_equal(planar[0, 0, :, :, 0], [[1, 0], [2, 0], [0, 4]])

    selected = _native.select_virtual_subarray_complex(data, 2, (3, 1))
    np.testing.assert_array_equal(selected[0, 0, :, 0], [4, 2])


def test_native_cube_rejects_noncontiguous_input_and_invalid_selection() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _native.remove_static_clutter_complex(
            np.ones((2, 2), dtype=np.complex64).T,
            0,
        )
    with pytest.raises(ValueError, match="outside axis length"):
        _native.select_virtual_subarray_complex(
            np.ones((1, 1, 2, 1), dtype=np.complex64),
            2,
            (2,),
        )


def test_native_cube_matches_numpy_for_nonleading_axes() -> None:
    random = np.random.default_rng(7)

    clutter_data = _complex_data(random, (2, 3, 4, 5))
    np.testing.assert_allclose(
        _native.remove_static_clutter_complex(clutter_data, 2),
        clutter_data - clutter_data.mean(axis=2, keepdims=True),
        rtol=1e-6,
        atol=1e-6,
    )

    calibration_data = _complex_data(random, (2, 4, 2, 5, 3))
    tx_axis, rx_axis, sample_axis = 2, 4, 1
    frequencies = random.normal(size=(2, 3)).astype(np.float32)
    corrections = _complex_data(random, (2, 3))
    canonical = np.moveaxis(
        calibration_data,
        (tx_axis, rx_axis, sample_axis),
        (-3, -2, -1),
    )
    sample_indices = np.arange(canonical.shape[-1], dtype=np.float32)
    expected_calibration = (
        canonical
        * corrections[..., np.newaxis]
        * np.exp(1j * frequencies[..., np.newaxis] * sample_indices).astype(np.complex64)
    )
    expected_calibration = np.moveaxis(
        expected_calibration,
        (-3, -2, -1),
        (tx_axis, rx_axis, sample_axis),
    )
    np.testing.assert_allclose(
        _native.apply_time_domain_channel_calibration_complex(
            calibration_data,
            tx_axis,
            rx_axis,
            sample_axis,
            frequencies.reshape(-1),
            corrections.reshape(-1),
        ),
        expected_calibration,
        rtol=1e-6,
        atol=1e-6,
    )

    tdm_data = _complex_data(random, (2, 3, 4, 5, 2))
    chirp_axis, rx_axis = 2, 0
    moved = np.moveaxis(tdm_data, (chirp_axis, rx_axis), (0, 1))
    expected_tdm = moved.reshape(2, 2, 2, *moved.shape[2:])
    expected_tdm = expected_tdm.reshape(2, 4, *expected_tdm.shape[3:])
    expected_tdm = np.moveaxis(expected_tdm, (0, 1), (chirp_axis, rx_axis))
    np.testing.assert_allclose(
        _native.map_tdm_virtual_array_complex(tdm_data, chirp_axis, rx_axis, 2),
        expected_tdm,
        rtol=0,
        atol=0,
    )

    phase_data = _complex_data(random, (4, 2, 3, 5))
    doppler_axis, virtual_axis = 3, 0
    signed_bins = np.fft.fftshift(np.fft.fftfreq(5) * 5)
    tx_slots = np.repeat(np.arange(2), 2)
    phase = np.exp(-2j * np.pi * signed_bins[:, np.newaxis] * tx_slots / 10).astype(np.complex64)
    phase_by_axis = np.moveaxis(
        phase.reshape(5, 4, 1, 1),
        (0, 1),
        (doppler_axis, virtual_axis),
    )
    np.testing.assert_allclose(
        _native.compensate_tdm_doppler_phase_complex(
            phase_data,
            doppler_axis,
            virtual_axis,
            2,
            2,
            True,
        ),
        phase_data * phase_by_axis,
        rtol=1e-6,
        atol=1e-6,
    )

    planar_data = _complex_data(random, (2, 4, 3))
    grid_indices = ((0, 0), (1, 0), (1, 0), (2, 1))
    planar_moved = np.moveaxis(planar_data, 1, -1)
    expected_planar = np.zeros((*planar_moved.shape[:-1], 3, 2), dtype=np.complex64)
    expected_planar[..., (0, 1, 2), (0, 0, 1)] = planar_moved[..., (0, 1, 3)]
    expected_planar = np.moveaxis(expected_planar, (-2, -1), (1, 2))
    np.testing.assert_allclose(
        _native.map_planar_aperture_complex(planar_data, 1, grid_indices),
        expected_planar,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        _native.select_virtual_subarray_complex(planar_data, 1, (3, 1)),
        np.take(planar_data, (3, 1), axis=1),
        rtol=0,
        atol=0,
    )


def _complex_data(random: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (
        random.normal(size=shape).astype(np.float32)
        + 1j * random.normal(size=shape).astype(np.float32)
    ).astype(np.complex64)
