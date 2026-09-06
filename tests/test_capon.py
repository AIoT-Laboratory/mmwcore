"""Physical and independent numerical checks for the complete dynamic chain."""

import json

import numpy as np
import pytest
from _capon_reference import MU, NU, evaluate, steering

from mmwcore import _native
from mmwcore.core import ADCDecodeSpec, ADCFrameSpec, FFTWindow, RadarCube, RangeFFTSpec
from mmwcore.dsp import IskCaponSpec, isk_capon_from_range_cube, isk_capon_point_cloud, range_fft

SPEC = IskCaponSpec(0.043, 0.09, 32)


def cube(data):
    return RadarCube(
        np.asarray(data, dtype=np.complex64)[None],
        axes=("frame", "loop", "virtual_rx", "range_bin"),
    )


def scene():
    rng = np.random.default_rng(412)
    x = (rng.normal(size=(32, 12, 64)) + 1j * rng.normal(size=(32, 12, 64))) * 0.2
    t = np.arange(32)
    for r, a, e, d in [(25, 65, 18, -5), (40, 120, 8, 7)]:
        x[:, :, r] += 20 * np.exp(2j * np.pi * d * t[:, None] / 32) * steering(NU[a], MU[e])
    return x.astype(np.complex64)


def test_full_chain_against_independent_numpy_reference():
    data = scene()
    actual = isk_capon_from_range_cube(cube(data), SPEC)
    power, expected, total = evaluate(data, SPEC.doppler_bins)
    np.testing.assert_allclose(actual.ra_power, power, rtol=3e-6, atol=1e-8)
    assert actual.diagnostics["accepted_before_capacity"] == total
    assert len(actual.detections) == len(expected) > 0
    for got, want in zip(actual.detections, expected, strict=True):
        for key in ("range_bin", "azimuth_bin", "elevation_bin", "doppler_bin"):
            assert getattr(got, key) == want[key]
        for key in ("elevation_interpolated_bin", "snr", "range_noise", "elevation_power"):
            np.testing.assert_allclose(getattr(got, key), want[key], rtol=3e-6, atol=1e-8)
        np.testing.assert_allclose(
            got.xyz_m,
            np.asarray(want["direction"]) * got.range_bin * SPEC.range_resolution_m,
            atol=1e-7,
        )
    for r, a, e, d in [(25, 65, 18, -5), (40, 120, 8, 7)]:
        match = next(p for p in actual.detections if p.range_bin == r and p.azimuth_bin == a)
        assert match.doppler_bin == d
        assert match.elevation_bin == e
        assert np.sign(match.xyz_m[1]) == np.sign(-NU[a])
        assert np.sign(match.xyz_m[2]) == np.sign(MU[e])


def test_static_is_annihilated_and_zero_background_is_explicit():
    static = np.broadcast_to(np.arange(12)[None, :, None] * (1 + 2j), (32, 12, 64))
    result = isk_capon_from_range_cube(cube(static), SPEC)
    assert result.points.points.shape == (0, 5)
    assert not result.ra_power.any()
    assert result.diagnostics["zero_covariance_ranges"] == 64
    raw = np.zeros((32, 12, 64), np.complex64)
    raw[:, :, 30] = np.exp(2j * np.pi * np.arange(32)[:, None] / 32) * steering(NU[65], 0)
    result = isk_capon_from_range_cube(cube(raw), SPEC)
    assert not len(result.detections)
    assert result.diagnostics["zero_noise_dropped"] > 0


@pytest.mark.parametrize("doppler", [-8, 8, 16])
def test_doppler_sign_padding_and_source_nyquist(doppler):
    rng = np.random.default_rng(71)
    data = (rng.normal(size=(32, 12, 64)) + 1j * rng.normal(size=(32, 12, 64))) * 0.01
    data[:, :, 30] += (
        10 * np.exp(2j * np.pi * doppler * np.arange(32)[:, None] / 32) * steering(NU[65], MU[18])
    )
    result = isk_capon_from_range_cube(cube(data), SPEC)
    match = next(p for p in result.detections if p.range_bin == 30 and p.azimuth_bin == 65)
    assert match.doppler_bin == doppler
    if abs(doppler) < 16:
        padded = isk_capon_from_range_cube(cube(data), IskCaponSpec(0.043, 0.045, 64))
        match2 = next(p for p in padded.detections if p.range_bin == 30 and p.azimuth_bin == 65)
        assert match2.doppler_bin == doppler * 2
        assert match2.velocity_mps == match.velocity_mps


def test_blackman_full_complex_range_fft():
    rng = np.random.default_rng(3)
    data = (rng.normal(size=(1, 2, 4, 64)) + 1j * rng.normal(size=(1, 2, 4, 64))).astype(
        np.complex64
    )
    result = range_fft(
        RadarCube(data, axes=("frame", "chirp", "rx", "sample")),
        RangeFFTSpec(window=FFTWindow.BLACKMAN),
    )
    expected = np.fft.fft(data * np.blackman(64), axis=-1)
    assert result.data.shape == data.shape
    np.testing.assert_allclose(result.data, expected, atol=8e-6)


def test_checked_adc_entry_and_unsupported_contracts():
    decode = ADCDecodeSpec(ADCFrameSpec(96, 4, 64))
    raw = np.zeros(decode.adc.raw_values_per_frame, np.int16)
    result = isk_capon_point_cloud(raw, decode, SPEC, tx_order=(0, 1, 2))
    assert result.ra_power.shape == (187, 64)
    with pytest.raises(ValueError, match="TX order"):
        isk_capon_point_cloud(raw, decode, SPEC, tx_order=(0, 2, 1))
    with pytest.raises(ValueError, match="power of two"):
        IskCaponSpec(0.1, 0.1, 31)
    with pytest.raises(ValueError, match="finite"):
        isk_capon_from_range_cube(cube(np.full((32, 12, 64), np.nan)), SPEC)
    with pytest.raises(ValueError, match="shape"):
        _native.isk_capon_complex(np.zeros((32, 11, 64), np.complex64), 0.1, 0.1, 32)
    with pytest.raises(ValueError, match="ranges"):
        _native.isk_capon_complex(np.zeros((32, 12, 32), np.complex64), 0.1, 0.1, 32)
    _, report = _native.isk_capon_complex(np.zeros((32, 12, 64), np.complex64), 0.1, 0.1, 32)
    assert json.loads(report)["diagnostics"]["zero_covariance_ranges"] == 64


def test_nonzero_adc_to_points_preserves_range_angle_and_doppler():
    rng = np.random.default_rng(27)
    decode = ADCDecodeSpec(ADCFrameSpec(96, 4, 64))
    signal = np.empty((32, 3, 4, 64), dtype=np.complex128)
    a = steering(NU[65], MU[18]).reshape(3, 4)
    fast = np.exp(2j * np.pi * 25 * np.arange(64) / 64)
    for t in range(32):
        signal[t] = 800 * np.exp(-2j * np.pi * 5 * t / 32) * a[:, :, None] * fast
    signal += rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
    # Explicit IQ-interleaved int16 capture, with TX0/1/2 chirps each loop.
    raw = np.stack((signal.real, signal.imag), axis=-1).round().astype(np.int16).ravel()
    result = isk_capon_point_cloud(raw, decode, SPEC, tx_order=(0, 1, 2))
    match = next(d for d in result.detections if d.range_bin == 25 and d.azimuth_bin == 65)
    assert match.doppler_bin == -5
    assert match.elevation_bin == 18
    assert match.xyz_m[1] > 0 and match.xyz_m[2] > 0
    np.testing.assert_allclose(np.linalg.norm(match.xyz_m), 25 * SPEC.range_resolution_m)
