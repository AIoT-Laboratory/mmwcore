import numpy as np
import pytest
from numpy.typing import NDArray

from mmwcore import _native
from mmwcore.config import RadarProfile, iwr6843_isk_point_cloud_pipeline
from mmwcore.core import ADCComplexLayout, ADCFrame, FFTWindow
from mmwcore.dsp import point_cloud


@pytest.mark.parametrize("layout", [2, 4])
def test_grouped_iq_decoder_preserves_components_frames_and_receivers(layout: int) -> None:
    expected = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 4) - 16
    values = expected + 1j * (expected * 3 + 1)
    pairs = values.reshape(2, 2, 2, 2, 2)
    first, second = (pairs.real, pairs.imag) if layout == 2 else (pairs.imag, pairs.real)
    raw = np.concatenate((first, second), axis=-1).astype(np.int16).ravel()
    np.testing.assert_array_equal(_native.decode_adc_i16(raw, 2, 2, 4, layout, False), values)


@pytest.mark.parametrize("direction", [-1, 0, 1], ids=["approaching", "stationary", "receding"])
@pytest.mark.parametrize("tx_order", [(0, 1, 2), (0, 2, 1)])
@pytest.mark.parametrize(
    "layout", [ADCComplexLayout.GROUP2_I_THEN_Q, ADCComplexLayout.GROUP2_Q_THEN_I]
)
def test_physical_range_rate_matches_rpc_velocity(direction, tx_order, layout) -> None:
    # Positive-slope dechirped FMCW: phase = 4*pi*(S*r*t_fast/c + r/lambda).
    # Generate from r(t), not from the Doppler bin or a chosen projection sign.
    profile = RadarProfile(num_adc_samples=64, num_chirps_per_tx=32, idle_time_s=7e-6)
    velocity = direction * 2 * profile.velocity_resolution_mps
    initial_range = 8 * profile.range_resolution_m
    chirp_time = np.arange(profile.chirps_per_frame) * profile.chirp_period_s
    ranges = initial_range + velocity * chirp_time
    fast_time = np.arange(profile.num_adc_samples) / profile.adc_sample_rate_hz
    phase = (
        4
        * np.pi
        * (
            profile.frequency_slope_hz_per_s
            * ranges[:, None]
            * fast_time
            / profile.speed_of_light_mps
            + ranges[:, None] / profile.wavelength_m
        )
    )
    signal = np.broadcast_to((1000 * np.exp(1j * phase))[:, None, :], (96, 4, 64))
    grouped: NDArray[np.complex128] = signal.reshape(96, 4, 32, 2)
    first, second = (
        (grouped.real, grouped.imag)
        if layout is ADCComplexLayout.GROUP2_I_THEN_Q
        else (grouped.imag, grouped.real)
    )
    raw = ADCFrame(np.rint(np.concatenate((first, second), axis=-1)).astype(np.int16).ravel())
    recipe = iwr6843_isk_point_cloud_pipeline(
        100_000,
        profile,
        adc_layout=layout,
        tx_order=tx_order,
        range_window=FFTWindow.HANN,
        doppler_window=FFTWindow.HANN,
        angle_window=FFTWindow.NONE,
        angle_n_fft=8,
    )
    cloud = point_cloud(raw, recipe)
    strongest = cloud.points[np.argmax(cloud.points[:, cloud.channels.index("magnitude")])]
    assert strongest[1] == pytest.approx(initial_range, abs=profile.range_resolution_m / 2)
    assert strongest[3] == pytest.approx(velocity, abs=1e-6)
    assert np.sign(strongest[3]) == np.sign(ranges[-1] - ranges[0])
