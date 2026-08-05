"""Pure radar profile contracts for offline mmwcore processing."""

from __future__ import annotations

from dataclasses import dataclass

from mmwcore.core import ADCComplexLayout, ADCFrameSpec, PointCloudProjectionSpec


@dataclass(frozen=True)
class RadarProfile:
    """Physical radar profile independent of hardware control side effects."""

    start_frequency_hz: float = 60e9
    frequency_slope_hz_per_s: float = 60.012e12
    adc_sample_rate_hz: float = 4.4e6
    adc_start_time_s: float = 6e-6
    ramp_end_time_s: float = 65e-6
    idle_time_s: float = 360e-6
    num_adc_samples: int = 256
    num_chirps_per_tx: int = 64
    num_tx: int = 3
    num_rx: int = 4
    speed_of_light_mps: float = 299_792_458.0

    def __post_init__(self) -> None:
        for name, value in (
            ("start_frequency_hz", self.start_frequency_hz),
            ("frequency_slope_hz_per_s", self.frequency_slope_hz_per_s),
            ("adc_sample_rate_hz", self.adc_sample_rate_hz),
            ("adc_start_time_s", self.adc_start_time_s),
            ("ramp_end_time_s", self.ramp_end_time_s),
            ("idle_time_s", self.idle_time_s),
            ("speed_of_light_mps", self.speed_of_light_mps),
        ):
            if value <= 0:
                raise ValueError(f"RadarProfile.{name} must be positive; got {value}.")

        for name, value in (
            ("num_adc_samples", self.num_adc_samples),
            ("num_chirps_per_tx", self.num_chirps_per_tx),
            ("num_tx", self.num_tx),
            ("num_rx", self.num_rx),
        ):
            if value <= 0:
                raise ValueError(f"RadarProfile.{name} must be positive; got {value}.")

        if self.adc_start_time_s >= self.ramp_end_time_s:
            raise ValueError("RadarProfile.adc_start_time_s must be before ramp_end_time_s.")

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_light_mps / self.start_frequency_hz

    @property
    def adc_sample_time_s(self) -> float:
        return self.num_adc_samples / self.adc_sample_rate_hz

    @property
    def usable_ramp_time_s(self) -> float:
        return self.ramp_end_time_s - self.adc_start_time_s

    @property
    def bandwidth_hz(self) -> float:
        return self.frequency_slope_hz_per_s * min(
            self.adc_sample_time_s,
            self.usable_ramp_time_s,
        )

    @property
    def chirp_period_s(self) -> float:
        return self.idle_time_s + self.ramp_end_time_s

    @property
    def tdm_loop_period_s(self) -> float:
        return self.chirp_period_s * self.num_tx

    @property
    def chirps_per_frame(self) -> int:
        return self.num_chirps_per_tx * self.num_tx

    @property
    def virtual_antennas(self) -> int:
        return self.num_tx * self.num_rx

    @property
    def range_resolution_m(self) -> float:
        return self.speed_of_light_mps / (2 * self.bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        return self.range_resolution_m * self.num_adc_samples

    @property
    def max_velocity_mps(self) -> float:
        return self.wavelength_m / (4 * self.tdm_loop_period_s)

    @property
    def velocity_resolution_mps(self) -> float:
        return (2 * self.max_velocity_mps) / self.num_chirps_per_tx

    def to_adc_frame_spec(
        self,
        *,
        layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED,
    ) -> ADCFrameSpec:
        """Build the ADC frame shape used by offline cube organization."""

        return ADCFrameSpec(
            num_chirps=self.chirps_per_frame,
            num_rx=self.num_rx,
            num_samples=self.num_adc_samples,
            layout=layout,
        )

    def to_point_cloud_projection_spec(
        self,
        *,
        center_doppler: bool = True,
        doppler_bins: int | None = None,
        doppler_fftshifted: bool = False,
    ) -> PointCloudProjectionSpec:
        """Build the simple range-Doppler projection spec from profile geometry."""

        bins = doppler_bins if doppler_bins is not None else self.num_chirps_per_tx
        return PointCloudProjectionSpec(
            range_resolution_m=self.range_resolution_m,
            doppler_resolution_mps=self.velocity_resolution_mps,
            center_doppler=center_doppler,
            doppler_bins=bins if center_doppler else doppler_bins,
            doppler_fftshifted=doppler_fftshifted,
        )
