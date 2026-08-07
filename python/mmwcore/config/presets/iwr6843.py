"""IWR6843 radar profile presets."""

from __future__ import annotations

from dataclasses import replace

from mmwcore.config.profiles import RadarProfile
from mmwcore.core import (
    ADCComplexLayout,
    ADCDecodeRecipe,
    AngleFFTSpec,
    AntennaArrayGeometry,
    CFARDetectionSpec,
    DetectionMethod,
    DetectionQualitySpec,
    DetectionRecipe,
    DopplerFFTSpec,
    FFTWindow,
    PeakDetectionSpec,
    PeakGroupingSpec,
    PlanarApertureLayout,
    PointCloudRecipe,
    RangeDopplerCFARSpec,
    RangeDopplerRecipe,
    RangeFFTSpec,
    TDMVirtualArraySpec,
    VirtualAntennaLayout,
    VirtualChannelCalibration,
    VirtualSubarraySpec,
)


def iwr6843_profile(**overrides: object) -> RadarProfile:
    """Return the default IWR6843-style RadarProfile with optional overrides."""

    profile = RadarProfile(
        start_frequency_hz=60e9,
        frequency_slope_hz_per_s=60.012e12,
        adc_sample_rate_hz=4.4e6,
        idle_time_s=360e-6,
        adc_start_time_s=6e-6,
        ramp_end_time_s=65e-6,
        num_adc_samples=256,
        num_chirps_per_tx=64,
        num_tx=3,
        num_rx=4,
    )
    return replace(profile, **overrides)


def iwr6843_isk_antenna_geometry() -> AntennaArrayGeometry:
    """Return standard IWR6843 ISK/EVM phase centers in wavelength units."""

    half_wavelength = 0.5
    return AntennaArrayGeometry(
        tx_positions_wavelengths=tuple(
            (azimuth * half_wavelength, 0.0, elevation * half_wavelength)
            for azimuth, elevation in ((0, 1), (2, 0), (4, 1))
        ),
        rx_positions_wavelengths=tuple(
            (azimuth * half_wavelength, 0.0, 0.0) for azimuth in range(4)
        ),
        name="iwr6843_isk",
    )


def iwr6843_isk_tdm_virtual_array(
    *,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> TDMVirtualArraySpec:
    """Return the standard ISK geometry with an explicit TDM transmit order."""

    return TDMVirtualArraySpec(iwr6843_isk_antenna_geometry(), tx_order)


def iwr6843_isk_planar_aperture_layout(
    *,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> PlanarApertureLayout:
    """Return the sparse half-wavelength ISK planar aperture."""

    positions = iwr6843_isk_tdm_virtual_array(tx_order=tx_order).virtual_layout()
    grid_indices: list[tuple[int, int]] = []
    for azimuth, depth, elevation in positions.positions_wavelengths:
        if depth != 0.0:
            raise ValueError("IWR6843 ISK aperture phase centers must share one depth.")
        azimuth_index = round(azimuth / 0.5)
        elevation_index = round(elevation / 0.5)
        if azimuth != azimuth_index * 0.5 or elevation != elevation_index * 0.5:
            raise ValueError("IWR6843 ISK aperture must lie on a half-wavelength grid.")
        grid_indices.append((azimuth_index, elevation_index))
    return PlanarApertureLayout(
        tuple(grid_indices),
        name="iwr6843_isk_planar",
    )


def iwr6843_isk_azimuth_subarray(
    *,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> VirtualSubarraySpec:
    """Return the 8-channel horizontal ULA for an explicit ISK Tx order."""

    tdm = iwr6843_isk_tdm_virtual_array(tx_order=tx_order)
    azimuth_slots = tuple(tx_order.index(tx_index) for tx_index in (0, 2))
    indices = tuple(
        slot * tdm.geometry.num_rx + rx_index
        for slot in azimuth_slots
        for rx_index in range(tdm.geometry.num_rx)
    )
    full_layout = tdm.virtual_layout()
    positions = tuple(full_layout.positions_wavelengths[index] for index in indices)
    return VirtualSubarraySpec(
        antenna_indices=indices,
        layout=VirtualAntennaLayout(
            positions,
            name="iwr6843_isk_azimuth",
            angle_axis="azimuth",
        ),
    )


def iwr6843_isk_elevation_subarray(
    *,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> VirtualSubarraySpec:
    """Return the four-channel ISK row displaced below the azimuth ULA."""

    tdm = iwr6843_isk_tdm_virtual_array(tx_order=tx_order)
    elevation_slot = tx_order.index(1)
    indices = tuple(
        elevation_slot * tdm.geometry.num_rx + rx_index for rx_index in range(tdm.geometry.num_rx)
    )
    full_layout = tdm.virtual_layout()
    positions = tuple(full_layout.positions_wavelengths[index] for index in indices)
    return VirtualSubarraySpec(
        antenna_indices=indices,
        layout=VirtualAntennaLayout(
            positions,
            name="iwr6843_isk_elevation_row",
            angle_axis="azimuth",
        ),
    )


def iwr6843_isk_range_doppler_recipe(
    profile: RadarProfile | None = None,
    *,
    adc_layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED,
    range_window: FFTWindow = FFTWindow.HANN,
    doppler_window: FFTWindow = FFTWindow.HANN,
    remove_range_dc: bool = False,
    remove_static_clutter: bool = False,
    channel_calibration: VirtualChannelCalibration | None = None,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> RangeDopplerRecipe:
    """Build the standard ISK ADC-to-range-Doppler recipe."""

    radar = profile or iwr6843_profile()
    _require_isk_shape(radar, tx_order=tx_order)
    return RangeDopplerRecipe(
        decode=ADCDecodeRecipe(radar.to_adc_frame_spec(layout=adc_layout)),
        range_fft=RangeFFTSpec(
            n_fft=radar.num_adc_samples,
            window=range_window,
            one_sided=True,
            remove_dc=remove_range_dc,
        ),
        doppler_fft=DopplerFFTSpec(
            n_fft=radar.num_chirps_per_tx,
            window=doppler_window,
            fftshift=True,
            input_axis="loop",
        ),
        tdm_virtual_array=iwr6843_isk_tdm_virtual_array(tx_order=tx_order),
        channel_calibration=channel_calibration,
        remove_static_clutter=remove_static_clutter,
    )


def iwr6843_isk_detection_recipe(
    threshold: float,
    profile: RadarProfile | None = None,
    *,
    adc_layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED,
    range_window: FFTWindow = FFTWindow.HANN,
    doppler_window: FFTWindow = FFTWindow.HANN,
    angle_window: FFTWindow = FFTWindow.HANN,
    angle_n_fft: int = 64,
    azimuth_peak_radius: int = 1,
    azimuth_peak_strict: bool = True,
    remove_static_clutter: bool = False,
    channel_calibration: VirtualChannelCalibration | None = None,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> DetectionRecipe:
    """Build the standard ISK threshold-detection and azimuth-AoA recipe."""

    subarray = iwr6843_isk_azimuth_subarray(tx_order=tx_order)
    return DetectionRecipe(
        transform=iwr6843_isk_range_doppler_recipe(
            profile,
            adc_layout=adc_layout,
            range_window=range_window,
            doppler_window=doppler_window,
            remove_static_clutter=remove_static_clutter,
            channel_calibration=channel_calibration,
            tx_order=tx_order,
        ),
        peak_detection=PeakDetectionSpec(
            threshold=threshold,
            azimuth_peak_radius=azimuth_peak_radius,
            azimuth_peak_strict=azimuth_peak_strict,
        ),
        angle_fft=AngleFFTSpec(
            n_fft=angle_n_fft,
            window=angle_window,
            fftshift=True,
            input_axis="virtual_rx",
            virtual_layout=subarray.layout,
        ),
        virtual_subarray=subarray,
    )


def iwr6843_isk_point_cloud_recipe(
    threshold: float,
    profile: RadarProfile | None = None,
    *,
    adc_layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED,
    range_window: FFTWindow = FFTWindow.HANN,
    doppler_window: FFTWindow = FFTWindow.HANN,
    angle_window: FFTWindow = FFTWindow.HANN,
    angle_n_fft: int = 64,
    azimuth_peak_radius: int = 1,
    azimuth_peak_strict: bool = True,
    remove_static_clutter: bool = False,
    channel_calibration: VirtualChannelCalibration | None = None,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> PointCloudRecipe:
    """Build the standard ISK ADC-to-calibrated-point-cloud recipe."""

    radar = profile or iwr6843_profile()
    return PointCloudRecipe(
        detection=iwr6843_isk_detection_recipe(
            threshold,
            radar,
            adc_layout=adc_layout,
            range_window=range_window,
            doppler_window=doppler_window,
            angle_window=angle_window,
            angle_n_fft=angle_n_fft,
            azimuth_peak_radius=azimuth_peak_radius,
            azimuth_peak_strict=azimuth_peak_strict,
            remove_static_clutter=remove_static_clutter,
            channel_calibration=channel_calibration,
            tx_order=tx_order,
        ),
        projection=radar.to_point_cloud_projection_spec(doppler_fftshifted=True),
    )


def iwr6843_isk_3d_point_cloud_recipe(
    threshold: float,
    profile: RadarProfile | None = None,
    *,
    adc_layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED,
    range_window: FFTWindow = FFTWindow.HANN,
    doppler_window: FFTWindow = FFTWindow.HANN,
    angle_window: FFTWindow = FFTWindow.HANN,
    angle_n_fft: int = 64,
    azimuth_peak_radius: int = 1,
    azimuth_peak_strict: bool = True,
    remove_static_clutter: bool = False,
    channel_calibration: VirtualChannelCalibration | None = None,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> PointCloudRecipe:
    """Build the ISK point-cloud recipe with coupled azimuth/elevation AoA."""

    recipe = iwr6843_isk_point_cloud_recipe(
        threshold,
        profile,
        adc_layout=adc_layout,
        range_window=range_window,
        doppler_window=doppler_window,
        angle_window=angle_window,
        angle_n_fft=angle_n_fft,
        azimuth_peak_radius=azimuth_peak_radius,
        azimuth_peak_strict=azimuth_peak_strict,
        remove_static_clutter=remove_static_clutter,
        channel_calibration=channel_calibration,
        tx_order=tx_order,
    )
    return replace(
        recipe,
        detection=replace(
            recipe.detection,
            elevation_subarray=iwr6843_isk_elevation_subarray(tx_order=tx_order),
        ),
    )


def iwr6843_isk_cfar_point_cloud_recipe(
    cfar_detection: CFARDetectionSpec | RangeDopplerCFARSpec,
    peak_grouping: PeakGroupingSpec | None = None,
    profile: RadarProfile | None = None,
    *,
    quality_filter: DetectionQualitySpec | None = None,
    adc_layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED,
    range_window: FFTWindow = FFTWindow.HANN,
    doppler_window: FFTWindow = FFTWindow.HANN,
    angle_window: FFTWindow = FFTWindow.HANN,
    angle_n_fft: int = 64,
    remove_static_clutter: bool = False,
    channel_calibration: VirtualChannelCalibration | None = None,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> PointCloudRecipe:
    """Build two-stage ISK range-Doppler CFAR and candidate-AoA processing."""

    radar = profile or iwr6843_profile()
    subarray = iwr6843_isk_azimuth_subarray(tx_order=tx_order)
    return PointCloudRecipe(
        detection=DetectionRecipe(
            transform=iwr6843_isk_range_doppler_recipe(
                radar,
                adc_layout=adc_layout,
                range_window=range_window,
                doppler_window=doppler_window,
                remove_static_clutter=remove_static_clutter,
                channel_calibration=channel_calibration,
                tx_order=tx_order,
            ),
            detection_method=DetectionMethod.CFAR,
            cfar_detection=cfar_detection,
            peak_grouping=peak_grouping,
            quality_filter=quality_filter,
            angle_fft=AngleFFTSpec(
                n_fft=angle_n_fft,
                window=angle_window,
                fftshift=True,
                input_axis="virtual_rx",
                virtual_layout=subarray.layout,
            ),
            virtual_subarray=subarray,
        ),
        projection=radar.to_point_cloud_projection_spec(doppler_fftshifted=True),
    )


def iwr6843_isk_3d_cfar_point_cloud_recipe(
    cfar_detection: CFARDetectionSpec | RangeDopplerCFARSpec,
    peak_grouping: PeakGroupingSpec | None = None,
    profile: RadarProfile | None = None,
    *,
    quality_filter: DetectionQualitySpec | None = None,
    adc_layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED,
    range_window: FFTWindow = FFTWindow.HANN,
    doppler_window: FFTWindow = FFTWindow.HANN,
    angle_window: FFTWindow = FFTWindow.HANN,
    angle_n_fft: int = 64,
    remove_static_clutter: bool = False,
    channel_calibration: VirtualChannelCalibration | None = None,
    tx_order: tuple[int, ...] = (0, 2, 1),
) -> PointCloudRecipe:
    """Build ISK range-Doppler CFAR with coupled azimuth/elevation AoA."""

    recipe = iwr6843_isk_cfar_point_cloud_recipe(
        cfar_detection,
        peak_grouping,
        profile,
        quality_filter=quality_filter,
        adc_layout=adc_layout,
        range_window=range_window,
        doppler_window=doppler_window,
        angle_window=angle_window,
        angle_n_fft=angle_n_fft,
        remove_static_clutter=remove_static_clutter,
        channel_calibration=channel_calibration,
        tx_order=tx_order,
    )
    return replace(
        recipe,
        detection=replace(
            recipe.detection,
            elevation_subarray=iwr6843_isk_elevation_subarray(tx_order=tx_order),
        ),
    )


def _require_isk_shape(profile: RadarProfile, *, tx_order: tuple[int, ...]) -> None:
    if profile.num_rx != 4:
        raise ValueError("IWR6843 ISK recipes require 4 Rx antennas.")
    if profile.num_tx != len(tx_order):
        raise ValueError("IWR6843 ISK profile active-Tx count must match the configured Tx order.")
