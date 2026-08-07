"""TI SDK source-backed antenna geometry presets for offline processing."""

from __future__ import annotations

from mmwcore.core import AntennaArrayGeometry

_HalfWavelengthOffset = tuple[int, int]


def xwr1642_antenna_geometry() -> AntennaArrayGeometry:
    """Return TI SDK XWR1642 phase centers in wavelength units.

    This source-backed preset is covered by offline tests; it does not claim hardware validation.
    """

    return _geometry_from_half_wavelength_offsets(
        name="xwr1642",
        tx_offsets=((0, 0), (4, 0)),
        rx_offsets=((0, 0), (1, 0), (2, 0), (3, 0)),
    )


def xwr1843_evm_antenna_geometry() -> AntennaArrayGeometry:
    """Return TI SDK standard XWR1843 EVM phase centers in wavelength units.

    This source-backed preset is covered by offline tests; it does not claim hardware validation.
    """

    return _geometry_from_half_wavelength_offsets(
        name="xwr1843_evm",
        tx_offsets=((0, 1), (2, 0), (4, 1)),
        rx_offsets=((0, 0), (1, 0), (2, 0), (3, 0)),
    )


def iwr6843_aop_antenna_geometry() -> AntennaArrayGeometry:
    """Return TI SDK IWR6843 AOP phase centers in wavelength units.

    This source-backed preset is covered by offline tests; it does not claim hardware validation.
    """

    return _geometry_from_half_wavelength_offsets(
        name="iwr6843_aop",
        tx_offsets=((0, 0), (2, 2), (0, 2)),
        rx_offsets=((1, 1), (1, 0), (0, 1), (0, 0)),
    )


def awr1843_aop_antenna_geometry() -> AntennaArrayGeometry:
    """Return TI SDK AWR1843 AOP phase centers in wavelength units.

    This source-backed preset is covered by offline tests; it does not claim hardware validation.
    """

    return _geometry_from_half_wavelength_offsets(
        name="awr1843_aop",
        tx_offsets=((0, 0), (0, 1), (0, 2)),
        rx_offsets=((3, 0), (2, 0), (1, 0), (0, 0)),
    )


def _geometry_from_half_wavelength_offsets(
    *,
    name: str,
    tx_offsets: tuple[_HalfWavelengthOffset, ...],
    rx_offsets: tuple[_HalfWavelengthOffset, ...],
) -> AntennaArrayGeometry:
    return AntennaArrayGeometry(
        tx_positions_wavelengths=tuple(_wavelength_position(offset) for offset in tx_offsets),
        rx_positions_wavelengths=tuple(_wavelength_position(offset) for offset in rx_offsets),
        name=name,
    )


def _wavelength_position(offset: _HalfWavelengthOffset) -> tuple[float, float, float]:
    azimuth, elevation = offset
    return azimuth * 0.5, 0.0, elevation * 0.5
