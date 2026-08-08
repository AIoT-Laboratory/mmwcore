"""ADC frame and virtual antenna layout specs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from operator import index as integer_index
from sys import maxsize as _MAX_PLATFORM_INDEX

from .spec_enums import ADCComplexLayout

_ANGLE_AXES = frozenset({"azimuth", "elevation"})


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool, string, or complex.")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite; got {normalized}.")
    return normalized


def _positive_real(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive; got {normalized}.")
    return normalized


def _non_empty_name(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _angle_axis(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value not in _ANGLE_AXES:
        raise ValueError(f"Unsupported angle axis: {value}.")
    return value


@dataclass(frozen=True)
class VirtualAntennaLayout:
    """Virtual antenna positions expressed in wavelengths."""

    positions_wavelengths: tuple[tuple[float, float, float], ...]
    name: str = "virtual_array"
    angle_axis: str = "azimuth"

    def __post_init__(self) -> None:
        positions = _positions(
            self.positions_wavelengths,
            name="VirtualAntennaLayout.positions_wavelengths",
        )
        name = _non_empty_name(self.name, name="VirtualAntennaLayout.name")
        angle_axis = _angle_axis(
            self.angle_axis,
            name="VirtualAntennaLayout.angle_axis",
        )

        object.__setattr__(self, "positions_wavelengths", positions)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "angle_axis", angle_axis)

    @classmethod
    def uniform_linear(
        cls,
        num_antennas: int,
        *,
        spacing_wavelengths: float = 0.5,
        name: str = "uniform_linear",
        angle_axis: str = "azimuth",
    ) -> VirtualAntennaLayout:
        """Create a linear virtual array along the x-axis."""

        num_antennas = _positive_dimension(num_antennas, name="num_antennas")
        spacing_wavelengths = _positive_real(
            spacing_wavelengths,
            name="VirtualAntennaLayout.uniform_linear.spacing_wavelengths",
        )
        return cls(
            positions_wavelengths=tuple(
                (index * spacing_wavelengths, 0.0, 0.0) for index in range(num_antennas)
            ),
            name=name,
            angle_axis=angle_axis,
        )

    @property
    def num_antennas(self) -> int:
        return len(self.positions_wavelengths)

    def as_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "angle_axis": self.angle_axis,
            "num_antennas": self.num_antennas,
            "positions_wavelengths": [list(position) for position in self.positions_wavelengths],
        }


@dataclass(frozen=True)
class PlanarApertureLayout:
    """Sparse virtual-channel locations on a zero-based planar FFT grid."""

    grid_indices: tuple[tuple[int, int], ...]
    name: str = "planar_aperture"

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.grid_indices)
        if not rows:
            raise ValueError("PlanarApertureLayout.grid_indices must not be empty.")
        if any(len(row) != 2 for row in rows):
            raise ValueError("Planar aperture indices must contain azimuth/elevation pairs.")
        indices = tuple(
            _integer_indices(row, name="PlanarApertureLayout.grid_indices") for row in rows
        )
        if any(value < 0 for index in indices for value in index):
            raise ValueError("Planar aperture indices must be non-negative.")
        name = _non_empty_name(self.name, name="PlanarApertureLayout.name")
        object.__setattr__(self, "grid_indices", indices)
        object.__setattr__(self, "name", name)

    @property
    def num_antennas(self) -> int:
        return len(self.grid_indices)

    @property
    def num_unique_positions(self) -> int:
        return len(set(self.grid_indices))

    @property
    def aperture_shape(self) -> tuple[int, int]:
        return (
            max(index[0] for index in self.grid_indices) + 1,
            max(index[1] for index in self.grid_indices) + 1,
        )

    def as_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "num_antennas": self.num_antennas,
            "num_unique_positions": self.num_unique_positions,
            "aperture_shape": list(self.aperture_shape),
            "grid_indices": [list(index) for index in self.grid_indices],
            "duplicate_policy": "first",
        }


@dataclass(frozen=True)
class AntennaArrayGeometry:
    """Physical Tx/Rx phase-center coordinates expressed in wavelengths."""

    tx_positions_wavelengths: tuple[tuple[float, float, float], ...]
    rx_positions_wavelengths: tuple[tuple[float, float, float], ...]
    name: str = "antenna_array"

    def __post_init__(self) -> None:
        tx = _positions(
            self.tx_positions_wavelengths,
            name="AntennaArrayGeometry.tx_positions_wavelengths",
        )
        rx = _positions(
            self.rx_positions_wavelengths,
            name="AntennaArrayGeometry.rx_positions_wavelengths",
        )
        name = _non_empty_name(self.name, name="AntennaArrayGeometry.name")
        object.__setattr__(self, "tx_positions_wavelengths", tx)
        object.__setattr__(self, "rx_positions_wavelengths", rx)
        object.__setattr__(self, "name", name)

    @property
    def num_tx(self) -> int:
        return len(self.tx_positions_wavelengths)

    @property
    def num_rx(self) -> int:
        return len(self.rx_positions_wavelengths)

    def virtual_layout(
        self,
        tx_order: tuple[int, ...],
        *,
        angle_axis: str = "azimuth",
    ) -> VirtualAntennaLayout:
        """Build ordered virtual phase centers as Tx + Rx coordinates."""

        order = _validate_tx_order(tx_order, self.num_tx)
        positions = tuple(
            (
                tx[0] + rx[0],
                tx[1] + rx[1],
                tx[2] + rx[2],
            )
            for tx_index in order
            for tx in (self.tx_positions_wavelengths[tx_index],)
            for rx in self.rx_positions_wavelengths
        )
        return VirtualAntennaLayout(
            positions,
            name=f"{self.name}_virtual",
            angle_axis=angle_axis,
        )


@dataclass(frozen=True)
class TDMVirtualArraySpec:
    """TDM-MIMO chirp order and physical antenna geometry."""

    geometry: AntennaArrayGeometry
    tx_order: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, AntennaArrayGeometry):
            raise TypeError("TDMVirtualArraySpec.geometry must be an AntennaArrayGeometry.")
        order = _validate_tx_order(self.tx_order, self.geometry.num_tx)
        object.__setattr__(self, "tx_order", order)

    @property
    def num_tx(self) -> int:
        return len(self.tx_order)

    @property
    def num_virtual_antennas(self) -> int:
        return self.num_tx * self.geometry.num_rx

    def virtual_layout(self, *, angle_axis: str = "azimuth") -> VirtualAntennaLayout:
        return self.geometry.virtual_layout(self.tx_order, angle_axis=angle_axis)


@dataclass(frozen=True)
class VirtualSubarraySpec:
    """Ordered virtual-channel selection with its physical phase centers."""

    antenna_indices: tuple[int, ...]
    layout: VirtualAntennaLayout

    def __post_init__(self) -> None:
        if not isinstance(self.layout, VirtualAntennaLayout):
            raise TypeError("VirtualSubarraySpec.layout must be a VirtualAntennaLayout.")
        indices = _integer_indices(
            self.antenna_indices,
            name="VirtualSubarraySpec.antenna_indices",
        )
        if not indices:
            raise ValueError("VirtualSubarraySpec.antenna_indices must not be empty.")
        if len(set(indices)) != len(indices):
            raise ValueError("VirtualSubarraySpec.antenna_indices must not contain duplicates.")
        if any(index < 0 for index in indices):
            raise ValueError("VirtualSubarraySpec.antenna_indices must be non-negative.")
        if len(indices) != self.layout.num_antennas:
            raise ValueError("VirtualSubarraySpec index count must match layout antenna count.")
        object.__setattr__(self, "antenna_indices", indices)


def _positions(
    values: tuple[tuple[float, float, float], ...],
    *,
    name: str,
) -> tuple[tuple[float, float, float], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of 3D coordinates.")
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of 3D coordinates.") from exc
    if not rows:
        raise ValueError(f"{name} must not be empty.")

    positions: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"{name}[{row_index}] must be a 3D coordinate.")
        try:
            raw_position = tuple(row)
        except TypeError as exc:
            raise TypeError(f"{name}[{row_index}] must be a 3D coordinate.") from exc
        if len(raw_position) != 3:
            raise ValueError(
                f"{name}[{row_index}] must be a 3D coordinate with exactly three values."
            )
        position = tuple(
            _finite_real(
                value,
                name=f"{name}[{row_index}][{coordinate_index}]",
            )
            for coordinate_index, value in enumerate(raw_position)
        )
        positions.append((position[0], position[1], position[2]))
    return tuple(positions)


def _platform_index(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        normalized = integer_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if not -_MAX_PLATFORM_INDEX - 1 <= normalized <= _MAX_PLATFORM_INDEX:
        raise OverflowError(f"{name} must fit the platform index range.")
    return normalized


def _positive_dimension(value: int, *, name: str) -> int:
    normalized = _platform_index(value, name=name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive; got {normalized}.")
    return normalized


def _require_index_product(values: tuple[int, ...], *, name: str) -> None:
    product = 1
    for value in values:
        if product > _MAX_PLATFORM_INDEX // value:
            raise OverflowError(f"{name} exceeds the platform index range.")
        product *= value


def _integer_indices(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    indices: list[int] = []
    for value in values:
        try:
            indices.append(_platform_index(value, name=name))
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"{name} must contain integers within the platform range.") from exc
    return tuple(indices)


def _validate_tx_order(tx_order: tuple[int, ...], num_tx: int) -> tuple[int, ...]:
    order = _integer_indices(tx_order, name="TDMVirtualArraySpec.tx_order")
    if not order:
        raise ValueError("TDM Tx order must not be empty.")
    if len(set(order)) != len(order):
        raise ValueError("TDM Tx order must not contain duplicates.")
    if any(index < 0 or index >= num_tx for index in order):
        raise ValueError(f"TDM Tx order indices must be within [0, {num_tx}).")
    return order


@dataclass(frozen=True)
class ADCFrameSpec:
    """Shape and layout specification for one ADC frame."""

    num_chirps: int
    num_rx: int
    num_samples: int
    layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED

    def __post_init__(self) -> None:
        for name, value in (
            ("num_chirps", self.num_chirps),
            ("num_rx", self.num_rx),
            ("num_samples", self.num_samples),
        ):
            object.__setattr__(
                self,
                name,
                _positive_dimension(value, name=f"ADCFrameSpec.{name}"),
            )
        _require_index_product(
            (self.num_chirps, self.num_rx, self.num_samples, 2),
            name="ADCFrameSpec raw frame size",
        )

        if not isinstance(self.layout, ADCComplexLayout):
            object.__setattr__(self, "layout", ADCComplexLayout(self.layout))

    @property
    def complex_values_per_frame(self) -> int:
        return self.num_chirps * self.num_rx * self.num_samples

    @property
    def raw_values_per_frame(self) -> int:
        return self.complex_values_per_frame * 2


@dataclass(frozen=True)
class CascadeADCFrameSpec:
    """Shape and device order for one multi-device TDM-MIMO ADC frame."""

    num_samples: int
    num_loops: int
    num_tx: int
    num_rx_per_device: int
    device_names: tuple[str, ...]
    layout: ADCComplexLayout = ADCComplexLayout.IQ_INTERLEAVED

    def __post_init__(self) -> None:
        for name, value in (
            ("num_samples", self.num_samples),
            ("num_loops", self.num_loops),
            ("num_tx", self.num_tx),
            ("num_rx_per_device", self.num_rx_per_device),
        ):
            object.__setattr__(
                self,
                name,
                _positive_dimension(value, name=f"CascadeADCFrameSpec.{name}"),
            )
        _require_index_product(
            (
                self.num_samples,
                self.num_loops,
                self.num_tx,
                self.num_rx_per_device,
                4,
            ),
            name="CascadeADCFrameSpec per-device frame size",
        )

        devices = tuple(str(name) for name in self.device_names)
        if not devices or any(not name for name in devices):
            raise ValueError("CascadeADCFrameSpec.device_names must contain non-empty names.")
        if len(set(devices)) != len(devices):
            raise ValueError("CascadeADCFrameSpec.device_names must be unique.")
        if not isinstance(self.layout, ADCComplexLayout):
            object.__setattr__(self, "layout", ADCComplexLayout(self.layout))
        object.__setattr__(self, "device_names", devices)

    @property
    def num_devices(self) -> int:
        return len(self.device_names)

    @property
    def num_rx(self) -> int:
        return self.num_devices * self.num_rx_per_device

    @property
    def complex_values_per_device_frame(self) -> int:
        return self.num_samples * self.num_loops * self.num_tx * self.num_rx_per_device

    @property
    def bytes_per_device_frame(self) -> int:
        return self.complex_values_per_device_frame * 2 * 2
