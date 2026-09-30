"""Shared clustering, geometry and track-status contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from operator import index as integer_index
from sys import maxsize as _MAX_PLATFORM_INDEX


class TrackStatus(StrEnum):
    """Lifecycle state for one target track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        normalized = int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if not -_MAX_PLATFORM_INDEX - 1 <= normalized <= _MAX_PLATFORM_INDEX:
        raise OverflowError(f"{name} must fit the platform index range.")
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _require_finite_positive(value: float, *, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _require_finite_non_negative(value: float, *, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class DBSCANSpec:
    """DBSCAN policy for Cartesian radar points and optional radial velocity."""

    eps_m: float
    min_samples: int
    velocity_scale_s: float = 0.0
    use_z: bool = True

    def __post_init__(self) -> None:
        _require_finite_positive(self.eps_m, name="DBSCANSpec.eps_m")
        object.__setattr__(
            self,
            "min_samples",
            _positive_integer(self.min_samples, name="DBSCANSpec.min_samples"),
        )
        _require_finite_non_negative(
            self.velocity_scale_s,
            name="DBSCANSpec.velocity_scale_s",
        )
        if type(self.use_z) is not bool:
            raise TypeError("DBSCANSpec.use_z must be a bool.")


@dataclass(frozen=True)
class Box2D:
    """Inclusive Cartesian tracking region in radar x/y coordinates."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x_min_m", self.x_min_m),
            ("x_max_m", self.x_max_m),
            ("y_min_m", self.y_min_m),
            ("y_max_m", self.y_max_m),
        ):
            if isinstance(value, bool):
                raise TypeError(f"Box2D.{name} must be a real number, not bool.")
            if not isfinite(value):
                raise ValueError(f"Box2D.{name} must be finite.")
        if self.x_min_m >= self.x_max_m or self.y_min_m >= self.y_max_m:
            raise ValueError("Box2D minimum bounds must be below maximum bounds.")

    def contains(self, x_m: float, y_m: float) -> bool:
        return self.x_min_m <= x_m <= self.x_max_m and self.y_min_m <= y_m <= self.y_max_m


@dataclass(frozen=True)
class Box3D:
    """Inclusive Cartesian tracking region in three dimensions."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x_min_m", self.x_min_m),
            ("x_max_m", self.x_max_m),
            ("y_min_m", self.y_min_m),
            ("y_max_m", self.y_max_m),
            ("z_min_m", self.z_min_m),
            ("z_max_m", self.z_max_m),
        ):
            if isinstance(value, bool):
                raise TypeError(f"Box3D.{name} must be a real number, not bool.")
            if not isfinite(value):
                raise ValueError(f"Box3D.{name} must be finite.")
        if (
            self.x_min_m >= self.x_max_m
            or self.y_min_m >= self.y_max_m
            or self.z_min_m >= self.z_max_m
        ):
            raise ValueError("Box3D minimum bounds must be below maximum bounds.")

    def contains(self, x_m: float, y_m: float, z_m: float) -> bool:
        return (
            self.x_min_m <= x_m <= self.x_max_m
            and self.y_min_m <= y_m <= self.y_max_m
            and self.z_min_m <= z_m <= self.z_max_m
        )
