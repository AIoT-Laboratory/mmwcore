"""Typed, physically explicit mmWave radar processing."""

from __future__ import annotations

from importlib.metadata import version

from .io import open_capture

__all__ = ["__version__", "open_capture"]

__version__ = version("mmwcore")
