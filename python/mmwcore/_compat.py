"""Small standard-library compatibility surface for supported Python versions."""

# ruff: noqa: UP035, UP036, UP042

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import Self
else:
    from enum import Enum

    from typing_extensions import Self

    class StrEnum(str, Enum):
        """Python 3.10 subset of enum.StrEnum used by mmwcore."""

        __str__ = str.__str__
        __format__ = str.__format__


__all__ = ["Self", "StrEnum"]
