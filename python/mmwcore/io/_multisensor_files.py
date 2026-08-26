"""Validate immutable files in published multi-sensor capture directories."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from typing import BinaryIO


def _session_root(path: str | Path) -> Path:
    try:
        root = Path(path).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Multisensor capture directory is unavailable: {path}.") from exc
    if not root.is_dir() or root.name.endswith(".part"):
        raise ValueError(f"Multisensor capture path is not a published directory: {root}.")
    return root


def _regular_leaf(root: Path, name: str, label: str) -> Path:
    path = root / name
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}.") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}.")
    return path


def _directory_leaf(root: Path, name: str, label: str) -> Path:
    path = root / name
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}.") from exc
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{label} is not a directory: {path}.")
    return path


def _require_directory_names(root: Path, expected: set[str], label: str) -> None:
    try:
        actual = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise ValueError(f"{label} cannot be listed: {root}.") from exc
    if actual != expected:
        raise ValueError(f"{label} has undeclared or missing leaves.")


def _read_bounded_regular(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its regular-file bound.")
    payload = path.read_bytes()
    if len(payload) != status.st_size or len(payload) > maximum_bytes:
        raise ValueError(f"{label} changed while it was read.")
    return payload


def _require_file_size(path: Path, expected: int, label: str) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}.") from exc
    if not stat.S_ISREG(status.st_mode) or status.st_size != expected:
        raise ValueError(f"{label} size does not match session.json.")


def _sha256_file(path: Path, expected_size: int) -> str:
    _require_file_size(path, expected_size, "source artifact")
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    _require_file_size(path, expected_size, "source artifact")
    return digest


def _read_exact(file: BinaryIO, size: int, label: str) -> bytes:
    payload = file.read(size)
    if type(payload) is not bytes or len(payload) != size:
        raise ValueError(f"{label} is truncated.")
    return payload
