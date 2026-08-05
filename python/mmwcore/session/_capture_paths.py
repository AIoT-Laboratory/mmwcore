"""Safe path resolution shared by synchronized capture artifacts."""

from __future__ import annotations

from pathlib import Path


def manifest_path(path: str | Path) -> Path:
    value = Path(path)
    return value / "manifest.json" if value.is_dir() else value


def validate_relative_reference(reference: str, name: str) -> None:
    path = Path(reference)
    if not reference or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Synchronized capture {name} must be a safe relative path reference.")


def resolve_relative_reference(
    root: Path,
    reference: str,
    name: str,
) -> Path:
    validate_relative_reference(reference, name)
    resolved = (root / reference).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Synchronized capture {name} escapes its manifest root.") from exc
    return resolved


__all__ = [
    "manifest_path",
    "resolve_relative_reference",
    "validate_relative_reference",
]
