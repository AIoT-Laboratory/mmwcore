from __future__ import annotations

import ast
from pathlib import Path


def test_mmwcore_source_does_not_import_upper_layers() -> None:
    repository_root = Path(__file__).parents[1]
    package_root = repository_root / "python" / "mmwcore"
    source_paths = sorted(package_root.rglob("*.py"))

    violations: list[str] = []
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for module in _imported_modules(node):
                if module.split(".", 1)[0] in {"openmmw", "mmwapps", "mmwlab"}:
                    line = getattr(node, "lineno", 0)
                    violations.append(f"{path.relative_to(repository_root)}:{line} {module}")

    assert violations == []


def _imported_modules(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.ImportFrom):
        return (node.module,) if node.module is not None else ()
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    return ()
