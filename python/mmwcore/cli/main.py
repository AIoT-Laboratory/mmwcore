"""Unified mmwcore command line."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    target: Callable[[list[str]], int | None]
    help: str


def _inspect(argv: list[str]) -> int:
    from .inspect import main

    return main(argv)


_COMMANDS = {
    "inspect": Command(_inspect, "Inspect raw radar inputs."),
}


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        _print_help()
        return 0
    name, *remaining = values
    command = _COMMANDS.get(name)
    if command is None:
        print(f"mmwcore: unknown command {name!r}", file=sys.stderr)
        print("Run 'mmwcore --help' for available commands.", file=sys.stderr)
        return 2
    previous = sys.argv
    sys.argv = [f"mmwcore {name}", *remaining]
    try:
        result = command.target(remaining)
    finally:
        sys.argv = previous
    return int(result) if result is not None else 0


def _print_help() -> None:
    print("usage: mmwcore <command> [args]")
    print()
    print("Low-level mmWave radar input inspection.")
    print()
    print("commands:")
    width = max(len(name) for name in _COMMANDS)
    for name, command in _COMMANDS.items():
        print(f"  {name:<{width}}  {command.help}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
