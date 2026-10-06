"""Check one wheel and its source distribution outside the checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def check_distribution(artifacts: Path) -> None:
    artifacts = artifacts.resolve(strict=True)
    (wheel,) = artifacts.glob("*.whl")
    (sdist,) = artifacts.glob("*.tar.gz")
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mmwcore-installed-") as directory:
        work = Path(directory)
        environment = work / "venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--only-binary", ":all:", str(wheel)],
            check=True,
        )
        subprocess.run(
            [str(python), "-I", str(root / "tests/distribution_smoke.py"), str(artifacts)],
            cwd=work,
            check=True,
        )
        source = work / "source"
        with tarfile.open(sdist) as archive:
            archive.extractall(source, filter="data")
        (project,) = source.iterdir()
        # Building a wheel does not compile the Rust tests or check their fixtures.
        subprocess.run(
            ["cargo", "test", "--workspace", "--locked"],
            cwd=project,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="Directory with exactly one wheel and sdist")
    check_distribution(parser.parse_args().artifacts)
