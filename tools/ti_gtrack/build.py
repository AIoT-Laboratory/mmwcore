"""Build a local TI-device-only GTRACK plugin from a pinned external SDK installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk-packages",
        type=Path,
        required=True,
        help="The pinned trackerproc_overhead/packages directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local build directory, outside distributable packages",
    )
    parser.add_argument("--cc", default="gcc", help="GCC-compatible host C compiler")
    args = parser.parse_args()
    packages = args.sdk_packages.resolve(strict=True)
    base = packages / "ti/alg/gtrack"
    pin = json.loads((HERE / "source-lock.json").read_text(encoding="utf-8"))
    for relative, expected in pin["sha256"].items():
        path = base / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(
                f"TI source mismatch: {path}; expected pinned Radar Toolbox {pin['version']}"
            )
    output = args.output.resolve()
    if output == base or base in output.parents:
        raise ValueError("Build output must not overwrite the TI SDK source installation")
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    library = output / ("mmw_ti_gtrack" + suffix)
    compiler = shutil.which(args.cc)
    if compiler is None:
        raise ValueError(f"Compiler not found: {args.cc}")
    sources = sorted(base / p for p in pin["sha256"] if p.endswith(".c"))
    command = [
        compiler,
        "-shared",
        "-O2",
        "-std=c11",
        "-DGTRACK_3D",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-I",
        str(packages),
        "-I",
        str(HERE),
        "-o",
        str(library),
        str(HERE / "bridge.c"),
        *map(str, sources),
        "-lm",
    ]
    if sys.platform == "win32":
        command.insert(2, "-static-libgcc")
    else:
        command.insert(2, "-fPIC")
    subprocess.run(command, check=True)
    license_text = (base / "src/gtrack_create.c").read_text(encoding="utf-8").split("*/", 1)[
        0
    ] + "*/\n"
    (output / "TI-LICENSE.txt").write_text(license_text, encoding="utf-8")
    manifest = {
        "schema": "mmwcore.ti-gtrack-plugin.v1",
        "abi": 1,
        "source_version": pin["version"],
        "source_root": str(base),
        "source_lock_sha256": sha(HERE / "source-lock.json"),
        "source_sha256": pin["sha256"],
        "default_tracking_profile": pin["default_tracking_profile"],
        "library": library.name,
        "library_sha256": sha(library),
        "bridge_sha256": {name: sha(HERE / name) for name in ["bridge.c", "bridge.h", "build.py"]},
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "platform": platform.platform(),
        "command": command,
        "license": "TI Devices only; see TI-LICENSE.txt",
        "numerics": "Unmodified TI GTRACK_3D; host compiler, not board-binary equivalence",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
