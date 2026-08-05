from __future__ import annotations

import importlib
import sys


def test_stable_mmwcore_imports_do_not_load_optional_hardware_modules() -> None:
    importlib.import_module("mmwcore")
    importlib.import_module("mmwcore.config")
    importlib.import_module("mmwcore.io")
    importlib.import_module("mmwcore.session")

    assert "serial" not in sys.modules
    assert "cv2" not in sys.modules
