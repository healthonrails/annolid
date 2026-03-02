from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path


def test_sitecustomize_injects_pkg_resources_stub_when_missing() -> None:
    sitecustomize_path = (
        Path(__file__).resolve().parents[1]
        / "annolid"
        / "utils"
        / "tensorboard_sitecustomize"
        / "sitecustomize.py"
    )
    assert sitecustomize_path.exists()

    original_pkg_resources = sys.modules.pop("pkg_resources", None)
    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pkg_resources":
            raise ModuleNotFoundError("No module named 'pkg_resources'")
        return original_import(name, globals, locals, fromlist, level)

    module_name = "_annolid_tb_sitecustomize_test"
    try:
        builtins.__import__ = _patched_import
        spec = importlib.util.spec_from_file_location(module_name, sitecustomize_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        shim = sys.modules.get("pkg_resources")
        assert shim is not None
        assert hasattr(shim, "iter_entry_points")
        assert list(shim.iter_entry_points("tensorboard_plugins")) == []
    finally:
        builtins.__import__ = original_import
        sys.modules.pop(module_name, None)
        if original_pkg_resources is not None:
            sys.modules["pkg_resources"] = original_pkg_resources
        else:
            sys.modules.pop("pkg_resources", None)
