from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from annolid.yolo import runtime


def test_yolo_package_imports_without_torch_or_ultralytics() -> None:
    script = """
import importlib.abc
import sys


class BlockHeavyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        if fullname == "ultralytics" or fullname.startswith("ultralytics."):
            raise ModuleNotFoundError("No module named 'ultralytics'")
        return None


sys.meta_path.insert(0, BlockHeavyRuntime())
import annolid.yolo
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.splitlines()


def test_import_ultralytics_symbol_checks_yolo_capability(monkeypatch) -> None:
    calls = []
    marker = object()
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = marker
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(
        runtime,
        "ensure_yolo_runtime",
        lambda: calls.append("yolo") or (),
    )

    loaded = runtime.import_ultralytics_symbol("YOLO")

    assert loaded is marker
    assert calls == ["yolo"]


def test_import_ultralytics_symbol_reports_missing_symbol(monkeypatch) -> None:
    fake_ultralytics = types.ModuleType("ultralytics")
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(runtime, "ensure_yolo_runtime", lambda: ())

    with pytest.raises(RuntimeError, match="does not provide ultralytics.YOLOE"):
        runtime.import_ultralytics_symbol("YOLOE")


def test_resolve_weight_path_prefers_user_search_roots(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    expected = weights_dir / "custom.pt"
    expected.write_bytes(b"fake weights")

    resolved = runtime.resolve_weight_path("custom", search_roots=[weights_dir])

    assert resolved == expected


def test_resolve_weight_path_returns_cache_target_for_ultralytics_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    resolved = runtime.resolve_weight_path("yolo11n")

    assert resolved == tmp_path / "annolid" / "ultralytics" / "weights" / "yolo11n.pt"


def test_resolve_weight_path_keeps_explicit_unresolved_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    unresolved = Path("models") / "yolo11n.pt"

    assert runtime.resolve_weight_path(str(unresolved)) == unresolved
