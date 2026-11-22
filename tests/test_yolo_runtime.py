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


def test_resolve_yolo_device_prefers_explicit_then_cuda_mps_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    assert runtime.resolve_yolo_device("cpu") == "cpu"
    assert runtime.resolve_yolo_device(0) == "0"
    assert runtime.resolve_yolo_device("mps") == "mps"
    assert runtime.resolve_yolo_device("cuda") == "0"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "_mps_is_available", lambda _torch: True)
    assert runtime.resolve_yolo_device(None) == "0"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime, "_mps_is_available", lambda _torch: True)
    assert runtime.resolve_yolo_device("") == "mps"

    monkeypatch.setattr(runtime, "_mps_is_available", lambda _torch: False)
    assert runtime.resolve_yolo_device(None) == "cpu"


def test_ensure_mps_fallback_env_sets_default_on_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    assert runtime.ensure_mps_fallback_env() is True
    assert runtime.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

    # Explicit user override is preserved
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    assert runtime.ensure_mps_fallback_env() is False
    assert runtime.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_select_backend_supports_mps_and_explicit_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    pt_file = tmp_path / "yolo11n.pt"
    pt_file.write_bytes(b"pt")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime, "_mps_is_available", lambda _torch: True)

    backend = runtime.select_backend(str(pt_file))
    assert backend.backend == "PyTorch (MPS)"
    assert backend.device == torch.device("mps")

    cpu_backend = runtime.select_backend(str(pt_file), device="cpu")
    assert cpu_backend.device == torch.device("cpu")


def test_export_yolo_weights_forwards_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class _FakeYOLO:
        def __init__(self, weights):
            captured["weights"] = weights

        def export(self, **kwargs):
            captured["export_kwargs"] = kwargs
            return "/tmp/best.onnx"

    monkeypatch.setattr(runtime, "configure_ultralytics_cache", lambda: None)
    monkeypatch.setattr(
        runtime, "import_ultralytics_symbol", lambda name, purpose="": _FakeYOLO
    )
    monkeypatch.setattr(runtime, "resolve_weight_path", lambda w: Path(w))

    result = runtime.export_yolo_weights(
        "/tmp/best.pt",
        export_format="onnx",
        imgsz=640,
        half=True,
        device="0",
    )

    assert result == "/tmp/best.onnx"
    assert captured["weights"] == "/tmp/best.pt"
    assert captured["export_kwargs"] == {
        "format": "onnx",
        "imgsz": 640,
        "half": True,
        "device": "0",
    }
