from __future__ import annotations

import io
import json
import subprocess
import sys
import threading

import numpy as np
import pytest

from annolid.infrastructure.capabilities import (
    AUTO_INSTALL_OPTIONAL_DEPS_ENV,
    CAPABILITIES,
    INSTALL_PROFILES,
    ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL,
    PYTORCH_CPU_INDEX_URL,
    PYTORCH_CUDA12_INDEX_URL,
    RuntimeDependency,
    RuntimeInstallCancelledError,
    _build_smart_install_plan,
    _pin_abi_sensitive_specs,
    capability_message,
    check_capability,
    ensure_capability,
    format_capability_report,
    install_capability_stream,
    list_capabilities,
)


def test_capability_available_when_all_imports_resolve() -> None:
    status = check_capability("cutie", find_spec=lambda name: object())

    assert status.available is True
    assert status.state == "available"
    assert status.missing_imports == ()
    assert "annolid[cutie]" in status.install_command


def test_capability_reports_missing_imports_and_install_hint() -> None:
    missing = {"torch", "hydra"}
    status = check_capability(
        "cutie", find_spec=lambda name: None if name in missing else object()
    )

    assert status.available is False
    assert status.state == "missing"
    assert status.missing_imports == ("torch", "hydra")
    message = capability_message(status)
    assert "torch, hydra" in message
    assert "python -m pip install" in message
    assert "annolid[cutie]" in message


def test_format_capability_report_is_deterministic_for_selected_names() -> None:
    report = format_capability_report(("sam3", "yolo"), find_spec=lambda name: object())

    assert report.splitlines() == ["sam3: available", "yolo: available"]


def test_unknown_capability_lists_known_names() -> None:
    with pytest.raises(KeyError, match="Known capabilities"):
        check_capability("missing-feature", find_spec=lambda name: object())


def test_list_capabilities_exposes_professional_install_tiers() -> None:
    capabilities = set(list_capabilities())

    assert {
        "ml",
        "tracking",
        "cutie",
        "sam3",
        "yolo",
        "bot",
        "onnx",
        "scientific",
        "webengine",
        "pdf",
        "model_downloads",
    }.issubset(capabilities)
    assert set(INSTALL_PROFILES) == {
        "minimal",
        "gui",
        "tracking",
        "workstation",
        "full",
    }
    assert INSTALL_PROFILES["gui"]["capabilities"] == ()
    assert set(INSTALL_PROFILES["tracking"]["capabilities"]) == {
        "ml",
        "tracking",
        "cutie",
    }


def test_ensure_capability_installs_missing_runtime_packages(monkeypatch) -> None:
    calls = []
    available = {"lap"}

    monkeypatch.setattr(
        "annolid.infrastructure.capabilities.shutil.which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )

    def _finder(name: str):
        return object() if name in available else None

    def _runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        available.add("ultralytics")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    installed = ensure_capability("yolo", find_spec=_finder, runner=_runner)

    assert installed == ("ultralytics>=8.4.0",)
    assert calls
    assert calls[0][0][:5] == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
    ]
    assert calls[0][1]["capture_output"] is True


def test_ensure_capability_can_disable_auto_install(monkeypatch) -> None:
    monkeypatch.setenv(AUTO_INSTALL_OPTIONAL_DEPS_ENV, "0")

    with pytest.raises(RuntimeError) as exc_info:
        ensure_capability("sam3", find_spec=lambda _name: None)

    message = str(exc_info.value)
    assert "sam3: missing optional runtime packages" in message
    assert "annolid[sam3]" in message


def test_ensure_capability_does_not_mutate_frozen_bundle(monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_capability("yolo", find_spec=lambda _name: None)

    assert "Frozen desktop bundles are read-only" in str(exc_info.value)


def test_cutie_capability_allows_pillow_12_plus_wheels() -> None:
    pillow_specs = [
        dep.package_spec
        for dep in CAPABILITIES["cutie"].dependencies
        if dep.import_name == "PIL"
    ]
    assert pillow_specs == ["Pillow>=9.3.0"]


def test_pin_abi_sensitive_specs_locks_pyside6_addons_to_essentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._installed_pyside6_version",
        lambda: "6.8.1",
    )

    pinned = _pin_abi_sensitive_specs(("PySide6-Addons>=6.7.0", "pymupdf"))

    assert pinned == ("PySide6-Addons==6.8.1", "pymupdf")


def test_smart_install_plan_uses_linux_cpu_and_cuda12_wheel_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = (
        RuntimeDependency("torch", "torch>=2.5.0"),
        RuntimeDependency("onnxruntime", "onnxruntime>=1.23.2"),
    )
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities.shutil.which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )
    monkeypatch.setattr(sys, "platform", "linux")

    # Non-GPU Linux: lightweight CPU PyTorch wheel index + standard CPU onnxruntime
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities.detect_accelerator_summary",
        lambda: {
            "platform": "linux",
            "arch": "x86_64",
            "has_nvidia_gpu": False,
            "has_cuda12": False,
        },
    )
    specs_cpu, steps_cpu = _build_smart_install_plan(deps)
    assert specs_cpu == ("torch>=2.5.0", "onnxruntime>=1.23.2")
    assert PYTORCH_CPU_INDEX_URL in steps_cpu[0][0]
    assert steps_cpu[0][1] is not None  # PyPI fallback command exists

    # CUDA 12 Linux: CUDA 12 PyTorch wheel index + onnxruntime-gpu extra index
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities.detect_accelerator_summary",
        lambda: {
            "platform": "linux",
            "arch": "x86_64",
            "has_nvidia_gpu": True,
            "has_cuda12": True,
        },
    )
    specs_gpu, steps_gpu = _build_smart_install_plan(deps)
    assert specs_gpu == ("torch>=2.5.0", "onnxruntime-gpu>=1.23.2")
    assert PYTORCH_CUDA12_INDEX_URL in steps_gpu[0][0]
    assert ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL in steps_gpu[1][0]
    assert steps_gpu[1][1] is not None  # CPU onnxruntime fallback exists


def test_install_capability_stream_streams_output_and_honors_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available: set[str] = set()
    lines: list[str] = []

    def _finder(name: str):
        return object() if name in available else None

    def _fake_stream_cmd(
        command, *, on_output=None, cancel_event=None, capability_name=""
    ):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeInstallCancelledError("cancelled")
        if on_output is not None:
            on_output("Resolved 1 package in 12ms")
        available.add("fitz")
        return 0, "Resolved 1 package in 12ms"

    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._run_streaming_command",
        _fake_stream_cmd,
    )

    installed = install_capability_stream(
        "pdf",
        auto_install=True,
        on_output=lines.append,
        find_spec=_finder,
    )
    assert installed == ("pymupdf",)
    assert "Resolved 1 package in 12ms" in lines

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeInstallCancelledError):
        install_capability_stream(
            "model_downloads",
            auto_install=True,
            cancel_event=cancel,
            find_spec=lambda _name: None,
        )


def test_annolid_run_runtime_cli_status_profiles_and_dry_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from annolid.engine.cli import main as cli_main

    rc = cli_main(["runtime", "status", "--json"])
    assert rc == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert "environment" in status_payload
    assert "python_version" in status_payload["environment"]
    cap_names = {row["name"] for row in status_payload["capabilities"]}
    assert {"webengine", "onnx", "scientific", "pdf", "model_downloads"}.issubset(
        cap_names
    )

    rc_text = cli_main(["runtime", "status"])
    assert rc_text == 0
    assert "CAPABILITY" in capsys.readouterr().out

    rc_profiles = cli_main(["runtime", "profiles", "--json"])
    assert rc_profiles == 0
    profiles_payload = json.loads(capsys.readouterr().out)
    profile_names = {row["profile"] for row in profiles_payload}
    assert profile_names == {"minimal", "gui", "tracking", "workstation", "full"}

    rc_dry = cli_main(["runtime", "install", "tracking", "--dry-run"])
    assert rc_dry == 0
    dry_out = capsys.readouterr().out.strip()
    assert dry_out


def test_stdlib_gdown_shim_handles_https_and_drive_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from annolid.utils import model_assets

    class _FakeHttpResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.close()

    monkeypatch.setattr(
        model_assets.urllib.request,
        "urlopen",
        lambda req, timeout=120: _FakeHttpResponse(b"onnx-weights-bytes"),
    )

    target_https = tmp_path / "weights.onnx"
    out_path = model_assets._StdlibGdownShim.cached_download(
        "https://example.org/models/weights.onnx",
        str(target_https),
    )
    assert out_path == str(target_https)
    assert target_https.read_bytes() == b"onnx-weights-bytes"

    ensured: list[str] = []

    class _FakeRealGdown:
        @staticmethod
        def cached_download(url: str, path: str, **_kwargs) -> str:
            from pathlib import Path

            Path(path).write_bytes(b"gdrive-checkpoint")
            return path

    monkeypatch.setattr(
        "annolid.infrastructure.capabilities.ensure_capability",
        lambda name, **_kwargs: ensured.append(name) or ("gdown",),
    )
    monkeypatch.setitem(sys.modules, "gdown", _FakeRealGdown)

    target_drive = tmp_path / "drive.pth"
    drive_out = model_assets._StdlibGdownShim.cached_download(
        "https://drive.google.com/uc?id=12345",
        str(target_drive),
    )
    assert ensured == ["model_downloads"]
    assert drive_out == str(target_drive)
    assert target_drive.read_bytes() == b"gdrive-checkpoint"


def test_pure_numpy_fallbacks_for_mask_centroid_and_pca_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from annolid.annotation.masks import (
        binary_mask_to_coco_rle,
        calculate_mask_centroid,
    )
    from annolid.postprocessing import skeletonization

    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[6:11, 10:15] = 1
    rle = binary_mask_to_coco_rle(mask)

    # Hide scipy from calculate_mask_centroid to exercise the np.nonzero fallback.
    monkeypatch.setitem(sys.modules, "scipy", None)
    cx, cy = calculate_mask_centroid(rle)
    assert (cx, cy) == (12, 8)

    # Force SVD fallback in skeletonization._principal_axis when sklearn is absent.
    monkeypatch.setattr(skeletonization, "_SklearnPCA", None)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    axis = skeletonization._principal_axis(pts)
    assert abs(abs(float(axis[0])) - 1.0) < 1e-6
    assert abs(float(axis[1])) < 1e-6
