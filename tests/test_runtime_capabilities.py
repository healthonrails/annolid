from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import numpy as np
import pytest

from annolid.infrastructure.capabilities import (
    AUTO_INSTALL_OPTIONAL_DEPS_ENV,
    CAPABILITIES,
    INSTALL_PROFILES,
    ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL,
    PYTORCH_CPU_INDEX_URL,
    PYTORCH_CUDA12_INDEX_URL,
    USER_RUNTIME_DIR_ENV,
    WHEELHOUSE_MANIFEST_NAME,
    RuntimeCapability,
    RuntimeDependency,
    RuntimeInstallCancelledError,
    _build_smart_install_plan,
    _extract_verified_wheel,
    _pin_abi_sensitive_specs,
    capability_message,
    check_capability,
    ensure_capability,
    export_wheelhouse_bundle,
    format_capability_report,
    install_capability_stream,
    list_capabilities,
    user_runtime_site_dir,
    verify_wheelhouse_bundle,
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


def _make_pep427_wheel(
    dest_dir: Path,
    dist_name: str,
    version: str,
    *,
    py_tag: str = "py3",
    abi_tag: str = "none",
    plat_tag: str = "any",
    files: dict[str, bytes] | None = None,
    requires_dist: tuple[str, ...] = (),
    tamper_member: str | None = None,
    extra_unsafe_member: tuple[str, bytes] | None = None,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    norm_dist = dist_name.replace("-", "_")
    whl_path = dest_dir / f"{norm_dist}-{version}-{py_tag}-{abi_tag}-{plat_tag}.whl"
    dist_info = f"{norm_dist}-{version}.dist-info"

    meta_lines = [
        "Metadata-Version: 2.1",
        f"Name: {dist_name}",
        f"Version: {version}",
    ]
    for req in requires_dist:
        meta_lines.append(f"Requires-Dist: {req}")
    metadata_bytes = ("\n".join(meta_lines) + "\n\n").encode("utf-8")
    wheel_bytes = (
        "Wheel-Version: 1.0\n"
        "Generator: annolid-test\n"
        "Root-Is-Purelib: true\n"
        f"Tag: {py_tag}-{abi_tag}-{plat_tag}\n"
    ).encode("utf-8")

    payload: dict[str, bytes] = dict(files or {})
    payload[f"{dist_info}/METADATA"] = metadata_bytes
    payload[f"{dist_info}/WHEEL"] = wheel_bytes

    record_lines: list[str] = []
    for rel_path, content in payload.items():
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        record_lines.append(f"{rel_path},sha256={digest},{len(content)}")
    record_member = f"{dist_info}/RECORD"
    record_lines.append(f"{record_member},,")
    payload[record_member] = ("\n".join(record_lines) + "\n").encode("utf-8")

    if tamper_member is not None and tamper_member in payload:
        # Keep the same byte length so SHA256 verification specifically catches tampering.
        orig = payload[tamper_member]
        payload[tamper_member] = bytes([orig[0] ^ 0xFF]) + orig[1:]

    with zipfile.ZipFile(whl_path, "w") as zf:
        for rel_path, content in payload.items():
            zf.writestr(rel_path, content)
        if extra_unsafe_member is not None:
            zf.writestr(extra_unsafe_member[0], extra_unsafe_member[1])

    return whl_path


def test_export_and_verify_wheelhouse_bundle_detects_tampering(
    tmp_path: Path,
) -> None:
    wh_dir = tmp_path / "offline-bundle"
    whl_holder: list[Path] = []

    def _fake_download(cmd, **_kwargs):
        assert "download" in cmd
        whl = _make_pep427_wheel(
            wh_dir,
            "PyMuPDF",
            "1.25.0",
            files={"fitz/__init__.py": b"__version__ = '1.25.0'\n"},
        )
        whl_holder.append(whl)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    manifest = export_wheelhouse_bundle(
        wh_dir,
        capability_names=("pdf",),
        runner=_fake_download,
    )
    assert manifest["wheel_count"] == 1
    assert manifest["capabilities"] == ["pdf"]
    assert (wh_dir / WHEELHOUSE_MANIFEST_NAME).is_file()

    report = verify_wheelhouse_bundle(wh_dir)
    assert report["valid"] is True
    assert report["manifest_present"] is True
    assert report["wheel_count"] == 1

    # Flip one byte inside the wheel without changing file size -> SHA256 mismatch
    raw = bytearray(whl_holder[0].read_bytes())
    raw[-1] ^= 0x01
    whl_holder[0].write_bytes(bytes(raw))

    with pytest.raises(RuntimeError, match="SHA256 checksum mismatch"):
        verify_wheelhouse_bundle(wh_dir)


def test_smart_install_plan_supports_offline_and_cached_wheelhouse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._find_uv_binary",
        lambda: "/usr/bin/uv",
    )
    wh_dir = tmp_path / "wheelhouse"
    _make_pep427_wheel(wh_dir, "pymupdf", "1.25.0")

    deps = (RuntimeDependency("fitz", "pymupdf"),)

    # Strict offline mode: --no-index + --find-links, no PyPI fallback
    specs_off, steps_off = _build_smart_install_plan(
        deps, wheelhouse=wh_dir, offline=True
    )
    assert specs_off == ("pymupdf",)
    assert len(steps_off) == 1
    primary_off, fallback_off = steps_off[0]
    assert "--no-index" in primary_off
    assert "--find-links" in primary_off
    assert str(wh_dir.resolve()) in primary_off
    assert fallback_off is None

    # Cached wheelhouse mode (offline=False): --find-links primary + offline fallback
    _, steps_cached = _build_smart_install_plan(deps, wheelhouse=wh_dir, offline=False)
    primary_cached, fallback_cached = steps_cached[0]
    assert "--find-links" in primary_cached
    assert "--no-index" not in primary_cached
    assert fallback_cached is not None
    assert "--no-index" in fallback_cached

    # Offline mode without wheelhouse fails fast with actionable guidance
    with pytest.raises(RuntimeError, match="Offline mode is enabled"):
        _build_smart_install_plan(deps, wheelhouse=None, offline=True)


def test_pure_python_offline_wheelhouse_installer_in_airgapped_frozen_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "Annolid.app" / "Contents" / "MacOS"
    bundle_dir.mkdir(parents=True)
    frozen_exe = bundle_dir / "annolid"
    frozen_exe.write_text("#!/bin/sh\n", encoding="utf-8")
    bundle_snapshot_before = sorted(
        p.relative_to(tmp_path / "Annolid.app")
        for p in (tmp_path / "Annolid.app").rglob("*")
    )

    user_rt_root = tmp_path / "isolated_user_rt"
    wh_dir = tmp_path / "airgap_wheelhouse"

    # Root wheel uses cp39-abi3-any tag + PEP 508 conditional transitive dependency
    _make_pep427_wheel(
        wh_dir,
        "airgap-leaf-pkg",
        "0.4.0",
        files={"airgap_leaf_mod/__init__.py": b"LEAF_VALUE = 42\n"},
    )
    _make_pep427_wheel(
        wh_dir,
        "ignored-other-os-pkg",
        "9.9.9",
        files={"ignored_other_os_mod/__init__.py": b"SHOULD_NOT_INSTALL = True\n"},
    )
    _make_pep427_wheel(
        wh_dir,
        "airgap-root-pkg",
        "1.2.0",
        py_tag="cp39",
        abi_tag="abi3",
        plat_tag="any",
        requires_dist=(
            "airgap-leaf-pkg>=0.4.0; python_version >= '3.10'",
            "ignored-other-os-pkg>=1.0; sys_platform == 'nonexistent_os'",
        ),
        files={
            "airgap_root_mod/__init__.py": (
                b"from airgap_leaf_mod import LEAF_VALUE\nROOT_VALUE = LEAF_VALUE + 8\n"
            )
        },
    )

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(frozen_exe))
    monkeypatch.setenv(USER_RUNTIME_DIR_ENV, str(user_rt_root))
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._find_uv_binary", lambda: None
    )
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._find_matching_host_python",
        lambda: None,
    )
    monkeypatch.setitem(
        CAPABILITIES,
        "_test_airgap_ext",
        RuntimeCapability(
            name="_test_airgap_ext",
            title="Airgap Test Extension",
            dependencies=(
                RuntimeDependency("airgap_root_mod", "airgap-root-pkg>=1.0.0"),
            ),
            package_specs=("airgap-root-pkg>=1.0.0",),
            docs_url="https://annolid.com/installation/",
            description="Synthetic capability for zero-toolchain frozen airgap test.",
        ),
    )

    site_dir = user_runtime_site_dir()
    try:
        installed = ensure_capability(
            "_test_airgap_ext",
            wheelhouse=wh_dir,
            offline=True,
            allow_frozen_user_runtime=True,
        )
        assert installed == ("airgap-root-pkg>=1.0.0",)
        assert (site_dir / "airgap_root_mod" / "__init__.py").is_file()
        assert (site_dir / "airgap_leaf_mod" / "__init__.py").is_file()
        assert not (site_dir / "ignored_other_os_mod").exists()

        mod = importlib.import_module("airgap_root_mod")
        assert mod.ROOT_VALUE == 50

        # Bundled app directory must remain 100% untouched (read-only contract).
        bundle_snapshot_after = sorted(
            p.relative_to(tmp_path / "Annolid.app")
            for p in (tmp_path / "Annolid.app").rglob("*")
        )
        assert bundle_snapshot_after == bundle_snapshot_before
    finally:
        for mod_name in ("airgap_root_mod", "airgap_leaf_mod"):
            sys.modules.pop(mod_name, None)
        resolved_site = str(site_dir.resolve())
        while resolved_site in sys.path:
            sys.path.remove(resolved_site)


def test_extract_verified_wheel_rejects_bad_record_hash_and_zipslip(
    tmp_path: Path,
) -> None:
    wh_dir = tmp_path / "bad_wheels"
    target_site = tmp_path / "site"

    bad_hash_whl = _make_pep427_wheel(
        wh_dir,
        "corrupt-pkg",
        "1.0.0",
        files={"corrupt_pkg/__init__.py": b"VALUE = 1\n"},
        tamper_member="corrupt_pkg/__init__.py",
    )
    with pytest.raises(RuntimeError, match="RECORD SHA256 mismatch"):
        _extract_verified_wheel(bad_hash_whl, target_site)
    assert not list(target_site.rglob("*.annolid-tmp"))
    assert not (target_site / "corrupt_pkg" / "__init__.py").exists()

    zipslip_whl = _make_pep427_wheel(
        wh_dir,
        "evil-pkg",
        "1.0.0",
        files={"evil_pkg/__init__.py": b"VALUE = 1\n"},
        extra_unsafe_member=("../escaped.py", b"raise SystemExit(1)\n"),
    )
    with pytest.raises(RuntimeError, match="Unsafe archive path"):
        _extract_verified_wheel(zipslip_whl, target_site)
    assert not (tmp_path / "escaped.py").exists()


def test_annolid_run_runtime_cli_bundle_verify_and_offline_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from annolid.engine.cli import main as cli_main

    wh_dir = tmp_path / "cli_wheelhouse"

    def _fake_export(out_dir, *, capability_names=(), **_kwargs):
        _make_pep427_wheel(Path(out_dir), "pymupdf", "1.25.0")
        manifest = {
            "schema_version": 1,
            "wheel_count": 1,
            "capabilities": list(capability_names),
            "wheels": [
                {
                    "filename": "pymupdf-1.25.0-py3-none-any.whl",
                    "size_bytes": (Path(out_dir) / "pymupdf-1.25.0-py3-none-any.whl")
                    .stat()
                    .st_size,
                    "sha256": hashlib.sha256(
                        (Path(out_dir) / "pymupdf-1.25.0-py3-none-any.whl").read_bytes()
                    ).hexdigest(),
                }
            ],
        }
        (Path(out_dir) / WHEELHOUSE_MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(
        "annolid.infrastructure.capabilities.export_wheelhouse_bundle",
        _fake_export,
    )

    rc_bundle = cli_main(["runtime", "bundle", "pdf", "--out", str(wh_dir), "--json"])
    assert rc_bundle == 0
    bundle_json = json.loads(capsys.readouterr().out)
    assert bundle_json["wheel_count"] == 1

    rc_verify = cli_main(["runtime", "verify-bundle", str(wh_dir), "--json"])
    assert rc_verify == 0
    verify_json = json.loads(capsys.readouterr().out)
    assert verify_json["valid"] is True
    assert verify_json["manifest_present"] is True

    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._missing_dependencies",
        lambda cap, **_kw: cap.dependencies,
    )
    rc_dry = cli_main(
        [
            "runtime",
            "install",
            "pdf",
            "--wheelhouse",
            str(wh_dir),
            "--offline",
            "--dry-run",
        ]
    )
    assert rc_dry == 0
    dry_out = capsys.readouterr().out
    assert "--no-index" in dry_out
    assert str(wh_dir.resolve()) in dry_out
