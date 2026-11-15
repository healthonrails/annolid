from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
from qtpy import QtWidgets

from annolid.gui.runtime_installer import (
    RuntimeInstallDialog,
    ensure_gui_capability,
)
from annolid.gui.widgets.runtime_manager_dialog import RuntimeManagerDialog
from annolid.infrastructure.capabilities import (
    AUTO_INSTALL_OPTIONAL_DEPS_ENV,
    CapabilityStatus,
    check_capability,
    list_capabilities,
)


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp() -> QtWidgets.QApplication:
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_runtime_manager_dialog_renders_capabilities_and_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_qapp()

    def _fake_check(name: str, **_kwargs) -> CapabilityStatus:
        real = check_capability(
            name, find_spec=lambda mod: object() if mod == "numpy" else None
        )
        return real

    monkeypatch.setattr(
        "annolid.gui.widgets.runtime_manager_dialog.check_capability",
        _fake_check,
    )

    dialog = RuntimeManagerDialog()
    try:
        all_caps = list_capabilities()
        assert dialog._table.rowCount() == len(all_caps)

        # Filtering hides non-matching rows
        dialog._filter_edit.setText("webengine")
        visible_rows = [
            row
            for row in range(dialog._table.rowCount())
            if not dialog._table.isRowHidden(row)
        ]
        assert len(visible_rows) >= 1
        dialog._filter_edit.setText("")

        # Selecting the 'tracking' Quick Profile checks ml, tracking, cutie
        idx = dialog._profile_combo.findData("tracking")
        assert idx >= 0
        dialog._profile_combo.setCurrentIndex(idx)
        assert set(dialog.checked_capabilities()) == {"ml", "tracking", "cutie"}
    finally:
        dialog.close()


def test_runtime_manager_dialog_frozen_bundle_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _ensure_qapp()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._find_uv_binary", lambda: None
    )
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._find_matching_host_python",
        lambda: None,
    )

    dialog = RuntimeManagerDialog()
    try:
        # Air-gapped frozen bundle with no uv/host-python and no wheelhouse -> disabled
        assert not dialog._install_btn.isEnabled()
        assert dialog.checked_capabilities() == []

        # Selecting a local wheelhouse directory containing .whl files enables User Runtime install
        wh_dir = tmp_path / "wheelhouse"
        wh_dir.mkdir()
        (wh_dir / "pymupdf-1.25.0-py3-none-any.whl").write_bytes(b"fake-whl")
        dialog.set_wheelhouse_dir(str(wh_dir), offline=True)
        assert dialog._install_btn.isEnabled()
        assert "User Runtime" in dialog._env_label.text()

        # Explicit ANNOLID_FROZEN_USER_RUNTIME=0 locks down user-space installs
        monkeypatch.setenv("ANNOLID_FROZEN_USER_RUNTIME", "0")
        dialog.refresh_table()
        assert not dialog._install_btn.isEnabled()
    finally:
        dialog.close()


def test_runtime_install_dialog_enables_airgapped_frozen_install_with_wheelhouse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _ensure_qapp()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._find_uv_binary", lambda: None
    )
    monkeypatch.setattr(
        "annolid.infrastructure.capabilities._find_matching_host_python",
        lambda: None,
    )
    status = check_capability("pdf", find_spec=lambda _name: None)

    dialog = RuntimeInstallDialog(
        (status,),
        feature_title="PDF Import",
        auto_start=False,
        allow_frozen_user_runtime=True,
    )
    try:
        assert not dialog._install_btn.isEnabled()
        assert "Target User Runtime" in dialog._summary_label.text()

        wh_dir = tmp_path / "offline_wh"
        wh_dir.mkdir()
        (wh_dir / "pymupdf-1.25.0-py3-none-any.whl").write_bytes(b"fake-whl")
        dialog.set_wheelhouse_dir(str(wh_dir), offline=True)

        assert dialog._install_btn.isEnabled()
        assert "Offline Wheelhouse" in dialog._summary_label.text()
    finally:
        dialog.close()


def test_runtime_install_dialog_displays_missing_specs_and_cancels_cleanly() -> None:
    _ensure_qapp()
    status = check_capability("webengine", find_spec=lambda _name: None)

    dialog = RuntimeInstallDialog(
        (status,),
        feature_title="Embedded PDF Viewer",
        auto_start=False,
    )
    try:
        assert "Embedded PDF Viewer" in dialog._header_label.text()
        assert "PySide6-Addons" in dialog._summary_label.text()
        assert "annolid[webengine]" in dialog._manual_command_text()
        dialog._copy_install_command()
        dialog._on_cancel_clicked()
        assert dialog.result() == QtWidgets.QDialog.Rejected
    finally:
        dialog.close()


def test_ensure_gui_capability_respects_readonly_env_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_qapp()
    shown_titles: list[str] = []

    monkeypatch.setenv(AUTO_INSTALL_OPTIONAL_DEPS_ENV, "0")
    monkeypatch.setattr(
        "annolid.gui.runtime_installer.check_capability",
        lambda name: check_capability(name, find_spec=lambda _m: None),
    )
    monkeypatch.setattr(
        "annolid.gui.runtime_installer._show_readonly_capability_message",
        lambda _parent, _statuses, *, feature_title=None: shown_titles.append(
            str(feature_title or "")
        ),
    )

    ok = ensure_gui_capability(None, "pdf", feature_title="PDF Import")
    assert ok is False
    assert shown_titles == ["PDF Import"]


def test_all_cli_model_plugins_load_without_heavy_ml_or_scientific_packages() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {
            "anthropic",
            "diffusers",
            "fitz",
            "gdown",
            "h5py",
            "hydra",
            "librosa",
            "matplotlib",
            "mcp",
            "numba",
            "omegaconf",
            "onnxruntime",
            "openai",
            "scipy",
            "skimage",
            "sklearn",
            "sounddevice",
            "tensorboard",
            "torch",
            "torchvision",
            "transformers",
            "ultralytics",
        }

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".", 1)[0]
                if root in blocked:
                    raise ModuleNotFoundError(
                        f"blocked optional runtime import: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, Blocker())
        from annolid.engine.registry import list_models

        models = list_models()
        assert len(models) >= 11, f"expected >=11 plugins, got {len(models)}"
        by_name = {m.name: m for m in models}
        assert "ml" in by_name["behavior_classifier"].required_capabilities
        assert "yolo" in by_name["yolo"].required_capabilities
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
