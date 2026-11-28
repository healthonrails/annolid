"""Regression tests for Annolid update hardening, service helpers, and Update Dialog."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

from qtpy import QtWidgets

from annolid.core.agent.update_manager.history import (
    last_known_good_version,
    load_update_history,
    record_update_history,
)
from annolid.core.agent.update_manager.manager import (
    SignedUpdateManager,
    SignedUpdatePlan,
    _build_apply_commands,
    _build_post_check_commands,
    _detect_install_mode,
    _run_step,
)
from annolid.core.agent.update_manager.manifest import (
    UpdateManifest,
    fetch_channel_manifest,
)
from annolid.core.agent.update_manager.rollback import build_rollback_plan
from annolid.core.agent.update_manager.service import UpdateManagerService
from annolid.core.agent.update_manager.verify import VerificationResult
from annolid.gui.widgets.update_dialog import AnnolidUpdateDialog
from annolid.services.agent_tooling import validate_agent_tools
from annolid.services.agent_update import (
    check_auto_update_if_due,
    format_update_reason,
    run_agent_update,
)


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_plan(
    *,
    current_version: str = "1.7.1",
    target_version: str = "1.8.0",
    install_mode: str = "package",
    verification_ok: bool = True,
    verification_reason: str = "ok",
    commands: list[list[str]] | None = None,
    preflight: dict | None = None,
) -> SignedUpdatePlan:
    manifest = UpdateManifest(
        project="annolid",
        channel="stable",
        version=target_version,
        release_date="2026-11-01T00:00:00Z",
        artifact_url="https://example.invalid/annolid.whl",
        artifact_sha256="abc123",
        signature="",
        signature_alg="none",
        source="https://pypi.org/pypi/annolid/json",
    )
    verification = VerificationResult(
        ok=verification_ok,
        reason=verification_reason,
        details={},
    )
    return SignedUpdatePlan(
        channel="stable",
        project="annolid",
        current_version=current_version,
        manifest=manifest,
        install_mode=install_mode,
        source_root=".",
        commands=list(commands or []),
        verification=verification,
        preflight=dict(preflight or {}),
    )


def test_validate_agent_tools_passes_cleanly(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    summary, exit_code = validate_agent_tools()
    assert exit_code == 0
    assert summary["status"] == "ok"
    checks = {item["name"]: item for item in summary["checks"]}
    assert checks["registry"]["ok"] is True


def test_fetch_channel_manifest_skips_empty_and_yanked_releases(monkeypatch) -> None:
    import annolid.core.agent.update_manager.manifest as manifest_mod

    payload = {
        "releases": {
            "1.7.0": [
                {
                    "filename": "annolid-1.7.0-py3-none-any.whl",
                    "url": "https://files.example/annolid-1.7.0.whl",
                    "upload_time_iso_8601": "2026-10-01T00:00:00Z",
                    "yanked": False,
                    "digests": {"sha256": "aaa111"},
                }
            ],
            "1.8.0": [],
            "1.9.0": [
                {
                    "filename": "annolid-1.9.0-py3-none-any.whl",
                    "url": "https://files.example/annolid-1.9.0.whl",
                    "yanked": True,
                    "digests": {"sha256": "bbb222"},
                }
            ],
        }
    }

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int = -1) -> bytes:
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        manifest_mod, "urlopen", lambda _req, timeout=4.0: _FakeResponse()
    )

    manifest = fetch_channel_manifest(project="annolid", channel="stable")
    assert manifest.version == "1.7.0"
    assert manifest.artifact_sha256 == "aaa111"


def test_manager_up_to_date_returns_success_not_blocked_failure(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.core.agent.update_manager.manager as manager_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    plan = _make_plan(
        current_version="1.7.1",
        target_version="1.7.1",
        verification_ok=False,
        verification_reason="not_newer_than_current",
    )

    monkeypatch.setattr(
        manager_mod.SignedUpdateManager,
        "stage",
        lambda self, **kwargs: plan,
    )

    payload, exit_code = run_agent_update(execute=True)
    assert exit_code == 0
    assert payload["status"] == "up_to_date"
    assert payload["reason"] == "already_up_to_date"
    assert "already up to date" in payload["message"].lower()


def test_frozen_install_mode_detected_and_requires_manual_download(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert _detect_install_mode(tmp_path) == "frozen"
    assert (
        _build_apply_commands(
            install_mode="frozen", channel="stable", project="annolid"
        )
        == []
    )

    manager = SignedUpdateManager(project="annolid")
    plan = _make_plan(install_mode="frozen", verification_ok=True)
    payload = manager.run(plan, execute=True, run_post_check=False)
    assert payload["status"] == "manual_required"
    assert payload["reason"] == "frozen_bundle_manual_update"
    assert "github.com/healthonrails/annolid/releases" in payload["release_url"]


def test_package_mode_and_rollback_fall_back_to_uv_when_pip_missing(
    monkeypatch,
) -> None:
    import annolid.core.agent.update_manager.manager as manager_mod
    import annolid.core.agent.update_manager.rollback as rollback_mod

    monkeypatch.setattr(manager_mod, "_has_pip_module", lambda: False)
    monkeypatch.setattr(
        manager_mod.shutil,
        "which",
        lambda name: "/usr/local/bin/uv" if name == "uv" else None,
    )
    cmds = _build_apply_commands(
        install_mode="package", channel="beta", project="annolid"
    )
    assert cmds == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            "annolid",
            "--prerelease=allow",
        ]
    ]

    monkeypatch.setattr(rollback_mod.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(
        rollback_mod.shutil,
        "which",
        lambda name: "/usr/local/bin/uv" if name == "uv" else None,
    )
    rb = build_rollback_plan(
        install_mode="package", project="annolid", previous_version="1.7.0"
    )
    assert rb.manual_required is False
    assert rb.commands == [
        ["uv", "pip", "install", "--python", sys.executable, "annolid==1.7.0"]
    ]


def test_post_checks_use_sys_executable_and_run_step_catches_missing_binary() -> None:
    checks = _build_post_check_commands()
    assert checks
    for cmd in checks:
        assert cmd[0] == sys.executable
    assert any("validate-agent-tools" in cmd for cmd in checks)
    assert not any("agent-security-check" in cmd for cmd in checks)

    step = _run_step(
        name="apply",
        command=["definitely-missing-annolid-binary-xyz"],
        cwd=None,
        execute=True,
    )
    assert step["ok"] is False
    assert step["returncode"] == -1
    assert step.get("error")


def test_source_mode_rejects_dirty_checkout_and_supports_force_sync(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = SignedUpdateManager(project="annolid")

    dirty_plan = _make_plan(
        install_mode="source",
        verification_ok=True,
        commands=[[sys.executable, "-V"]],
        preflight={"git_dirty": True, "missing_binaries": []},
    )
    dirty_res = manager.run(dirty_plan, execute=True, run_post_check=False)
    assert dirty_res["status"] == "failed"
    assert dirty_res["reason"] == "dirty_working_tree"
    assert dirty_res["rollback"] is None

    lines: list[str] = []
    sync_plan = _make_plan(
        current_version="1.7.1",
        target_version="1.7.1",
        install_mode="source",
        verification_ok=False,
        verification_reason="not_newer_than_current",
        commands=[[sys.executable, "-c", "print('synced ok')"]],
        preflight={"git_dirty": False, "git_commit": "abc1234", "missing_binaries": []},
    )
    sync_res = manager.run(
        sync_plan,
        execute=True,
        run_post_check=False,
        on_output=lines.append,
        cancel_event=threading.Event(),
        force_source_sync=True,
    )
    assert sync_res["status"] == "updated"
    assert any("synced ok" in line for line in lines)


def test_service_init_does_not_create_stage_dir_eagerly(tmp_path: Path) -> None:
    stage_dir = tmp_path / "lazy_stage"
    service = UpdateManagerService(project="annolid", stage_dir=stage_dir)
    assert service.stage_dir == stage_dir.resolve()
    assert not stage_dir.exists()


def test_update_history_and_scheduled_auto_update_if_due(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    record_update_history(
        project="annolid",
        channel="stable",
        install_mode="package",
        previous_version="1.7.0",
        target_version="1.7.1",
        status="updated",
    )
    assert last_known_good_version(project="annolid") == "1.7.0"
    assert len(load_update_history()) == 1

    monkeypatch.setenv("ANNOLID_AUTO_UPDATE_ENABLED", "0")
    disabled = check_auto_update_if_due(now_epoch_s=1000.0)
    assert disabled["status"] == "disabled"

    monkeypatch.setenv("ANNOLID_AUTO_UPDATE_ENABLED", "1")
    monkeypatch.setenv("ANNOLID_AUTO_UPDATE_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("ANNOLID_AUTO_UPDATE_JITTER_SECONDS", "0")
    calls: list[dict] = []
    monkeypatch.setattr(
        "annolid.services.agent_update.inspect_gui_update_status",
        lambda **kwargs: calls.append(kwargs)
        or {
            "current_version": "1.7.1",
            "target_version": "1.8.0",
            "channel": "stable",
            "update_available": True,
        },
    )

    first = check_auto_update_if_due(now_epoch_s=1000.0)
    assert first["status"] == "update_available"
    assert first["update_available"] is True
    assert len(calls) == 1

    second = check_auto_update_if_due(now_epoch_s=1100.0)
    assert second["status"] == "not_due"
    assert len(calls) == 1


def test_update_dialog_renders_states_and_restart_button() -> None:
    _ensure_qapp()
    dialog = AnnolidUpdateDialog(auto_check=False)

    # 1. Up-to-date package install
    dialog._apply_check_payload(
        {
            "current_version": "1.7.1",
            "target_version": "1.7.1",
            "channel": "stable",
            "install_mode": "package",
            "update_available": False,
            "up_to_date": True,
            "verification_reason": "not_newer_than_current",
        }
    )
    assert "up to date" in dialog._headline_label.text().lower()
    assert dialog._update_btn.isEnabled() is False

    # 2. Up-to-date clean source checkout offers Pull Latest (Git)
    dialog._status_payload = {
        "current_version": "1.7.1",
        "target_version": "1.7.1",
        "channel": "stable",
        "install_mode": "source",
        "git_branch": "main",
        "git_commit": "8585d82e",
        "git_dirty": False,
        "update_available": False,
        "up_to_date": True,
        "verification_reason": "not_newer_than_current",
    }
    dialog._apply_check_payload(dialog._status_payload)
    assert dialog._update_btn.isEnabled() is True
    assert "Pull Latest" in dialog._update_btn.text()

    # 3. Dirty source checkout disables update button
    dialog._status_payload = {
        **dialog._status_payload,
        "git_dirty": True,
    }
    dialog._apply_check_payload(dialog._status_payload)
    assert dialog._update_btn.isEnabled() is False

    # 4. Frozen bundle offers Download Release
    dialog._status_payload = {
        "current_version": "1.7.1",
        "target_version": "1.8.0",
        "channel": "stable",
        "install_mode": "frozen",
        "update_available": True,
        "up_to_date": False,
        "verification_reason": "ok",
        "release_url": "https://github.com/healthonrails/annolid/releases/tag/v1.8.0",
    }
    dialog._apply_check_payload(dialog._status_payload)
    assert dialog._update_btn.isEnabled() is True
    assert dialog._update_btn.text() == "Download Release"

    # 5. Completed update reveals Restart button
    dialog._on_run_completed(
        {
            "status": "updated",
            "restart_required": True,
            "message": "Updated annolid to 1.8.0 successfully.",
        },
        0,
    )
    assert dialog._restart_btn.isHidden() is False
    assert "completed successfully" in dialog._headline_label.text().lower()

    # 6. Human-readable reason helper
    assert "Require signed manifest" in format_update_reason(
        "signature_required_missing"
    )
    dialog.close()
