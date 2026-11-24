"""Offscreen Qt GUI tests for ColabStudioDialog, WorkflowActionsMixin, and Wizard Colab bridges."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtWidgets

from annolid.colab.auth import ColabAuthManager, ColabAuthRequiredError
from annolid.colab.models import ColabJobRecord
from annolid.colab.service import ColabWorkflowService
from annolid.gui.mixins.workflow_actions_mixin import WorkflowActionsMixin
from annolid.gui.widgets.colab_dialog import ColabSignInDialog, ColabStudioDialog
from annolid.gui.widgets.inference_wizard import InferenceWizard
from annolid.gui.widgets.training_wizard import TrainingWizard


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_colab_studio_dialog_prefill_and_notebook_generation(qapp, tmp_path: Path):
    ds_root = tmp_path / "yolo_ds"
    (ds_root / "images" / "train").mkdir(parents=True)
    (ds_root / "images" / "train" / "f1.jpg").write_bytes(b"img")
    data_yaml = ds_root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(ds_root),
                "train": "images/train",
                "val": "images/train",
                "names": {0: "mouse"},
            }
        ),
        encoding="utf-8",
    )

    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "missing_creds.json"),
        token_file=str(tmp_path / "missing_token.json"),
    )
    svc = ColabWorkflowService(auth_manager=auth, home_dir=tmp_path / "colab_home")
    dlg = ColabStudioDialog(service=svc)

    # Prefill from training config
    dlg.prefill_training_config(
        {
            "backend": "yolo",
            "dataset_path": str(data_yaml),
            "model": "yolo11s-seg.pt",
            "epochs": 42,
            "batch": 12,
            "imgsz": 640,
        }
    )
    assert dlg.tabs.currentIndex() == 1
    assert dlg.train_dataset_edit.text() == str(data_yaml)
    assert dlg.train_epochs_spin.value() == 42
    assert dlg.train_batch_spin.value() == 12

    # Prefill from inference config
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"vid")
    dlg.prefill_inference_config(
        {
            "model_type": "yolo",
            "videos": [str(video)],
            "model_path": "yolo11n-seg.pt",
            "score_threshold": 0.4,
            "save_csv": True,
        }
    )
    assert dlg.tabs.currentIndex() == 2
    assert dlg.predict_source_edit.text() == str(video)
    assert dlg.predict_conf_spin.value() == pytest.approx(0.4)

    # Generate training notebook via background worker
    dlg.train_open_browser_check.setChecked(False)
    dlg.train_upload_drive_check.setChecked(False)
    dlg._start_train_job(force_notebook=True)

    # Wait for QThread to finish cleanly
    for _ in range(200):
        qapp.processEvents()
        if dlg._worker_thread is None:
            break
        dlg._worker_thread.wait(25)

    jobs = svc.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status == "notebook_ready"
    assert Path(jobs[0].notebook_path).is_file()
    assert dlg.jobs_table.rowCount() == 1

    # Emitting openPredictionsRequested works when selecting a job with outputs
    out_dir = tmp_path / "pred_out"
    (out_dir / "predictions").mkdir(parents=True)
    jobs[0].local_output_dir = str(out_dir)
    svc.job_store.save(jobs[0])
    dlg.refresh_jobs()
    dlg.jobs_table.selectRow(0)

    emitted: list[str] = []
    dlg.openPredictionsRequested.connect(emitted.append)
    dlg._on_load_selected_predictions()
    assert emitted == [str(out_dir / "predictions")]
    dlg.close()


def test_workflow_actions_mixin_train_on_colab_opens_studio(qapp, tmp_path: Path):
    class _DummyWindow(WorkflowActionsMixin, QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.video_file = str(tmp_path / "session.mp4")
            self.annotation_dir = str(tmp_path)
            self.imported_dirs: list[str] = []

        def importDirImages(self, path: str):
            self.imported_dirs.append(path)

    win = _DummyWindow()
    dlg = win.train_on_colab()
    assert isinstance(dlg, ColabStudioDialog)
    assert dlg.tabs.currentIndex() == 1
    assert dlg.predict_source_edit.text() == str(tmp_path / "session.mp4")
    dlg.close()
    win.close()


def test_training_and_inference_wizards_emit_colab_requested(qapp, tmp_path: Path):
    train_wiz = TrainingWizard()
    captured_train: list[dict] = []
    train_wiz.colab_requested.connect(captured_train.append)
    train_wiz.request_colab_launch()
    assert len(captured_train) == 1
    assert "backend" in captured_train[0]
    train_wiz.close()

    inf_wiz = InferenceWizard(recent_models=[])
    captured_inf: list[dict] = []
    inf_wiz.colab_requested.connect(captured_inf.append)
    inf_wiz.request_colab_launch()
    assert len(captured_inf) == 1
    assert "model_type" in captured_inf[0]
    inf_wiz.close()


def test_colab_gui_sign_in_prompt_and_auto_resume(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ds_root = tmp_path / "yolo_ds"
    (ds_root / "images" / "train").mkdir(parents=True)
    (ds_root / "images" / "train" / "f1.jpg").write_bytes(b"img")
    data_yaml = ds_root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(ds_root),
                "train": "images/train",
                "val": "images/train",
                "names": {0: "mouse"},
            }
        ),
        encoding="utf-8",
    )

    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "missing_creds.json"),
        token_file=str(tmp_path / "missing_token.json"),
    )
    monkeypatch.setattr(auth, "find_colab_cli", lambda: None)
    monkeypatch.setattr(auth, "find_gcloud_cli", lambda: None)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "no_adc.json"))

    svc = ColabWorkflowService(auth_manager=auth, home_dir=tmp_path / "colab_home")

    # Verify ColabSignInDialog options & default selection
    sign_in_dlg = ColabSignInDialog(
        reason="No credentials found.",
        auth_status=svc.auth_status(),
        allow_notebook_fallback=True,
    )
    assert sign_in_dlg.selected_action() == "personal"
    sign_in_dlg.close()

    dlg = ColabStudioDialog(service=svc)
    assert dlg.install_cli_btn.isEnabled() is True
    dlg.train_dataset_edit.setText(str(data_yaml))
    dlg.train_open_browser_check.setChecked(False)
    dlg.train_upload_drive_check.setChecked(False)

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(
            lambda _parent, title, msg: (_ for _ in ()).throw(
                AssertionError(f"Unexpected QMessageBox.warning({title!r}, {msg!r})")
            )
        ),
    )

    events: list[tuple[str, str]] = []

    def _fake_login(**kwargs):
        events.append(("login", str(kwargs.get("method"))))
        return {"ok": True, "method": kwargs.get("method"), "status": svc.auth_status()}

    def _fake_run_train(req, **_kwargs):
        events.append(("train", req.backend))
        ckpt = tmp_path / "out" / "train" / "weights" / "best.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"pt")
        rec = ColabJobRecord(
            job_id="train_resumed_1",
            kind="train",
            model=req.model,
            backend=req.backend,
            accelerator=req.accelerator,
            session_name="warm-gpu-1",
            status="completed",
            local_output_dir=str(tmp_path / "out"),
            downloaded_files=[str(ckpt)],
        )
        svc.job_store.save(rec)
        return rec

    monkeypatch.setattr(svc, "login", _fake_login)
    monkeypatch.setattr(svc, "run_train", _fake_run_train)
    monkeypatch.setattr(dlg, "_prompt_sign_in", lambda **_k: "colab-cli")
    dlg.train_keep_rt_check.setChecked(True)

    dlg._start_train_job(force_notebook=False)

    # Process both login thread and auto-resumed training thread
    for _ in range(300):
        qapp.processEvents()
        if ("train", "cli") in events and dlg._worker_thread is None:
            break
        if dlg._worker_thread is not None:
            dlg._worker_thread.wait(15)

    assert events == [("login", "colab-cli"), ("train", "cli")]
    assert dlg.jobs_table.rowCount() == 1
    # Auto-handoff fills Inference tab with trained best.pt and kept session name
    assert dlg.predict_weights_combo.currentText().endswith("best.pt")
    assert dlg.predict_session_edit.text() == "warm-gpu-1"

    # Verify mid-flight ColabAuthRequiredError triggers prompt and notebook fallback
    events.clear()
    monkeypatch.setattr(
        auth,
        "get_auth_status",
        lambda: {
            "recommended_backend": "api",
            "summary": "API ready",
            "colab_cli": {"installed": False},
            "oauth_colab": {"ready": True},
            "oauth_drive": {"ready": False},
            "adc": {"ready": False, "gcloud_installed": False},
            "cli_ready": False,
            "api_ready": True,
            "drive_ready": False,
            "credentials_file": str(tmp_path / "c.json"),
            "token_file": str(tmp_path / "t.json"),
        },
    )

    def _stale_then_notebook(req, **_kwargs):
        events.append(("train", req.backend))
        if req.backend != "notebook":
            raise ColabAuthRequiredError("Expired token", backend="api")
        rec = ColabJobRecord(
            job_id="train_fallback_nb",
            kind="train",
            model=req.model,
            backend="notebook",
            accelerator=req.accelerator,
            status="notebook_ready",
        )
        svc.job_store.save(rec)
        return rec

    monkeypatch.setattr(svc, "run_train", _stale_then_notebook)
    monkeypatch.setattr(dlg, "_prompt_sign_in", lambda **_k: "notebook")

    dlg._start_train_job(force_notebook=False)
    for _ in range(300):
        qapp.processEvents()
        if ("train", "notebook") in events and dlg._worker_thread is None:
            break
        if dlg._worker_thread is not None:
            dlg._worker_thread.wait(15)

    assert events == [("train", "auto"), ("train", "notebook")]
    dlg.close()


def test_colab_gui_personal_account_banner_switch_and_live_monitor(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from qtpy import QtGui

    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "missing_creds.json"),
        token_file=str(tmp_path / "missing_token.json"),
    )
    monkeypatch.setattr(auth, "find_colab_cli", lambda: None)
    monkeypatch.setattr(auth, "find_gcloud_cli", lambda: None)
    svc = ColabWorkflowService(auth_manager=auth, home_dir=tmp_path / "colab_home")
    dlg = ColabStudioDialog(service=svc)

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(
            lambda _parent, title, msg: (_ for _ in ()).throw(
                AssertionError(f"Unexpected warning: {title}: {msg}")
            )
        ),
    )

    # 1. Check personal account refresh populates header & Account tab
    acct_state = {
        "signed_in": True,
        "email": "alice@gmail.com",
        "provider": "colab-cli (oauth2)",
        "expires_in": "55m",
        "compute_balance": "12.50 compute units",
        "usage_rate": "0.00/hr",
        "active_assignments": 0,
    }
    monkeypatch.setattr(svc, "get_account_info", lambda: dict(acct_state))
    dlg._on_refresh_account_clicked()
    for _ in range(200):
        qapp.processEvents()
        if (
            "alice@gmail.com" in dlg.header_account_label.text()
            and dlg._worker_thread is None
        ):
            break
        if dlg._worker_thread is not None:
            dlg._worker_thread.wait(15)
    assert "alice@gmail.com" in dlg.header_account_label.text()
    assert "12.50 compute units" in dlg.account_balance_label.text()

    # 2. Switch Account clears previous credentials, signs in new personal account, and refreshes
    switch_events: list[str] = []

    def _fake_logout(**kwargs):
        switch_events.append("logout")
        assert kwargs.get("include_colab_cli") is True
        return {
            "ok": True,
            "removed_token": True,
            "token_file": "t",
            "status": svc.auth_status(),
        }

    def _fake_login(**kwargs):
        switch_events.append(f"login:{kwargs.get('method')}")
        acct_state["email"] = "bob@gmail.com"
        return {"ok": True, "method": kwargs.get("method"), "status": svc.auth_status()}

    monkeypatch.setattr(svc, "logout", _fake_logout)
    monkeypatch.setattr(svc, "login", _fake_login)
    dlg._on_switch_account_clicked()
    for _ in range(400):
        qapp.processEvents()
        if (
            "bob@gmail.com" in dlg.header_account_label.text()
            and dlg._worker_thread is None
        ):
            break
        if dlg._worker_thread is not None:
            dlg._worker_thread.wait(15)
    assert switch_events == ["logout", "login:personal"]
    assert "bob@gmail.com" in dlg.header_account_label.text()

    # 3. Live Monitor updates GPU, Colab URL, epoch bar, KPI cards, chart, and log filter
    dlg._reset_monitor(
        kind="train", model="yolo", session_name="annolid-gpu", total_steps=3
    )
    dlg._append_log(
        "Colab URL: https://colab.research.google.com/notebooks/empty.ipynb"
    )
    dlg._append_log(
        "|   0  Tesla T4                       Off |   00000000:00:04.0 Off | 0 |"
    )
    dlg._append_log(
        "| N/A   48C    P8             10W /   70W |       0MiB /  15360MiB | 0% |"
    )
    dlg._append_log(
        "        2/3     0.450G      1.800      3.400      3.600      1.500 0 40 320: 100% 9/9"
    )
    dlg._append_log(
        "                   all         18        198      0.400      0.350      0.380     0.2100      0.390      0.340      0.365     0.1950"
    )
    assert dlg.monitor_gpu_label.text() == "Tesla T4 (15360 MiB)"
    assert dlg.monitor_open_colab_btn.isEnabled() is True
    assert dlg._kpi_values["step"].text() == "2/3"
    assert dlg._kpi_values["map50"].text() == "0.3800"
    assert dlg._kpi_values["mask_map50"].text() == "0.3650"
    dlg.monitor_chart.repaint()

    # Log filter narrows displayed text without losing raw history
    dlg.log_filter_edit.setText("Tesla T4")
    assert "Tesla T4" in dlg.log_view.toPlainText()
    assert "0.450G" not in dlg.log_view.toPlainText()
    dlg.log_filter_edit.clear()
    assert "0.450G" in dlg.log_view.toPlainText()

    # 4. Inspect completed job loads results.csv + results.png into Live Monitor
    run_out = tmp_path / "completed_run" / "train"
    run_out.mkdir(parents=True)
    (run_out / "results.csv").write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B)\n"
        "1,2.1,3.2,1.5,0.22\n"
        "2,1.4,2.1,1.1,0.58\n",
        encoding="utf-8",
    )
    img = QtGui.QImage(64, 48, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.QColor(30, 120, 200))
    img.save(str(run_out / "results.png"))

    rec = ColabJobRecord(
        job_id="train_inspect_1",
        kind="train",
        model="yolo",
        backend="cli",
        accelerator="T4",
        status="completed",
        local_output_dir=str(run_out.parent),
        downloaded_files=[str(run_out / "results.csv"), str(run_out / "results.png")],
        result_summary={"elapsed_seconds": 41.0},
    )
    dlg.inspect_job_in_monitor(rec)
    assert dlg.tabs.currentIndex() == 3
    assert dlg._kpi_values["map50"].text() == "0.5800"
    assert dlg.monitor_plot_combo.count() == 1
    assert dlg.monitor_open_output_btn.isEnabled() is True
    dlg.close()
