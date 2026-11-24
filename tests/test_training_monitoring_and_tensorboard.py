from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image
import pytest
from qtpy import QtCore, QtWidgets

from annolid.gui import tensorboard as tb_mod
from annolid.gui.mixins.project_workflow_mixin import ProjectWorkflowMixin
from annolid.gui.mixins.training_workflow_mixin import TrainingWorkflowMixin
from annolid.gui.mixins.workflow_actions_mixin import WorkflowActionsMixin
from annolid.gui.widgets.training_charts import (
    ChartSeries,
    RunComparisonMatrixWidget,
    SystemTelemetryWidget,
    TrainingArtifactsGalleryWidget,
    TrainingCurvePlotWidget,
    TrainingMetricsCanvas,
    series_color,
)
from annolid.gui.widgets.training_dashboard import (
    TrainingDashboardDialog,
    TrainingDashboardWidget,
)
from annolid.services.training_interop import (
    detect_available_trackers,
    export_run_to_rerun,
    export_run_to_wandb,
    export_runs_comparison_csv,
    export_runs_to_mlflow,
    export_training_experiment_report,
)
from annolid.services.training_monitor import (
    RUN_METADATA_FILENAME,
    RunMetadata,
    analyze_run_health,
    build_tensorboard_logdir_spec,
    canonicalize_run_dir,
    compute_hparam_diff,
    discover_training_runs,
    inspect_run_dir,
    load_run_metadata,
    normalize_run_tags,
    read_tensorboard_scalars,
    save_run_metadata,
    smooth_ema,
)
from annolid.services.training_telemetry import (
    SystemTelemetrySampler,
    TelemetrySample,
    collect_telemetry_sample,
    format_telemetry_summary,
)

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _create_sample_yolo_run(root: Path, name: str = "20261124_120000") -> Path:
    run_dir = root / "yolo" / "yolo11s-seg" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Include Ultralytics-style whitespace padding around column names.
    _write_csv(
        run_dir / "results.csv",
        [
            {
                "  epoch": "1",
                "  train/box_loss": "1.20",
                "  train/seg_loss": "0.80",
                "  val/box_loss": "1.05",
                "  val/seg_loss": "0.70",
                "  metrics/mAP50(B)": "0.42",
                "  metrics/mAP50(M)": "0.39",
                "  lr/pg0": "0.002",
            },
            {
                "  epoch": "2",
                "  train/box_loss": "0.90",
                "  train/seg_loss": "0.60",
                "  val/box_loss": "0.85",
                "  val/seg_loss": "0.55",
                "  metrics/mAP50(B)": "0.61",
                "  metrics/mAP50(M)": "0.58",
                "  lr/pg0": "0.0015",
            },
        ],
    )
    (run_dir / "args.yaml").write_text(
        "epochs: 20\nmodel: yolo11s-seg.pt\n", encoding="utf-8"
    )
    (run_dir / "train.log").write_text(
        "Epoch 1/20 loss=2.0\nEpoch 2/20 loss=1.5\n", encoding="utf-8"
    )
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"pt")
    Image.new("RGB", (96, 64), color=(35, 99, 229)).save(run_dir / "results.png")
    (run_dir / "events.out.tfevents.1000.host").write_bytes(b"")
    return run_dir


def _create_sample_maskrcnn_run(root: Path, name: str = "20261124_130000") -> Path:
    run_dir = root / "maskrcnn" / "train" / name
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / "training_log.csv",
        [
            {
                "iteration": "50",
                "train_loss": "1.45",
                "loss_classifier": "0.45",
                "loss_box_reg": "0.30",
                "loss_mask": "0.70",
                "lr": "0.0025",
            },
            {
                "iteration": "100",
                "train_loss": "0.95",
                "loss_classifier": "0.25",
                "loss_box_reg": "0.20",
                "loss_mask": "0.50",
                "lr": "0.0025",
            },
        ],
    )
    (run_dir / "command.txt").write_text(
        "annolid-run train maskrcnn_detectron2 --max-iterations=500\n",
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text("iter 50/500\niter 100/500\n", encoding="utf-8")
    (run_dir / "model_final.pth").write_bytes(b"pth")
    (tb_dir / "events.out.tfevents.2000.host").write_bytes(b"")
    (tb_dir / "projector_config.pbtxt").write_text("embeddings {}\n", encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Layer 1: Domain & IO monitoring service tests
# ---------------------------------------------------------------------------


def test_canonicalize_and_discover_deduplicates_tensorboard_subdirs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    yolo_dir = _create_sample_yolo_run(runs_root)
    mrcnn_dir = _create_sample_maskrcnn_run(runs_root)

    assert (
        canonicalize_run_dir(mrcnn_dir / "tensorboard", runs_root=runs_root)
        == mrcnn_dir.resolve()
    )
    # Never step above runs_root if user explicitly sets a folder named "tensorboard" as runs_root.
    explicit_tb_root = tmp_path / "tensorboard"
    explicit_tb_root.mkdir()
    assert (
        canonicalize_run_dir(explicit_tb_root, runs_root=explicit_tb_root)
        == explicit_tb_root.resolve()
    )

    discovered = discover_training_runs(runs_root)
    discovered_dirs = {r.run_dir for r in discovered}
    assert discovered_dirs == {yolo_dir.resolve(), mrcnn_dir.resolve()}

    by_dir = {r.run_dir: r for r in discovered}

    yolo = by_dir[yolo_dir.resolve()]
    assert yolo.task == "yolo"
    assert yolo.model == "yolo11s-seg"
    assert yolo.step_unit == "epoch"
    assert yolo.last_step == 2
    assert yolo.total_steps == 20
    assert yolo.train_loss == pytest.approx(1.5)
    assert yolo.val_loss == pytest.approx(1.4)
    assert yolo.primary_metric_name == "metrics/mAP50(M)"
    assert yolo.primary_metric_value == pytest.approx(0.58)
    assert yolo.checkpoints[0].name == "best.pt"
    assert any(p.name == "results.png" for p in yolo.plot_images)

    mrcnn = by_dir[mrcnn_dir.resolve()]
    assert mrcnn.task == "maskrcnn"
    assert mrcnn.step_unit == "iteration"
    assert mrcnn.last_step == 100
    assert mrcnn.total_steps == 500
    assert mrcnn.train_loss == pytest.approx(0.95)
    assert mrcnn.tensorboard_dir == (mrcnn_dir / "tensorboard").resolve()
    assert mrcnn.has_projector is True
    assert mrcnn.checkpoints[0].name == "model_final.pth"


def test_smooth_ema_and_logdir_spec_builder(tmp_path: Path) -> None:
    raw = [1.0, 2.0, 3.0, 4.0]
    assert smooth_ema(raw, 0.0) == (1.0, 2.0, 3.0, 4.0)
    smoothed = smooth_ema(raw, 0.6)
    assert len(smoothed) == 4
    assert smoothed[0] == pytest.approx(1.0)
    assert smoothed[-1] < 4.0

    r1 = _create_sample_yolo_run(tmp_path / "runs", "run_a")
    r2 = _create_sample_maskrcnn_run(tmp_path / "runs", "run_b")
    spec = build_tensorboard_logdir_spec([("dup:name", r1), ("dup:name", r2)])
    parts = spec.split(",")
    assert len(parts) == 2
    assert parts[0].startswith("dup_name:")
    assert parts[1].startswith("dup_name_2:")
    assert parts[1].endswith(str((r2 / "tensorboard").resolve()))


def test_read_tensorboard_scalars_merges_and_degrades_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "tb_only_run"
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True)

    # Write a real tfevents file when torch SummaryWriter + tensorboard are installed.
    wrote_real = False
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_scalar("loss/total", 1.25, 1)
        writer.add_scalar("loss/total", 0.75, 2)
        writer.add_scalar("metrics/bbox_AP", 0.64, 2)
        writer.flush()
        writer.close()
        wrote_real = True
    except Exception:
        (tb_dir / "events.out.tfevents.999.host").write_bytes(b"")

    if wrote_real:
        scalars = read_tensorboard_scalars(tb_dir)
        assert "loss/total" in scalars
        assert scalars["loss/total"].steps == (1, 2)
        assert scalars["loss/total"].latest_value == pytest.approx(0.75)

        inspected = inspect_run_dir(run_dir, include_tb_scalars=True)
        assert inspected.train_loss == pytest.approx(0.75)
        assert inspected.primary_metric_name == "metrics/bbox_AP"
        assert inspected.primary_metric_value == pytest.approx(0.64)

    # Verify graceful empty fallback when tensorboard cannot be imported.
    import builtins

    real_import = builtins.__import__

    def _block_tb(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorboard" or name.startswith("tensorboard."):
            raise ModuleNotFoundError("blocked tensorboard")
        return real_import(name, globals, locals, fromlist, level)

    tb_dir2 = tmp_path / "tb_blocked"
    tb_dir2.mkdir()
    (tb_dir2 / "events.out.tfevents.111.host").write_bytes(b"")
    monkeypatch.setattr(builtins, "__import__", _block_tb)
    assert read_tensorboard_scalars(tb_dir2) == {}


# ---------------------------------------------------------------------------
# Layer 2: TensorBoard process lifecycle & embed tests
# ---------------------------------------------------------------------------


def test_ensure_tensorboard_session_reports_stderr_and_cleans_up_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tb_mod.stop_tensorboard()

    class _CrashingProc:
        def __init__(self) -> None:
            self.stdout = io.StringIO("ImportError: broken protobuf\n")
            self.killed = False
            self.terminated = False

        def poll(self):
            return 2

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 2

        def kill(self):
            self.killed = True

    monkeypatch.setattr(tb_mod.subprocess, "Popen", lambda *a, **kw: _CrashingProc())
    monkeypatch.setattr(tb_mod, "_pick_free_port", lambda **kw: 6123)

    with pytest.raises(RuntimeError) as exc_info:
        tb_mod.ensure_tensorboard_session(
            log_dir=tmp_path,
            startup_timeout_s=1.0,
            ensure_runtime=False,
        )
    assert "exit code 2" in str(exc_info.value)
    assert "broken protobuf" in str(exc_info.value)

    # Timeout case must terminate the spawned process instead of leaking it.
    class _HungProc:
        def __init__(self) -> None:
            self.stdout = io.StringIO("still loading...\n")
            self._rc = None
            self.terminated = False

        def poll(self):
            return self._rc

        def terminate(self):
            self.terminated = True
            self._rc = -15

        def wait(self, timeout=None):
            return self._rc

        def kill(self):
            self._rc = -9

    hung = _HungProc()
    monkeypatch.setattr(tb_mod.subprocess, "Popen", lambda *a, **kw: hung)
    monkeypatch.setattr(tb_mod, "_can_connect", lambda *a, **kw: False)

    with pytest.raises(TimeoutError):
        tb_mod.ensure_tensorboard_session(
            logdir_spec=f"r1:{tmp_path}",
            startup_timeout_s=0.15,
            ensure_runtime=False,
        )
    assert hung.terminated is True


def test_visualization_window_and_embed_idle_and_fast_close() -> None:
    _ensure_qapp()
    win = tb_mod.VisualizationWindow(auto_start=False)
    try:
        assert win.embed.state == "idle"
        assert win.embed._web_view is None
        assert win.embed._stack.currentIndex() == 0
        t0 = time.monotonic()
        win.close()
        elapsed = time.monotonic() - t0
        # Legacy VisualizationWindow had time.sleep(3) in closeEvent.
        assert elapsed < 1.0
    finally:
        win.deleteLater()


# ---------------------------------------------------------------------------
# Layer 3: Native charts & artifacts gallery tests
# ---------------------------------------------------------------------------


def test_training_charts_and_gallery_render_across_modes(tmp_path: Path) -> None:
    _ensure_qapp()
    runs_root = tmp_path / "runs"
    yolo_dir = _create_sample_yolo_run(runs_root, "run_1")
    mrcnn_dir = _create_sample_maskrcnn_run(runs_root, "run_2")

    r1 = inspect_run_dir(yolo_dir, runs_root=runs_root)
    r2 = inspect_run_dir(mrcnn_dir, runs_root=runs_root)

    plot = TrainingCurvePlotWidget(title="Test Loss")
    plot.resize(480, 260)
    plot.set_chart_data(
        series=[
            ChartSeries("Run A", (1, 2, 3), (1.2, 0.8, 0.4), series_color(0)),
            ChartSeries(
                "Run B", (1, 2, 3), (1.0, 0.6, 0.0), series_color(1), dashed=True
            ),
        ]
    )
    plot.set_options(smoothing=0.6, log_scale=False, show_markers=True)
    pm = plot.grab()
    assert not pm.isNull()

    # Log scale with zero values must not raise
    plot.set_options(log_scale=True)
    pm_log = plot.grab()
    assert not pm_log.isNull()

    canvas = TrainingMetricsCanvas()
    canvas.resize(760, 340)
    canvas.set_runs(r1, [r1, r2])
    assert (
        canvas.compare_banner.isVisible() is True
        or "Comparing 2 runs" in canvas.compare_banner.text()
    )
    for idx in range(canvas.view_mode_combo.count()):
        canvas.view_mode_combo.setCurrentIndex(idx)
        assert not canvas.grab().isNull()

    gallery = TrainingArtifactsGalleryWidget()
    gallery.set_run(r1)
    assert gallery.file_list.count() == 1
    assert gallery.open_img_btn.isEnabled() is True
    assert gallery.open_tb_images_btn.isEnabled() is True


# ---------------------------------------------------------------------------
# Layer 4 & 5: Unified dashboard & GUI workflow integration tests
# ---------------------------------------------------------------------------


def test_training_dashboard_discovers_compares_and_tracks_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure_qapp()
    runs_root = tmp_path / "runs"
    _create_sample_yolo_run(runs_root, "run_yolo")
    mrcnn_dir = _create_sample_maskrcnn_run(runs_root, "run_mrcnn")
    monkeypatch.setenv("ANNOLID_RUNS_ROOT", str(runs_root))

    dash = TrainingDashboardWidget()
    try:
        assert dash.active_table.rowCount() == 2
        assert dash.runs_list.count() == 2

        # Select MaskRCNN run and verify KPI / metrics / projector discovery
        dash.select_run_dir(mrcnn_dir)
        assert dash._selected_run_key == str(mrcnn_dir.resolve())
        assert dash.epoch_table.rowCount() == 2
        assert "projector_config.pbtxt" in dash.info_view.toPlainText()

        # Check both rows for multi-run comparison
        for row in range(dash.active_table.rowCount()):
            item = dash.active_table.item(row, dash._COL_COMPARE)
            assert item is not None
            item.setCheckState(QtCore.Qt.Checked)
        assert len(dash._compared_keys) == 2
        assert dash.compare_in_tb_btn.isEnabled() is True

        # Scope "compared" resolves to a valid --logdir_spec
        idx = dash.tb_scope_combo.findData("compared")
        dash.tb_scope_combo.setCurrentIndex(idx)
        log_dir, spec = dash._resolve_tb_scope()
        assert log_dir is None
        assert spec is not None and "," in spec

        # Subprocess-tracked run lifecycle (e.g. Mask R-CNN / Keypoint R-CNN)
        sub_run_dir = runs_root / "maskrcnn" / "train" / "live_sub"
        sub_run_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            dash.register_subprocess_run(
                task="maskrcnn",
                model="maskrcnn_resnet50_fpn",
                run_dir=sub_run_dir,
                process=proc,
                total_steps=200,
                step_unit="iteration",
            )
            assert dash._runs[str(sub_run_dir.resolve())].status == "running"
            assert dash.stop_training_btn.isEnabled() is True

            # Stopping terminates the subprocess cleanly
            dash._stop_selected_training()
            assert proc.poll() is not None
            assert dash._runs[str(sub_run_dir.resolve())].status == "failed"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
    finally:
        dash.close()
        dash.deleteLater()


def test_workflow_visualization_opens_unified_dashboard_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure_qapp()
    runs_root = tmp_path / "runs"
    _create_sample_yolo_run(runs_root)
    monkeypatch.setenv("ANNOLID_RUNS_ROOT", str(runs_root))

    # Ensure legacy blocking ensure_tensorboard is NOT called by visualization()
    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "visualization() must not synchronously launch TensorBoard"
        )

    monkeypatch.setattr(tb_mod, "ensure_tensorboard", _fail_if_called)

    class _FakeMainWindow(
        WorkflowActionsMixin, ProjectWorkflowMixin, TrainingWorkflowMixin
    ):
        def __init__(self) -> None:
            self.settings = None
            self._training_dashboard_dialog = None
            self._active_training_subprocesses = []

        def ensure_yolo_training_manager(self):
            return None

        def ensure_dino_kpseg_training_manager(self):
            return None

    win = _FakeMainWindow()
    dlg = win.visualization()
    try:
        assert isinstance(dlg, TrainingDashboardDialog)
        assert win._training_dashboard_dialog is dlg
        assert dlg.dashboard.tabs.currentWidget() is dlg.dashboard.active_tab

        # Launching a monitored CLI training subprocess records command + log and registers in dashboard
        cli_run = tmp_path / "cli_run"
        proc = win._launch_monitored_cli_training(
            cmd=[sys.executable, "-c", "print('hello training')"],
            task="keypoint_rcnn",
            model="keypointrcnn_resnet50_fpn",
            run_dir=cli_run,
            total_steps=100,
            step_unit="iteration",
        )
        assert proc is not None
        proc.wait(timeout=5)
        assert (cli_run / "command.txt").is_file()
        assert (cli_run / "train.log").is_file()
        assert str(cli_run.resolve()) in dlg.dashboard._runs
    finally:
        dlg.close()
        dlg.deleteLater()


def test_project_workflow_open_tensorboard_flag_routes_to_unified_dashboard() -> None:
    recorded = {}

    class _FakeWin(ProjectWorkflowMixin):
        def _start_yolo_training_from_wizard(self, _cfg):
            recorded["started"] = True

        def open_training_dashboard(self, **kwargs):
            recorded["dashboard_kwargs"] = kwargs

    win = _FakeWin()
    win._on_training_requested(
        {
            "backend": "yolo",
            "open_dashboard": True,
            "open_tensorboard": True,
        }
    )
    assert recorded["started"] is True
    assert recorded["dashboard_kwargs"] == {
        "initial_tab": "tensorboard",
        "run_dir": None,
        "start_tensorboard": True,
    }


def test_tensorboard_embed_transitions_and_cancel_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure_qapp()
    tb_mod.stop_tensorboard()

    # Pre-set cancel_event must abort immediately without launching a process.
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        tb_mod.ensure_tensorboard_session(
            log_dir=tmp_path,
            cancel_event=cancel,
            ensure_runtime=False,
        )

    embed = tb_mod.TensorBoardEmbedWidget()
    try:
        states: list[tuple[str, str]] = []
        embed.status_changed.connect(lambda st, msg: states.append((st, msg)))

        gen = embed._launch_generation

        # Simulate error state
        embed._on_worker_failed(gen, "port blocked")
        assert embed.state == "error"
        assert "port blocked" in embed.placeholder_detail.text()

        # Simulate running state without QtWebEngine (headless fallback)
        class _FakeProc:
            def __init__(self) -> None:
                self.terminated = False

            def poll(self):
                return None if not self.terminated else 0

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.terminated = True

        # Stale generation must be discarded and its process terminated cleanly.
        stale_proc = _FakeProc()
        stale_sess = tb_mod.TensorBoardSession(
            process=stale_proc,  # type: ignore[arg-type]
            url="http://127.0.0.1:6098/",
            host="127.0.0.1",
            port=6098,
            log_dir=tmp_path,
        )
        embed._on_worker_launched(gen - 1, stale_sess)
        assert stale_proc.terminated is True
        assert embed.session is None

        fake_proc = _FakeProc()
        monkeypatch.setattr(embed, "_ensure_web_view", lambda: None)
        embed._pending_open_external_fallback = False
        embed._pending_initial_plugin = "images"
        sess = tb_mod.TensorBoardSession(
            process=fake_proc,  # type: ignore[arg-type]
            url="http://127.0.0.1:6099/",
            host="127.0.0.1",
            port=6099,
            log_dir=tmp_path,
            has_projector=True,
        )
        embed._on_worker_launched(gen, sess)
        assert embed.state == "running"
        assert embed._target_url_for_session(sess) == "http://127.0.0.1:6099/#images"
        assert embed.browser_btn.isEnabled() is True
        assert embed.projector_btn.isEnabled() is True

        # Verify failed WebEngine page load switches cleanly back to placeholder card.
        class _FakeWebView:
            def url(self):
                return QtCore.QUrl("http://127.0.0.1:6099/#images")

        embed._web_view = _FakeWebView()  # type: ignore[assignment]
        embed._stack.setCurrentIndex(1)
        embed._on_web_load_finished(False)
        assert embed._stack.currentIndex() == 0
        assert "failed to render" in embed.placeholder_detail.text().lower()
        embed._web_view = None

        embed.stop_server()
        assert embed.state == "idle"
        assert fake_proc.terminated is True
    finally:
        embed.cleanup()
        embed.deleteLater()


def test_colab_dialog_monitor_opens_unified_training_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure_qapp()
    colab_mod = pytest.importorskip("annolid.colab")
    colab_dlg_mod = pytest.importorskip("annolid.gui.widgets.colab_dialog")
    ColabAuthManager = colab_mod.ColabAuthManager
    ColabWorkflowService = colab_mod.ColabWorkflowService
    ColabStudioDialog = colab_dlg_mod.ColabStudioDialog

    out_dir = _create_sample_yolo_run(tmp_path / "colab_runs", "colab_job_1")
    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "creds.json"),
        token_file=str(tmp_path / "token.json"),
    )
    svc = ColabWorkflowService(auth_manager=auth, home_dir=tmp_path / "colab_home")
    dlg = ColabStudioDialog(service=svc, auto_refresh_account=False)
    try:
        emitted: list[str] = []
        dlg.openTrainingDashboardRequested.connect(emitted.append)
        assert dlg.monitor_open_dashboard_btn.isEnabled() is False

        dlg._populate_monitor_outputs(str(out_dir))
        assert dlg.monitor_open_dashboard_btn.isEnabled() is True
        dlg.monitor_open_dashboard_btn.click()
        assert emitted == [str(out_dir)]
    finally:
        dlg.close()
        dlg.deleteLater()


def test_live_tensorboard_embed_widget_async_launch_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end non-blocking QThread launch of real TensorBoard via TensorBoardEmbedWidget."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                pass
    except OSError:
        pytest.skip("Loopback sockets are blocked in this sandbox environment.")

    app = _ensure_qapp()
    run_dir = tmp_path / "embed_live_run"
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_scalar("loss/total", 0.42, 1)
        writer.flush()
        writer.close()
    except Exception:
        pytest.skip("torch.utils.tensorboard unavailable")

    tb_mod.stop_tensorboard()
    embed = tb_mod.TensorBoardEmbedWidget()
    # Keep headless minimal-QPA test deterministic while exercising real worker + TensorBoard server.
    monkeypatch.setattr(embed, "_ensure_web_view", lambda: None)
    try:
        started_sessions: list[tb_mod.TensorBoardSession] = []
        errors: list[str] = []
        embed.session_started.connect(lambda s: started_sessions.append(s))
        embed.status_changed.connect(
            lambda st, msg: errors.append(msg) if st == "error" else None
        )

        ok = embed.start_server(
            log_dir=run_dir,
            preferred_port=16026,
            initial_plugin="images",
            open_external_if_no_webengine=False,
            check_capability=False,
        )
        assert ok is True
        assert embed.state == "starting"

        deadline = time.monotonic() + 20.0
        while not started_sessions and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.05)

        assert not errors, f"TensorBoard embed failed: {errors}"
        assert len(started_sessions) == 1
        sess = started_sessions[0]
        assert embed.state == "running"
        assert embed._target_url_for_session(sess).endswith("/#images")
        assert tb_mod._can_connect(sess.url, timeout=2.0) is True

        embed.stop_server()
        assert embed.state == "idle"
        assert sess.process.poll() is not None
    finally:
        embed.cleanup()
        embed.deleteLater()


def test_live_tensorboard_server_startup_and_teardown(tmp_path: Path) -> None:
    """Real end-to-end check launching tensorboard.main, probing HTTP, and tearing down."""
    import socket

    # Skip cleanly only when running inside a network-restricted sandbox that blocks loopback.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                pass
    except OSError:
        pytest.skip("Loopback sockets are blocked in this sandbox environment.")

    run_dir = tmp_path / "live_tb_run"
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_scalar("loss/total", 0.5, 1)
        writer.flush()
        writer.close()
    except Exception:
        pytest.skip("torch.utils.tensorboard unavailable")

    tb_mod.stop_tensorboard()
    session = tb_mod.ensure_tensorboard_session(
        log_dir=run_dir,
        preferred_port=16006,
        host="127.0.0.1",
        startup_timeout_s=20.0,
        ensure_runtime=False,
    )
    try:
        assert session.process.poll() is None
        assert tb_mod._can_connect(session.url, timeout=2.0) is True
        # Reusing with the same scope returns the same live process.
        reused = tb_mod.ensure_tensorboard_session(
            log_dir=run_dir,
            preferred_port=16006,
            host="127.0.0.1",
            startup_timeout_s=5.0,
            ensure_runtime=False,
        )
        assert reused.process is session.process
        assert reused.url == session.url
    finally:
        tb_mod.stop_tensorboard(session.process)
        assert session.process.poll() is not None


# ---------------------------------------------------------------------------
# Layer 6: Run metadata, hparam diff, health diagnostics, telemetry & interop
# ---------------------------------------------------------------------------


def test_run_metadata_roundtrip_and_best_metrics(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    yolo_dir = _create_sample_yolo_run(runs_root, "exp_meta")

    assert normalize_run_tags("baseline, Mouse Top , baseline, aug v2") == (
        "baseline",
        "Mouse-Top",
        "aug-v2",
    )

    meta = RunMetadata(
        alias="mouse-seg-baseline",
        tags=normalize_run_tags("baseline, Top View, baseline"),
        notes="Frozen backbone 10 layers; clean validation masks.",
        starred=True,
    )
    saved = save_run_metadata(yolo_dir, meta)
    assert saved == (yolo_dir / RUN_METADATA_FILENAME).resolve()
    assert saved.is_file()

    loaded = load_run_metadata(yolo_dir)
    assert loaded.alias == "mouse-seg-baseline"
    assert loaded.tags == ("baseline", "Top-View")
    assert loaded.notes == "Frozen backbone 10 layers; clean validation masks."
    assert loaded.starred is True
    assert loaded.updated_at is not None

    inspected = inspect_run_dir(yolo_dir, runs_root=runs_root)
    assert inspected.metadata.alias == "mouse-seg-baseline"
    assert inspected.metadata.starred is True
    assert inspected.min_train_loss == pytest.approx(1.5)
    assert inspected.min_val_loss == pytest.approx(1.4)
    assert inspected.best_metric_value == pytest.approx(0.58)
    assert inspected.best_metric_step == 2


def test_hparam_diff_merges_config_and_command_flags(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    r1_dir = _create_sample_yolo_run(runs_root, "hp_a")
    r2_dir = _create_sample_yolo_run(runs_root, "hp_b")
    (r1_dir / "args.yaml").write_text(
        "epochs: 50\nlr0: 0.002\nimgsz: 640\nmodel: yolo11s-seg.pt\n",
        encoding="utf-8",
    )
    (r1_dir / "command.txt").write_text(
        "annolid-run train yolo --freeze 10 --cos-lr\n", encoding="utf-8"
    )
    (r2_dir / "args.yaml").write_text(
        "epochs: 100\nlr0: 0.005\nimgsz: 640\nmodel: yolo11s-seg.pt\n",
        encoding="utf-8",
    )
    (r2_dir / "command.txt").write_text(
        "annolid-run train yolo --freeze 0\n", encoding="utf-8"
    )

    r1 = inspect_run_dir(r1_dir, runs_root=runs_root)
    r2 = inspect_run_dir(r2_dir, runs_root=runs_root)

    diff_only = compute_hparam_diff([r1, r2], only_diffs=True)
    diff_keys = {row.key for row in diff_only}
    assert {"epochs", "lr0", "freeze", "cos_lr"}.issubset(diff_keys)
    assert "imgsz" not in diff_keys
    assert all(row.differs for row in diff_only)

    all_rows = compute_hparam_diff([r1, r2], only_diffs=False)
    all_keys = {row.key for row in all_rows}
    assert "imgsz" in all_keys
    # Differing rows must appear before identical rows.
    first_identical = next(i for i, row in enumerate(all_rows) if not row.differs)
    assert all(row.differs for row in all_rows[:first_identical])


def test_analyze_run_health_detects_all_anomaly_types(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"

    # 1. NaN + loss spike + missing checkpoint + low disk
    bad_dir = runs_root / "yolo" / "yolo11s" / "bad_run"
    bad_dir.mkdir(parents=True)
    _write_csv(
        bad_dir / "results.csv",
        [
            {
                "epoch": "1",
                "train_loss": "1.0",
                "val_loss": "1.0",
                "metrics/mAP50(B)": "0.3",
            },
            {
                "epoch": "2",
                "train_loss": "0.9",
                "val_loss": "0.9",
                "metrics/mAP50(B)": "0.4",
            },
            {
                "epoch": "3",
                "train_loss": "0.8",
                "val_loss": "0.8",
                "metrics/mAP50(B)": "0.5",
            },
            {
                "epoch": "4",
                "train_loss": "0.7",
                "val_loss": "0.7",
                "metrics/mAP50(B)": "0.55",
            },
            {
                "epoch": "5",
                "train_loss": "8.5",
                "val_loss": "NaN",
                "metrics/mAP50(B)": "0.2",
            },
        ],
    )
    bad_run = inspect_run_dir(bad_dir, runs_root=runs_root, status="finished")
    findings = analyze_run_health(bad_run, disk_free_gb=0.3)
    codes = {f.code: f.severity for f in findings}
    assert codes["nan_metric"] == "critical"
    assert codes["loss_spike"] == "critical"
    assert codes["no_checkpoint"] == "warning"
    assert codes["low_disk"] == "critical"

    # 2. Overfitting + metric plateau
    overfit_dir = runs_root / "yolo" / "yolo11s" / "overfit_run"
    overfit_dir.mkdir(parents=True)
    rows = []
    for ep in range(1, 23):
        train_l = max(0.05, 1.5 - ep * 0.06)
        val_l = 0.8 if ep <= 4 else 0.8 + (ep - 4) * 0.05
        map_val = 0.82 if ep == 4 else 0.78
        rows.append(
            {
                "epoch": str(ep),
                "train_loss": f"{train_l:.4f}",
                "val_loss": f"{val_l:.4f}",
                "metrics/mAP50(B)": f"{map_val:.4f}",
            }
        )
    _write_csv(overfit_dir / "results.csv", rows)
    (overfit_dir / "weights").mkdir()
    (overfit_dir / "weights" / "best.pt").write_bytes(b"ok")
    overfit_run = inspect_run_dir(overfit_dir, runs_root=runs_root, status="finished")
    of_codes = {f.code: f.severity for f in overfit_run.health_findings}
    assert of_codes["overfitting"] == "warning"
    assert of_codes["metric_plateau"] == "info"
    assert "no_checkpoint" not in of_codes


def test_system_telemetry_sampler_and_widget_render(tmp_path: Path) -> None:
    _ensure_qapp()
    sample = collect_telemetry_sample(disk_path=tmp_path, probe_gpu=False)
    assert sample.disk_free_gb is not None and sample.disk_free_gb > 0
    assert "Disk" in format_telemetry_summary(sample)

    sampler = SystemTelemetrySampler(max_samples=16, disk_path=tmp_path)
    s1 = sampler.sample_now()
    s2 = sampler.sample_now()
    assert len(sampler.history()) == 2
    assert sampler.latest() == s2
    assert s1.timestamp <= s2.timestamp

    widget = SystemTelemetryWidget()
    widget.resize(640, 280)
    synthetic = [
        TelemetrySample(
            timestamp=1.0,
            cpu_percent=35.0,
            ram_used_gb=8.0,
            ram_total_gb=32.0,
            ram_percent=25.0,
            disk_free_gb=120.0,
            disk_total_gb=500.0,
            gpu_name="NVIDIA RTX 4090",
            gpu_util_percent=82.0,
            gpu_mem_used_mib=10240.0,
            gpu_mem_total_mib=24576.0,
            gpu_temp_c=68.0,
        ),
        TelemetrySample(
            timestamp=4.0,
            cpu_percent=42.0,
            ram_used_gb=8.5,
            ram_total_gb=32.0,
            ram_percent=26.6,
            disk_free_gb=119.8,
            disk_total_gb=500.0,
            gpu_name="NVIDIA RTX 4090",
            gpu_util_percent=91.0,
            gpu_mem_used_mib=11264.0,
            gpu_mem_total_mib=24576.0,
            gpu_temp_c=71.0,
        ),
    ]
    widget.set_samples(synthetic)
    assert "GPU 91%" in widget.summary_label.text()
    assert (
        "68" not in widget.summary_label.text()
        and "71°C" in widget.summary_label.text()
    )
    assert not widget.grab().isNull()


def test_experiment_report_export_and_tracker_interop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import types

    runs_root = tmp_path / "runs"
    r1_dir = _create_sample_yolo_run(runs_root, "rep_a")
    r2_dir = _create_sample_maskrcnn_run(runs_root, "rep_b")
    save_run_metadata(
        r1_dir,
        RunMetadata(
            alias="yolo-baseline",
            tags=("baseline", "mouse"),
            notes="Best validation mask AP.",
            starred=True,
        ),
    )
    r1 = inspect_run_dir(r1_dir, runs_root=runs_root)
    r2 = inspect_run_dir(r2_dir, runs_root=runs_root)

    # Report export (HTML5 + inline SVG + Markdown + JSON + CSV)
    out_dir = tmp_path / "report_out"
    artifacts = export_training_experiment_report(
        [r1, r2], out_dir, title="Lab Comparison"
    )
    assert artifacts.html_path.is_file()
    assert artifacts.markdown_path.is_file()
    assert artifacts.json_path.is_file()
    assert artifacts.csv_path.is_file()

    html_txt = artifacts.html_path.read_text(encoding="utf-8")
    assert "<svg " in html_txt
    assert "yolo-baseline" in html_txt
    assert "Best validation mask AP." in html_txt
    assert 'class="diff"' in html_txt

    md_txt = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "# Lab Comparison" in md_txt
    assert "## Lab Notes" in md_txt
    assert "## Hyperparameter Differences" in md_txt

    csv_standalone = export_runs_comparison_csv([r1, r2], tmp_path / "flat.csv")
    assert "yolo-baseline" in csv_standalone.read_text(encoding="utf-8")

    # detect_available_trackers probes all 4 backends cleanly
    trackers = detect_available_trackers()
    assert [t.name for t in trackers] == ["tensorboard", "mlflow", "wandb", "rerun"]

    # Graceful actionable error when optional backends are not installed
    real_import = builtins.__import__

    def _block_optional(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"mlflow", "wandb", "rerun"}:
            raise ImportError(f"blocked {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_optional)
    with pytest.raises(RuntimeError, match="pip install mlflow"):
        export_runs_to_mlflow([r1])
    with pytest.raises(RuntimeError, match="pip install wandb"):
        export_run_to_wandb(r1)
    with pytest.raises(RuntimeError, match="pip install rerun-sdk"):
        export_run_to_rerun(r1, tmp_path / "out.rrd")
    monkeypatch.setattr(builtins, "__import__", real_import)

    # Mocked happy paths for MLflow, W&B, and Rerun exporters
    mlflow_calls: dict[str, object] = {"metrics": [], "artifacts": [], "tags": {}}

    class _FakeMlflowRun:
        class _Info:
            run_id = "mlf-run-1"

        info = _Info()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda uri: mlflow_calls.__setitem__("uri", uri)
    fake_mlflow.get_tracking_uri = lambda: mlflow_calls.get("uri", "file:///tmp/mlruns")
    fake_mlflow.set_experiment = lambda name: mlflow_calls.__setitem__("exp", name)
    fake_mlflow.start_run = lambda run_name=None: _FakeMlflowRun()
    fake_mlflow.set_tags = lambda d: mlflow_calls["tags"].update(d)
    fake_mlflow.set_tag = lambda k, v: mlflow_calls["tags"].__setitem__(k, v)
    fake_mlflow.log_params = lambda d: mlflow_calls.__setitem__("params", d)
    fake_mlflow.log_metric = lambda k, v, step=0: mlflow_calls["metrics"].append(
        (k, v, step)
    )
    fake_mlflow.log_artifact = lambda p, artifact_path=None: mlflow_calls[
        "artifacts"
    ].append((p, artifact_path))
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    mlf_res = export_runs_to_mlflow([r1], tracking_uri="file:///tmp/test_mlruns")
    assert mlf_res["exported_run_ids"] == ["mlf-run-1"]
    assert mlflow_calls["metrics"]
    assert mlflow_calls["artifacts"]
    assert mlflow_calls["tags"]["mlflow.note.content"] == "Best validation mask AP."

    wb_logged: list[tuple[dict, object]] = []

    class _FakeWbRun:
        id = "wb-1"
        dir = str(tmp_path / "wandb_offline")
        finished = False

        def log(self, payload, step=None):
            wb_logged.append((payload, step))

        def finish(self):
            self.finished = True

    fake_wb_run = _FakeWbRun()
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kw: fake_wb_run
    fake_wandb.Image = lambda path: f"WBImage({path})"
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wb_res = export_run_to_wandb(r1, project="lab-proj", mode="offline")
    assert wb_res["run_id"] == "wb-1"
    assert fake_wb_run.finished is True
    assert len(wb_logged) >= 2

    rr_logged: list[tuple[str, object]] = []
    fake_rr = types.ModuleType("rerun")
    fake_rr.init = lambda app_id, spawn=False: None
    fake_rr.save = lambda path: Path(path).write_bytes(b"RRD")
    fake_rr.set_time_sequence = lambda name, step: None
    fake_rr.Scalars = lambda v: ("scalar", v)
    fake_rr.Image = lambda arr: ("image", arr.shape)
    fake_rr.log = lambda entity, obj: rr_logged.append((entity, obj))
    monkeypatch.setitem(sys.modules, "rerun", fake_rr)

    rrd_path = export_run_to_rerun(r1, tmp_path / "trace.rrd")
    assert rrd_path.is_file()
    assert any(ent.startswith("metrics/") for ent, _ in rr_logged)
    assert any(ent.startswith("plots/") for ent, _ in rr_logged)


def test_dashboard_notes_tags_star_health_compare_and_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure_qapp()
    runs_root = tmp_path / "runs"
    healthy_dir = _create_sample_yolo_run(runs_root, "healthy_run")
    bad_dir = runs_root / "yolo" / "yolo11s-seg" / "nan_run"
    bad_dir.mkdir(parents=True)
    _write_csv(
        bad_dir / "results.csv",
        [
            {"epoch": "1", "train_loss": "1.0", "val_loss": "1.0"},
            {"epoch": "2", "train_loss": "NaN", "val_loss": "1.1"},
        ],
    )
    (bad_dir / "args.yaml").write_text(
        "epochs: 50\nmodel: yolo11m-seg.pt\n", encoding="utf-8"
    )
    monkeypatch.setenv("ANNOLID_RUNS_ROOT", str(runs_root))

    dash = TrainingDashboardWidget()
    try:
        assert dash.active_table.rowCount() == 2
        assert "Disk" in dash.telemetry_status_label.text()

        # Select anomalous run -> Health diagnostics banner must appear
        dash.select_run_dir(bad_dir)
        assert not dash.health_banner.isHidden()
        assert "Non-finite" in dash.health_banner.text()

        # Select healthy run -> Health banner hidden, save notes/tags/star
        dash.select_run_dir(healthy_dir)
        assert dash.health_banner.isHidden()
        dash.meta_alias_edit.setText("prod-candidate")
        dash.meta_tags_edit.setText("mouse, top-view")
        dash.meta_starred_check.setChecked(True)
        dash.meta_notes_edit.setPlainText("Validated on held-out cage videos.")
        dash.save_meta_btn.click()

        assert (healthy_dir / RUN_METADATA_FILENAME).is_file()
        assert dash.star_run_btn.isChecked() is True

        # Starred-only filter and tag search filter work end-to-end
        dash.starred_only_check.setChecked(True)
        assert dash.active_table.rowCount() == 1
        dash.starred_only_check.setChecked(False)
        assert dash.active_table.rowCount() == 2

        dash.run_filter_edit.setText("top-view")
        assert dash.active_table.rowCount() == 1
        dash.run_filter_edit.setText("")

        # Compare & Diff widget shows both runs and toggles diff-only cleanly
        dash.select_tab("compare")
        assert dash.details_tabs.currentWidget() is dash.compare_tab
        assert isinstance(dash.compare_matrix, RunComparisonMatrixWidget)
        assert dash.compare_matrix.metrics_table.rowCount() == 2
        diff_rows = dash.compare_matrix.hparam_table.rowCount()
        dash.compare_matrix.only_diffs_check.setChecked(False)
        all_rows = dash.compare_matrix.hparam_table.rowCount()
        assert all_rows >= diff_rows >= 1

        # Tab navigation helpers for notes & system telemetry
        dash.select_tab("notes")
        assert dash.details_tabs.currentWidget() is dash.notes_tab
        dash.select_tab("system")
        assert dash.details_tabs.currentWidget() is dash.system_tab
    finally:
        dash.close()
        dash.deleteLater()

    # Metadata survives a fresh dashboard reload
    dash2 = TrainingDashboardWidget()
    try:
        dash2.select_run_dir(healthy_dir)
        assert dash2.meta_alias_edit.text() == "prod-candidate"
        assert "top-view" in dash2.meta_tags_edit.text()
        assert dash2.meta_starred_check.isChecked() is True
        assert "held-out cage videos" in dash2.meta_notes_edit.toPlainText()
    finally:
        dash2.close()
        dash2.deleteLater()


def test_audit_fixes_step_aligned_overfitting_and_tb_prefixed_metrics(
    tmp_path: Path,
) -> None:
    from annolid.services.training_monitor import (
        DiscoveredRun,
        MetricSeries,
        summarize_run_best_metrics,
    )

    # Validation logged every 5 epochs while train logged every epoch.
    # Best val is at epoch 10 (index 1 in val_series, index 9 in train_series).
    # Train loss at epoch 10 is 0.50 and drops to 0.30 by epoch 30 -> true overfitting.
    train_steps = tuple(range(1, 31))
    train_vals = tuple(max(0.25, 1.0 - s * 0.025) for s in train_steps)
    val_steps = (5, 10, 15, 20, 25, 30)
    val_vals = (0.70, 0.50, 0.55, 0.62, 0.72, 0.85)
    run = DiscoveredRun(
        run_dir=tmp_path / "sparse_val",
        display_name="sparse_val",
        status="running",
        metrics={
            "ultralytics/train/box_loss": MetricSeries(
                "ultralytics/train/box_loss",
                train_steps,
                train_vals,
                source="tensorboard",
            ),
            "ultralytics/val/box_loss": MetricSeries(
                "ultralytics/val/box_loss", val_steps, val_vals, source="tensorboard"
            ),
        },
    )
    min_tr, min_va, _best_m, _best_st = summarize_run_best_metrics(run.metrics)
    assert min_tr == pytest.approx(min(train_vals))
    assert min_va == pytest.approx(0.50)

    findings = analyze_run_health(run)
    codes = {f.code for f in findings}
    assert "overfitting" in codes

    # Unreadable metadata JSON must degrade cleanly to default RunMetadata.
    corrupt_dir = tmp_path / "corrupt_meta"
    corrupt_dir.mkdir()
    (corrupt_dir / RUN_METADATA_FILENAME).write_text(
        "{not valid json", encoding="utf-8"
    )
    assert load_run_metadata(corrupt_dir) == RunMetadata()


def test_audit_fixes_multigpu_nvidia_smi_parsing() -> None:
    from annolid.services.training_telemetry import _parse_nvidia_smi_csv

    stdout = (
        "NVIDIA RTX A6000, 3, 512, 49140, 38\nNVIDIA RTX A6000, 94, 18432, 49140, 74\n"
    )
    name, util, mem_used, mem_total, temp = _parse_nvidia_smi_csv(stdout)
    assert name == "NVIDIA RTX A6000 [GPU 1/2]"
    assert util == pytest.approx(94.0)
    assert mem_used == pytest.approx(18432.0)
    assert mem_total == pytest.approx(49140.0)
    assert temp == pytest.approx(74.0)


def test_audit_fixes_interop_escaping_nan_safety_and_mlflow_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types
    from annolid.services.training_interop import _sanitize_mlflow_metric_key
    from annolid.services.training_monitor import DiscoveredRun, MetricSeries

    assert _sanitize_mlflow_metric_key("metrics/mAP@50:95(B)%") == "metrics/mAP_50_95_B"

    run = DiscoveredRun(
        run_dir=tmp_path / "tricky_run",
        display_name="tricky_run",
        task="yolo",
        model="yolo11s",
        status="finished",
        metadata=RunMetadata(
            alias="v1 | fast\nexp",
            tags=("a|b",),
            notes="Note with | pipe",
            starred=True,
        ),
        metrics={
            "train_loss": MetricSeries(
                "train_loss", (1, 2, 3), (1.0, float("nan"), 0.5)
            ),
            "metrics/mAP@50:95(B)%": MetricSeries(
                "metrics/mAP@50:95(B)%", (1, 2), (0.4, float("inf"))
            ),
        },
        hparams={"custom": "left | right\nnext"},
    )
    artifacts = export_training_experiment_report([run], tmp_path / "rep_tricky")
    html_txt = artifacts.html_path.read_text(encoding="utf-8")
    assert 'points="' in html_txt
    assert "nan" not in html_txt.lower()
    assert "inf," not in html_txt.lower()

    md_txt = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "v1 \\| fast exp" in md_txt

    logged_metrics: list[tuple[str, float, int]] = []

    class _FakeRun:
        class _Info:
            run_id = "id-1"

        info = _Info()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda _u: None
    fake_mlflow.get_tracking_uri = lambda: "file:///tmp/mlruns"
    fake_mlflow.set_experiment = lambda _e: None
    fake_mlflow.start_run = lambda run_name=None: _FakeRun()
    fake_mlflow.set_tags = lambda _d: None
    fake_mlflow.set_tag = lambda _k, _v: None
    fake_mlflow.log_params = lambda _d: None
    fake_mlflow.log_metric = lambda k, v, step=0: logged_metrics.append((k, v, step))
    fake_mlflow.log_artifact = lambda *_a, **_kw: None
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    export_runs_to_mlflow([run])
    assert all(k == _sanitize_mlflow_metric_key(k) for k, _v, _s in logged_metrics)
    assert len(logged_metrics) == 3  # NaN and Inf points were skipped


def test_audit_fixes_dashboard_dirty_notes_star_revert_filter_and_shlex_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from annolid.gui.widgets import training_dashboard as td_mod
    from annolid.services.training_monitor import extract_effective_hparams

    _ensure_qapp()
    runs_root = tmp_path / "runs"
    r1 = _create_sample_yolo_run(runs_root, "run_alpha")
    r2 = _create_sample_maskrcnn_run(runs_root, "run_beta")
    save_run_metadata(r1, RunMetadata(alias="alpha-best", starred=True))
    monkeypatch.setenv("ANNOLID_RUNS_ROOT", str(runs_root))

    dash = TrainingDashboardWidget()
    try:
        # Alias propagates to curves legend & TensorBoard logdir_spec
        dash.select_run_dir(r1)
        assert (
            TrainingMetricsCanvas._short_run_label(
                dash._runs[str(r1.resolve())].discovered
            )
            == "alpha-best"
        )
        dash._compared_keys = {str(r1.resolve()), str(r2.resolve())}
        idx = dash.tb_scope_combo.findData("compared")
        dash.tb_scope_combo.setCurrentIndex(idx)
        _ld, spec = dash._resolve_tb_scope()
        assert spec is not None and "alpha-best:" in spec

        # Typing unsaved notes marks dirty and survives polling refresh
        dash.meta_notes_edit.setPlainText("Unsaved lab observation")
        assert dash._meta_dirty is True
        assert "*" in dash.save_meta_btn.text()
        dash._refresh_active_metrics()
        assert dash.meta_notes_edit.toPlainText() == "Unsaved lab observation"

        # If saving star fails, star_run_btn reverts its checked state
        monkeypatch.setattr(
            td_mod,
            "save_run_metadata",
            lambda *_a, **_kw: (_ for _ in ()).throw(PermissionError("read-only")),
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_kw: None)
        dash.star_run_btn.setChecked(False)
        dash._toggle_selected_run_star(False)
        assert dash.star_run_btn.isChecked() is True

        # Filtering out selected run switches selection coherently (and clears on 0 matches)
        dash.select_run_dir(r2)
        assert dash._selected_run_key == str(r2.resolve())
        dash.starred_only_check.setChecked(True)
        assert dash.active_table.rowCount() == 1
        assert dash._selected_run_key == str(r1.resolve())

        dash.run_filter_edit.setText("nonexistent_filter_token")
        assert dash.active_table.rowCount() == 0
        assert dash._selected_run_key is None
    finally:
        dash.close()
        dash.deleteLater()

    # CLI launcher shell-quotes paths with spaces so command.txt parses back cleanly
    class _FakeCliWin(TrainingWorkflowMixin):
        pass

    spaced_dir = tmp_path / "My Lab Experiments" / "run 1"
    win = _FakeCliWin()
    proc = win._launch_monitored_cli_training(
        cmd=[
            sys.executable,
            "-c",
            "pass",
            "--dataset-dir",
            str(tmp_path / "My Dataset Folder"),
            "--max-iterations=250",
        ],
        task="maskrcnn",
        model="maskrcnn_resnet50_fpn",
        run_dir=spaced_dir,
    )
    assert proc is not None
    proc.wait(timeout=5)
    inspected = inspect_run_dir(spaced_dir)
    hp = extract_effective_hparams(inspected)
    assert hp.get("dataset_dir") == str(tmp_path / "My Dataset Folder")
    assert hp.get("max_iterations") == "250"


def test_residual_fixes_async_background_export_lifecycle_and_error_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _ensure_qapp()
    runs_root = tmp_path / "runs"
    _create_sample_yolo_run(runs_root, "async_exp")
    monkeypatch.setenv("ANNOLID_RUNS_ROOT", str(runs_root))

    warnings: list[tuple[str, str]] = []
    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _p, title, text, *_a: warnings.append((str(title), str(text))),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda _p, title, text, *_a: infos.append((str(title), str(text))),
    )

    dash = TrainingDashboardWidget()
    try:
        # 1. Success path runs off the Qt main thread and restores button state
        worker_threads: list[bool] = []
        done_payloads: list[object] = []
        gate = threading.Event()

        def _bg_ok() -> object:
            worker_threads.append(
                threading.current_thread() is not threading.main_thread()
            )
            gate.wait(timeout=2.0)
            return {"ok": True}

        started = dash._start_background_export(
            label="Exporting Test",
            error_title="Export Failed",
            fn=_bg_ok,
            on_success=done_payloads.append,
        )
        assert started is True
        assert dash.export_menu_btn.isEnabled() is False

        # Concurrent export while in-flight must be rejected cleanly
        second = dash._start_background_export(
            label="Duplicate",
            error_title="Err",
            fn=lambda: None,
            on_success=lambda _r: None,
        )
        assert second is False
        assert any("Export In Progress" in t for t, _ in infos)

        gate.set()
        deadline = time.monotonic() + 5.0
        while not done_payloads and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)

        assert worker_threads == [True]
        assert done_payloads == [{"ok": True}]
        assert dash.export_menu_btn.isEnabled() is True
        assert dash.export_menu_btn.text() == "Export / Trackers"

        # Wait briefly for QThread finished cleanup
        deadline = time.monotonic() + 2.0
        while dash._export_thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)

        # 2. Failure path surfaces warning and re-enables button
        started_err = dash._start_background_export(
            label="Exporting Broken",
            error_title="Tracker Error",
            fn=lambda: (_ for _ in ()).throw(RuntimeError("remote 503")),
            on_success=lambda _r: None,
        )
        assert started_err is True
        deadline = time.monotonic() + 5.0
        while not warnings and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        assert warnings == [("Tracker Error", "remote 503")]
        assert dash.export_menu_btn.isEnabled() is True

        # Wait briefly for QThread finished cleanup
        deadline = time.monotonic() + 2.0
        while dash._export_thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)

        # 3. Closing/cleanup mid-export discards stale result safely
        late_results: list[object] = []
        slow_gate = threading.Event()
        dash._start_background_export(
            label="Slow Export",
            error_title="Slow Err",
            fn=lambda: (slow_gate.wait(timeout=0.3), "late")[1],
            on_success=late_results.append,
        )
        dash.cleanup()
        slow_gate.set()
        app.processEvents()
        assert late_results == []
    finally:
        dash.close()
        dash.deleteLater()


def test_residual_fixes_real_subprocess_nvidia_smi_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat
    from annolid.services import training_telemetry as tel_mod

    if sys.platform.startswith("win"):
        pytest.skip("POSIX mock nvidia-smi script test")

    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir()
    fake_smi = bin_dir / "nvidia-smi"
    fake_smi.write_text(
        "#!/bin/sh\n"
        'echo "NVIDIA L40S, 5, 1024, 46068, 36"\n'
        'echo "NVIDIA L40S, 88, 22528, 46068, 72"\n',
        encoding="utf-8",
    )
    fake_smi.chmod(fake_smi.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    name, util, mem_used, mem_total, temp = tel_mod._probe_nvidia_gpu(timeout_s=2.0)
    assert name == "NVIDIA L40S [GPU 1/2]"
    assert util == pytest.approx(88.0)
    assert mem_used == pytest.approx(22528.0)
    assert mem_total == pytest.approx(46068.0)
    assert temp == pytest.approx(72.0)

    # Async sampler picks up the real subprocess probe without blocking sample_now()
    sampler = SystemTelemetrySampler(disk_path=tmp_path)
    sampler.sample_now()
    deadline = time.monotonic() + 3.0
    sample = None
    while time.monotonic() < deadline:
        sample = sampler.sample_now()
        if sample.gpu_util_percent is not None:
            break
        time.sleep(0.05)
    assert sample is not None
    assert sample.gpu_util_percent == pytest.approx(88.0)
    assert "GPU 88%" in format_telemetry_summary(sample)

    # Hanging nvidia-smi must time out cleanly without leaving _gpu_probe_in_flight stuck
    fake_smi.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    t0 = time.monotonic()
    res = tel_mod._probe_nvidia_gpu(timeout_s=0.15)
    assert time.monotonic() - t0 < 2.0
    assert res == ("", None, None, None, None)


def test_residual_fixes_mlflow_batching_uri_restore_wandb_dir_modern_rerun_and_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types
    from annolid.services.training_monitor import DiscoveredRun, MetricSeries

    run_dir = _create_sample_yolo_run(tmp_path / "runs", "sdk_hardening")
    run = inspect_run_dir(run_dir, runs_root=tmp_path / "runs")
    # Inject 135 hyperparameters (> MLflow's 100-param batch limit)
    run.hparams = {f"hp_{i}": i for i in range(135)}

    state: dict[str, object] = {
        "uri": "http://original-mlflow:5000",
        "param_chunks": [],
        "batches": [],
    }

    class _FakeMetric:
        def __init__(self, key, value, timestamp, step):
            self.key = key
            self.value = value
            self.timestamp = timestamp
            self.step = step

    class _FakeClient:
        def log_batch(self, run_id, metrics=()):
            state["batches"].append((run_id, list(metrics)))

    class _FakeRun:
        class _Info:
            run_id = "batch-run-1"

        info = _Info()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.tracking = types.SimpleNamespace(MlflowClient=_FakeClient)
    fake_mlflow.entities = types.SimpleNamespace(Metric=_FakeMetric)
    fake_mlflow.get_tracking_uri = lambda: state["uri"]
    fake_mlflow.set_tracking_uri = lambda u: state.__setitem__("uri", u)
    fake_mlflow.set_experiment = lambda _e: None
    fake_mlflow.start_run = lambda run_name=None: _FakeRun()
    fake_mlflow.set_tags = lambda _d: None
    fake_mlflow.set_tag = lambda _k, _v: None
    fake_mlflow.log_params = lambda d: state["param_chunks"].append(dict(d))
    fake_mlflow.log_metric = lambda *_a, **_kw: (_ for _ in ()).throw(
        AssertionError("Should use MlflowClient.log_batch when available")
    )
    fake_mlflow.log_artifact = lambda *_a, **_kw: None
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    mlf_res = export_runs_to_mlflow([run], tracking_uri="file:///tmp/override_mlruns")
    assert mlf_res["tracking_uri"] == "file:///tmp/override_mlruns"
    # Previous tracking URI must be restored after export
    assert state["uri"] == "http://original-mlflow:5000"
    assert len(state["param_chunks"]) == 2
    assert all(len(c) <= 100 for c in state["param_chunks"])
    assert len(state["batches"]) == 1 and len(state["batches"][0][1]) > 0

    # W&B offline colocation passes dir=str(run.run_dir)
    wb_init_kwargs: dict[str, object] = {}

    class _FakeWbRun:
        id = "wb-coloc"
        dir = str(run_dir / "wandb")

        def log(self, *_a, **_kw):
            pass

        def finish(self):
            pass

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kw: (wb_init_kwargs.update(kw), _FakeWbRun())[1]
    fake_wandb.Image = lambda p: p
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    export_run_to_wandb(run, mode="offline")
    assert wb_init_kwargs.get("dir") == str(run.run_dir.resolve())

    # Modern rerun-sdk >=0.22 uses set_time(timeline, sequence=...) + disconnect()
    rr_events: list[tuple[str, object]] = []
    fake_rr = types.ModuleType("rerun")
    fake_rr.init = lambda app_id, spawn=False: None
    fake_rr.save = lambda path: Path(path).write_bytes(b"RRD2")
    fake_rr.set_time = lambda timeline, sequence=0: rr_events.append(
        ("set_time", (timeline, sequence))
    )
    fake_rr.Scalars = lambda v: ("scalar", v)
    fake_rr.Image = lambda arr: ("img", arr.shape)
    fake_rr.log = lambda ent, obj: rr_events.append(("log", ent))
    fake_rr.disconnect = lambda: rr_events.append(("disconnect", True))
    monkeypatch.setitem(sys.modules, "rerun", fake_rr)

    out_rrd = export_run_to_rerun(run, tmp_path / "modern.rrd")
    assert out_rrd.is_file()
    assert any(k == "set_time" for k, _ in rr_events)
    assert ("disconnect", True) in rr_events

    # Configurable health thresholds
    mild_run = DiscoveredRun(
        run_dir=tmp_path / "mild",
        display_name="mild",
        metrics={
            "train_loss": MetricSeries(
                "train_loss", (1, 2, 3, 4, 5), (1.0, 0.9, 0.9, 0.9, 2.2)
            )
        },
    )
    assert not any(f.code == "loss_spike" for f in analyze_run_health(mild_run))
    assert any(
        f.code == "loss_spike" for f in analyze_run_health(mild_run, spike_factor=2.0)
    )
    assert any(
        f.code == "low_disk"
        for f in analyze_run_health(mild_run, disk_free_gb=4.0, low_disk_gb=5.0)
    )
