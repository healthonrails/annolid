"""Unified Training Monitor & Visualization Dashboard for Annolid.

Combines:
- Live & historical training run tracking across YOLO, DINO KPSEG, Mask R-CNN,
  Keypoint R-CNN, Behavior classifiers, and Colab outputs.
- Native pure-Qt interactive metric curves (with EMA smoothing, hover readout,
  log Y-scale, PNG export, and multi-run comparison) backed by both CSV logs
  and TensorBoard event files.
- Training plot & visual artifact gallery.
- Non-blocking embedded & external TensorBoard + Embedding Projector integration.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.tensorboard import (
    TensorBoardEmbedWidget,
    TensorBoardSession,
    _terminate_process,
)
from annolid.gui.widgets.training_charts import (
    RunComparisonMatrixWidget,
    SystemTelemetryWidget,
    TrainingArtifactsGalleryWidget,
    TrainingMetricsCanvas,
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
    DiscoveredRun,
    RunHealthFinding,
    RunMetadata,
    analyze_run_health,
    build_tensorboard_logdir_spec,
    canonicalize_run_dir,
    discover_training_runs,
    find_config_file,
    find_results_csv,
    find_train_log,
    format_timeline_line,
    format_timeline_lines,
    inspect_run_dir,
    normalize_run_tags,
    read_csv_rows,
    save_run_metadata,
    tail_log_text,
)
from annolid.services.training_telemetry import (
    SystemTelemetrySampler,
    format_telemetry_summary,
)
from annolid.utils.runs import shared_runs_root


# Backward-compatible helper aliases for callers/tests importing them directly.
_find_results_csv = find_results_csv
_find_args_yaml = find_config_file
_find_train_log = find_train_log
_read_csv_rows = read_csv_rows
_tail_text = tail_log_text
_format_timeline_line = format_timeline_line


@dataclass
class TrainingRunState:
    """Live or discovered training run tracked by the dashboard."""

    run_dir: Path
    task: str
    model: str
    status: str  # running | finished | failed | recorded
    started_at: float
    last_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    updated_at: Optional[float] = None
    results_csv: Optional[Path] = None
    csv_columns: Tuple[str, ...] = ()
    csv_rows: List[Dict[str, str]] = field(default_factory=list)
    log_path: Optional[Path] = None
    timeline_last_epoch: Optional[int] = None
    epoch_timeline: List[str] = field(default_factory=list)
    step_unit: str = "epoch"
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None
    tensorboard_dir: Optional[Path] = None
    has_projector: bool = False
    checkpoints: Tuple[Path, ...] = ()
    plot_images: Tuple[Path, ...] = ()
    discovered: Optional[DiscoveredRun] = None
    process: Optional[subprocess.Popen] = None
    is_live: bool = False


def _format_metric(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "—"
    if abs(value) > 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _format_timestamp(ts: Optional[float]) -> str:
    if ts is None or ts <= 0:
        return ""
    now = time.time()
    if abs(now - ts) < 86400:
        return time.strftime("%H:%M:%S", time.localtime(ts))
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


class _DashboardExportWorker(QtCore.QObject):
    """Runs experiment report or external tracker exports off the Qt UI thread."""

    succeeded = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(self, generation: int, fn: Callable[[], object]) -> None:
        super().__init__()
        self._generation = int(generation)
        self._fn = fn

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._fn()
            self.succeeded.emit(self._generation, result)
        except Exception as exc:
            self.failed.emit(self._generation, str(exc))


class TrainingDashboardWidget(QtWidgets.QWidget):
    """Unified training monitoring, native curves, artifacts, and TensorBoard dashboard."""

    _COL_COMPARE = 0
    _COL_TASK = 1
    _COL_MODEL = 2
    _COL_STATUS = 3
    _COL_PROGRESS = 4
    _COL_TRAIN_LOSS = 5
    _COL_VAL_LOSS = 6
    _COL_METRIC = 7
    _COL_ARTIFACTS = 8
    _COL_UPDATED = 9
    _COL_RUN_DIR = 10

    def __init__(
        self, *, settings: Optional[QtCore.QSettings] = None, parent=None
    ) -> None:
        super().__init__(parent=parent)
        self._settings = settings
        self._tb_process: Optional[subprocess.Popen] = None
        self._tb_url = "http://127.0.0.1:6006/"
        self._runs: Dict[str, TrainingRunState] = {}
        self._compared_keys: set[str] = set()
        self._training_managers: List[QtCore.QObject] = []
        self._selected_run_key: Optional[str] = None
        self._metadata_loaded_key: Optional[str] = None
        self._meta_dirty = False
        self._loading_metadata = False
        self._follow_log = True
        self._raw_log_text = ""
        self._rebuilding_table = False
        self._cleaned_up = False
        self._export_generation = 0
        self._export_thread: Optional[QtCore.QThread] = None
        self._export_worker: Optional[_DashboardExportWorker] = None
        self._export_callbacks: Dict[int, Tuple[Callable[[object], None], str]] = {}

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._refresh_active_metrics)

        self._telemetry_sampler = SystemTelemetrySampler(disk_path=shared_runs_root())
        self._telemetry_timer = QtCore.QTimer(self)
        self._telemetry_timer.setInterval(3000)
        self._telemetry_timer.timeout.connect(self._sample_telemetry)

        self._build_ui()
        self._load_settings()
        self._sync_env_runs_root()
        self._telemetry_sampler.set_disk_path(self.runs_root())
        self._sample_telemetry()
        self._telemetry_timer.start()
        self._refresh_runs_list()

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        with contextlib.suppress(Exception):
            self._timer.stop()
        with contextlib.suppress(Exception):
            self._telemetry_timer.stop()
        with contextlib.suppress(Exception):
            self._cancel_export_worker()
        with contextlib.suppress(Exception):
            self.tb_embed.cleanup()
        self._tb_process = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cleanup()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # Public integration API
    # ------------------------------------------------------------------ #

    def register_training_manager(self, manager: QtCore.QObject) -> None:
        if manager not in self._training_managers:
            self._training_managers.append(manager)
        started = getattr(manager, "training_started", None)
        finished = getattr(manager, "training_finished", None)
        if hasattr(started, "connect"):
            with contextlib.suppress(Exception):
                started.connect(self._on_training_started)
        if hasattr(finished, "connect"):
            with contextlib.suppress(Exception):
                finished.connect(self._on_training_finished)

    def register_subprocess_run(
        self,
        *,
        task: str,
        model: str,
        run_dir: Path | str,
        process: subprocess.Popen,
        total_steps: Optional[int] = None,
        step_unit: str = "iteration",
    ) -> None:
        """Track a CLI subprocess training job (e.g. Mask R-CNN, Keypoint R-CNN, YOLACT)."""
        resolved = canonicalize_run_dir(Path(run_dir))
        key = str(resolved)
        state = TrainingRunState(
            run_dir=resolved,
            task=str(task or "train"),
            model=str(model or resolved.name),
            status="running",
            started_at=time.time(),
            total_epochs=int(total_steps) if total_steps else None,
            step_unit=str(step_unit or "iteration"),
            process=process,
            is_live=True,
        )
        self._runs[key] = state
        self._sync_state_from_disk(state, include_tb_scalars=True)
        self._refresh_active_table()
        self._select_run(key)
        if not self._timer.isActive():
            self._timer.start()

    def runs_root(self) -> Path:
        raw = self.runs_root_edit.text().strip()
        if not raw:
            return shared_runs_root()
        return Path(raw).expanduser().resolve()

    def select_tab(self, tab_key: str) -> None:
        key = str(tab_key or "").strip().lower()
        sub_mapping = {
            "compare": self.compare_tab,
            "diff": self.compare_tab,
            "notes": self.notes_tab,
            "tags": self.notes_tab,
            "system": self.system_tab,
            "telemetry": self.system_tab,
            "plots": self.plots_tab,
            "log": self.log_tab,
        }
        if key in sub_mapping:
            self.tabs.setCurrentWidget(self.active_tab)
            self.details_tabs.setCurrentWidget(sub_mapping[key])
            return
        mapping = {
            "monitor": self.active_tab,
            "active": self.active_tab,
            "curves": self.active_tab,
            "tensorboard": self.tb_tab,
            "tb": self.tb_tab,
            "runs": self.runs_tab,
        }
        widget = mapping.get(key)
        if widget is not None:
            self.tabs.setCurrentWidget(widget)

    def select_run_dir(self, run_dir: Path | str) -> None:
        """Load (even if outside runs_root) and select a specific run directory."""
        resolved = canonicalize_run_dir(Path(run_dir))
        if not resolved.exists():
            return
        key = str(resolved)
        state = self._runs.get(key)
        if state is None:
            disc = inspect_run_dir(
                resolved,
                runs_root=self.runs_root(),
                include_tb_scalars=True,
            )
            state = self._state_from_discovered(disc)
            self._runs[key] = state
        else:
            self._sync_state_from_disk(state, include_tb_scalars=True)
        self._refresh_active_table()
        self._select_run(key)

    # ------------------------------------------------------------------ #
    # UI Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header row
        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)
        title = QtWidgets.QLabel(self.tr("Training Monitor & Visualization"))
        t_font = title.font()
        t_font.setPointSize(max(13, t_font.pointSize() + 3))
        t_font.setBold(True)
        title.setFont(t_font)
        self.summary_subtitle = QtWidgets.QLabel(
            self.tr(
                "Monitor live training runs, compare curves, browse plots, and launch TensorBoard."
            )
        )
        self.summary_subtitle.setProperty("muted", True)
        title_box.addWidget(title)
        title_box.addWidget(self.summary_subtitle)
        header.addLayout(title_box, 1)

        right_header = QtWidgets.QVBoxLayout()
        right_header.setSpacing(2)
        badge_row = QtWidgets.QHBoxLayout()
        badge_row.setSpacing(8)
        self.live_summary_badge = QtWidgets.QLabel(self.tr("0 active · 0 runs"))
        badge_font = self.live_summary_badge.font()
        badge_font.setBold(True)
        self.live_summary_badge.setFont(badge_font)

        self.export_menu_btn = QtWidgets.QPushButton(self.tr("Export / Trackers"), self)
        export_menu = QtWidgets.QMenu(self.export_menu_btn)
        export_menu.addAction(
            self.tr("Export HTML5 + SVG Report…"), self._export_experiment_report
        )
        export_menu.addAction(
            self.tr("Export Comparison CSV…"), self._export_comparison_csv
        )
        export_menu.addSeparator()
        export_menu.addAction(self.tr("Export Runs to MLflow…"), self._export_to_mlflow)
        export_menu.addAction(
            self.tr("Export Selected Run to W&B (Offline)…"), self._export_to_wandb
        )
        export_menu.addAction(
            self.tr("Export Selected Run to Rerun (.rrd)…"), self._export_to_rerun
        )
        export_menu.addSeparator()
        export_menu.addAction(
            self.tr("External Trackers Status…"), self._show_trackers_status
        )
        self.export_menu_btn.setMenu(export_menu)

        badge_row.addStretch(1)
        badge_row.addWidget(self.live_summary_badge)
        badge_row.addWidget(self.export_menu_btn)
        right_header.addLayout(badge_row)

        self.telemetry_status_label = QtWidgets.QLabel(self.tr("System: —"), self)
        self.telemetry_status_label.setProperty("muted", True)
        self.telemetry_status_label.setAlignment(QtCore.Qt.AlignRight)
        right_header.addWidget(self.telemetry_status_label)
        header.addLayout(right_header)
        layout.addLayout(header)

        # Runs root bar
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(6)
        self.runs_root_edit = QtWidgets.QLineEdit(self)
        self.runs_root_edit.setPlaceholderText(
            self.tr(
                "Runs root (shared across YOLO, DINO KPSEG, Mask R-CNN, Keypoint R-CNN)"
            )
        )
        self.runs_root_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        self.runs_root_browse.clicked.connect(self._browse_runs_root)
        self.refresh_btn = QtWidgets.QPushButton(self.tr("Rescan Runs"))
        self.refresh_btn.clicked.connect(self._refresh_runs_list)
        self.open_root_top_btn = QtWidgets.QPushButton(self.tr("Open Root"))
        self.open_root_top_btn.clicked.connect(self._open_runs_root)

        bar.addWidget(QtWidgets.QLabel(self.tr("Runs root:")))
        bar.addWidget(self.runs_root_edit, 1)
        bar.addWidget(self.runs_root_browse)
        bar.addWidget(self.refresh_btn)
        bar.addWidget(self.open_root_top_btn)
        layout.addLayout(bar)

        # TensorBoard control strip
        tb_frame = QtWidgets.QFrame(self)
        tb_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        tb_bar = QtWidgets.QHBoxLayout(tb_frame)
        tb_bar.setContentsMargins(8, 6, 8, 6)
        tb_bar.setSpacing(6)

        tb_bar.addWidget(QtWidgets.QLabel(self.tr("TensorBoard Scope:")))
        self.tb_scope_combo = QtWidgets.QComboBox(tb_frame)
        self.tb_scope_combo.addItem(self.tr("Selected Run"), "selected")
        self.tb_scope_combo.addItem(self.tr("Compare Checked Runs"), "compared")
        self.tb_scope_combo.addItem(self.tr("All Runs (Root)"), "root")
        tb_bar.addWidget(self.tb_scope_combo)

        self.start_tb_btn = QtWidgets.QPushButton(self.tr("Start TensorBoard"))
        self.start_tb_btn.clicked.connect(self._start_tensorboard)
        self.stop_tb_btn = QtWidgets.QPushButton(self.tr("Stop"))
        self.stop_tb_btn.setEnabled(False)
        self.stop_tb_btn.clicked.connect(self._stop_tensorboard)
        self.open_projector_btn = QtWidgets.QPushButton(self.tr("Projector"))
        self.open_projector_btn.setToolTip(
            self.tr("Launch TensorBoard directly on the Embedding Projector tab")
        )
        self.open_projector_btn.clicked.connect(
            lambda: self.start_tensorboard_for_current_scope(open_projector=True)
        )
        self.open_tb_btn = QtWidgets.QPushButton(self.tr("Open in Browser"))
        self.open_tb_btn.setEnabled(False)
        self.open_tb_btn.clicked.connect(self._open_tensorboard_in_browser)
        self.tb_status = QtWidgets.QLabel(self.tr("TensorBoard: idle"))
        self.tb_status.setProperty("muted", True)
        self.tb_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        tb_bar.addWidget(self.start_tb_btn)
        tb_bar.addWidget(self.stop_tb_btn)
        tb_bar.addWidget(self.open_projector_btn)
        tb_bar.addWidget(self.open_tb_btn)
        tb_bar.addWidget(self.tb_status, 1)
        layout.addWidget(tb_frame)

        # Primary tabs
        self.tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        self.active_tab = QtWidgets.QWidget(self)
        self.tb_tab = QtWidgets.QWidget(self)
        self.runs_tab = QtWidgets.QWidget(self)
        self.tabs.addTab(self.active_tab, self.tr("Monitor && Curves"))
        self.tabs.addTab(self.tb_tab, self.tr("TensorBoard"))
        self.tabs.addTab(self.runs_tab, self.tr("Runs Explorer"))

        self._build_active_tab()
        self._build_tb_tab()
        self._build_runs_tab()

        self.runs_root_edit.editingFinished.connect(self._on_runs_root_changed)

    def _build_active_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.active_tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Filter & comparison bar above runs table
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setSpacing(6)
        self.run_filter_edit = QtWidgets.QLineEdit(self.active_tab)
        self.run_filter_edit.setPlaceholderText(
            self.tr("Filter runs by task, model, status, tag, alias, or folder…")
        )
        self.run_filter_edit.textChanged.connect(
            lambda *_: self._refresh_active_table(sync_selected=False)
        )
        self.active_only_check = QtWidgets.QCheckBox(
            self.tr("Active session only"), self.active_tab
        )
        self.active_only_check.toggled.connect(
            lambda *_: self._refresh_active_table(sync_selected=False)
        )
        self.starred_only_check = QtWidgets.QCheckBox(
            self.tr("Starred only"), self.active_tab
        )
        self.starred_only_check.toggled.connect(
            lambda *_: self._refresh_active_table(sync_selected=False)
        )
        self.clear_compare_btn = QtWidgets.QPushButton(
            self.tr("Clear Compare"), self.active_tab
        )
        self.clear_compare_btn.setEnabled(False)
        self.clear_compare_btn.clicked.connect(self._clear_compared_runs)
        self.compare_in_tb_btn = QtWidgets.QPushButton(
            self.tr("Compare Checked in TensorBoard"), self.active_tab
        )
        self.compare_in_tb_btn.setEnabled(False)
        self.compare_in_tb_btn.clicked.connect(self._launch_tensorboard_for_compared)

        filter_row.addWidget(self.run_filter_edit, 1)
        filter_row.addWidget(self.active_only_check)
        filter_row.addWidget(self.starred_only_check)
        filter_row.addWidget(self.clear_compare_btn)
        filter_row.addWidget(self.compare_in_tb_btn)
        layout.addLayout(filter_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.active_tab)

        self.active_table = QtWidgets.QTableWidget(0, 11, splitter)
        self.active_table.setHorizontalHeaderLabels(
            [
                "Cmp",
                "Task",
                "Model",
                "Status",
                "Epoch / Step",
                "Train Loss",
                "Val Loss",
                "Metric",
                "Signals",
                "Updated",
                "Run Dir",
            ]
        )
        self.active_table.verticalHeader().setVisible(False)
        self.active_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.active_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.active_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.active_table.setAlternatingRowColors(True)
        self.active_table.horizontalHeader().setStretchLastSection(True)
        self.active_table.cellDoubleClicked.connect(self._open_selected_run_dir)
        self.active_table.itemChanged.connect(self._on_table_item_changed)
        splitter.addWidget(self.active_table)

        # Lower inspector pane
        details = QtWidgets.QWidget(splitter)
        details_layout = QtWidgets.QVBoxLayout(details)
        details_layout.setContentsMargins(0, 6, 0, 0)
        details_layout.setSpacing(6)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(6)
        self.selected_run_label = QtWidgets.QLabel(
            self.tr("Select a run to inspect metrics, curves, plots, and logs."),
            details,
        )
        self.selected_run_label.setProperty("muted", True)
        self.selected_run_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        header_row.addWidget(self.selected_run_label, 1)

        self.star_run_btn = QtWidgets.QPushButton(self.tr("☆ Star"), details)
        self.star_run_btn.setCheckable(True)
        self.star_run_btn.setEnabled(False)
        self.star_run_btn.setToolTip(
            self.tr(
                "Bookmark this run as a starred favorite (saved in annolid_run_meta.json)"
            )
        )
        self.star_run_btn.clicked.connect(self._toggle_selected_run_star)
        header_row.addWidget(self.star_run_btn)

        self.stop_training_btn = QtWidgets.QPushButton(
            self.tr("Stop Training"), details
        )
        self.stop_training_btn.clicked.connect(self._stop_selected_training)
        self.stop_training_btn.setEnabled(False)
        header_row.addWidget(self.stop_training_btn)

        self.run_inference_btn = QtWidgets.QPushButton(
            self.tr("Run Inference…"), details
        )
        self.run_inference_btn.clicked.connect(self._run_inference_for_selected_run)
        self.run_inference_btn.setEnabled(False)
        header_row.addWidget(self.run_inference_btn)

        self.open_run_tb_btn = QtWidgets.QPushButton(
            self.tr("View in TensorBoard"), details
        )
        self.open_run_tb_btn.clicked.connect(self._launch_tensorboard_for_selected_run)
        self.open_run_tb_btn.setEnabled(False)
        header_row.addWidget(self.open_run_tb_btn)

        self.open_log_btn = QtWidgets.QPushButton(self.tr("Open Log"), details)
        self.open_log_btn.clicked.connect(self._open_selected_log)
        self.open_log_btn.setEnabled(False)
        header_row.addWidget(self.open_log_btn)

        self.open_run_btn = QtWidgets.QPushButton(self.tr("Open Folder"), details)
        self.open_run_btn.clicked.connect(self._open_selected_run_dir)
        self.open_run_btn.setEnabled(False)
        header_row.addWidget(self.open_run_btn)
        details_layout.addLayout(header_row)

        # Progress + KPI strip
        kpi_row = QtWidgets.QHBoxLayout()
        kpi_row.setSpacing(6)
        self.progress_bar = QtWidgets.QProgressBar(details)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progress: —")
        self.progress_bar.setEnabled(False)
        self.progress_bar.setMinimumWidth(180)
        kpi_row.addWidget(self.progress_bar, 2)

        self._kpi_cards: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}
        for kpi_key, caption in (
            ("train_loss", "Train Loss"),
            ("val_loss", "Val Loss"),
            ("metric", "Primary Metric"),
            ("artifacts", "Artifacts"),
        ):
            card = QtWidgets.QFrame(details)
            card.setFrameShape(QtWidgets.QFrame.StyledPanel)
            cl = QtWidgets.QVBoxLayout(card)
            cl.setContentsMargins(8, 4, 8, 4)
            cl.setSpacing(1)
            cap = QtWidgets.QLabel(caption, card)
            cap.setProperty("muted", True)
            cap_f = cap.font()
            cap_f.setPointSize(max(8, cap_f.pointSize() - 1))
            cap.setFont(cap_f)
            val = QtWidgets.QLabel("—", card)
            val_f = val.font()
            val_f.setBold(True)
            val.setFont(val_f)
            cl.addWidget(cap)
            cl.addWidget(val)
            self._kpi_cards[kpi_key] = (cap, val)
            kpi_row.addWidget(card, 1)
        details_layout.addLayout(kpi_row)

        # Automated training health & anomaly diagnostics banner
        self.health_banner = QtWidgets.QLabel("", details)
        self.health_banner.setWordWrap(True)
        self.health_banner.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.health_banner.setContentsMargins(8, 5, 8, 5)
        self.health_banner.setVisible(False)
        details_layout.addWidget(self.health_banner)

        # Inspector sub-tabs (Curves first!)
        self.details_tabs = QtWidgets.QTabWidget(details)
        details_layout.addWidget(self.details_tabs, 1)

        self.curves_tab = QtWidgets.QWidget(self.details_tabs)
        curves_layout = QtWidgets.QVBoxLayout(self.curves_tab)
        curves_layout.setContentsMargins(2, 2, 2, 2)
        self.metrics_canvas = TrainingMetricsCanvas(self.curves_tab)
        curves_layout.addWidget(self.metrics_canvas, 1)
        self.details_tabs.addTab(self.curves_tab, self.tr("Curves"))

        self.plots_tab = QtWidgets.QWidget(self.details_tabs)
        plots_layout = QtWidgets.QVBoxLayout(self.plots_tab)
        plots_layout.setContentsMargins(2, 2, 2, 2)
        self.artifacts_gallery = TrainingArtifactsGalleryWidget(self.plots_tab)
        self.artifacts_gallery.open_tensorboard_images_requested.connect(
            self._launch_tensorboard_images_for_selected_run
        )
        plots_layout.addWidget(self.artifacts_gallery, 1)
        self.details_tabs.addTab(self.plots_tab, self.tr("Plots && Images"))

        self.compare_tab = QtWidgets.QWidget(self.details_tabs)
        compare_layout = QtWidgets.QVBoxLayout(self.compare_tab)
        compare_layout.setContentsMargins(2, 2, 2, 2)
        self.compare_matrix = RunComparisonMatrixWidget(self.compare_tab)
        self.compare_matrix.export_report_requested.connect(
            self._export_experiment_report
        )
        self.compare_matrix.export_csv_requested.connect(self._export_comparison_csv)
        compare_layout.addWidget(self.compare_matrix, 1)
        self.details_tabs.addTab(self.compare_tab, self.tr("Compare && Diff"))

        self.notes_tab = QtWidgets.QWidget(self.details_tabs)
        notes_layout = QtWidgets.QVBoxLayout(self.notes_tab)
        notes_layout.setContentsMargins(6, 6, 6, 6)
        notes_layout.setSpacing(6)
        meta_form = QtWidgets.QFormLayout()
        meta_form.setSpacing(6)
        self.meta_alias_edit = QtWidgets.QLineEdit(self.notes_tab)
        self.meta_alias_edit.setPlaceholderText(
            self.tr("Optional short experiment name (e.g. 'baseline-lr2e-3')")
        )
        self.meta_alias_edit.textChanged.connect(self._on_metadata_edited)
        self.meta_tags_edit = QtWidgets.QLineEdit(self.notes_tab)
        self.meta_tags_edit.setPlaceholderText(
            self.tr("Comma-separated tags (e.g. baseline, mouse-topview, aug-v2)")
        )
        self.meta_tags_edit.textChanged.connect(self._on_metadata_edited)
        self.meta_starred_check = QtWidgets.QCheckBox(
            self.tr("Star this run as a favorite / best candidate"), self.notes_tab
        )
        self.meta_starred_check.toggled.connect(self._on_metadata_edited)
        meta_form.addRow(self.tr("Alias:"), self.meta_alias_edit)
        meta_form.addRow(self.tr("Tags:"), self.meta_tags_edit)
        meta_form.addRow("", self.meta_starred_check)
        notes_layout.addLayout(meta_form)

        notes_layout.addWidget(QtWidgets.QLabel(self.tr("Lab Notes / Observations:")))
        self.meta_notes_edit = QtWidgets.QPlainTextEdit(self.notes_tab)
        self.meta_notes_edit.setPlaceholderText(
            self.tr(
                "Record dataset changes, augmentation hypotheses, failure cases, or next steps…\n"
                "Saved non-destructively to annolid_run_meta.json inside the run folder."
            )
        )
        self.meta_notes_edit.textChanged.connect(self._on_metadata_edited)
        notes_layout.addWidget(self.meta_notes_edit, 1)

        notes_actions = QtWidgets.QHBoxLayout()
        self.save_meta_btn = QtWidgets.QPushButton(
            self.tr("Save Notes && Tags"), self.notes_tab
        )
        self.save_meta_btn.setEnabled(False)
        self.save_meta_btn.clicked.connect(self._save_selected_run_metadata)
        self.meta_status_label = QtWidgets.QLabel("", self.notes_tab)
        self.meta_status_label.setProperty("muted", True)
        notes_actions.addWidget(self.save_meta_btn)
        notes_actions.addWidget(self.meta_status_label, 1)
        notes_layout.addLayout(notes_actions)
        self.details_tabs.addTab(self.notes_tab, self.tr("Notes && Tags"))

        self.epochs_tab = QtWidgets.QWidget(self.details_tabs)
        epochs_layout = QtWidgets.QVBoxLayout(self.epochs_tab)
        self.epoch_table = QtWidgets.QTableWidget(0, 0, self.epochs_tab)
        self.epoch_table.verticalHeader().setVisible(False)
        self.epoch_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.epoch_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.epoch_table.setAlternatingRowColors(True)
        self.epoch_table.horizontalHeader().setStretchLastSection(True)
        epochs_layout.addWidget(self.epoch_table, 1)
        self.details_tabs.addTab(self.epochs_tab, self.tr("Metrics Table"))

        self.timeline_tab = QtWidgets.QWidget(self.details_tabs)
        timeline_layout = QtWidgets.QVBoxLayout(self.timeline_tab)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.timeline_view = QtWidgets.QPlainTextEdit(self.timeline_tab)
        self.timeline_view.setReadOnly(True)
        self.timeline_view.setFont(mono)
        timeline_layout.addWidget(self.timeline_view, 1)
        self.details_tabs.addTab(self.timeline_tab, self.tr("Timeline"))

        self.log_tab = QtWidgets.QWidget(self.details_tabs)
        log_layout = QtWidgets.QVBoxLayout(self.log_tab)
        log_bar = QtWidgets.QHBoxLayout()
        self.log_filter_edit = QtWidgets.QLineEdit(self.log_tab)
        self.log_filter_edit.setPlaceholderText(
            self.tr("Filter log lines (e.g. epoch, loss, mAP, error)…")
        )
        self.log_filter_edit.textChanged.connect(self._apply_log_filter)
        self.follow_log_checkbox = QtWidgets.QCheckBox(
            self.tr("Follow log"), self.log_tab
        )
        self.follow_log_checkbox.setChecked(True)
        self.follow_log_checkbox.stateChanged.connect(
            lambda _=None: setattr(
                self, "_follow_log", bool(self.follow_log_checkbox.isChecked())
            )
        )
        copy_log_btn = QtWidgets.QPushButton(self.tr("Copy"), self.log_tab)
        copy_log_btn.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(
                self.log_view.toPlainText()
            )
        )
        log_bar.addWidget(self.log_filter_edit, 1)
        log_bar.addWidget(self.follow_log_checkbox)
        log_bar.addWidget(copy_log_btn)
        log_layout.addLayout(log_bar)

        self.log_view = QtWidgets.QPlainTextEdit(self.log_tab)
        self.log_view.setReadOnly(True)
        self.log_view.setFont(mono)
        log_layout.addWidget(self.log_view, 1)
        self.details_tabs.addTab(self.log_tab, self.tr("Log"))

        self.info_tab = QtWidgets.QWidget(self.details_tabs)
        info_layout = QtWidgets.QVBoxLayout(self.info_tab)
        self.info_view = QtWidgets.QPlainTextEdit(self.info_tab)
        self.info_view.setReadOnly(True)
        self.info_view.setFont(mono)
        info_layout.addWidget(self.info_view, 1)
        self.details_tabs.addTab(self.info_tab, self.tr("Config && Artifacts"))

        self.system_tab = QtWidgets.QWidget(self.details_tabs)
        system_layout = QtWidgets.QVBoxLayout(self.system_tab)
        system_layout.setContentsMargins(2, 2, 2, 2)
        self.telemetry_widget = SystemTelemetryWidget(self.system_tab)
        system_layout.addWidget(self.telemetry_widget, 1)
        self.details_tabs.addTab(self.system_tab, self.tr("System"))

        splitter.addWidget(details)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        with contextlib.suppress(Exception):
            self.active_table.selectionModel().selectionChanged.connect(
                self._on_active_selection_changed
            )

    def _build_tb_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tb_tab)
        layout.setContentsMargins(6, 6, 6, 6)
        self.tb_embed = TensorBoardEmbedWidget(self.tb_tab)
        self.tb_embed.start_requested.connect(self._start_tensorboard)
        self.tb_embed.status_changed.connect(self._on_tb_status_changed)
        self.tb_embed.session_started.connect(self._on_tb_session_started)
        self.tb_embed.session_stopped.connect(self._on_tb_session_stopped)
        layout.addWidget(self.tb_embed, 1)

    def _build_runs_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.runs_tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        hint = QtWidgets.QLabel(
            self.tr(
                "Discovered training runs under Runs Root. Double-click a run to "
                "inspect its curves and plots, or open its folder / TensorBoard."
            )
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        layout.addWidget(hint)

        self.runs_list = QtWidgets.QListWidget(self.runs_tab)
        self.runs_list.itemDoubleClicked.connect(self._inspect_run_list_item)
        layout.addWidget(self.runs_list, 1)

        row = QtWidgets.QHBoxLayout()
        self.inspect_list_run_btn = QtWidgets.QPushButton(self.tr("Inspect in Monitor"))
        self.inspect_list_run_btn.clicked.connect(self._inspect_selected_list_run)
        self.tb_list_run_btn = QtWidgets.QPushButton(self.tr("Open in TensorBoard"))
        self.tb_list_run_btn.clicked.connect(self._tb_selected_list_run)
        self.open_list_folder_btn = QtWidgets.QPushButton(self.tr("Open Run Folder"))
        self.open_list_folder_btn.clicked.connect(self._open_selected_list_run_folder)
        self.open_runs_root_btn = QtWidgets.QPushButton(self.tr("Open Runs Root"))
        self.open_runs_root_btn.clicked.connect(self._open_runs_root)

        row.addWidget(self.inspect_list_run_btn)
        row.addWidget(self.tb_list_run_btn)
        row.addWidget(self.open_list_folder_btn)
        row.addWidget(self.open_runs_root_btn)
        row.addStretch(1)
        layout.addLayout(row)

    # ------------------------------------------------------------------ #
    # Settings & Runs Root
    # ------------------------------------------------------------------ #

    def _load_settings(self) -> None:
        root = None
        if self._settings is not None:
            root = self._settings.value("training/runs_root", "", type=str)
        if root:
            self.runs_root_edit.setText(root)
            return
        self.runs_root_edit.setText(str(shared_runs_root()))

    def _save_settings(self) -> None:
        if self._settings is None:
            return
        self._settings.setValue(
            "training/runs_root", self.runs_root_edit.text().strip()
        )

    def _sync_env_runs_root(self) -> None:
        root = self.runs_root_edit.text().strip()
        if root:
            os.environ["ANNOLID_RUNS_ROOT"] = root

    def _on_runs_root_changed(self) -> None:
        self._save_settings()
        self._sync_env_runs_root()
        self._telemetry_sampler.set_disk_path(self.runs_root())
        self._sample_telemetry()
        self._refresh_runs_list()

    def _browse_runs_root(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select runs root"), str(self.runs_root())
        )
        if not folder:
            return
        self.runs_root_edit.setText(folder)
        self._on_runs_root_changed()

    def _open_runs_root(self) -> None:
        root = self.runs_root()
        if not root.exists():
            QtWidgets.QMessageBox.information(
                self, self.tr("Not found"), f"Runs root does not exist:\n{root}"
            )
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(root)))

    # ------------------------------------------------------------------ #
    # TensorBoard Controls (Non-Blocking)
    # ------------------------------------------------------------------ #

    def _resolve_tb_scope(self) -> Tuple[Optional[Path], Optional[str]]:
        scope_mode = str(self.tb_scope_combo.currentData() or "selected")
        if scope_mode == "compared":
            entries: List[Tuple[str, Path]] = []
            for key in sorted(self._compared_keys):
                st = self._runs.get(key)
                if st is not None and st.run_dir.exists():
                    alias = (
                        st.discovered.metadata.alias
                        if st.discovered is not None
                        else ""
                    )
                    label = (
                        alias
                        if alias
                        else (
                            f"{st.model}_{st.run_dir.name}"
                            if st.model and st.model != st.run_dir.name
                            else st.run_dir.name
                        )
                    )
                    entries.append((label, st.run_dir))
            spec = build_tensorboard_logdir_spec(entries)
            if spec:
                return None, spec
        if scope_mode == "selected":
            key = self._selected_run_key or self._current_selected_run_key()
            if key and key in self._runs:
                run_dir = self._runs[key].run_dir
                if run_dir.exists():
                    return run_dir, None
        root = self.runs_root()
        root.mkdir(parents=True, exist_ok=True)
        return root, None

    def start_tensorboard_for_current_scope(
        self,
        *,
        open_projector: bool = False,
        initial_plugin: Optional[str] = None,
        switch_to_tab: bool = True,
        log_dir_override: Optional[Path] = None,
        logdir_spec_override: Optional[str] = None,
    ) -> bool:
        if log_dir_override is not None or logdir_spec_override is not None:
            log_dir, spec = log_dir_override, logdir_spec_override
        else:
            log_dir, spec = self._resolve_tb_scope()

        parsed = QtCore.QUrl(self._tb_url)
        preferred_port = parsed.port() if parsed.port() != -1 else 6006
        started = self.tb_embed.start_server(
            log_dir=log_dir,
            logdir_spec=spec,
            preferred_port=preferred_port,
            host="127.0.0.1",
            open_projector=open_projector,
            initial_plugin=initial_plugin,
            open_external_if_no_webengine=True,
            check_capability=True,
        )
        if started and switch_to_tab:
            self.tabs.setCurrentWidget(self.tb_tab)
        return started

    def _start_tensorboard(self) -> None:
        self.start_tensorboard_for_current_scope(
            open_projector=False, switch_to_tab=True
        )

    def _stop_tensorboard(self) -> None:
        self.tb_embed.stop_server()
        self._tb_process = None

    def _open_tensorboard_in_browser(self) -> None:
        self.tb_embed.open_in_browser()

    @QtCore.Slot(str, str)
    def _on_tb_status_changed(self, state: str, message: str) -> None:
        self.tb_status.setText(
            message
            if message.lower().startswith("tensorboard")
            else f"TensorBoard: {message}"
        )
        running = state == "running"
        starting = state == "starting"
        self.start_tb_btn.setEnabled(not starting)
        self.start_tb_btn.setText(
            self.tr("Restart TensorBoard") if running else self.tr("Start TensorBoard")
        )
        self.stop_tb_btn.setEnabled(running or starting)
        self.open_tb_btn.setEnabled(running)

    @QtCore.Slot(object)
    def _on_tb_session_started(self, session: object) -> None:
        if isinstance(session, TensorBoardSession):
            self._tb_process = session.process
            self._tb_url = session.url

    @QtCore.Slot()
    def _on_tb_session_stopped(self) -> None:
        self._tb_process = None

    def _launch_tensorboard_for_selected_run(self) -> None:
        key = self._selected_run_key or self._current_selected_run_key()
        if not key or key not in self._runs:
            return
        idx = self.tb_scope_combo.findData("selected")
        if idx >= 0:
            self.tb_scope_combo.setCurrentIndex(idx)
        self.start_tensorboard_for_current_scope(
            log_dir_override=self._runs[key].run_dir,
            switch_to_tab=True,
        )

    def _launch_tensorboard_images_for_selected_run(self) -> None:
        key = self._selected_run_key or self._current_selected_run_key()
        if not key or key not in self._runs:
            return
        run_dir = self._runs[key].run_dir
        sess = self.tb_embed.session
        if sess is not None and sess.log_dir == run_dir:
            self.tb_embed.navigate_plugin("images")
            self.tabs.setCurrentWidget(self.tb_tab)
            return
        self.start_tensorboard_for_current_scope(
            log_dir_override=run_dir,
            initial_plugin="images",
            switch_to_tab=True,
        )

    def _launch_tensorboard_for_compared(self) -> None:
        if not self._compared_keys:
            return
        idx = self.tb_scope_combo.findData("compared")
        if idx >= 0:
            self.tb_scope_combo.setCurrentIndex(idx)
        self.start_tensorboard_for_current_scope(switch_to_tab=True)

    # ------------------------------------------------------------------ #
    # Run Discovery, Live Sync & Table Rendering
    # ------------------------------------------------------------------ #

    def _state_from_discovered(self, disc: DiscoveredRun) -> TrainingRunState:
        timeline = format_timeline_lines(
            disc.csv_rows, step_unit=disc.step_unit, max_lines=800
        )
        return TrainingRunState(
            run_dir=disc.run_dir,
            task=disc.task,
            model=disc.model,
            status=disc.status,
            started_at=disc.started_at,
            last_epoch=disc.last_step,
            total_epochs=disc.total_steps,
            train_loss=disc.train_loss,
            val_loss=disc.val_loss,
            updated_at=disc.updated_at,
            results_csv=disc.results_csv,
            csv_columns=disc.csv_columns,
            csv_rows=list(disc.csv_rows),
            log_path=disc.log_path,
            timeline_last_epoch=disc.last_step,
            epoch_timeline=timeline,
            step_unit=disc.step_unit,
            primary_metric_name=disc.primary_metric_name,
            primary_metric_value=disc.primary_metric_value,
            tensorboard_dir=disc.tensorboard_dir,
            has_projector=disc.has_projector,
            checkpoints=disc.checkpoints,
            plot_images=disc.plot_images,
            discovered=disc,
            is_live=False,
        )

    def _sync_state_from_disk(
        self, state: TrainingRunState, *, include_tb_scalars: bool = False
    ) -> None:
        if state.process is not None and state.status == "running":
            rc = state.process.poll()
            if rc is not None:
                state.status = "finished" if rc == 0 else "failed"
                state.updated_at = time.time()

        disc = inspect_run_dir(
            state.run_dir,
            runs_root=self.runs_root(),
            include_tb_scalars=include_tb_scalars,
            status=state.status,
            task_override=state.task if state.is_live else None,
            model_override=state.model if (state.is_live and state.model) else None,
        )
        state.discovered = disc
        state.step_unit = disc.step_unit or state.step_unit
        if disc.last_step is not None:
            state.last_epoch = disc.last_step
        if disc.total_steps is not None and state.total_epochs is None:
            state.total_epochs = disc.total_steps
        if disc.train_loss is not None:
            state.train_loss = disc.train_loss
        if disc.val_loss is not None:
            state.val_loss = disc.val_loss
        if disc.primary_metric_name is not None:
            state.primary_metric_name = disc.primary_metric_name
            state.primary_metric_value = disc.primary_metric_value
        if disc.updated_at is not None:
            state.updated_at = disc.updated_at
        state.results_csv = disc.results_csv
        state.csv_columns = disc.csv_columns
        state.csv_rows = list(disc.csv_rows)
        state.log_path = disc.log_path
        state.tensorboard_dir = disc.tensorboard_dir
        state.has_projector = disc.has_projector
        state.checkpoints = disc.checkpoints
        state.plot_images = disc.plot_images
        if disc.csv_rows:
            state.epoch_timeline = format_timeline_lines(
                disc.csv_rows, step_unit=state.step_unit, max_lines=800
            )
            state.timeline_last_epoch = disc.last_step

    def _refresh_runs_list(self) -> None:
        root = self.runs_root()
        self.runs_list.clear()

        discovered_runs = discover_training_runs(
            root, max_runs=200, include_tb_scalars=False
        )
        live_keys = {k for k, st in self._runs.items() if st.is_live}
        # Remove stale non-live runs that no longer exist on disk
        for k in list(self._runs.keys()):
            if k not in live_keys and not self._runs[k].run_dir.exists():
                self._runs.pop(k, None)
                self._compared_keys.discard(k)

        for disc in discovered_runs:
            key = str(disc.run_dir)
            if key in self._runs and self._runs[key].is_live:
                self._sync_state_from_disk(self._runs[key], include_tb_scalars=False)
            else:
                self._runs[key] = self._state_from_discovered(disc)

            item = QtWidgets.QListWidgetItem(str(disc.run_dir))
            item.setData(QtCore.Qt.UserRole, key)
            self.runs_list.addItem(item)

        self._refresh_active_table()
        if self._selected_run_key is None and self.active_table.rowCount() > 0:
            first_key = self._run_key_at_row(0)
            if first_key:
                self._select_run(first_key)

    def _filtered_ordered_runs(self) -> List[TrainingRunState]:
        query = self.run_filter_edit.text().strip().lower()
        active_only = bool(self.active_only_check.isChecked())
        starred_only = bool(self.starred_only_check.isChecked())

        items: List[TrainingRunState] = []
        for state in self._runs.values():
            if active_only and not state.is_live:
                continue
            meta = (
                state.discovered.metadata
                if state.discovered is not None
                else RunMetadata()
            )
            if starred_only and not meta.starred:
                continue
            if query:
                tags_blob = " ".join(meta.tags)
                haystack = (
                    f"{state.task} {state.model} {state.status} {state.run_dir} "
                    f"{meta.alias} {tags_blob} {meta.notes}"
                ).lower()
                if query not in haystack:
                    continue
            items.append(state)

        def _sort_key(st: TrainingRunState) -> Tuple[int, float]:
            tier = 0 if st.status == "running" else (1 if st.is_live else 2)
            ts = st.updated_at or st.started_at or 0.0
            return (tier, -ts)

        items.sort(key=_sort_key)
        return items

    def _run_key_at_row(self, row: int) -> Optional[str]:
        if row < 0 or row >= self.active_table.rowCount():
            return None
        item = self.active_table.item(row, self._COL_COMPARE)
        if item is not None:
            raw = item.data(QtCore.Qt.UserRole)
            if raw:
                return str(raw)
        dir_item = self.active_table.item(row, self._COL_RUN_DIR)
        if dir_item is not None and dir_item.text().strip():
            return str(canonicalize_run_dir(Path(dir_item.text().strip())))
        return None

    def _current_selected_run_key(self) -> Optional[str]:
        return self._run_key_at_row(self.active_table.currentRow())

    def _refresh_active_table(self, *, sync_selected: bool = True) -> None:
        selected_key = self._selected_run_key or self._current_selected_run_key()
        ordered = self._filtered_ordered_runs()
        visible_keys = {str(st.run_dir) for st in ordered}
        running_count = sum(1 for st in self._runs.values() if st.status == "running")
        self.live_summary_badge.setText(
            self.tr("%d active · %d total runs") % (running_count, len(self._runs))
        )

        self._rebuilding_table = True
        try:
            self.active_table.setRowCount(0)
            for idx, state in enumerate(ordered):
                key = str(state.run_dir)
                self.active_table.insertRow(idx)

                cmp_item = QtWidgets.QTableWidgetItem("")
                cmp_item.setFlags(
                    QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsSelectable
                    | QtCore.Qt.ItemIsUserCheckable
                )
                cmp_item.setCheckState(
                    QtCore.Qt.Checked
                    if key in self._compared_keys
                    else QtCore.Qt.Unchecked
                )
                cmp_item.setData(QtCore.Qt.UserRole, key)
                self.active_table.setItem(idx, self._COL_COMPARE, cmp_item)

                meta = (
                    state.discovered.metadata
                    if state.discovered is not None
                    else RunMetadata()
                )
                findings = (
                    state.discovered.health_findings
                    if state.discovered is not None
                    else ()
                )

                self.active_table.setItem(
                    idx, self._COL_TASK, QtWidgets.QTableWidgetItem(state.task)
                )
                model_display = state.model
                if meta.alias:
                    model_display = (
                        f"{meta.alias} ({state.model})"
                        if state.model and meta.alias != state.model
                        else meta.alias
                    )
                if meta.starred:
                    model_display = f"★ {model_display}"
                model_item = QtWidgets.QTableWidgetItem(model_display)
                if meta.notes:
                    model_item.setToolTip(meta.notes)
                self.active_table.setItem(idx, self._COL_MODEL, model_item)

                has_crit = any(f.severity == "critical" for f in findings)
                has_warn = any(f.severity == "warning" for f in findings)
                status_text = state.status
                if has_crit or has_warn:
                    status_text = f"{state.status} ⚠"
                status_item = QtWidgets.QTableWidgetItem(status_text)
                if findings:
                    status_item.setToolTip(
                        "\n".join(
                            f"[{f.severity.upper()}] {f.message}" for f in findings
                        )
                    )
                if state.status == "failed" or has_crit:
                    status_item.setForeground(QtGui.QBrush(QtGui.QColor(220, 38, 38)))
                elif has_warn:
                    status_item.setForeground(QtGui.QBrush(QtGui.QColor(234, 88, 12)))
                elif state.status == "running":
                    status_item.setForeground(QtGui.QBrush(QtGui.QColor(37, 99, 235)))
                elif state.status == "finished":
                    status_item.setForeground(QtGui.QBrush(QtGui.QColor(22, 163, 74)))
                self.active_table.setItem(idx, self._COL_STATUS, status_item)

                epoch_txt = ""
                if state.last_epoch is not None:
                    if state.total_epochs is not None and state.total_epochs > 0:
                        epoch_txt = f"{state.last_epoch}/{state.total_epochs}"
                    else:
                        epoch_txt = str(state.last_epoch)
                self.active_table.setItem(
                    idx, self._COL_PROGRESS, QtWidgets.QTableWidgetItem(epoch_txt)
                )
                self.active_table.setItem(
                    idx,
                    self._COL_TRAIN_LOSS,
                    QtWidgets.QTableWidgetItem(_format_metric(state.train_loss, 4)),
                )
                self.active_table.setItem(
                    idx,
                    self._COL_VAL_LOSS,
                    QtWidgets.QTableWidgetItem(_format_metric(state.val_loss, 4)),
                )

                metric_txt = "—"
                if state.primary_metric_value is not None:
                    short_name = (state.primary_metric_name or "").split("/")[-1]
                    metric_txt = (
                        f"{short_name}={_format_metric(state.primary_metric_value, 4)}"
                        if short_name
                        else _format_metric(state.primary_metric_value, 4)
                    )
                self.active_table.setItem(
                    idx, self._COL_METRIC, QtWidgets.QTableWidgetItem(metric_txt)
                )

                signals: List[str] = []
                if state.results_csv is not None:
                    signals.append("CSV")
                if state.tensorboard_dir is not None:
                    signals.append("TB")
                if state.has_projector:
                    signals.append("Proj")
                if state.plot_images:
                    signals.append(f"{len(state.plot_images)} img")
                if meta.tags:
                    signals.append(" ".join(f"#{t}" for t in meta.tags[:2]))
                sig_item = QtWidgets.QTableWidgetItem(" · ".join(signals))
                if meta.tags:
                    sig_item.setToolTip("Tags: " + ", ".join(meta.tags))
                self.active_table.setItem(idx, self._COL_ARTIFACTS, sig_item)
                self.active_table.setItem(
                    idx,
                    self._COL_UPDATED,
                    QtWidgets.QTableWidgetItem(_format_timestamp(state.updated_at)),
                )
                self.active_table.setItem(
                    idx,
                    self._COL_RUN_DIR,
                    QtWidgets.QTableWidgetItem(str(state.run_dir)),
                )

            self.active_table.resizeColumnsToContents()
            self.active_table.setColumnWidth(self._COL_COMPARE, 44)
        finally:
            self._rebuilding_table = False

        self._update_compare_buttons()
        if selected_key and selected_key in visible_keys:
            self._select_run(str(selected_key), sync_from_disk=sync_selected)
        elif ordered:
            self._select_run(str(ordered[0].run_dir), sync_from_disk=True)
        else:
            self._selected_run_key = None
            self._refresh_selected_details()

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._rebuilding_table or item.column() != self._COL_COMPARE:
            return
        key = str(item.data(QtCore.Qt.UserRole) or "")
        if not key:
            return
        if item.checkState() == QtCore.Qt.Checked:
            self._compared_keys.add(key)
            st = self._runs.get(key)
            if st is not None:
                self._sync_state_from_disk(st, include_tb_scalars=True)
        else:
            self._compared_keys.discard(key)
        self._update_compare_buttons()
        self._refresh_curves_canvas()
        self._refresh_comparison_matrix()

    def _clear_compared_runs(self) -> None:
        self._compared_keys.clear()
        self._refresh_active_table(sync_selected=False)

    def _update_compare_buttons(self) -> None:
        count = len(self._compared_keys)
        self.clear_compare_btn.setEnabled(count > 0)
        self.compare_in_tb_btn.setEnabled(count > 0)
        if count > 0:
            self.compare_in_tb_btn.setText(self.tr("Compare %d in TensorBoard") % count)
        else:
            self.compare_in_tb_btn.setText(self.tr("Compare Checked in TensorBoard"))

    def _select_run(self, key: str, *, sync_from_disk: bool = True) -> None:
        self._selected_run_key = key
        current = self._current_selected_run_key()
        if current != key:
            for row in range(self.active_table.rowCount()):
                if self._run_key_at_row(row) == key:
                    self.active_table.blockSignals(True)
                    self.active_table.setCurrentCell(row, self._COL_TASK)
                    self.active_table.blockSignals(False)
                    break
        state = self._runs.get(key)
        if state is not None and sync_from_disk:
            self._sync_state_from_disk(state, include_tb_scalars=True)
        self._refresh_selected_details()

    def _on_active_selection_changed(self, *_args) -> None:
        if self._rebuilding_table:
            return
        key = self._current_selected_run_key()
        if key == self._selected_run_key:
            return
        self._selected_run_key = key
        if key and key in self._runs:
            self._sync_state_from_disk(self._runs[key], include_tb_scalars=True)
        self._refresh_selected_details()

    # ------------------------------------------------------------------ #
    # Training Manager Signal Handlers
    # ------------------------------------------------------------------ #

    @QtCore.Slot(object)
    def _on_training_started(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        run_dir = payload.get("run_dir")
        if not run_dir:
            return
        resolved = canonicalize_run_dir(Path(str(run_dir)))
        key = str(resolved)
        state = TrainingRunState(
            run_dir=resolved,
            task=str(payload.get("task") or "train"),
            model=str(payload.get("model") or resolved.name),
            status="running",
            started_at=time.time(),
            total_epochs=(
                int(payload["total_steps"]) if payload.get("total_steps") else None
            ),
            step_unit=str(payload.get("step_unit") or "epoch"),
            is_live=True,
        )
        self._runs[key] = state
        self._refresh_active_table()
        self._select_run(key)
        if not self._timer.isActive():
            self._timer.start()

    @QtCore.Slot(object)
    def _on_training_finished(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        run_dir = payload.get("run_dir")
        if not run_dir:
            return
        resolved = canonicalize_run_dir(Path(str(run_dir)))
        key = str(resolved)
        state = self._runs.get(key)
        if state is None:
            return
        state.status = "finished" if payload.get("ok", True) else "failed"
        state.updated_at = time.time()
        self._refresh_active_metrics()

    def _refresh_active_metrics(self) -> None:
        any_running = False
        for key, state in list(self._runs.items()):
            if (
                state.status == "running"
                or key == self._selected_run_key
                or key in self._compared_keys
            ):
                include_tb = key == self._selected_run_key or key in self._compared_keys
                self._sync_state_from_disk(state, include_tb_scalars=include_tb)
            if state.status == "running":
                any_running = True

        self._refresh_active_table(sync_selected=False)
        if not any_running:
            self._timer.stop()

    # ------------------------------------------------------------------ #
    # Selected Run Inspector Rendering
    # ------------------------------------------------------------------ #

    def _find_run_checkpoint(self, state: TrainingRunState) -> Optional[Path]:
        if state.checkpoints:
            for ckpt in state.checkpoints:
                if ckpt.is_file():
                    return ckpt
        for filename in ("best.pt", "last.pt"):
            candidate = state.run_dir / "weights" / filename
            if candidate.is_file():
                return candidate
        for filename in ("model_final.pth", "best_model.pth"):
            candidate = state.run_dir / filename
            if candidate.is_file():
                return candidate
        return None

    def _refresh_curves_canvas(self) -> None:
        primary_state = (
            self._runs.get(self._selected_run_key) if self._selected_run_key else None
        )
        primary_disc = primary_state.discovered if primary_state is not None else None
        compared_discs: List[DiscoveredRun] = []
        for k in sorted(self._compared_keys):
            st = self._runs.get(k)
            if st is not None and st.discovered is not None:
                compared_discs.append(st.discovered)
        self.metrics_canvas.set_runs(primary_disc, compared_discs)

    def _runs_for_comparison(self) -> Tuple[List[DiscoveredRun], Optional[str]]:
        checked: List[DiscoveredRun] = []
        for k in sorted(self._compared_keys):
            st = self._runs.get(k)
            if st is not None and st.discovered is not None:
                checked.append(st.discovered)
        if len(checked) >= 2:
            return (
                checked,
                self.tr("Comparing %d checked runs · highlighted rows differ")
                % len(checked),
            )
        if len(checked) == 1:
            sel = (
                self._runs.get(self._selected_run_key)
                if self._selected_run_key
                else None
            )
            if (
                sel is not None
                and sel.discovered is not None
                and sel.discovered.run_dir != checked[0].run_dir
            ):
                return (
                    [checked[0], sel.discovered],
                    self.tr(
                        "Comparing checked + selected run · highlighted rows differ"
                    ),
                )
            return checked, None

        # Default: show up to 5 most recent visible runs (selected run first).
        visible = self._filtered_ordered_runs()
        ordered_discs: List[DiscoveredRun] = []
        seen_dirs: set[Path] = set()
        if self._selected_run_key and self._selected_run_key in self._runs:
            sel_disc = self._runs[self._selected_run_key].discovered
            if sel_disc is not None:
                ordered_discs.append(sel_disc)
                seen_dirs.add(sel_disc.run_dir)
        for st in visible:
            if st.discovered is not None and st.discovered.run_dir not in seen_dirs:
                ordered_discs.append(st.discovered)
                seen_dirs.add(st.discovered.run_dir)
            if len(ordered_discs) >= 5:
                break
        if len(ordered_discs) >= 2:
            return (
                ordered_discs,
                self.tr(
                    "Showing %d recent runs (check Cmp boxes to pick specific runs)"
                )
                % len(ordered_discs),
            )
        return ordered_discs, None

    def _refresh_comparison_matrix(self) -> None:
        runs, subtitle = self._runs_for_comparison()
        self.compare_matrix.set_runs(runs, subtitle=subtitle)

    def _render_health_banner(self, state: Optional[TrainingRunState]) -> None:
        if state is None or state.discovered is None:
            self.health_banner.setVisible(False)
            self.health_banner.setText("")
            return
        latest_sample = self._telemetry_sampler.latest()
        disk_free = latest_sample.disk_free_gb if latest_sample is not None else None
        findings: List[RunHealthFinding] = analyze_run_health(
            state.discovered, disk_free_gb=disk_free
        )
        if not findings:
            self.health_banner.setVisible(False)
            self.health_banner.setText("")
            return

        has_crit = any(f.severity == "critical" for f in findings)
        has_warn = any(f.severity == "warning" for f in findings)
        if has_crit:
            border, bg = "#DC2626", "rgba(220, 38, 38, 0.16)"
            prefix = "Health Alert"
        elif has_warn:
            border, bg = "#EA580C", "rgba(234, 88, 12, 0.16)"
            prefix = "Health Warning"
        else:
            border, bg = "#2563EB", "rgba(37, 99, 235, 0.14)"
            prefix = "Training Insight"

        summary = " · ".join(f.message for f in findings[:3])
        if len(findings) > 3:
            summary += f" (+{len(findings) - 3} more)"
        self.health_banner.setStyleSheet(
            f"QLabel {{ border: 1px solid {border}; border-radius: 6px; "
            f"background-color: {bg}; padding: 5px 8px; }}"
        )
        self.health_banner.setText(f"{prefix}: {summary}")
        self.health_banner.setVisible(True)

    def _on_metadata_edited(self, *_args) -> None:
        if self._loading_metadata or not self._selected_run_key:
            return
        self._meta_dirty = True
        self.save_meta_btn.setText(self.tr("Save Notes && Tags *"))

    def _load_metadata_editor(
        self, state: Optional[TrainingRunState], *, force: bool = False
    ) -> None:
        if state is None:
            self._metadata_loaded_key = None
            self._meta_dirty = False
            self._loading_metadata = True
            try:
                self.meta_alias_edit.setText("")
                self.meta_tags_edit.setText("")
                self.meta_starred_check.setChecked(False)
                self.meta_notes_edit.setPlainText("")
            finally:
                self._loading_metadata = False
            self.meta_alias_edit.setEnabled(False)
            self.meta_tags_edit.setEnabled(False)
            self.meta_starred_check.setEnabled(False)
            self.meta_notes_edit.setEnabled(False)
            self.save_meta_btn.setEnabled(False)
            self.save_meta_btn.setText(self.tr("Save Notes && Tags"))
            self.meta_status_label.setText("")
            return

        key = str(state.run_dir)
        self.meta_alias_edit.setEnabled(True)
        self.meta_tags_edit.setEnabled(True)
        self.meta_starred_check.setEnabled(True)
        self.meta_notes_edit.setEnabled(True)
        self.save_meta_btn.setEnabled(True)
        if not force and self._metadata_loaded_key == key and self._meta_dirty:
            return
        self._metadata_loaded_key = key
        meta = (
            state.discovered.metadata if state.discovered is not None else RunMetadata()
        )
        self._loading_metadata = True
        try:
            self.meta_alias_edit.setText(meta.alias)
            self.meta_tags_edit.setText(", ".join(meta.tags))
            self.meta_starred_check.setChecked(bool(meta.starred))
            self.meta_notes_edit.setPlainText(meta.notes)
        finally:
            self._loading_metadata = False
        self._meta_dirty = False
        self.save_meta_btn.setText(self.tr("Save Notes && Tags"))
        if meta.updated_at:
            self.meta_status_label.setText(
                self.tr("Last saved: %s") % _format_timestamp(meta.updated_at)
            )
        else:
            self.meta_status_label.setText(
                self.tr("Stored in %s") % (state.run_dir / "annolid_run_meta.json")
            )

    def _refresh_selected_details(self) -> None:
        key = self._selected_run_key
        state = self._runs.get(key) if key else None
        if state is None:
            self.selected_run_label.setText(
                self.tr("Select a run to inspect metrics, curves, plots, and logs.")
            )
            self.star_run_btn.setEnabled(False)
            self.star_run_btn.setChecked(False)
            self.star_run_btn.setText(self.tr("☆ Star"))
            self.open_log_btn.setEnabled(False)
            self.open_run_btn.setEnabled(False)
            self.open_run_tb_btn.setEnabled(False)
            self.stop_training_btn.setEnabled(False)
            self.run_inference_btn.setEnabled(False)
            self.progress_bar.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Progress: —")
            for _cap, val in self._kpi_cards.values():
                val.setText("—")
            self._render_health_banner(None)
            self._load_metadata_editor(None)
            self.epoch_table.setRowCount(0)
            self.epoch_table.setColumnCount(0)
            self._raw_log_text = ""
            self.log_view.setPlainText("")
            self.timeline_view.setPlainText("")
            self.info_view.setPlainText("")
            self.artifacts_gallery.set_run(None)
            self._refresh_curves_canvas()
            self._refresh_comparison_matrix()
            return

        meta = (
            state.discovered.metadata if state.discovered is not None else RunMetadata()
        )
        title_prefix = f"{meta.alias} · " if meta.alias else ""
        self.selected_run_label.setText(
            f"{title_prefix}{state.task} · {state.model} · {state.status} · {state.run_dir}"
        )
        self.star_run_btn.setEnabled(True)
        self.star_run_btn.setChecked(bool(meta.starred))
        self.star_run_btn.setText(
            self.tr("★ Starred") if meta.starred else self.tr("☆ Star")
        )
        self.open_log_btn.setEnabled(
            state.log_path is not None and state.log_path.exists()
        )
        self.open_run_btn.setEnabled(state.run_dir.exists())
        self.open_run_tb_btn.setEnabled(state.run_dir.exists())
        self.open_projector_btn.setToolTip(
            self.tr(
                "Open TensorBoard Embedding Projector for this run (projector_config.pbtxt detected)"
            )
            if state.has_projector
            else self.tr("Launch TensorBoard directly on the Embedding Projector tab")
        )

        has_running_proc = state.process is not None and state.process.poll() is None
        has_running_mgr = any(
            callable(getattr(m, "stop_training", None))
            and (not callable(getattr(m, "is_running", None)) or m.is_running())
            for m in self._training_managers
        )
        self.stop_training_btn.setEnabled(
            state.status == "running" and (has_running_proc or has_running_mgr)
        )
        self.run_inference_btn.setEnabled(self._find_run_checkpoint(state) is not None)

        # KPI cards
        self._kpi_cards["train_loss"][1].setText(_format_metric(state.train_loss, 4))
        self._kpi_cards["val_loss"][1].setText(_format_metric(state.val_loss, 4))
        metric_cap, metric_val = self._kpi_cards["metric"]
        if state.primary_metric_name:
            metric_cap.setText(state.primary_metric_name.split("/")[-1])
            metric_val.setText(_format_metric(state.primary_metric_value, 4))
        else:
            metric_cap.setText("Primary Metric")
            metric_val.setText("—")

        art_parts: List[str] = []
        if state.tensorboard_dir is not None:
            art_parts.append("TB")
        if state.has_projector:
            art_parts.append("Projector")
        if state.plot_images:
            art_parts.append(f"{len(state.plot_images)} plots")
        if state.checkpoints:
            art_parts.append(f"{len(state.checkpoints)} ckpt")
        self._kpi_cards["artifacts"][1].setText(
            " · ".join(art_parts) if art_parts else "—"
        )

        self._render_progress(state)
        self._render_health_banner(state)
        self._load_metadata_editor(state, force=False)
        self._refresh_curves_canvas()
        self._refresh_comparison_matrix()
        self.artifacts_gallery.set_run(state.discovered)
        self._render_epoch_table(state)
        self._render_timeline(state)
        self._render_log_tail(state)
        self._render_run_info(state)

    def _render_progress(self, state: TrainingRunState) -> None:
        unit_label = (
            "Iter"
            if state.step_unit == "iteration"
            else ("Step" if state.step_unit == "step" else "Epoch")
        )
        if state.last_epoch is None:
            self.progress_bar.setEnabled(state.status == "running")
            if state.status == "running":
                self.progress_bar.setRange(0, 0)
                self.progress_bar.setFormat(self.tr("Running…"))
            else:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("Progress: —")
            return
        if state.total_epochs is None or state.total_epochs <= 0:
            self.progress_bar.setEnabled(True)
            if state.status == "running":
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setRange(0, max(1, int(state.last_epoch)))
                self.progress_bar.setValue(int(state.last_epoch))
            self.progress_bar.setFormat(f"{unit_label} {state.last_epoch}")
            return

        self.progress_bar.setEnabled(True)
        self.progress_bar.setRange(0, int(state.total_epochs))
        self.progress_bar.setValue(
            int(max(0, min(state.last_epoch, state.total_epochs)))
        )
        self.progress_bar.setFormat(
            f"{unit_label} {state.last_epoch}/{state.total_epochs}"
        )

    def _render_epoch_table(self, state: TrainingRunState) -> None:
        if not state.csv_rows or not state.csv_columns:
            self.epoch_table.setRowCount(0)
            self.epoch_table.setColumnCount(0)
            return

        columns = list(state.csv_columns)
        for step_key in ("epoch", "Epoch", "iteration", "iter", "step"):
            if step_key in columns:
                columns.remove(step_key)
                columns.insert(0, step_key)
                break

        current_headers = [
            ""
            if self.epoch_table.horizontalHeaderItem(i) is None
            else self.epoch_table.horizontalHeaderItem(i).text()
            for i in range(self.epoch_table.columnCount())
        ]
        if self.epoch_table.columnCount() != len(columns) or current_headers != columns:
            self.epoch_table.clear()
            self.epoch_table.setColumnCount(len(columns))
            self.epoch_table.setHorizontalHeaderLabels(columns)

        self.epoch_table.setRowCount(len(state.csv_rows))
        for r, row in enumerate(state.csv_rows):
            for c, col in enumerate(columns):
                self.epoch_table.setItem(
                    r, c, QtWidgets.QTableWidgetItem(str(row.get(col, "")))
                )
        self.epoch_table.resizeColumnsToContents()
        with contextlib.suppress(Exception):
            self.epoch_table.scrollToBottom()

    def _render_timeline(self, state: TrainingRunState) -> None:
        text = "\n".join(state.epoch_timeline)
        if self.timeline_view.toPlainText() == text:
            return
        self.timeline_view.setPlainText(text)
        with contextlib.suppress(Exception):
            cursor = self.timeline_view.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            self.timeline_view.setTextCursor(cursor)
            self.timeline_view.centerCursor()

    def _render_log_tail(self, state: TrainingRunState) -> None:
        if state.log_path is None or not state.log_path.exists():
            self._raw_log_text = ""
            self.log_view.setPlainText("")
            return
        text = tail_log_text(state.log_path, max_lines=300)
        if text == self._raw_log_text:
            return
        self._raw_log_text = text
        self._apply_log_filter()

    def _apply_log_filter(self) -> None:
        needle = self.log_filter_edit.text().strip().lower()
        if not needle:
            shown = self._raw_log_text
        else:
            shown = "\n".join(
                line
                for line in self._raw_log_text.splitlines()
                if needle in line.lower()
            )
        if self.log_view.toPlainText() == shown:
            return
        self.log_view.setPlainText(shown)
        if self._follow_log:
            with contextlib.suppress(Exception):
                cursor = self.log_view.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                self.log_view.setTextCursor(cursor)

    def _render_run_info(self, state: TrainingRunState) -> None:
        disc = state.discovered
        lines = [
            f"Run directory : {state.run_dir}",
            f"Task / Model  : {state.task} / {state.model}",
            f"Status        : {state.status}",
            f"Metrics CSV   : {state.results_csv or '(none)'}",
            f"TensorBoard   : {state.tensorboard_dir or '(none)'}",
            f"Projector     : {'available (projector_config.pbtxt)' if state.has_projector else 'not generated'}",
            f"Log file      : {state.log_path or '(none)'}",
            "",
        ]
        if disc is not None and (
            disc.metadata.alias
            or disc.metadata.tags
            or disc.metadata.starred
            or disc.metadata.notes
        ):
            lines.append("Run Metadata (annolid_run_meta.json):")
            if disc.metadata.alias:
                lines.append(f"  Alias   : {disc.metadata.alias}")
            if disc.metadata.starred:
                lines.append("  Starred : yes")
            if disc.metadata.tags:
                lines.append(f"  Tags    : {', '.join(disc.metadata.tags)}")
            if disc.metadata.notes:
                lines.append(f"  Notes   : {disc.metadata.notes.strip()}")
            lines.append("")
        if disc is not None and disc.health_findings:
            lines.append("Health Diagnostics:")
            for finding in disc.health_findings:
                lines.append(
                    f"  - [{finding.severity.upper()}] ({finding.code}) {finding.message}"
                )
            lines.append("")
        if state.checkpoints:
            lines.append("Checkpoints:")
            for ckpt in state.checkpoints:
                lines.append(f"  - {ckpt}")
            lines.append("")
        if (
            disc is not None
            and disc.command_path is not None
            and disc.command_path.is_file()
        ):
            with contextlib.suppress(OSError):
                lines.append("Command:")
                lines.append(
                    disc.command_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                )
                lines.append("")
        if disc is not None and disc.hparams:
            lines.append("Configuration / Hyperparameters:")
            for k in sorted(disc.hparams.keys()):
                lines.append(f"  {k}: {disc.hparams[k]}")
            lines.append("")
        if disc is not None and disc.metrics:
            lines.append(f"Available Metric Tags ({len(disc.metrics)}):")
            for tag, metric_series in sorted(disc.metrics.items()):
                lines.append(
                    f"  - {tag} [{metric_series.source}] ({len(metric_series.steps)} pts, latest={_format_metric(metric_series.latest_value, 4)})"
                )
        text = "\n".join(lines)
        if self.info_view.toPlainText() != text:
            self.info_view.setPlainText(text)

    # ------------------------------------------------------------------ #
    # Actions: Stop / Inference / Open Folder / Runs Explorer
    # ------------------------------------------------------------------ #

    def _stop_selected_training(self) -> None:
        stopped_any = False
        key = self._selected_run_key
        state = self._runs.get(key) if key else None
        if (
            state is not None
            and state.process is not None
            and state.process.poll() is None
        ):
            _terminate_process(state.process, grace_s=1.5)
            state.status = "failed"
            state.updated_at = time.time()
            stopped_any = True

        for manager in list(self._training_managers):
            is_running = getattr(manager, "is_running", None)
            stop_fn = getattr(manager, "stop_training", None)
            if callable(stop_fn) and (not callable(is_running) or is_running()):
                with contextlib.suppress(Exception):
                    if stop_fn():
                        stopped_any = True

        if stopped_any:
            self.stop_training_btn.setEnabled(False)
            self._refresh_active_table()

    def _run_inference_for_selected_run(self) -> None:
        key = self._selected_run_key
        if not key or key not in self._runs:
            return
        ckpt = self._find_run_checkpoint(self._runs[key])
        if ckpt is None:
            return
        win = self.window()
        parent_win = win.parent() if win is not None else None
        for target in (parent_win, win):
            if target is not None and hasattr(target, "open_inference_wizard"):
                with contextlib.suppress(TypeError, Exception):
                    target.open_inference_wizard(model_path=str(ckpt))
                    return
        from annolid.gui.widgets.inference_wizard import InferenceWizard

        wizard = InferenceWizard(model_path=str(ckpt), parent=self)
        wizard.exec_()

    def _open_selected_run_dir(self, *_args) -> None:
        key = self._selected_run_key or self._current_selected_run_key()
        if not key or key not in self._runs:
            return
        path = self._runs[key].run_dir
        if path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _open_selected_log(self) -> None:
        key = self._selected_run_key
        if not key or key not in self._runs:
            return
        path = self._runs[key].log_path
        if path is not None and path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _selected_list_run_path(self) -> Optional[Path]:
        item = self.runs_list.currentItem()
        if item is None:
            return None
        raw = item.data(QtCore.Qt.UserRole) or item.text()
        return Path(str(raw)) if raw else None

    def _inspect_run_list_item(self, item: QtWidgets.QListWidgetItem) -> None:
        raw = item.data(QtCore.Qt.UserRole) or item.text()
        if not raw:
            return
        self.select_run_dir(Path(str(raw)))
        self.tabs.setCurrentWidget(self.active_tab)

    def _inspect_selected_list_run(self) -> None:
        path = self._selected_list_run_path()
        if path is not None:
            self.select_run_dir(path)
            self.tabs.setCurrentWidget(self.active_tab)

    def _tb_selected_list_run(self) -> None:
        path = self._selected_list_run_path()
        if path is not None:
            self.select_run_dir(path)
            self.start_tensorboard_for_current_scope(
                log_dir_override=path, switch_to_tab=True
            )

    def _open_selected_list_run_folder(self) -> None:
        path = self._selected_list_run_path()
        if path is not None and path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------ #
    # System Telemetry, Run Notes/Tags, Reports & External Tracker Interop
    # ------------------------------------------------------------------ #

    def _sample_telemetry(self) -> None:
        sample = self._telemetry_sampler.sample_now()
        self.telemetry_status_label.setText(format_telemetry_summary(sample))
        self.telemetry_widget.set_samples(self._telemetry_sampler.history())

    def _toggle_selected_run_star(self, checked: bool) -> None:
        key = self._selected_run_key
        state = self._runs.get(key) if key else None
        if state is None:
            return
        cur = (
            state.discovered.metadata if state.discovered is not None else RunMetadata()
        )
        updated = RunMetadata(
            alias=cur.alias,
            tags=cur.tags,
            notes=cur.notes,
            starred=bool(checked),
            updated_at=time.time(),
        )
        try:
            save_run_metadata(state.run_dir, updated)
        except Exception as exc:
            self.star_run_btn.setChecked(not bool(checked))
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Could Not Save Metadata"),
                self.tr("Failed to update starred status:\n%s") % exc,
            )
            return
        self._sync_state_from_disk(state, include_tb_scalars=True)
        self._loading_metadata = True
        try:
            self.meta_starred_check.setChecked(bool(checked))
        finally:
            self._loading_metadata = False
        self._refresh_active_table(sync_selected=False)

    def _save_selected_run_metadata(self) -> None:
        key = self._selected_run_key
        state = self._runs.get(key) if key else None
        if state is None:
            return
        meta = RunMetadata(
            alias=self.meta_alias_edit.text().strip(),
            tags=normalize_run_tags(self.meta_tags_edit.text()),
            notes=self.meta_notes_edit.toPlainText(),
            starred=bool(self.meta_starred_check.isChecked()),
            updated_at=time.time(),
        )
        try:
            saved_path = save_run_metadata(state.run_dir, meta)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Could Not Save Notes & Tags"),
                self.tr("Failed to write run metadata:\n%s") % exc,
            )
            return
        self._sync_state_from_disk(state, include_tb_scalars=True)
        self._metadata_loaded_key = key
        self._load_metadata_editor(state, force=True)
        self.meta_status_label.setText(self.tr("Saved to %s") % saved_path.name)
        self._refresh_active_table(sync_selected=False)

    def _runs_for_export(self) -> List[DiscoveredRun]:
        if self._compared_keys:
            out: List[DiscoveredRun] = []
            for k in sorted(self._compared_keys):
                st = self._runs.get(k)
                if st is not None:
                    self._sync_state_from_disk(st, include_tb_scalars=True)
                    if st.discovered is not None:
                        out.append(st.discovered)
            if out:
                return out
        if self._selected_run_key and self._selected_run_key in self._runs:
            st = self._runs[self._selected_run_key]
            self._sync_state_from_disk(st, include_tb_scalars=True)
            if st.discovered is not None:
                return [st.discovered]
        return [
            st.discovered
            for st in self._filtered_ordered_runs()[:50]
            if st.discovered is not None
        ]

    def _cancel_export_worker(self) -> None:
        self._export_generation += 1
        self._export_callbacks.clear()
        worker = self._export_worker
        thread = self._export_thread
        self._export_worker = None
        self._export_thread = None
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.succeeded.disconnect(self._on_export_succeeded)
            with contextlib.suppress(Exception):
                worker.failed.disconnect(self._on_export_failed)
        still_running = False
        if thread is not None:
            with contextlib.suppress(Exception):
                if thread.isRunning():
                    thread.quit()
                    thread.wait(1500)
                still_running = bool(thread.isRunning())
            if still_running:
                with contextlib.suppress(Exception):
                    thread.finished.connect(thread.deleteLater)
                if worker is not None:
                    with contextlib.suppress(Exception):
                        thread.finished.connect(worker.deleteLater)
            else:
                with contextlib.suppress(Exception):
                    thread.deleteLater()
        if worker is not None and not still_running:
            with contextlib.suppress(Exception):
                worker.deleteLater()

    def _start_background_export(
        self,
        *,
        label: str,
        error_title: str,
        fn: Callable[[], object],
        on_success: Callable[[object], None],
    ) -> bool:
        """Run report or tracker export asynchronously on a QThread."""
        if self._export_thread is not None and self._export_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Export In Progress"),
                self.tr(
                    "Another export is currently running. Please wait for it to finish."
                ),
            )
            return False

        self._export_generation += 1
        gen = self._export_generation
        self._export_callbacks[gen] = (on_success, error_title)
        self.export_menu_btn.setEnabled(False)
        self.export_menu_btn.setText(f"{label}…")

        worker = _DashboardExportWorker(gen, fn)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_export_succeeded)
        worker.failed.connect(self._on_export_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_export_thread_finished)

        self._export_worker = worker
        self._export_thread = thread
        thread.start()
        return True

    def _on_export_thread_finished(self) -> None:
        thread = self.sender()
        if thread is self._export_thread:
            worker = self._export_worker
            self._export_worker = None
            self._export_thread = None
            if worker is not None:
                with contextlib.suppress(Exception):
                    worker.deleteLater()
        if thread is not None:
            with contextlib.suppress(Exception):
                thread.deleteLater()

    @QtCore.Slot(int, object)
    def _on_export_succeeded(self, generation: int, result: object) -> None:
        self.export_menu_btn.setEnabled(True)
        self.export_menu_btn.setText(self.tr("Export / Trackers"))
        entry = self._export_callbacks.pop(int(generation), None)
        if (
            entry is None
            or self._cleaned_up
            or int(generation) != self._export_generation
        ):
            return
        on_success, _error_title = entry
        on_success(result)

    @QtCore.Slot(int, str)
    def _on_export_failed(self, generation: int, error_text: str) -> None:
        self.export_menu_btn.setEnabled(True)
        self.export_menu_btn.setText(self.tr("Export / Trackers"))
        entry = self._export_callbacks.pop(int(generation), None)
        if (
            entry is None
            or self._cleaned_up
            or int(generation) != self._export_generation
        ):
            return
        _on_success, error_title = entry
        QtWidgets.QMessageBox.warning(self, error_title, error_text)

    def _export_experiment_report(self) -> None:
        runs = self._runs_for_export()
        if not runs:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Runs Available"),
                self.tr("Select or check at least one training run to export."),
            )
            return
        default_dir = str(self.runs_root() / "experiment_report")
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Folder for Experiment Report"), default_dir
        )
        if not folder:
            return
        out_dir = Path(folder)
        run_count = len(runs)

        def _on_done(artifacts: object) -> None:
            msg = (
                f"Exported report for {run_count} run(s):\n"
                f"- {artifacts.html_path}\n"
                f"- {artifacts.markdown_path}\n"
                f"- {artifacts.csv_path}\n"
                f"- {artifacts.json_path}"
            )
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle(self.tr("Experiment Report Exported"))
            box.setText(msg)
            open_btn = box.addButton(
                self.tr("Open HTML Report"), QtWidgets.QMessageBox.AcceptRole
            )
            box.addButton(QtWidgets.QMessageBox.Ok)
            box.exec_()
            if box.clickedButton() is open_btn:
                QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl.fromLocalFile(str(artifacts.html_path))
                )

        self._start_background_export(
            label=self.tr("Exporting Report"),
            error_title=self.tr("Report Export Failed"),
            fn=lambda: export_training_experiment_report(
                runs, out_dir, title="Annolid Training Experiment Report"
            ),
            on_success=_on_done,
        )

    def _export_comparison_csv(self) -> None:
        runs = self._runs_for_export()
        if not runs:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Runs Available"),
                self.tr("Select or check at least one training run to export."),
            )
            return
        default_path = str(self.runs_root() / "runs_comparison.csv")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export Runs Comparison CSV"),
            default_path,
            "CSV Files (*.csv)",
        )
        if not out_path:
            return
        try:
            written = export_runs_comparison_csv(runs, Path(out_path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("CSV Export Failed"),
                self.tr("Could not export CSV:\n%s") % exc,
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            self.tr("CSV Exported"),
            self.tr("Exported %d run(s) to:\n%s") % (len(runs), written),
        )

    def _export_to_mlflow(self) -> None:
        runs = self._runs_for_export()
        if not runs:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Runs Available"),
                self.tr("Select or check at least one training run to export."),
            )
            return
        default_uri = (self.runs_root() / "mlruns").resolve().as_uri()
        uri, ok = QtWidgets.QInputDialog.getText(
            self,
            self.tr("Export Runs to MLflow"),
            self.tr("MLflow tracking URI (local directory URI or remote server):"),
            text=default_uri,
        )
        if not ok or not uri.strip():
            return
        tracking_uri = uri.strip()

        def _on_done(res: object) -> None:
            if not isinstance(res, dict):
                return
            QtWidgets.QMessageBox.information(
                self,
                self.tr("MLflow Export Complete"),
                self.tr("Exported %d run(s) to MLflow (%s).\nRun IDs: %s")
                % (
                    len(res["exported_run_ids"]),
                    res["tracking_uri"],
                    ", ".join(res["exported_run_ids"][:5]),
                ),
            )

        self._start_background_export(
            label=self.tr("Exporting MLflow"),
            error_title=self.tr("MLflow Export Unavailable"),
            fn=lambda: export_runs_to_mlflow(runs, tracking_uri=tracking_uri),
            on_success=_on_done,
        )

    def _export_to_wandb(self) -> None:
        runs = self._runs_for_export()
        if not runs:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Run Selected"),
                self.tr("Select a training run to export to Weights & Biases."),
            )
            return
        project, ok = QtWidgets.QInputDialog.getText(
            self,
            self.tr("Export Run to Weights & Biases (Offline)"),
            self.tr("W&B project name:"),
            text="annolid",
        )
        if not ok or not project.strip():
            return
        target_run = runs[0]
        project_name = project.strip()

        def _on_done(res: object) -> None:
            if not isinstance(res, dict):
                return
            QtWidgets.QMessageBox.information(
                self,
                self.tr("W&B Offline Export Complete"),
                self.tr(
                    "Exported run '%s' to Weights & Biases (offline).\n"
                    "Sync later with `wandb sync` from:\n%s"
                )
                % (
                    target_run.run_dir.name,
                    res.get("wandb_dir") or "(default wandb dir)",
                ),
            )

        self._start_background_export(
            label=self.tr("Exporting W&B"),
            error_title=self.tr("Weights & Biases Export Unavailable"),
            fn=lambda: export_run_to_wandb(
                target_run, project=project_name, mode="offline"
            ),
            on_success=_on_done,
        )

    def _export_to_rerun(self) -> None:
        runs = self._runs_for_export()
        if not runs:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Run Selected"),
                self.tr("Select a training run to export to Rerun (.rrd)."),
            )
            return
        target_run = runs[0]
        default_path = str(target_run.run_dir / f"{target_run.run_dir.name}.rrd")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export Run to Rerun Recording"),
            default_path,
            "Rerun Recording (*.rrd)",
        )
        if not out_path:
            return
        out_file = Path(out_path)

        def _on_done(written: object) -> None:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Rerun Recording Exported"),
                self.tr("Saved Rerun recording to:\n%s\n\nOpen with: rerun %s")
                % (written, written),
            )

        self._start_background_export(
            label=self.tr("Exporting Rerun"),
            error_title=self.tr("Rerun Export Unavailable"),
            fn=lambda: export_run_to_rerun(target_run, out_file),
            on_success=_on_done,
        )

    def _show_trackers_status(self) -> None:
        trackers = detect_available_trackers()
        lines = [
            "Optional Experiment Tracking & Visualization Backends:",
            "",
        ]
        for trk in trackers:
            status = (
                f"Installed (v{trk.version})"
                if trk.installed and trk.version
                else (
                    "Installed"
                    if trk.installed
                    else f"Not installed — {trk.install_hint}"
                )
            )
            lines.append(f"• {trk.display_name}: {status}")
            lines.append(f"  {trk.description}")
        QtWidgets.QMessageBox.information(
            self,
            self.tr("External Trackers Status"),
            "\n".join(lines),
        )


class TrainingDashboardDialog(QtWidgets.QDialog):
    """Standalone non-modal window hosting the unified Training Monitor & Visualization Dashboard."""

    def __init__(
        self, *, settings: Optional[QtCore.QSettings] = None, parent=None
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("Training Monitor & Visualization"))
        self.setModal(False)
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.Window | QtCore.Qt.WindowMinMaxButtonsHint
        )
        self.resize(1220, 780)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.dashboard = TrainingDashboardWidget(settings=settings, parent=self)
        layout.addWidget(self.dashboard)

    def select_tab(self, tab_key: str) -> None:
        self.dashboard.select_tab(tab_key)

    def select_run_dir(self, run_dir: Path | str) -> None:
        self.dashboard.select_run_dir(run_dir)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        with contextlib.suppress(Exception):
            self.dashboard.cleanup()
        super().closeEvent(event)


__all__ = [
    "TrainingDashboardDialog",
    "TrainingDashboardWidget",
    "TrainingRunState",
]
