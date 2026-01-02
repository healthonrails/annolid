from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.tensorboard import ensure_tensorboard, stop_tensorboard
from annolid.utils.logger import logger
from annolid.utils.runs import shared_runs_root


@dataclass
class TrainingRunState:
    run_dir: Path
    task: str
    model: str
    status: str  # running|finished|failed
    started_at: float
    last_epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    updated_at: Optional[float] = None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _find_results_csv(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "results.csv"
    if direct.exists():
        return direct
    for candidate in run_dir.rglob("results.csv"):
        return candidate
    return None


def _read_last_csv_row(path: Path) -> Optional[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            last = None
            for row in reader:
                if row:
                    last = row
            return last
    except Exception:
        return None


class TrainingDashboardWidget(QtWidgets.QWidget):
    """Training dashboard with TensorBoard embed + lightweight progress monitoring."""

    def __init__(self, *, settings: Optional[QtCore.QSettings] = None, parent=None) -> None:
        super().__init__(parent=parent)
        self._settings = settings
        self._tb_process: Optional[subprocess.Popen] = None
        self._tb_url = "http://127.0.0.1:6006/"
        self._runs: Dict[str, TrainingRunState] = {}

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._refresh_active_metrics)

        self._build_ui()
        self._load_settings()
        self._sync_env_runs_root()
        self._refresh_runs_list()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._stop_tensorboard()
        super().closeEvent(event)

    def register_training_manager(self, manager: QtCore.QObject) -> None:
        started = getattr(manager, "training_started", None)
        finished = getattr(manager, "training_finished", None)
        if hasattr(started, "connect"):
            started.connect(self._on_training_started)
        if hasattr(finished, "connect"):
            finished.connect(self._on_training_finished)

    def runs_root(self) -> Path:
        return Path(self.runs_root_edit.text().strip()).expanduser().resolve()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Training Dashboard")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        bar = QtWidgets.QHBoxLayout()
        self.runs_root_edit = QtWidgets.QLineEdit()
        self.runs_root_edit.setPlaceholderText("Runs root (shared across trainings)")
        self.runs_root_browse = QtWidgets.QPushButton("Browse…")
        self.runs_root_browse.clicked.connect(self._browse_runs_root)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_runs_list)
        bar.addWidget(QtWidgets.QLabel("Runs root:"))
        bar.addWidget(self.runs_root_edit, 1)
        bar.addWidget(self.runs_root_browse)
        bar.addWidget(self.refresh_btn)
        layout.addLayout(bar)

        tb_bar = QtWidgets.QHBoxLayout()
        self.start_tb_btn = QtWidgets.QPushButton("Start TensorBoard")
        self.start_tb_btn.clicked.connect(self._start_tensorboard)
        self.stop_tb_btn = QtWidgets.QPushButton("Stop")
        self.stop_tb_btn.clicked.connect(self._stop_tensorboard)
        self.open_tb_btn = QtWidgets.QPushButton("Open in Browser")
        self.open_tb_btn.clicked.connect(self._open_tensorboard_in_browser)
        self.tb_status = QtWidgets.QLabel("TensorBoard: idle")
        self.tb_status.setStyleSheet("color: #666;")
        tb_bar.addWidget(self.start_tb_btn)
        tb_bar.addWidget(self.stop_tb_btn)
        tb_bar.addWidget(self.open_tb_btn)
        tb_bar.addWidget(self.tb_status, 1)
        layout.addLayout(tb_bar)

        self.tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        self.active_tab = QtWidgets.QWidget(self)
        self.runs_tab = QtWidgets.QWidget(self)
        self.tb_tab = QtWidgets.QWidget(self)
        self.tabs.addTab(self.active_tab, "Active")
        self.tabs.addTab(self.runs_tab, "Runs")
        self.tabs.addTab(self.tb_tab, "TensorBoard")

        self._build_active_tab()
        self._build_runs_tab()
        self._build_tb_tab()

        self.runs_root_edit.editingFinished.connect(self._on_runs_root_changed)

    def _build_active_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.active_tab)
        self.active_table = QtWidgets.QTableWidget(0, 8, self)
        self.active_table.setHorizontalHeaderLabels(
            ["Task", "Model", "Status", "Run Dir", "Epoch", "Train Loss", "Val Loss", "Updated"]
        )
        self.active_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.active_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.active_table.horizontalHeader().setStretchLastSection(True)
        self.active_table.cellDoubleClicked.connect(self._open_selected_run_dir)
        layout.addWidget(self.active_table)

        btns = QtWidgets.QHBoxLayout()
        self.open_run_btn = QtWidgets.QPushButton("Open Run Folder")
        self.open_run_btn.clicked.connect(self._open_selected_run_dir)
        btns.addWidget(self.open_run_btn)
        btns.addStretch(1)
        layout.addLayout(btns)

    def _build_runs_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.runs_tab)
        self.runs_list = QtWidgets.QListWidget(self)
        self.runs_list.itemDoubleClicked.connect(self._open_run_item)
        layout.addWidget(self.runs_list, 1)

        row = QtWidgets.QHBoxLayout()
        self.open_runs_root_btn = QtWidgets.QPushButton("Open Runs Root")
        self.open_runs_root_btn.clicked.connect(self._open_runs_root)
        row.addWidget(self.open_runs_root_btn)
        row.addStretch(1)
        layout.addLayout(row)

    def _build_tb_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tb_tab)
        try:
            from qtpy.QtWebEngineWidgets import QWebEngineView  # type: ignore
        except Exception:
            self.tb_view = None
            hint = QtWidgets.QLabel(
                "QtWebEngine is not available in this environment.\n"
                "Use 'Open in Browser' to view TensorBoard."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #666;")
            layout.addWidget(hint)
            return

        self.tb_view = QWebEngineView()
        self.tb_view.setUrl(QtCore.QUrl(self._tb_url))
        layout.addWidget(self.tb_view, 1)

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
        self._settings.setValue("training/runs_root", self.runs_root_edit.text().strip())

    def _sync_env_runs_root(self) -> None:
        root = self.runs_root_edit.text().strip()
        if root:
            os.environ["ANNOLID_RUNS_ROOT"] = root

    def _on_runs_root_changed(self) -> None:
        self._save_settings()
        self._sync_env_runs_root()
        self._refresh_runs_list()

    def _browse_runs_root(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select runs root")
        if not folder:
            return
        self.runs_root_edit.setText(folder)
        self._on_runs_root_changed()

    def _open_runs_root(self) -> None:
        root = self.runs_root()
        if not root.exists():
            QtWidgets.QMessageBox.information(self, "Not found", f"Runs root does not exist:\n{root}")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(root)))

    def _open_tensorboard_in_browser(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._tb_url))

    def _start_tensorboard(self) -> None:
        self._stop_tensorboard()
        log_dir = self.runs_root()
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            process, url = ensure_tensorboard(
                log_dir=log_dir,
                preferred_port=QtCore.QUrl(self._tb_url).port() if QtCore.QUrl(self._tb_url).port() != -1 else 6006,
                host="127.0.0.1",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "TensorBoard error", str(exc))
            return
        self._tb_process = process
        self._tb_url = str(url)
        self.tb_status.setText(f"TensorBoard: {self._tb_url} (logdir={log_dir})")
        if getattr(self, "tb_view", None) is not None:
            try:
                self.tb_view.setUrl(QtCore.QUrl(self._tb_url))
            except Exception:
                pass
        self.tabs.setCurrentWidget(self.tb_tab)

    def _stop_tensorboard(self) -> None:
        if self._tb_process is not None:
            try:
                stop_tensorboard(self._tb_process)
            except Exception:
                pass
            self._tb_process = None
        self.tb_status.setText("TensorBoard: idle")

    def _refresh_runs_list(self) -> None:
        root = self.runs_root()
        self.runs_list.clear()
        if not root.exists():
            return

        run_dirs = set()
        for path in root.rglob("events.out.tfevents.*"):
            run_dirs.add(path.parent)
        for path in root.rglob("results.csv"):
            run_dirs.add(path.parent)

        def sort_key(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        for run_dir in sorted(run_dirs, key=sort_key, reverse=True)[:500]:
            item = QtWidgets.QListWidgetItem(str(run_dir))
            self.runs_list.addItem(item)

    def _open_run_item(self, item: QtWidgets.QListWidgetItem) -> None:
        path = Path(item.text())
        if path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _open_selected_run_dir(self, *_args) -> None:
        row = self.active_table.currentRow()
        if row < 0:
            return
        item = self.active_table.item(row, 3)
        if item is None:
            return
        path = Path(item.text())
        if path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    @QtCore.Slot(object)
    def _on_training_started(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        run_dir = payload.get("run_dir")
        if not run_dir:
            return
        key = str(Path(str(run_dir)).expanduser().resolve())
        state = TrainingRunState(
            run_dir=Path(key),
            task=str(payload.get("task") or "train"),
            model=str(payload.get("model") or ""),
            status="running",
            started_at=time.time(),
        )
        self._runs[key] = state
        self._refresh_active_table()
        if not self._timer.isActive():
            self._timer.start()

    @QtCore.Slot(object)
    def _on_training_finished(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        run_dir = payload.get("run_dir")
        if not run_dir:
            return
        key = str(Path(str(run_dir)).expanduser().resolve())
        state = self._runs.get(key)
        if state is None:
            return
        state.status = "finished" if payload.get("ok", True) else "failed"
        state.updated_at = time.time()
        self._refresh_active_table()

    def _refresh_active_metrics(self) -> None:
        any_running = False
        for state in self._runs.values():
            if state.status == "running":
                any_running = True
            results_csv = _find_results_csv(state.run_dir)
            if results_csv is None:
                continue
            row = _read_last_csv_row(results_csv)
            if not row:
                continue
            state.last_epoch = _safe_int(row.get("epoch") or row.get("Epoch"))
            state.train_loss = _safe_float(row.get("train_loss") or row.get("train/box_loss"))
            state.val_loss = _safe_float(row.get("val_loss") or row.get("val/box_loss"))
            state.updated_at = time.time()

        self._refresh_active_table()
        if not any_running:
            self._timer.stop()

    def _refresh_active_table(self) -> None:
        self.active_table.setRowCount(0)
        for idx, state in enumerate(sorted(self._runs.values(), key=lambda s: s.started_at, reverse=True)):
            self.active_table.insertRow(idx)
            self.active_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(state.task))
            self.active_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(state.model))
            self.active_table.setItem(idx, 2, QtWidgets.QTableWidgetItem(state.status))
            self.active_table.setItem(idx, 3, QtWidgets.QTableWidgetItem(str(state.run_dir)))
            self.active_table.setItem(idx, 4, QtWidgets.QTableWidgetItem("" if state.last_epoch is None else str(state.last_epoch)))
            self.active_table.setItem(
                idx, 5, QtWidgets.QTableWidgetItem("" if state.train_loss is None else f"{state.train_loss:.6f}")
            )
            self.active_table.setItem(
                idx, 6, QtWidgets.QTableWidgetItem("" if state.val_loss is None else f"{state.val_loss:.6f}")
            )
            updated = ""
            if state.updated_at is not None:
                updated = time.strftime("%H:%M:%S", time.localtime(state.updated_at))
            self.active_table.setItem(idx, 7, QtWidgets.QTableWidgetItem(updated))

        self.active_table.resizeColumnsToContents()


class TrainingDashboardDialog(QtWidgets.QDialog):
    """Standalone window wrapper for the training dashboard widget."""

    def __init__(self, *, settings: Optional[QtCore.QSettings] = None, parent=None) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle("Training Dashboard")
        self.setModal(False)
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.Window
            | QtCore.Qt.WindowMinMaxButtonsHint
        )
        self.resize(1100, 700)

        layout = QtWidgets.QVBoxLayout(self)
        self.dashboard = TrainingDashboardWidget(settings=settings, parent=self)
        layout.addWidget(self.dashboard)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.dashboard._stop_tensorboard()
        except Exception:
            pass
        super().closeEvent(event)
