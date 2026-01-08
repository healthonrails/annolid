from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _find_args_yaml(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "args.yaml"
    if direct.exists():
        return direct
    for candidate in run_dir.rglob("args.yaml"):
        return candidate
    return None


def _find_train_log(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "train.log"
    if direct.exists():
        return direct
    for candidate in run_dir.rglob("train.log"):
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


def _read_csv_rows(path: Path, *, max_rows: int = 250) -> Tuple[Tuple[str, ...], List[Dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            columns = tuple(reader.fieldnames or ())
            rows: List[Dict[str, str]] = []
            for row in reader:
                if row:
                    rows.append({str(k): ("" if v is None else str(v))
                                for k, v in row.items() if k is not None})
            if max_rows and len(rows) > int(max_rows):
                rows = rows[-int(max_rows):]
            return columns, rows
    except Exception:
        return (), []


@lru_cache(maxsize=512)
def _has_yaml() -> bool:
    try:
        import yaml  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _read_total_epochs(args_yaml: Path) -> Optional[int]:
    try:
        if _has_yaml():
            import yaml  # type: ignore

            data = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                epochs = data.get("epochs")
                parsed = _safe_int(epochs)
                if parsed is not None:
                    return int(parsed)
    except Exception:
        pass

    try:
        for line in args_yaml.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if str(key).strip().lower() == "epochs":
                parsed = _safe_int(value)
                if parsed is not None:
                    return int(parsed)
    except Exception:
        pass
    return None


def _tail_text(path: Path, *, max_lines: int = 200) -> str:
    """Read a file tail efficiently without loading the whole file."""
    max_lines = max(1, int(max_lines))
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            if end <= 0:
                return ""

            block_size = 8192
            data = b""
            lines = 0
            pos = end
            while pos > 0 and lines <= max_lines:
                read_size = min(block_size, pos)
                pos -= read_size
                fh.seek(pos, os.SEEK_SET)
                chunk = fh.read(read_size)
                data = chunk + data
                lines = data.count(b"\n")
                if len(data) > 8_000_000:  # safety guard for pathological logs
                    break

        text = data.decode("utf-8", errors="replace")
        parts = text.splitlines()
        return "\n".join(parts[-max_lines:])
    except Exception:
        return ""


def _loss_from_row(row: Dict[str, str], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            value = _safe_float(row.get(key))
            if value is not None:
                return float(value)
    for key in keys:
        lower = key.lower()
        for k, v in row.items():
            if str(k).strip().lower() == lower:
                value = _safe_float(v)
                if value is not None:
                    return float(value)
    return None


_TIMELINE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("train_loss", "train"),
    ("val_loss", "val"),
    ("train/box_loss", "train_box"),
    ("train/seg_loss", "train_seg"),
    ("train/cls_loss", "train_cls"),
    ("train/dfl_loss", "train_dfl"),
    ("val/box_loss", "val_box"),
    ("val/seg_loss", "val_seg"),
    ("val/cls_loss", "val_cls"),
    ("val/dfl_loss", "val_dfl"),
    ("metrics/precision(B)", "prec_B"),
    ("metrics/recall(B)", "recall_B"),
    ("metrics/mAP50(B)", "mAP50_B"),
    ("metrics/mAP50-95(B)", "mAP5095_B"),
    ("metrics/mAP50(M)", "mAP50_M"),
    ("metrics/mAP50-95(M)", "mAP5095_M"),
    ("metrics/mAP50(P)", "mAP50_P"),
    ("metrics/mAP50-95(P)", "mAP5095_P"),
    ("metrics/precision(P)", "prec_P"),
    ("metrics/recall(P)", "recall_P"),
    ("metrics/precision(M)", "prec_M"),
    ("metrics/recall(M)", "recall_M"),
    ("seconds", "sec"),
)


def _format_timeline_line(epoch: int, row: Dict[str, str]) -> str:
    parts = [f"epoch={int(epoch)}"]
    for raw_key, label in _TIMELINE_KEYS:
        if raw_key not in row:
            continue
        val = _safe_float(row.get(raw_key))
        if val is None:
            continue
        if label == "sec":
            parts.append(f"{label}={val:.1f}")
        else:
            parts.append(f"{label}={val:.4f}")
    return "  ".join(parts)


class TrainingDashboardWidget(QtWidgets.QWidget):
    """Training dashboard with TensorBoard embed + lightweight progress monitoring."""

    def __init__(self, *, settings: Optional[QtCore.QSettings] = None, parent=None) -> None:
        super().__init__(parent=parent)
        self._settings = settings
        self._tb_process: Optional[subprocess.Popen] = None
        self._tb_url = "http://127.0.0.1:6006/"
        self._runs: Dict[str, TrainingRunState] = {}
        self._selected_run_key: Optional[str] = None
        self._follow_log = True

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
        title.setStyleSheet("font-size: 16px; font-weight: 650;")
        layout.addWidget(title)

        bar = QtWidgets.QHBoxLayout()
        self.runs_root_edit = QtWidgets.QLineEdit()
        self.runs_root_edit.setPlaceholderText(
            "Runs root (shared across trainings)")
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
            ["Task", "Model", "Status", "Run Dir", "Epoch",
                "Train Loss", "Val Loss", "Updated"]
        )
        self.active_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.active_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.active_table.horizontalHeader().setStretchLastSection(True)
        self.active_table.cellDoubleClicked.connect(
            self._open_selected_run_dir)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.active_tab)
        splitter.addWidget(self.active_table)

        details = QtWidgets.QWidget(self.active_tab)
        details_layout = QtWidgets.QVBoxLayout(details)
        details_layout.setContentsMargins(0, 10, 0, 0)

        header_row = QtWidgets.QHBoxLayout()
        self.selected_run_label = QtWidgets.QLabel(
            "Select a run to view details.", details)
        self.selected_run_label.setStyleSheet("color: #666;")
        header_row.addWidget(self.selected_run_label, 1)
        self.open_log_btn = QtWidgets.QPushButton("Open Log", details)
        self.open_log_btn.clicked.connect(self._open_selected_log)
        self.open_log_btn.setEnabled(False)
        header_row.addWidget(self.open_log_btn)
        details_layout.addLayout(header_row)

        progress_row = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar(details)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progress: —")
        self.progress_bar.setEnabled(False)
        progress_row.addWidget(self.progress_bar, 1)
        self.follow_log_checkbox = QtWidgets.QCheckBox("Follow log", details)
        self.follow_log_checkbox.setChecked(True)
        self.follow_log_checkbox.stateChanged.connect(
            lambda _=None: setattr(self, "_follow_log", bool(
                self.follow_log_checkbox.isChecked()))
        )
        progress_row.addWidget(self.follow_log_checkbox)
        details_layout.addLayout(progress_row)

        self.details_tabs = QtWidgets.QTabWidget(details)
        details_layout.addWidget(self.details_tabs, 1)

        self.epochs_tab = QtWidgets.QWidget(self.details_tabs)
        self.details_tabs.addTab(self.epochs_tab, "Epochs")
        epochs_layout = QtWidgets.QVBoxLayout(self.epochs_tab)
        self.epoch_table = QtWidgets.QTableWidget(0, 0, self.epochs_tab)
        self.epoch_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.epoch_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.epoch_table.horizontalHeader().setStretchLastSection(True)
        epochs_layout.addWidget(self.epoch_table, 1)

        self.log_tab = QtWidgets.QWidget(self.details_tabs)
        self.details_tabs.addTab(self.log_tab, "Log")
        log_layout = QtWidgets.QVBoxLayout(self.log_tab)
        self.log_view = QtWidgets.QPlainTextEdit(self.log_tab)
        self.log_view.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.log_view.setFont(mono)
        log_layout.addWidget(self.log_view, 1)

        self.timeline_tab = QtWidgets.QWidget(self.details_tabs)
        self.details_tabs.insertTab(1, self.timeline_tab, "Timeline")
        timeline_layout = QtWidgets.QVBoxLayout(self.timeline_tab)
        self.timeline_view = QtWidgets.QPlainTextEdit(self.timeline_tab)
        self.timeline_view.setReadOnly(True)
        self.timeline_view.setFont(mono)
        timeline_layout.addWidget(self.timeline_view, 1)

        splitter.addWidget(details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        btns = QtWidgets.QHBoxLayout()
        self.open_run_btn = QtWidgets.QPushButton("Open Run Folder")
        self.open_run_btn.clicked.connect(self._open_selected_run_dir)
        btns.addWidget(self.open_run_btn)
        btns.addStretch(1)
        layout.addLayout(btns)

        try:
            self.active_table.selectionModel().selectionChanged.connect(
                self._on_active_selection_changed)
        except Exception:
            pass

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
        self._settings.setValue("training/runs_root",
                                self.runs_root_edit.text().strip())

    def _sync_env_runs_root(self) -> None:
        root = self.runs_root_edit.text().strip()
        if root:
            os.environ["ANNOLID_RUNS_ROOT"] = root

    def _on_runs_root_changed(self) -> None:
        self._save_settings()
        self._sync_env_runs_root()
        self._refresh_runs_list()

    def _browse_runs_root(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select runs root")
        if not folder:
            return
        self.runs_root_edit.setText(folder)
        self._on_runs_root_changed()

    def _open_runs_root(self) -> None:
        root = self.runs_root()
        if not root.exists():
            QtWidgets.QMessageBox.information(
                self, "Not found", f"Runs root does not exist:\n{root}")
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
                preferred_port=QtCore.QUrl(self._tb_url).port() if QtCore.QUrl(
                    self._tb_url).port() != -1 else 6006,
                host="127.0.0.1",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "TensorBoard error", str(exc))
            return
        self._tb_process = process
        self._tb_url = str(url)
        self.tb_status.setText(
            f"TensorBoard: {self._tb_url} (logdir={log_dir})")
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
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(path)))

    def _open_selected_run_dir(self, *_args) -> None:
        row = self.active_table.currentRow()
        if row < 0:
            return
        item = self.active_table.item(row, 3)
        if item is None:
            return
        path = Path(item.text())
        if path.exists():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(path)))

    def _open_selected_log(self) -> None:
        key = self._selected_run_key
        if not key:
            return
        state = self._runs.get(key)
        if state is None or state.log_path is None:
            return
        path = state.log_path
        if path.exists():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(path)))

    def _on_active_selection_changed(self, *_args) -> None:
        self._selected_run_key = self._current_selected_run_key()
        self._refresh_selected_details()

    def _current_selected_run_key(self) -> Optional[str]:
        row = self.active_table.currentRow()
        if row < 0:
            return None
        item = self.active_table.item(row, 3)
        if item is None:
            return None
        try:
            return str(Path(item.text()).expanduser().resolve())
        except Exception:
            return None

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
        key = str(Path(str(run_dir)).expanduser().resolve())
        state = self._runs.get(key)
        if state is None:
            return
        state.status = "finished" if payload.get("ok", True) else "failed"
        state.updated_at = time.time()
        self._refresh_active_metrics()

    def _refresh_active_metrics(self) -> None:
        any_running = False
        for state in self._runs.values():
            if state.status == "running":
                any_running = True
            if state.results_csv is None or not state.results_csv.exists():
                state.results_csv = _find_results_csv(state.run_dir)
            if state.log_path is None or not state.log_path.exists():
                state.log_path = _find_train_log(state.run_dir)
            if state.total_epochs is None:
                args_yaml = _find_args_yaml(state.run_dir)
                if args_yaml is not None:
                    state.total_epochs = _read_total_epochs(args_yaml)

            if state.results_csv is not None:
                prev_epoch = state.timeline_last_epoch
                columns, rows = _read_csv_rows(state.results_csv, max_rows=250)
                if columns:
                    state.csv_columns = columns
                if rows:
                    state.csv_rows = rows
                    row = rows[-1]
                    state.last_epoch = _safe_int(
                        row.get("epoch") or row.get("Epoch"))
                    state.train_loss = _loss_from_row(
                        row,
                        (
                            "train_loss",
                            "train/box_loss",
                            "train/seg_loss",
                            "loss",
                            "train_loss_total",
                        ),
                    )
                    state.val_loss = _loss_from_row(
                        row,
                        (
                            "val_loss",
                            "val/box_loss",
                            "val/seg_loss",
                            "val_loss_total",
                        ),
                    )
                    state.updated_at = time.time()
                    for row in rows:
                        epoch = _safe_int(row.get("epoch") or row.get("Epoch"))
                        if epoch is None:
                            continue
                        if prev_epoch is not None and epoch <= prev_epoch:
                            continue
                        state.epoch_timeline.append(
                            _format_timeline_line(int(epoch), row))
                        prev_epoch = int(epoch)
                    state.timeline_last_epoch = prev_epoch
                    if len(state.epoch_timeline) > 800:
                        state.epoch_timeline = state.epoch_timeline[-800:]

        self._refresh_active_table()
        if not any_running:
            self._timer.stop()

    def _refresh_active_table(self) -> None:
        selected_key = self._selected_run_key or self._current_selected_run_key()
        self.active_table.setRowCount(0)
        for idx, state in enumerate(sorted(self._runs.values(), key=lambda s: s.started_at, reverse=True)):
            self.active_table.insertRow(idx)
            self.active_table.setItem(
                idx, 0, QtWidgets.QTableWidgetItem(state.task))
            self.active_table.setItem(
                idx, 1, QtWidgets.QTableWidgetItem(state.model))
            self.active_table.setItem(
                idx, 2, QtWidgets.QTableWidgetItem(state.status))
            self.active_table.setItem(
                idx, 3, QtWidgets.QTableWidgetItem(str(state.run_dir)))
            epoch_txt = ""
            if state.last_epoch is not None:
                if state.total_epochs is not None and state.total_epochs > 0:
                    epoch_txt = f"{state.last_epoch}/{state.total_epochs}"
                else:
                    epoch_txt = str(state.last_epoch)
            self.active_table.setItem(
                idx, 4, QtWidgets.QTableWidgetItem(epoch_txt))
            self.active_table.setItem(
                idx, 5, QtWidgets.QTableWidgetItem(
                    "" if state.train_loss is None else f"{state.train_loss:.6f}")
            )
            self.active_table.setItem(
                idx, 6, QtWidgets.QTableWidgetItem(
                    "" if state.val_loss is None else f"{state.val_loss:.6f}")
            )
            updated = ""
            if state.updated_at is not None:
                updated = time.strftime(
                    "%H:%M:%S", time.localtime(state.updated_at))
            self.active_table.setItem(
                idx, 7, QtWidgets.QTableWidgetItem(updated))

        self.active_table.resizeColumnsToContents()
        if selected_key:
            self._select_run(str(selected_key))

    def _select_run(self, key: str) -> None:
        """Try to highlight the provided run in the active table and refresh details."""
        self._selected_run_key = key
        current = self._current_selected_run_key()
        if current == key:
            self._refresh_selected_details()
            return
        for row in range(self.active_table.rowCount()):
            item = self.active_table.item(row, 3)
            if item is None:
                continue
            try:
                candidate = str(Path(item.text()).expanduser().resolve())
            except Exception:
                continue
            if candidate == key:
                self.active_table.setCurrentCell(row, 0)
                break
        self._refresh_selected_details()

    def _refresh_selected_details(self) -> None:
        key = self._selected_run_key
        if not key:
            self.selected_run_label.setText("Select a run to view details.")
            self.selected_run_label.setStyleSheet("color: #666;")
            self.open_log_btn.setEnabled(False)
            self.progress_bar.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Progress: —")
            self.epoch_table.setRowCount(0)
            self.epoch_table.setColumnCount(0)
            self.log_view.setPlainText("")
            if getattr(self, "timeline_view", None) is not None:
                self.timeline_view.setPlainText("")
            return

        state = self._runs.get(key)
        if state is None:
            return

        self.selected_run_label.setText(
            f"{state.task} · {state.model} · {state.status} · {state.run_dir}")
        self.selected_run_label.setStyleSheet("color: #222;")
        self.open_log_btn.setEnabled(
            state.log_path is not None and state.log_path.exists())

        self._render_epoch_table(state)
        self._render_progress(state)
        self._render_timeline(state)
        self._render_log_tail(state)

    def _render_progress(self, state: TrainingRunState) -> None:
        if state.last_epoch is None:
            self.progress_bar.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Progress: —")
            return
        if state.total_epochs is None or state.total_epochs <= 0:
            self.progress_bar.setEnabled(True)
            self.progress_bar.setRange(0, 0)  # indeterminate
            self.progress_bar.setFormat(f"Epoch {state.last_epoch}")
            return

        self.progress_bar.setEnabled(True)
        self.progress_bar.setRange(0, int(state.total_epochs))
        self.progress_bar.setValue(
            int(max(0, min(state.last_epoch, state.total_epochs))))
        self.progress_bar.setFormat(
            f"Epoch {state.last_epoch}/{state.total_epochs}")

    def _render_epoch_table(self, state: TrainingRunState) -> None:
        if not state.csv_rows or not state.csv_columns:
            self.epoch_table.setRowCount(0)
            self.epoch_table.setColumnCount(0)
            return

        columns = list(state.csv_columns)
        # Common "epoch" column first if present.
        for epoch_key in ("epoch", "Epoch"):
            if epoch_key in columns:
                columns.remove(epoch_key)
                columns.insert(0, epoch_key)
                break

        current_headers: List[str] = []
        for i in range(self.epoch_table.columnCount()):
            item = self.epoch_table.horizontalHeaderItem(i)
            current_headers.append("" if item is None else item.text())
        if self.epoch_table.columnCount() != len(columns) or current_headers != columns:
            self.epoch_table.clear()
            self.epoch_table.setColumnCount(len(columns))
            self.epoch_table.setHorizontalHeaderLabels(columns)

        self.epoch_table.setRowCount(len(state.csv_rows))
        for r, row in enumerate(state.csv_rows):
            for c, col in enumerate(columns):
                self.epoch_table.setItem(
                    r, c, QtWidgets.QTableWidgetItem(str(row.get(col, ""))))
        self.epoch_table.resizeColumnsToContents()
        try:
            self.epoch_table.scrollToBottom()
        except Exception:
            pass

    def _render_log_tail(self, state: TrainingRunState) -> None:
        if state.log_path is None or not state.log_path.exists():
            self.log_view.setPlainText("")
            return
        text = _tail_text(state.log_path, max_lines=250)
        if self.log_view.toPlainText() == text:
            return
        self.log_view.setPlainText(text)
        if self._follow_log:
            try:
                cursor = self.log_view.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                self.log_view.setTextCursor(cursor)
            except Exception:
                pass

    def _render_timeline(self, state: TrainingRunState) -> None:
        text = "\n".join(state.epoch_timeline)
        if getattr(self, "timeline_view", None) is None:
            return
        if self.timeline_view.toPlainText() == text:
            return
        self.timeline_view.setPlainText(text)
        try:
            cursor = self.timeline_view.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            self.timeline_view.setTextCursor(cursor)
            self.timeline_view.centerCursor()
        except Exception:
            pass


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
        self.dashboard = TrainingDashboardWidget(
            settings=settings, parent=self)
        layout.addWidget(self.dashboard)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.dashboard._stop_tensorboard()
        except Exception:
            pass
        super().closeEvent(event)
