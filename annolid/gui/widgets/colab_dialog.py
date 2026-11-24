"""Google Colab Studio dialog: personal account sign-in, runtime management, GPU training/inference, and live training monitor."""

from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.colab import (
    ColabAuthRequiredError,
    ColabJobRecord,
    ColabLogMonitorParser,
    ColabMetricPoint,
    ColabPredictRequest,
    ColabRuntimeInfo,
    ColabTrainRequest,
    ColabWorkflowService,
    PREDICT_MODEL_CHOICES,
    SUPPORTED_ACCELERATORS,
    TRAIN_MODEL_CHOICES,
    get_notebook_template,
    list_notebook_templates,
    load_metrics_from_results_csv,
)


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


class _ColabMetricsChartWidget(QtWidgets.QWidget):
    """Lightweight pure-Qt live training curves widget (Loss & mAP50)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._points: list[ColabMetricPoint] = []
        self.setMinimumHeight(185)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )

    def set_points(self, points: list[ColabMetricPoint]) -> None:
        self._points = list(points)
        self.update()

    def clear(self) -> None:
        self._points = []
        self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self.rect().adjusted(2, 2, -2, -2)
        gap = 12
        half_w = max(120, (rect.width() - gap) // 2)
        left_rect = QtCore.QRect(rect.left(), rect.top(), half_w, rect.height())
        right_rect = QtCore.QRect(
            left_rect.right() + gap,
            rect.top(),
            max(120, rect.right() - (left_rect.right() + gap)),
            rect.height(),
        )

        loss_series: list[tuple[str, QtGui.QColor, list[tuple[int, float]]]] = []
        train_pts = [
            (p.step, float(p.train_loss))
            for p in self._points
            if p.train_loss is not None
        ]
        val_pts = [
            (p.step, float(p.val_loss)) for p in self._points if p.val_loss is not None
        ]
        if train_pts:
            loss_series.append(("Train Loss", QtGui.QColor(37, 99, 235), train_pts))
        if val_pts:
            loss_series.append(("Val Loss", QtGui.QColor(234, 88, 12), val_pts))

        metric_series: list[tuple[str, QtGui.QColor, list[tuple[int, float]]]] = []
        box_pts = [
            (p.step, float(p.map50)) for p in self._points if p.map50 is not None
        ]
        mask_pts = [
            (p.step, float(p.mask_map50))
            for p in self._points
            if p.mask_map50 is not None
        ]
        if box_pts:
            metric_series.append(("Box mAP50", QtGui.QColor(22, 163, 74), box_pts))
        if mask_pts:
            metric_series.append(("Mask mAP50", QtGui.QColor(13, 148, 136), mask_pts))

        self._draw_panel(
            painter,
            left_rect,
            title="Training / Validation Loss",
            series=loss_series,
            empty_text="Waiting for epoch loss metrics…",
            fixed_unit_range=False,
        )
        self._draw_panel(
            painter,
            right_rect,
            title="Validation mAP50",
            series=metric_series,
            empty_text="Waiting for validation metrics…",
            fixed_unit_range=True,
        )
        painter.end()

    def _draw_panel(
        self,
        painter: QtGui.QPainter,
        panel: QtCore.QRect,
        *,
        title: str,
        series: list[tuple[str, QtGui.QColor, list[tuple[int, float]]]],
        empty_text: str,
        fixed_unit_range: bool,
    ) -> None:
        # Card background & border
        pal = self.palette()
        bg = pal.color(QtGui.QPalette.Base)
        fg = pal.color(QtGui.QPalette.Text)
        muted = QtGui.QColor(fg)
        muted.setAlpha(135)
        border = QtGui.QColor(fg)
        border.setAlpha(45)

        painter.setPen(QtGui.QPen(border, 1))
        painter.setBrush(bg)
        painter.drawRoundedRect(panel, 6, 6)

        header_font = painter.font()
        header_font.setBold(True)
        header_font.setPointSize(max(9, header_font.pointSize()))
        painter.setFont(header_font)
        painter.setPen(fg)
        painter.drawText(
            panel.adjusted(10, 6, -10, -panel.height() + 26),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            title,
        )

        # Latest values legend on top-right
        legend_font = painter.font()
        legend_font.setBold(False)
        legend_font.setPointSize(max(8, legend_font.pointSize() - 1))
        painter.setFont(legend_font)
        if series:
            summary_parts = [
                f"{name}: {pts[-1][1]:.4f}" for name, _c, pts in series if pts
            ]
            painter.setPen(muted)
            painter.drawText(
                panel.adjusted(10, 6, -10, -panel.height() + 26),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                "  |  ".join(summary_parts),
            )

        plot = panel.adjusted(46, 30, -14, -24)
        if plot.width() < 20 or plot.height() < 20:
            return

        if not series:
            painter.setPen(muted)
            painter.drawText(plot, QtCore.Qt.AlignCenter, empty_text)
            return

        all_x = [x for _n, _c, pts in series for x, _y in pts]
        all_y = [y for _n, _c, pts in series for _x, y in pts]
        min_x = min(all_x)
        max_x = max(all_x)
        if max_x == min_x:
            max_x = min_x + 1

        if fixed_unit_range:
            min_y = 0.0
            max_y = max(1.0, max(all_y) * 1.05)
        else:
            min_y = min(all_y)
            max_y = max(all_y)
            span = max_y - min_y
            if span <= 1e-9:
                pad = max(0.1, abs(max_y) * 0.1)
                min_y = max(0.0, min_y - pad)
                max_y = max_y + pad
            else:
                min_y = max(0.0, min_y - span * 0.08)
                max_y = max_y + span * 0.08

        # Grid lines + Y-axis labels
        grid_pen = QtGui.QPen(border, 1, QtCore.Qt.DotLine)
        for frac in (0.0, 0.5, 1.0):
            y_px = int(plot.bottom() - frac * plot.height())
            painter.setPen(grid_pen)
            painter.drawLine(plot.left(), y_px, plot.right(), y_px)
            y_val = min_y + frac * (max_y - min_y)
            painter.setPen(muted)
            painter.drawText(
                QtCore.QRect(panel.left() + 4, y_px - 8, 38, 16),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                f"{y_val:.2f}",
            )

        # X-axis endpoints
        painter.setPen(muted)
        painter.drawText(
            QtCore.QRect(plot.left(), plot.bottom() + 4, 60, 16),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"Ep {min_x}",
        )
        painter.drawText(
            QtCore.QRect(plot.right() - 60, plot.bottom() + 4, 60, 16),
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
            f"Ep {max(all_x)}",
        )

        # Draw curves
        for _name, color, pts in series:
            mapped: list[QtCore.QPointF] = []
            for x_val, y_val in pts:
                nx = (x_val - min_x) / float(max_x - min_x)
                ny = (y_val - min_y) / float(max_y - min_y)
                px = plot.left() + nx * plot.width()
                py = plot.bottom() - max(0.0, min(1.0, ny)) * plot.height()
                mapped.append(QtCore.QPointF(px, py))

            pen = QtGui.QPen(color, 2.0)
            painter.setPen(pen)
            if len(mapped) >= 2:
                poly = QtGui.QPolygonF(mapped)
                painter.drawPolyline(poly)
            painter.setBrush(color)
            for pt in mapped:
                painter.drawEllipse(pt, 3.0, 3.0)


class ColabSignInDialog(QtWidgets.QDialog):
    """Guided Google Colab sign-in prompt shown when credentials are missing or expired."""

    def __init__(
        self,
        *,
        reason: str,
        auth_status: dict[str, Any],
        allow_notebook_fallback: bool = True,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Sign In to Your Google Colab Account"))
        self.setMinimumWidth(560)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        title = QtWidgets.QLabel(self.tr("Sign In to Google Colab"))
        font = title.font()
        font.setPointSize(max(11, font.pointSize() + 2))
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        reason_label = QtWidgets.QLabel(str(reason or ""))
        reason_label.setWordWrap(True)
        layout.addWidget(reason_label)

        cli_installed = bool((auth_status.get("colab_cli") or {}).get("installed"))
        has_oauth_client = bool(
            (auth_status.get("oauth_colab") or {}).get("has_credentials_file")
        )
        has_gcloud = bool((auth_status.get("adc") or {}).get("gcloud_installed"))

        options_group = QtWidgets.QGroupBox(self.tr("Choose how to sign in"))
        opt_layout = QtWidgets.QVBoxLayout(options_group)
        self._button_group = QtWidgets.QButtonGroup(self)
        self._radios: list[tuple[str, QtWidgets.QRadioButton]] = []

        def _add_option(key: str, text: str, checked: bool = False) -> None:
            rb = QtWidgets.QRadioButton(text)
            rb.setChecked(checked)
            self._button_group.addButton(rb)
            self._radios.append((key, rb))
            opt_layout.addWidget(rb)

        personal_hint = (
            self.tr(
                "Sign in with my personal Google Colab account — automatic browser sign-in (Recommended)"
            )
            if has_gcloud
            else (
                self.tr(
                    "Sign in with my personal Google Colab account — opens browser sign-in (Recommended)"
                )
                if cli_installed
                else self.tr(
                    "Sign in with my personal Google Colab account — installs official Colab CLI automatically, no GCP setup needed (Recommended)"
                )
            )
        )
        _add_option("personal", personal_hint, checked=True)

        cli_label = (
            self.tr(
                "Colab CLI OAuth2 (`colab --auth=oauth2` — browser + paste authorization code)"
            )
            if cli_installed
            else self.tr(
                "Install & sign in via Colab CLI OAuth2 (personal Google account, no `gcloud` required)"
            )
        )
        _add_option("colab-cli", cli_label)

        adc_label = (
            self.tr(
                "Google Cloud ADC (`gcloud` detected — automatic localhost browser sign-in)"
            )
            if has_gcloud
            else self.tr("Google Cloud ADC (requires `gcloud` CLI)")
        )
        _add_option("adc", adc_label)

        oauth_label = (
            self.tr(
                "Shared Annolid Google OAuth2 (Colab REST API + Drive — OAuth client JSON detected)"
            )
            if has_oauth_client
            else self.tr(
                "Shared Annolid Google OAuth2 (Colab REST API + Drive — requires OAuth client JSON)"
            )
        )
        _add_option("oauth2", oauth_label)

        _add_option(
            "drive",
            self.tr("Google Drive OAuth only (for Drive-synced notebooks)"),
        )

        if allow_notebook_fallback:
            _add_option(
                "notebook",
                self.tr(
                    "Continue without signing in — generate a Colab .ipynb notebook instead"
                ),
            )

        layout.addWidget(options_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        ok_btn = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText(self.tr("Sign In && Continue"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_action(self) -> str:
        for key, rb in self._radios:
            if rb.isChecked():
                return key
        return "personal"


class _ColabTaskWorker(QtCore.QObject):
    """Background worker executing blocking Colab auth/runtime/job calls off the UI thread."""

    log_emitted = QtCore.Signal(str)
    progress_emitted = QtCore.Signal(int, str)
    auth_code_requested = QtCore.Signal(str, object, object)
    finished_ok = QtCore.Signal(str, object)
    finished_auth_required = QtCore.Signal(str, str)
    finished_error = QtCore.Signal(str, str)

    def __init__(
        self,
        task_name: str,
        fn: Callable[..., Any],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__()
        self._task_name = task_name
        self._fn = fn
        self._cancel_event = cancel_event or threading.Event()

    def _request_auth_code(self, auth_url: str) -> Optional[str]:
        done = threading.Event()
        holder: dict[str, Optional[str]] = {"code": None}
        self.auth_code_requested.emit(str(auth_url or ""), holder, done)
        done.wait(timeout=300)
        return holder.get("code")

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._fn(
                log_callback=self.log_emitted.emit,
                progress_callback=self.progress_emitted.emit,
                cancel_event=self._cancel_event,
                auth_code_callback=self._request_auth_code,
            )
        except ColabAuthRequiredError as exc:
            self.finished_auth_required.emit(self._task_name, str(exc))
            return
        except Exception as exc:
            self.finished_error.emit(self._task_name, str(exc))
            return
        self.finished_ok.emit(self._task_name, result)


class ColabStudioDialog(QtWidgets.QDialog):
    """Unified Google Colab Studio for personal sign-in, GPU training/inference, and live monitoring."""

    openPredictionsRequested = QtCore.Signal(str)
    openTrainingDashboardRequested = QtCore.Signal(str)

    _TAB_KEYS = ("auth", "train", "predict", "monitor", "jobs")

    def __init__(
        self,
        *,
        service: Optional[ColabWorkflowService] = None,
        initial_dataset_path: Optional[str] = None,
        initial_video_path: Optional[str] = None,
        initial_weights_path: Optional[str] = None,
        initial_tab: str = "auth",
        auto_refresh_account: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Google Colab Studio"))
        self.setMinimumSize(1000, 740)

        self._service = service or ColabWorkflowService()
        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_ColabTaskWorker] = None
        self._cancel_event: Optional[threading.Event] = None
        self._sessions_cache: list[ColabRuntimeInfo] = []
        self._jobs_cache: list[ColabJobRecord] = []
        self._action_buttons: list[QtWidgets.QPushButton] = []
        self._cli_installed: bool = False
        self._account_info: dict[str, Any] = {}
        self._active_retry_factory: Optional[Callable[[Optional[str]], None]] = None
        self._pending_after_cleanup: Optional[Callable[[], None]] = None
        self._run_pending_after_cleanup: bool = False

        # Live monitor state
        self._monitor_parser = ColabLogMonitorParser()
        self._monitor_started_at: Optional[float] = None
        self._monitor_raw_lines: list[str] = []
        self._monitor_output_dir: Optional[Path] = None
        self._monitor_plot_paths: list[Path] = []
        self._elapsed_timer = QtCore.QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed_timer)

        self._build_ui()
        self._apply_initial_paths(
            dataset_path=initial_dataset_path,
            video_path=initial_video_path,
            weights_path=initial_weights_path,
        )
        self.refresh_auth_status()
        self.refresh_jobs()
        self.select_tab(initial_tab)

        if auto_refresh_account:
            QtCore.QTimer.singleShot(150, self._on_refresh_account_clicked)

    # ------------------------------------------------------------------ #
    # UI Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)

        # Header banner
        header_row = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title_label = QtWidgets.QLabel(self.tr("Google Colab Studio"))
        title_font = title_label.font()
        title_font.setPointSize(max(13, title_font.pointSize() + 3))
        title_font.setBold(True)
        title_label.setFont(title_font)
        subtitle = QtWidgets.QLabel(
            self.tr(
                "Sign in with your own personal Google Colab account to train models, "
                "run GPU inference, and monitor training live from Annolid."
            )
        )
        subtitle.setWordWrap(True)
        title_box.addWidget(title_label)
        title_box.addWidget(subtitle)
        header_row.addLayout(title_box, 1)

        backend_box = QtWidgets.QHBoxLayout()
        backend_box.addWidget(QtWidgets.QLabel(self.tr("Execution Backend:")))
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItem(self.tr("Auto (Recommended)"), "auto")
        self.backend_combo.addItem(self.tr("Colab CLI (google-colab-cli)"), "cli")
        self.backend_combo.addItem(self.tr("Colab REST API (v1beta)"), "api")
        self.backend_combo.addItem(self.tr("Drive / .ipynb Notebook"), "notebook")
        backend_box.addWidget(self.backend_combo)
        header_row.addLayout(backend_box)
        root.addLayout(header_row)

        # Persistent Personal Account Bar across all tabs
        account_frame = QtWidgets.QFrame()
        account_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        acct_row = QtWidgets.QHBoxLayout(account_frame)
        acct_row.setContentsMargins(10, 6, 10, 6)
        self.header_account_dot = QtWidgets.QLabel("●")
        self.header_account_label = QtWidgets.QLabel(
            self.tr("Personal Colab Account: checking…")
        )
        acct_font = self.header_account_label.font()
        acct_font.setBold(True)
        self.header_account_label.setFont(acct_font)
        self.header_usage_label = QtWidgets.QLabel("")
        self.header_sign_in_btn = QtWidgets.QPushButton(self.tr("Sign In with Google"))
        self.header_sign_in_btn.clicked.connect(self._on_personal_sign_in_clicked)
        self.header_switch_account_btn = QtWidgets.QPushButton(
            self.tr("Switch Account")
        )
        self.header_switch_account_btn.clicked.connect(self._on_switch_account_clicked)
        self.header_refresh_account_btn = QtWidgets.QPushButton(
            self.tr("Check Account")
        )
        self.header_refresh_account_btn.clicked.connect(
            self._on_refresh_account_clicked
        )
        self.header_sim2real_btn = QtWidgets.QPushButton(self.tr("Sim-to-Real Studio…"))
        self.header_sim2real_btn.setToolTip(
            self.tr(
                "Generate environment-matched Mouse/Fly 3D synthetic datasets from example videos and iterate on Colab"
            )
        )
        self.header_sim2real_btn.clicked.connect(self._on_open_sim2real_studio_clicked)
        self._action_buttons.extend(
            [
                self.header_sign_in_btn,
                self.header_switch_account_btn,
                self.header_refresh_account_btn,
            ]
        )
        acct_row.addWidget(self.header_account_dot)
        acct_row.addWidget(self.header_account_label)
        acct_row.addWidget(self.header_usage_label, 1)
        acct_row.addWidget(self.header_sim2real_btn)
        acct_row.addWidget(self.header_sign_in_btn)
        acct_row.addWidget(self.header_switch_account_btn)
        acct_row.addWidget(self.header_refresh_account_btn)
        root.addWidget(account_frame)

        # Main tabs (preserve auth=0, train=1, predict=2 for compatibility)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_auth_tab(), self.tr("Account && Runtimes"))
        self.tabs.addTab(self._build_train_tab(), self.tr("Train on Colab"))
        self.tabs.addTab(self._build_predict_tab(), self.tr("Inference on Colab"))
        self.tabs.addTab(self._build_monitor_tab(), self.tr("Live Monitor"))
        self.tabs.addTab(self._build_jobs_tab(), self.tr("Jobs && Notebooks"))
        root.addWidget(self.tabs, 1)

        # Compact global progress & action bar at the bottom
        footer = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(260)
        self.status_label = QtWidgets.QLabel(self.tr("Ready"))
        self.view_monitor_btn = QtWidgets.QPushButton(self.tr("Open Monitor"))
        self.view_monitor_btn.clicked.connect(lambda: self.select_tab("monitor"))
        self.cancel_btn = QtWidgets.QPushButton(self.tr("Stop / Cancel"))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_active_task)
        close_btn = QtWidgets.QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.close)

        footer.addWidget(self.progress_bar)
        footer.addWidget(self.status_label, 1)
        footer.addWidget(self.view_monitor_btn)
        footer.addWidget(self.cancel_btn)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def _build_auth_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(10)

        # Friendly Personal Google Colab Account card
        personal_group = QtWidgets.QGroupBox(
            self.tr("Personal Google Colab Account (Recommended)")
        )
        p_layout = QtWidgets.QVBoxLayout(personal_group)
        p_intro = QtWidgets.QLabel(
            self.tr(
                "Sign in with your own personal Google account (Colab Free, Pro, or Pro+) "
                "to run Annolid GPU training and inference — no Google Cloud project setup required."
            )
        )
        p_intro.setWordWrap(True)
        p_layout.addWidget(p_intro)

        p_grid = QtWidgets.QGridLayout()
        self.account_email_label = QtWidgets.QLabel(self.tr("Not checked yet"))
        self.account_provider_label = QtWidgets.QLabel("—")
        self.account_balance_label = QtWidgets.QLabel("—")
        self.account_expiry_label = QtWidgets.QLabel("—")
        p_grid.addWidget(QtWidgets.QLabel(self.tr("Signed-In Account:")), 0, 0)
        p_grid.addWidget(self.account_email_label, 0, 1)
        p_grid.addWidget(QtWidgets.QLabel(self.tr("Auth Method:")), 0, 2)
        p_grid.addWidget(self.account_provider_label, 0, 3)
        p_grid.addWidget(QtWidgets.QLabel(self.tr("Compute Units / Usage:")), 1, 0)
        p_grid.addWidget(self.account_balance_label, 1, 1)
        p_grid.addWidget(QtWidgets.QLabel(self.tr("Session Token:")), 1, 2)
        p_grid.addWidget(self.account_expiry_label, 1, 3)
        p_grid.setColumnStretch(1, 1)
        p_grid.setColumnStretch(3, 1)
        p_layout.addLayout(p_grid)

        p_btns = QtWidgets.QHBoxLayout()
        self.personal_sign_in_btn = QtWidgets.QPushButton(
            self.tr("Sign In with Personal Google Account")
        )
        self.personal_sign_in_btn.clicked.connect(self._on_personal_sign_in_clicked)
        self.switch_account_btn = QtWidgets.QPushButton(
            self.tr("Sign Out / Switch Account")
        )
        self.switch_account_btn.clicked.connect(self._on_switch_account_clicked)
        self.live_probe_btn = QtWidgets.QPushButton(self.tr("Test Live Connection"))
        self.live_probe_btn.clicked.connect(lambda: self._on_run_diagnostics(live=True))
        manage_cu_btn = QtWidgets.QPushButton(self.tr("Manage Colab Compute Units…"))
        manage_cu_btn.clicked.connect(
            lambda: webbrowser.open("https://colab.research.google.com/signup")
        )
        self._action_buttons.extend(
            [self.personal_sign_in_btn, self.switch_account_btn, self.live_probe_btn]
        )
        p_btns.addWidget(self.personal_sign_in_btn)
        p_btns.addWidget(self.switch_account_btn)
        p_btns.addWidget(self.live_probe_btn)
        p_btns.addWidget(manage_cu_btn)
        p_btns.addStretch()
        p_layout.addLayout(p_btns)
        layout.addWidget(personal_group)

        # Collapsible Advanced OAuth / Cloud Settings (hidden by default for cleaner UX)
        self.advanced_auth_toggle = QtWidgets.QToolButton()
        self.advanced_auth_toggle.setText(
            self.tr("Advanced OAuth / Google Cloud Settings")
        )
        self.advanced_auth_toggle.setCheckable(True)
        self.advanced_auth_toggle.setChecked(False)
        self.advanced_auth_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.advanced_auth_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.advanced_auth_toggle.toggled.connect(self._on_toggle_advanced_auth)
        layout.addWidget(self.advanced_auth_toggle)

        self.advanced_auth_group = QtWidgets.QGroupBox()
        self.advanced_auth_group.setVisible(False)
        auth_layout = QtWidgets.QVBoxLayout(self.advanced_auth_group)

        self.auth_summary_label = QtWidgets.QLabel(self.tr("Checking authentication…"))
        self.auth_summary_label.setWordWrap(True)
        auth_layout.addWidget(self.auth_summary_label)

        grid = QtWidgets.QGridLayout()
        self.cli_status_label = QtWidgets.QLabel("—")
        self.oauth_colab_status_label = QtWidgets.QLabel("—")
        self.oauth_drive_status_label = QtWidgets.QLabel("—")
        self.adc_status_label = QtWidgets.QLabel("—")

        grid.addWidget(QtWidgets.QLabel(self.tr("Colab CLI (`colab`):")), 0, 0)
        grid.addWidget(self.cli_status_label, 0, 1)
        cli_btns = QtWidgets.QHBoxLayout()
        self.install_cli_btn = QtWidgets.QPushButton(self.tr("Install Colab CLI"))
        self.install_cli_btn.clicked.connect(self._on_install_cli_clicked)
        copy_cli_btn = QtWidgets.QPushButton(self.tr("Copy CLI Install Cmd"))
        copy_cli_btn.clicked.connect(
            lambda: self._copy_text("uv tool install google-colab-cli")
        )
        cli_btns.addWidget(self.install_cli_btn)
        cli_btns.addWidget(copy_cli_btn)
        grid.addLayout(cli_btns, 0, 2)

        grid.addWidget(QtWidgets.QLabel(self.tr("Google OAuth (Colab API):")), 1, 0)
        grid.addWidget(self.oauth_colab_status_label, 1, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel(self.tr("Google OAuth (Drive Sync):")), 2, 0)
        grid.addWidget(self.oauth_drive_status_label, 2, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel(self.tr("Google Cloud ADC (`gcloud`):")), 3, 0)
        grid.addWidget(self.adc_status_label, 3, 1)
        copy_adc_btn = QtWidgets.QPushButton(self.tr("Copy ADC Login Cmd"))
        copy_adc_btn.clicked.connect(self._copy_adc_command)
        grid.addWidget(copy_adc_btn, 3, 2)
        grid.setColumnStretch(1, 1)
        auth_layout.addLayout(grid)

        form = QtWidgets.QFormLayout()
        cred_row = QtWidgets.QHBoxLayout()
        self.credentials_edit = QtWidgets.QLineEdit(self._service.auth.credentials_file)
        cred_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        cred_browse.clicked.connect(self._browse_credentials_file)
        cred_row.addWidget(self.credentials_edit, 1)
        cred_row.addWidget(cred_browse)
        form.addRow(self.tr("OAuth Client JSON:"), cred_row)

        tok_row = QtWidgets.QHBoxLayout()
        self.token_edit = QtWidgets.QLineEdit(self._service.auth.token_file)
        tok_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        tok_browse.clicked.connect(self._browse_token_file)
        tok_row.addWidget(self.token_edit, 1)
        tok_row.addWidget(tok_browse)
        form.addRow(self.tr("OAuth Token File:"), tok_row)
        auth_layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.auth_method_combo = QtWidgets.QComboBox()
        self.auth_method_combo.addItem(
            self.tr("Personal Google Account (Recommended)"), "personal"
        )
        self.auth_method_combo.addItem(self.tr("Auto"), "auto")
        self.auth_method_combo.addItem(
            self.tr("Colab CLI OAuth2 (Code Flow)"), "colab-cli"
        )
        self.auth_method_combo.addItem(
            self.tr("Google OAuth2 (Colab + Drive)"), "oauth2"
        )
        self.auth_method_combo.addItem(self.tr("Google Drive OAuth Only"), "drive")
        self.auth_method_combo.addItem(self.tr("Google Cloud ADC (gcloud)"), "adc")

        self.login_btn = QtWidgets.QPushButton(self.tr("Sign In"))
        self.login_btn.clicked.connect(self._on_login_clicked)
        self.logout_btn = QtWidgets.QPushButton(self.tr("Clear OAuth Token"))
        self.logout_btn.clicked.connect(self._on_logout_clicked)
        self.refresh_auth_btn = QtWidgets.QPushButton(self.tr("Refresh Status"))
        self.refresh_auth_btn.clicked.connect(self.refresh_auth_status)
        self.doctor_btn = QtWidgets.QPushButton(self.tr("Run Diagnostics"))
        self.doctor_btn.clicked.connect(lambda: self._on_run_diagnostics(live=False))

        self._action_buttons.extend([self.login_btn, self.logout_btn, self.doctor_btn])
        btn_row.addWidget(QtWidgets.QLabel(self.tr("Method:")))
        btn_row.addWidget(self.auth_method_combo)
        btn_row.addWidget(self.login_btn)
        btn_row.addWidget(self.logout_btn)
        btn_row.addWidget(self.refresh_auth_btn)
        btn_row.addWidget(self.doctor_btn)
        btn_row.addStretch()
        auth_layout.addLayout(btn_row)
        layout.addWidget(self.advanced_auth_group)

        # Active runtimes table
        rt_group = QtWidgets.QGroupBox(self.tr("Active Colab Runtimes"))
        rt_layout = QtWidgets.QVBoxLayout(rt_group)

        self.sessions_table = QtWidgets.QTableWidget(0, 5)
        self.sessions_table.setHorizontalHeaderLabels(
            [
                self.tr("Name / Session"),
                self.tr("Backend"),
                self.tr("Accelerator"),
                self.tr("State"),
                self.tr("URL"),
            ]
        )
        self.sessions_table.horizontalHeader().setStretchLastSection(True)
        self.sessions_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sessions_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        rt_layout.addWidget(self.sessions_table)

        rt_btns = QtWidgets.QHBoxLayout()
        self.refresh_sessions_btn = QtWidgets.QPushButton(self.tr("Refresh Runtimes"))
        self.refresh_sessions_btn.clicked.connect(self._on_refresh_sessions)
        self.open_session_url_btn = QtWidgets.QPushButton(
            self.tr("Open Selected in Browser")
        )
        self.open_session_url_btn.clicked.connect(self._on_open_selected_session_url)
        self.check_specs_btn = QtWidgets.QPushButton(self.tr("Check Eligible GPUs"))
        self.check_specs_btn.clicked.connect(self._on_check_specs)
        self.stop_session_btn = QtWidgets.QPushButton(self.tr("Stop Selected Runtime"))
        self.stop_session_btn.clicked.connect(self._on_stop_selected_session)
        self._action_buttons.extend(
            [self.refresh_sessions_btn, self.check_specs_btn, self.stop_session_btn]
        )
        rt_btns.addWidget(self.refresh_sessions_btn)
        rt_btns.addWidget(self.open_session_url_btn)
        rt_btns.addWidget(self.check_specs_btn)
        rt_btns.addWidget(self.stop_session_btn)
        rt_btns.addStretch()
        rt_layout.addLayout(rt_btns)
        layout.addWidget(rt_group, 1)

        return page

    def _on_toggle_advanced_auth(self, checked: bool) -> None:
        self.advanced_auth_group.setVisible(bool(checked))
        self.advanced_auth_toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )

    def _build_train_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(10)

        form = QtWidgets.QFormLayout()

        self.train_model_combo = QtWidgets.QComboBox()
        for model_name in TRAIN_MODEL_CHOICES:
            self.train_model_combo.addItem(model_name, model_name)
        form.addRow(self.tr("Model:"), self.train_model_combo)

        ds_row = QtWidgets.QHBoxLayout()
        self.train_dataset_edit = QtWidgets.QLineEdit()
        self.train_dataset_edit.setPlaceholderText(
            self.tr("Path to data.yaml, spec.yaml, COCO folder, or .zip archive")
        )
        ds_file_btn = QtWidgets.QPushButton(self.tr("File…"))
        ds_file_btn.clicked.connect(self._browse_train_dataset_file)
        ds_dir_btn = QtWidgets.QPushButton(self.tr("Folder…"))
        ds_dir_btn.clicked.connect(self._browse_train_dataset_dir)
        ds_row.addWidget(self.train_dataset_edit, 1)
        ds_row.addWidget(ds_file_btn)
        ds_row.addWidget(ds_dir_btn)
        form.addRow(self.tr("Dataset:"), ds_row)

        wt_row = QtWidgets.QHBoxLayout()
        self.train_weights_combo = QtWidgets.QComboBox()
        self.train_weights_combo.setEditable(True)
        for preset in (
            "yolo11n-seg.pt",
            "yolo11s-seg.pt",
            "yolo11n-pose.pt",
            "yolo11s-pose.pt",
            "yolo11n.pt",
            "yolo26s-seg.pt",
            "yolo26s-pose.pt",
            "",
        ):
            self.train_weights_combo.addItem(preset)
        wt_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        wt_browse.clicked.connect(self._browse_train_weights)
        wt_row.addWidget(self.train_weights_combo, 1)
        wt_row.addWidget(wt_browse)
        form.addRow(self.tr("Initial Weights:"), wt_row)

        hp_row = QtWidgets.QHBoxLayout()
        self.train_epochs_spin = QtWidgets.QSpinBox()
        self.train_epochs_spin.setRange(1, 100000)
        self.train_epochs_spin.setValue(100)
        self.train_batch_spin = QtWidgets.QSpinBox()
        self.train_batch_spin.setRange(1, 512)
        self.train_batch_spin.setValue(16)
        self.train_imgsz_spin = QtWidgets.QSpinBox()
        self.train_imgsz_spin.setRange(64, 4096)
        self.train_imgsz_spin.setSingleStep(32)
        self.train_imgsz_spin.setValue(640)
        self.train_preset_combo = QtWidgets.QComboBox()
        self.train_preset_combo.addItem("(none)", "")
        for p in ("balanced", "small_dataset", "fast", "high_accuracy"):
            self.train_preset_combo.addItem(p, p)

        hp_row.addWidget(QtWidgets.QLabel(self.tr("Epochs/Iters:")))
        hp_row.addWidget(self.train_epochs_spin)
        hp_row.addWidget(QtWidgets.QLabel(self.tr("Batch:")))
        hp_row.addWidget(self.train_batch_spin)
        hp_row.addWidget(QtWidgets.QLabel(self.tr("Imgsz:")))
        hp_row.addWidget(self.train_imgsz_spin)
        hp_row.addWidget(QtWidgets.QLabel(self.tr("Preset:")))
        hp_row.addWidget(self.train_preset_combo)
        form.addRow(self.tr("Parameters:"), hp_row)

        hw_row = QtWidgets.QHBoxLayout()
        self.train_gpu_combo = QtWidgets.QComboBox()
        for acc in SUPPORTED_ACCELERATORS:
            self.train_gpu_combo.addItem(acc, acc)
        self.train_session_edit = QtWidgets.QLineEdit()
        self.train_session_edit.setPlaceholderText(
            self.tr("Optional session name (reuse or create, e.g. annolid-gpu)")
        )
        self.train_high_mem_check = QtWidgets.QCheckBox(self.tr("High-RAM"))
        self.train_keep_rt_check = QtWidgets.QCheckBox(self.tr("Keep runtime alive"))
        self.train_plots_check = QtWidgets.QCheckBox(self.tr("Save plots"))
        self.train_plots_check.setChecked(True)
        hw_row.addWidget(self.train_gpu_combo)
        hw_row.addWidget(self.train_session_edit, 1)
        hw_row.addWidget(self.train_high_mem_check)
        hw_row.addWidget(self.train_keep_rt_check)
        hw_row.addWidget(self.train_plots_check)
        form.addRow(self.tr("Colab Hardware:"), hw_row)

        out_row = QtWidgets.QHBoxLayout()
        self.train_output_edit = QtWidgets.QLineEdit()
        self.train_output_edit.setPlaceholderText(
            self.tr(
                "Optional local output folder (defaults to ~/.annolid/colab/outputs/<job>)"
            )
        )
        out_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        out_btn.clicked.connect(lambda: self._browse_dir_into(self.train_output_edit))
        out_row.addWidget(self.train_output_edit, 1)
        out_row.addWidget(out_btn)
        form.addRow(self.tr("Local Output Dir:"), out_row)

        drive_row = QtWidgets.QHBoxLayout()
        self.train_drive_folder_edit = QtWidgets.QLineEdit("Annolid/colab")
        self.train_upload_drive_check = QtWidgets.QCheckBox(self.tr("Sync to Drive"))
        self.train_upload_drive_check.setChecked(True)
        self.train_open_browser_check = QtWidgets.QCheckBox(self.tr("Open in browser"))
        self.train_open_browser_check.setChecked(True)
        drive_row.addWidget(self.train_drive_folder_edit, 1)
        drive_row.addWidget(self.train_upload_drive_check)
        drive_row.addWidget(self.train_open_browser_check)
        form.addRow(self.tr("Drive Folder:"), drive_row)

        self.train_install_spec_combo = QtWidgets.QComboBox()
        self.train_install_spec_combo.setEditable(True)
        self.train_install_spec_combo.addItem("annolid")
        self.train_install_spec_combo.addItem(
            "git+https://github.com/healthonrails/annolid.git"
        )
        self.train_install_spec_combo.setToolTip(
            self.tr(
                "PyPI release (`annolid`) or latest GitHub main (`git+https://...`) installed on the Colab VM."
            )
        )
        form.addRow(self.tr("Colab Pip Spec:"), self.train_install_spec_combo)

        self.train_extra_args_edit = QtWidgets.QLineEdit()
        self.train_extra_args_edit.setPlaceholderText(
            self.tr(
                "Optional extra annolid-run train args (e.g. --patience 30 --cos-lr)"
            )
        )
        form.addRow(self.tr("Extra CLI Args:"), self.train_extra_args_edit)

        layout.addLayout(form)

        actions = QtWidgets.QHBoxLayout()
        self.run_train_btn = QtWidgets.QPushButton(self.tr("Run Training on Colab"))
        self.run_train_btn.clicked.connect(
            lambda: self._start_train_job(force_notebook=False)
        )
        self.gen_train_nb_btn = QtWidgets.QPushButton(
            self.tr("Generate .ipynb Notebook")
        )
        self.gen_train_nb_btn.clicked.connect(
            lambda: self._start_train_job(force_notebook=True)
        )
        self._action_buttons.extend([self.run_train_btn, self.gen_train_nb_btn])
        actions.addWidget(self.run_train_btn)
        actions.addWidget(self.gen_train_nb_btn)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch()
        return page

    def _build_predict_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(10)

        form = QtWidgets.QFormLayout()

        self.predict_model_combo = QtWidgets.QComboBox()
        for model_name in PREDICT_MODEL_CHOICES:
            self.predict_model_combo.addItem(model_name, model_name)
        form.addRow(self.tr("Model:"), self.predict_model_combo)

        src_row = QtWidgets.QHBoxLayout()
        self.predict_source_edit = QtWidgets.QLineEdit()
        self.predict_source_edit.setPlaceholderText(
            self.tr("Video file (.mp4/.avi/.mov), image, or folder of images")
        )
        src_file_btn = QtWidgets.QPushButton(self.tr("Video/Image…"))
        src_file_btn.clicked.connect(self._browse_predict_source_file)
        src_dir_btn = QtWidgets.QPushButton(self.tr("Folder…"))
        src_dir_btn.clicked.connect(
            lambda: self._browse_dir_into(self.predict_source_edit)
        )
        src_row.addWidget(self.predict_source_edit, 1)
        src_row.addWidget(src_file_btn)
        src_row.addWidget(src_dir_btn)
        form.addRow(self.tr("Input Source:"), src_row)

        wt_row = QtWidgets.QHBoxLayout()
        self.predict_weights_combo = QtWidgets.QComboBox()
        self.predict_weights_combo.setEditable(True)
        for preset in (
            "yolo11n-seg.pt",
            "yolo11s-seg.pt",
            "yolo11n-pose.pt",
            "yoloe-26s-seg.pt",
            "",
        ):
            self.predict_weights_combo.addItem(preset)
        wt_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        wt_browse.clicked.connect(self._browse_predict_weights)
        wt_row.addWidget(self.predict_weights_combo, 1)
        wt_row.addWidget(wt_browse)
        form.addRow(self.tr("Model Weights:"), wt_row)

        opts_row = QtWidgets.QHBoxLayout()
        self.predict_conf_spin = QtWidgets.QDoubleSpinBox()
        self.predict_conf_spin.setRange(0.01, 1.0)
        self.predict_conf_spin.setSingleStep(0.05)
        self.predict_conf_spin.setValue(0.25)
        self.predict_iou_spin = QtWidgets.QDoubleSpinBox()
        self.predict_iou_spin.setRange(0.05, 1.0)
        self.predict_iou_spin.setSingleStep(0.05)
        self.predict_iou_spin.setValue(0.70)
        self.predict_imgsz_spin = QtWidgets.QSpinBox()
        self.predict_imgsz_spin.setRange(0, 4096)
        self.predict_imgsz_spin.setSingleStep(32)
        self.predict_imgsz_spin.setValue(640)
        opts_row.addWidget(QtWidgets.QLabel(self.tr("Conf:")))
        opts_row.addWidget(self.predict_conf_spin)
        opts_row.addWidget(QtWidgets.QLabel(self.tr("IoU:")))
        opts_row.addWidget(self.predict_iou_spin)
        opts_row.addWidget(QtWidgets.QLabel(self.tr("Imgsz (0=auto):")))
        opts_row.addWidget(self.predict_imgsz_spin)
        opts_row.addStretch()
        form.addRow(self.tr("Inference Params:"), opts_row)

        flags_row = QtWidgets.QHBoxLayout()
        self.predict_save_csv_check = QtWidgets.QCheckBox(
            self.tr("Export tracking CSV")
        )
        self.predict_save_csv_check.setChecked(True)
        self.predict_save_video_check = QtWidgets.QCheckBox(
            self.tr("Save preview video")
        )
        self.predict_no_tracking_check = QtWidgets.QCheckBox(
            self.tr("Disable tracking")
        )
        flags_row.addWidget(self.predict_save_csv_check)
        flags_row.addWidget(self.predict_save_video_check)
        flags_row.addWidget(self.predict_no_tracking_check)
        flags_row.addStretch()
        form.addRow(self.tr("Outputs:"), flags_row)

        hw_row = QtWidgets.QHBoxLayout()
        self.predict_gpu_combo = QtWidgets.QComboBox()
        for acc in SUPPORTED_ACCELERATORS:
            self.predict_gpu_combo.addItem(acc, acc)
        self.predict_session_edit = QtWidgets.QLineEdit()
        self.predict_session_edit.setPlaceholderText(
            self.tr("Optional session name (reuse warm session or create)")
        )
        self.predict_high_mem_check = QtWidgets.QCheckBox(self.tr("High-RAM"))
        self.predict_keep_rt_check = QtWidgets.QCheckBox(self.tr("Keep runtime alive"))
        hw_row.addWidget(self.predict_gpu_combo)
        hw_row.addWidget(self.predict_session_edit, 1)
        hw_row.addWidget(self.predict_high_mem_check)
        hw_row.addWidget(self.predict_keep_rt_check)
        form.addRow(self.tr("Colab Hardware:"), hw_row)

        out_row = QtWidgets.QHBoxLayout()
        self.predict_output_edit = QtWidgets.QLineEdit()
        self.predict_output_edit.setPlaceholderText(
            self.tr(
                "Optional local output folder (defaults to ~/.annolid/colab/outputs/<job>)"
            )
        )
        out_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        out_btn.clicked.connect(lambda: self._browse_dir_into(self.predict_output_edit))
        out_row.addWidget(self.predict_output_edit, 1)
        out_row.addWidget(out_btn)
        form.addRow(self.tr("Local Output Dir:"), out_row)

        drive_row = QtWidgets.QHBoxLayout()
        self.predict_drive_folder_edit = QtWidgets.QLineEdit("Annolid/colab")
        self.predict_upload_drive_check = QtWidgets.QCheckBox(self.tr("Sync to Drive"))
        self.predict_upload_drive_check.setChecked(True)
        self.predict_open_browser_check = QtWidgets.QCheckBox(
            self.tr("Open in browser")
        )
        self.predict_open_browser_check.setChecked(True)
        drive_row.addWidget(self.predict_drive_folder_edit, 1)
        drive_row.addWidget(self.predict_upload_drive_check)
        drive_row.addWidget(self.predict_open_browser_check)
        form.addRow(self.tr("Drive Folder:"), drive_row)

        self.predict_install_spec_combo = QtWidgets.QComboBox()
        self.predict_install_spec_combo.setEditable(True)
        self.predict_install_spec_combo.addItem("annolid")
        self.predict_install_spec_combo.addItem(
            "git+https://github.com/healthonrails/annolid.git"
        )
        self.predict_install_spec_combo.setToolTip(
            self.tr(
                "PyPI release (`annolid`) or latest GitHub main (`git+https://...`) installed on the Colab VM."
            )
        )
        form.addRow(self.tr("Colab Pip Spec:"), self.predict_install_spec_combo)

        self.predict_classes_edit = QtWidgets.QLineEdit()
        self.predict_classes_edit.setPlaceholderText(
            self.tr("Optional YOLOE text classes (e.g. mouse,arena)")
        )
        form.addRow(self.tr("Classes (optional):"), self.predict_classes_edit)

        self.predict_extra_args_edit = QtWidgets.QLineEdit()
        self.predict_extra_args_edit.setPlaceholderText(
            self.tr("Optional extra annolid-run predict args")
        )
        form.addRow(self.tr("Extra CLI Args:"), self.predict_extra_args_edit)

        layout.addLayout(form)

        actions = QtWidgets.QHBoxLayout()
        self.run_predict_btn = QtWidgets.QPushButton(self.tr("Run Inference on Colab"))
        self.run_predict_btn.clicked.connect(
            lambda: self._start_predict_job(force_notebook=False)
        )
        self.gen_predict_nb_btn = QtWidgets.QPushButton(
            self.tr("Generate .ipynb Notebook")
        )
        self.gen_predict_nb_btn.clicked.connect(
            lambda: self._start_predict_job(force_notebook=True)
        )
        self._action_buttons.extend([self.run_predict_btn, self.gen_predict_nb_btn])
        actions.addWidget(self.run_predict_btn)
        actions.addWidget(self.gen_predict_nb_btn)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch()
        return page

    def _build_monitor_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(8)

        # Live Run Summary Strip
        summary_frame = QtWidgets.QFrame()
        summary_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        s_grid = QtWidgets.QGridLayout(summary_frame)
        s_grid.setContentsMargins(10, 8, 10, 8)

        self.monitor_stage_label = QtWidgets.QLabel(self.tr("IDLE"))
        st_font = self.monitor_stage_label.font()
        st_font.setBold(True)
        self.monitor_stage_label.setFont(st_font)
        self.monitor_gpu_label = QtWidgets.QLabel("—")
        self.monitor_session_label = QtWidgets.QLabel("—")
        self.monitor_elapsed_label = QtWidgets.QLabel("00:00:00")
        self.monitor_job_label = QtWidgets.QLabel("—")

        s_grid.addWidget(QtWidgets.QLabel(self.tr("Stage:")), 0, 0)
        s_grid.addWidget(self.monitor_stage_label, 0, 1)
        s_grid.addWidget(QtWidgets.QLabel(self.tr("GPU:")), 0, 2)
        s_grid.addWidget(self.monitor_gpu_label, 0, 3)
        s_grid.addWidget(QtWidgets.QLabel(self.tr("Elapsed:")), 0, 4)
        s_grid.addWidget(self.monitor_elapsed_label, 0, 5)

        s_grid.addWidget(QtWidgets.QLabel(self.tr("Job:")), 1, 0)
        s_grid.addWidget(self.monitor_job_label, 1, 1, 1, 3)
        s_grid.addWidget(QtWidgets.QLabel(self.tr("Session:")), 1, 4)
        s_grid.addWidget(self.monitor_session_label, 1, 5)
        s_grid.setColumnStretch(3, 1)

        ctrl_row = QtWidgets.QHBoxLayout()
        self.monitor_open_colab_btn = QtWidgets.QPushButton(
            self.tr("Open Runtime in Colab Browser")
        )
        self.monitor_open_colab_btn.setEnabled(False)
        self.monitor_open_colab_btn.clicked.connect(self._on_monitor_open_colab)
        self.monitor_open_output_btn = QtWidgets.QPushButton(
            self.tr("Open Output Folder")
        )
        self.monitor_open_output_btn.setEnabled(False)
        self.monitor_open_output_btn.clicked.connect(self._on_monitor_open_output)
        self.monitor_open_dashboard_btn = QtWidgets.QPushButton(
            self.tr("Training Dashboard…")
        )
        self.monitor_open_dashboard_btn.setToolTip(
            self.tr(
                "Inspect & compare this run in the unified Training Monitor & Visualization Dashboard"
            )
        )
        self.monitor_open_dashboard_btn.setEnabled(False)
        self.monitor_open_dashboard_btn.clicked.connect(
            self._on_monitor_open_training_dashboard
        )
        self.monitor_cancel_btn = QtWidgets.QPushButton(self.tr("Stop / Cancel Run"))
        self.monitor_cancel_btn.setEnabled(False)
        self.monitor_cancel_btn.clicked.connect(self._cancel_active_task)
        ctrl_row.addWidget(self.monitor_open_colab_btn)
        ctrl_row.addWidget(self.monitor_open_output_btn)
        ctrl_row.addWidget(self.monitor_open_dashboard_btn)
        ctrl_row.addWidget(self.monitor_cancel_btn)
        ctrl_row.addStretch()
        s_grid.addLayout(ctrl_row, 2, 0, 1, 6)
        layout.addWidget(summary_frame)

        # Epoch / Step Progress bar
        ep_row = QtWidgets.QHBoxLayout()
        ep_row.addWidget(QtWidgets.QLabel(self.tr("Step / Epoch:")))
        self.monitor_epoch_bar = QtWidgets.QProgressBar()
        self.monitor_epoch_bar.setRange(0, 100)
        self.monitor_epoch_bar.setValue(0)
        self.monitor_epoch_bar.setFormat(self.tr("Waiting for training/inference…"))
        ep_row.addWidget(self.monitor_epoch_bar, 1)
        layout.addLayout(ep_row)

        # KPI Metric Cards
        kpi_row = QtWidgets.QHBoxLayout()
        self._kpi_values: dict[str, QtWidgets.QLabel] = {}
        for key, caption in (
            ("step", self.tr("Epoch / Step")),
            ("train_loss", self.tr("Train Loss")),
            ("map50", self.tr("Box mAP50")),
            ("mask_map50", self.tr("Mask mAP50")),
            ("gpu_mem", self.tr("GPU Mem")),
            ("artifacts", self.tr("Artifacts")),
        ):
            card = QtWidgets.QFrame()
            card.setFrameShape(QtWidgets.QFrame.StyledPanel)
            cl = QtWidgets.QVBoxLayout(card)
            cl.setContentsMargins(8, 5, 8, 5)
            cap_lbl = QtWidgets.QLabel(caption)
            val_lbl = QtWidgets.QLabel("—")
            vf = val_lbl.font()
            vf.setBold(True)
            vf.setPointSize(max(10, vf.pointSize() + 1))
            val_lbl.setFont(vf)
            cl.addWidget(cap_lbl)
            cl.addWidget(val_lbl)
            self._kpi_values[key] = val_lbl
            kpi_row.addWidget(card, 1)
        layout.addLayout(kpi_row)

        # Live Curves + Plots Preview sub-tabs
        self.monitor_visual_tabs = QtWidgets.QTabWidget()
        self.monitor_chart = _ColabMetricsChartWidget()
        self.monitor_visual_tabs.addTab(
            self.monitor_chart, self.tr("Live Loss && mAP Curves")
        )

        plots_page = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots_page)
        plot_bar = QtWidgets.QHBoxLayout()
        plot_bar.addWidget(QtWidgets.QLabel(self.tr("Output Plot:")))
        self.monitor_plot_combo = QtWidgets.QComboBox()
        self.monitor_plot_combo.currentIndexChanged.connect(
            self._on_monitor_plot_selected
        )
        plot_bar.addWidget(self.monitor_plot_combo, 1)
        plots_layout.addLayout(plot_bar)

        self.monitor_plot_preview = QtWidgets.QLabel(
            self.tr(
                "Training plots (results.png, confusion_matrix.png, val_batch0_pred.jpg) "
                "will appear here once downloaded."
            )
        )
        self.monitor_plot_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.monitor_plot_preview.setMinimumHeight(170)
        plots_layout.addWidget(self.monitor_plot_preview, 1)
        self.monitor_visual_tabs.addTab(plots_page, self.tr("Training Plots Preview"))
        layout.addWidget(self.monitor_visual_tabs)

        # Filterable Live Console Log
        log_bar = QtWidgets.QHBoxLayout()
        log_bar.addWidget(QtWidgets.QLabel(self.tr("Live Log:")))
        self.log_filter_edit = QtWidgets.QLineEdit()
        self.log_filter_edit.setPlaceholderText(
            self.tr("Filter lines (e.g. Epoch, WARN, mAP, error)…")
        )
        self.log_filter_edit.textChanged.connect(self._apply_log_filter)
        self.log_autoscroll_check = QtWidgets.QCheckBox(self.tr("Auto-scroll"))
        self.log_autoscroll_check.setChecked(True)
        copy_log_btn = QtWidgets.QPushButton(self.tr("Copy"))
        copy_log_btn.clicked.connect(
            lambda: self._copy_text(self.log_view.toPlainText())
        )
        save_log_btn = QtWidgets.QPushButton(self.tr("Save Log…"))
        save_log_btn.clicked.connect(self._on_save_log_clicked)
        clear_log_btn = QtWidgets.QPushButton(self.tr("Clear"))
        clear_log_btn.clicked.connect(self._clear_monitor_log)
        log_bar.addWidget(self.log_filter_edit, 1)
        log_bar.addWidget(self.log_autoscroll_check)
        log_bar.addWidget(copy_log_btn)
        log_bar.addWidget(save_log_btn)
        log_bar.addWidget(clear_log_btn)
        layout.addLayout(log_bar)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.log_view.setFont(mono)
        layout.addWidget(self.log_view, 1)

        return page

    def _build_jobs_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(10)

        jobs_group = QtWidgets.QGroupBox(self.tr("Recorded Colab Jobs"))
        jobs_layout = QtWidgets.QVBoxLayout(jobs_group)
        self.jobs_table = QtWidgets.QTableWidget(0, 6)
        self.jobs_table.setHorizontalHeaderLabels(
            [
                self.tr("Job ID"),
                self.tr("Kind"),
                self.tr("Model"),
                self.tr("Backend"),
                self.tr("Status"),
                self.tr("Local Output"),
            ]
        )
        self.jobs_table.horizontalHeader().setStretchLastSection(True)
        self.jobs_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.jobs_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        jobs_layout.addWidget(self.jobs_table)

        job_btns = QtWidgets.QHBoxLayout()
        refresh_jobs_btn = QtWidgets.QPushButton(self.tr("Refresh"))
        refresh_jobs_btn.clicked.connect(self.refresh_jobs)
        inspect_btn = QtWidgets.QPushButton(self.tr("Inspect in Monitor"))
        inspect_btn.clicked.connect(self._on_inspect_selected_job)
        self.pull_job_btn = QtWidgets.QPushButton(self.tr("Pull Results from Drive"))
        self.pull_job_btn.clicked.connect(self._on_pull_selected_job)
        open_colab_btn = QtWidgets.QPushButton(self.tr("Open in Colab"))
        open_colab_btn.clicked.connect(self._on_open_selected_job_url)
        open_out_btn = QtWidgets.QPushButton(self.tr("Open Output Folder"))
        open_out_btn.clicked.connect(self._on_open_selected_job_folder)
        load_preds_btn = QtWidgets.QPushButton(self.tr("Load Predictions in Annolid"))
        load_preds_btn.clicked.connect(self._on_load_selected_predictions)
        self._action_buttons.append(self.pull_job_btn)
        job_btns.addWidget(refresh_jobs_btn)
        job_btns.addWidget(inspect_btn)
        job_btns.addWidget(self.pull_job_btn)
        job_btns.addWidget(open_colab_btn)
        job_btns.addWidget(open_out_btn)
        job_btns.addWidget(load_preds_btn)
        job_btns.addStretch()
        jobs_layout.addLayout(job_btns)
        layout.addWidget(jobs_group, 2)

        tpl_group = QtWidgets.QGroupBox(self.tr("Official Annolid Colab Notebooks"))
        tpl_layout = QtWidgets.QVBoxLayout(tpl_group)
        self.templates_list = QtWidgets.QListWidget()
        for tpl in list_notebook_templates():
            item = QtWidgets.QListWidgetItem(
                f"[{tpl['category'].upper()}] {tpl['title']} — {tpl['description']}"
            )
            item.setData(QtCore.Qt.UserRole, tpl["key"])
            item.setToolTip(tpl["colab_url"])
            self.templates_list.addItem(item)
        if self.templates_list.count() > 0:
            self.templates_list.setCurrentRow(0)
        tpl_layout.addWidget(self.templates_list)

        tpl_btns = QtWidgets.QHBoxLayout()
        open_tpl_btn = QtWidgets.QPushButton(self.tr("Open Selected Notebook in Colab"))
        open_tpl_btn.clicked.connect(self._on_open_selected_template)
        tpl_btns.addWidget(open_tpl_btn)
        tpl_btns.addStretch()
        tpl_layout.addLayout(tpl_btns)
        layout.addWidget(tpl_group, 1)

        return page

    # ------------------------------------------------------------------ #
    # Prefill & Tab Navigation
    # ------------------------------------------------------------------ #

    def select_tab(self, tab_name: str) -> None:
        key = str(tab_name or "").strip().lower()
        if key in self._TAB_KEYS:
            self.tabs.setCurrentIndex(self._TAB_KEYS.index(key))

    def _apply_initial_paths(
        self,
        *,
        dataset_path: Optional[str],
        video_path: Optional[str],
        weights_path: Optional[str],
    ) -> None:
        if dataset_path:
            self.train_dataset_edit.setText(str(dataset_path))
        if video_path:
            self.predict_source_edit.setText(str(video_path))
        if weights_path:
            self.predict_weights_combo.setCurrentText(str(weights_path))

    def prefill_training_config(self, config: dict[str, Any]) -> None:
        backend = str(config.get("backend") or "yolo").strip().lower()
        model_map = {
            "yolo": "yolo",
            "dino_kpseg": "dino_kpseg",
            "maskrcnn": "maskrcnn_detectron2",
            "maskrcnn_detectron2": "maskrcnn_detectron2",
            "keypoint_rcnn": "keypoint_rcnn",
        }
        model_name = model_map.get(backend, backend)
        idx = self.train_model_combo.findData(model_name)
        if idx >= 0:
            self.train_model_combo.setCurrentIndex(idx)
        if config.get("dataset_path"):
            self.train_dataset_edit.setText(str(config["dataset_path"]))
        if config.get("model"):
            self.train_weights_combo.setCurrentText(str(config["model"]))
        if config.get("epochs"):
            self.train_epochs_spin.setValue(max(1, int(config["epochs"])))
        elif config.get("max_iterations"):
            self.train_epochs_spin.setValue(max(1, int(config["max_iterations"])))
        if config.get("batch"):
            self.train_batch_spin.setValue(max(1, int(config["batch"])))
        if config.get("imgsz"):
            self.train_imgsz_spin.setValue(max(64, int(config["imgsz"])))
        if config.get("output_dir"):
            self.train_output_edit.setText(str(config["output_dir"]))
        self.select_tab("train")

    def prefill_inference_config(self, config: dict[str, Any]) -> None:
        model_type = str(config.get("model_type") or "yolo").strip().lower()
        model_map = {
            "yolo": "yolo_labelme",
            "dino_kpseg": "dino_kpseg",
            "detectron2": "maskrcnn_detectron2",
            "maskrcnn_detectron2": "maskrcnn_detectron2",
            "keypoint_rcnn": "keypoint_rcnn",
        }
        model_name = model_map.get(model_type, "yolo_labelme")
        idx = self.predict_model_combo.findData(model_name)
        if idx >= 0:
            self.predict_model_combo.setCurrentIndex(idx)
        videos = list(config.get("videos") or [])
        if videos:
            self.predict_source_edit.setText(str(videos[0]))
        elif config.get("source_path"):
            self.predict_source_edit.setText(str(config["source_path"]))
        if config.get("model_path"):
            self.predict_weights_combo.setCurrentText(str(config["model_path"]))
        if config.get("score_threshold") is not None:
            self.predict_conf_spin.setValue(float(config["score_threshold"]))
        if config.get("iou_threshold") is not None:
            self.predict_iou_spin.setValue(float(config["iou_threshold"]))
        if config.get("imgsz"):
            self.predict_imgsz_spin.setValue(int(config["imgsz"]))
        if config.get("output_dir"):
            self.predict_output_edit.setText(str(config["output_dir"]))
        self.predict_save_csv_check.setChecked(bool(config.get("save_csv", True)))
        self.predict_save_video_check.setChecked(bool(config.get("save_video", False)))
        self.predict_no_tracking_check.setChecked(
            not bool(config.get("enable_tracking", True))
        )
        self.select_tab("predict")

    # ------------------------------------------------------------------ #
    # Live Monitor Helpers
    # ------------------------------------------------------------------ #

    def _reset_monitor(
        self,
        *,
        kind: str,
        model: str,
        session_name: str = "",
        total_steps: Optional[int] = None,
    ) -> None:
        self._monitor_parser.reset(
            kind=kind,
            model=model,
            session_name=session_name,
        )
        if total_steps:
            self._monitor_parser.state.total_steps = int(total_steps)
        self._monitor_started_at = time.monotonic()
        self._monitor_output_dir = None
        self._monitor_plot_paths = []
        self.monitor_plot_combo.clear()
        self.monitor_plot_preview.setPixmap(QtGui.QPixmap())
        self.monitor_plot_preview.setText(
            self.tr("Training plots will appear here once downloaded.")
        )
        self.monitor_open_colab_btn.setEnabled(False)
        self.monitor_open_output_btn.setEnabled(False)
        self.monitor_open_dashboard_btn.setEnabled(False)
        self.monitor_chart.clear()
        self._elapsed_timer.start()
        self._render_monitor_state()

    def _tick_elapsed_timer(self) -> None:
        if self._monitor_started_at is not None:
            self.monitor_elapsed_label.setText(
                _format_duration(time.monotonic() - self._monitor_started_at)
            )

    def _render_monitor_state(self) -> None:
        st = self._monitor_parser.state
        self.monitor_stage_label.setText(st.stage.upper())
        self.monitor_gpu_label.setText(st.gpu_name or "—")
        self.monitor_session_label.setText(st.session_name or "—")
        self.monitor_job_label.setText(
            f"{st.job_id} ({st.model})" if st.job_id else (st.model or "—")
        )
        self.monitor_open_colab_btn.setEnabled(bool(st.colab_url))

        if st.total_steps and st.total_steps > 0:
            pct = max(0, min(100, int(round((st.current_step / st.total_steps) * 100))))
            self.monitor_epoch_bar.setValue(pct)
            self.monitor_epoch_bar.setFormat(
                f"{st.step_unit.title()} {st.current_step} / {st.total_steps} ({pct}%)"
            )
            self._kpi_values["step"].setText(f"{st.current_step}/{st.total_steps}")
        elif st.current_step > 0:
            self.monitor_epoch_bar.setValue(100 if st.stage == "completed" else 50)
            self.monitor_epoch_bar.setFormat(
                f"{st.step_unit.title()} {st.current_step}"
            )
            self._kpi_values["step"].setText(str(st.current_step))
        else:
            self.monitor_epoch_bar.setValue(100 if st.stage == "completed" else 0)
            self.monitor_epoch_bar.setFormat(
                self.tr("Completed")
                if st.stage == "completed"
                else self.tr("Waiting for training/inference…")
            )
            self._kpi_values["step"].setText("—")

        latest = st.latest_metrics
        self._kpi_values["train_loss"].setText(
            f"{latest.train_loss:.4f}"
            if latest and latest.train_loss is not None
            else "—"
        )
        self._kpi_values["map50"].setText(
            f"{latest.map50:.4f}" if latest and latest.map50 is not None else "—"
        )
        self._kpi_values["mask_map50"].setText(
            f"{latest.mask_map50:.4f}"
            if latest and latest.mask_map50 is not None
            else "—"
        )
        self._kpi_values["gpu_mem"].setText(
            (latest.gpu_mem if latest and latest.gpu_mem else st.gpu_mem_used) or "—"
        )
        self._kpi_values["artifacts"].setText(
            str(st.artifact_count) if st.artifact_count else "—"
        )
        self.monitor_chart.set_points(st.history)

    def _populate_monitor_outputs(self, output_dir: Optional[str]) -> None:
        if not output_dir:
            return
        root = Path(output_dir)
        if not root.is_dir():
            return
        self._monitor_output_dir = root
        self.monitor_open_output_btn.setEnabled(True)
        self.monitor_open_dashboard_btn.setEnabled(True)

        # Replay full results.csv into live chart if available
        for csv_cand in sorted(root.rglob("results.csv")):
            pts = load_metrics_from_results_csv(csv_cand)
            if pts:
                self._monitor_parser.state.history = pts
                self._monitor_parser.state.latest_metrics = pts[-1]
                self._monitor_parser.state.current_step = pts[-1].step
                self._monitor_parser.state.total_steps = pts[-1].total_steps
                self._render_monitor_state()
                break

        # Collect generated plot images
        preferred_order = (
            "results.png",
            "confusion_matrix_normalized.png",
            "confusion_matrix.png",
            "BoxPR_curve.png",
            "MaskPR_curve.png",
            "val_batch0_pred.jpg",
            "train_batch0.jpg",
        )
        imgs = [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        imgs.sort(
            key=lambda p: (
                preferred_order.index(p.name)
                if p.name in preferred_order
                else len(preferred_order),
                p.name,
            )
        )
        self._monitor_plot_paths = imgs
        self.monitor_plot_combo.blockSignals(True)
        self.monitor_plot_combo.clear()
        for p in imgs:
            self.monitor_plot_combo.addItem(p.relative_to(root).as_posix(), str(p))
        self.monitor_plot_combo.blockSignals(False)
        if imgs:
            self._on_monitor_plot_selected(0)

    def _on_monitor_plot_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._monitor_plot_paths):
            return
        img_path = self._monitor_plot_paths[index]
        pix = QtGui.QPixmap(str(img_path))
        if pix.isNull():
            self.monitor_plot_preview.setText(f"Unable to load image: {img_path.name}")
            return
        target_w = max(480, self.monitor_plot_preview.width() - 12)
        target_h = max(170, self.monitor_plot_preview.height() - 8)
        scaled = pix.scaled(
            target_w,
            target_h,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.monitor_plot_preview.setPixmap(scaled)

    def _on_monitor_open_colab(self) -> None:
        url = self._monitor_parser.state.colab_url
        if url:
            webbrowser.open(url)

    def _on_monitor_open_output(self) -> None:
        if self._monitor_output_dir and self._monitor_output_dir.exists():
            self._open_local_folder(self._monitor_output_dir)

    def _on_monitor_open_training_dashboard(self) -> None:
        if self._monitor_output_dir and self._monitor_output_dir.exists():
            self.openTrainingDashboardRequested.emit(str(self._monitor_output_dir))

    def inspect_job_in_monitor(self, rec: ColabJobRecord) -> None:
        """Load a recorded Colab job's metrics, plots, and links into the Live Monitor tab."""
        self._elapsed_timer.stop()
        self._monitor_started_at = None
        self._monitor_parser.reset(
            job_id=rec.job_id,
            kind=rec.kind,
            model=rec.model,
            session_name=rec.session_name,
        )
        st = self._monitor_parser.state
        st.stage = rec.status or "completed"
        st.colab_url = rec.colab_url
        st.artifact_count = len(rec.downloaded_files)
        elapsed = (rec.result_summary or {}).get("elapsed_seconds")
        if elapsed is not None:
            self.monitor_elapsed_label.setText(_format_duration(float(elapsed)))
        else:
            self.monitor_elapsed_label.setText("—")
        self._render_monitor_state()
        self._populate_monitor_outputs(rec.local_output_dir)
        self.select_tab("monitor")

    def _clear_monitor_log(self) -> None:
        self._monitor_raw_lines.clear()
        self.log_view.clear()

    def _apply_log_filter(self, query: str) -> None:
        needle = str(query or "").strip().lower()
        if not needle:
            self.log_view.setPlainText("\n".join(self._monitor_raw_lines))
        else:
            matched = [
                line for line in self._monitor_raw_lines if needle in line.lower()
            ]
            self.log_view.setPlainText("\n".join(matched))
        if self.log_autoscroll_check.isChecked():
            sb = self.log_view.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())

    def _on_save_log_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Save Colab Log"),
            "colab_run.log",
            "Log Files (*.log *.txt);;All Files (*)",
        )
        if path:
            Path(path).write_text(
                "\n".join(self._monitor_raw_lines) + "\n", encoding="utf-8"
            )
            self._append_log(f"Saved log to {path}")

    # ------------------------------------------------------------------ #
    # Service sync & Background Task Runner
    # ------------------------------------------------------------------ #

    def _current_service(self) -> ColabWorkflowService:
        creds = self.credentials_edit.text().strip()
        tok = self.token_edit.text().strip()
        if creds and creds != self._service.auth.credentials_file:
            self._service.auth.credentials_file = creds
        if tok and tok != self._service.auth.token_file:
            self._service.auth.token_file = tok
        return self._service

    def _set_busy(self, busy: bool, label: str = "") -> None:
        for btn in self._action_buttons:
            btn.setEnabled(not busy)
        if hasattr(self, "install_cli_btn"):
            self.install_cli_btn.setEnabled((not busy) and (not self._cli_installed))
        self.cancel_btn.setEnabled(busy)
        if hasattr(self, "monitor_cancel_btn"):
            self.monitor_cancel_btn.setEnabled(busy)
        if label:
            self.status_label.setText(label)
        elif not busy:
            self.status_label.setText(self.tr("Ready"))

    def _append_log(self, text: str) -> None:
        if not text:
            return
        line = str(text)
        self._monitor_raw_lines.append(line)
        if self._monitor_parser.feed_line(line):
            self._render_monitor_state()
        needle = self.log_filter_edit.text().strip().lower()
        if not needle or needle in line.lower():
            self.log_view.appendPlainText(line)
            if self.log_autoscroll_check.isChecked():
                sb = self.log_view.verticalScrollBar()
                if sb is not None:
                    sb.setValue(sb.maximum())

    def _schedule_after_cleanup(self, fn: Callable[[], None]) -> None:
        if self._worker_thread is None or not self._worker_thread.isRunning():
            QtCore.QTimer.singleShot(0, fn)
        else:
            self._pending_after_cleanup = fn
            self._run_pending_after_cleanup = True

    def _run_background(
        self,
        task_name: str,
        fn: Callable[..., Any],
        *,
        status_text: str,
        retry_factory: Optional[Callable[[Optional[str]], None]] = None,
    ) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Task Running"),
                self.tr(
                    "Another Colab task is currently running. Please wait or cancel it."
                ),
            )
            return

        self._active_retry_factory = retry_factory
        self._cancel_event = threading.Event()
        self.progress_bar.setValue(0)
        self._set_busy(True, status_text)
        self._append_log(f"--- {status_text} ---")

        thread = QtCore.QThread(self)
        worker = _ColabTaskWorker(task_name, fn, cancel_event=self._cancel_event)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.log_emitted.connect(self._append_log)
        worker.progress_emitted.connect(self._on_worker_progress)
        worker.auth_code_requested.connect(self._on_worker_auth_code_requested)
        worker.finished_ok.connect(self._on_worker_ok)
        worker.finished_auth_required.connect(self._on_worker_auth_required)
        worker.finished_error.connect(self._on_worker_error)
        worker.finished_ok.connect(thread.quit)
        worker.finished_auth_required.connect(thread.quit)
        worker.finished_error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_cleaned)

        self._worker = worker
        self._worker_thread = thread
        thread.start()

    @QtCore.Slot(int, str)
    def _on_worker_progress(self, pct: int, label: str) -> None:
        self.progress_bar.setValue(max(0, min(100, int(pct))))
        if label:
            self.status_label.setText(label)

    @QtCore.Slot(str, object, object)
    def _on_worker_auth_code_requested(
        self, auth_url: str, holder: object, done_event: object
    ) -> None:
        code_val: Optional[str] = None
        try:
            prompt = self.tr(
                "Approve Google Colab access in the browser window that just opened, "
                "then paste the authorization code shown by Google below:"
            )
            if auth_url:
                prompt += f"\n\nSign-in URL:\n{auth_url}"
            text, ok = QtWidgets.QInputDialog.getText(
                self,
                self.tr("Google Colab Authorization Code"),
                prompt,
            )
            if ok and str(text or "").strip():
                code_val = str(text).strip()
        finally:
            if isinstance(holder, dict):
                holder["code"] = code_val
            if isinstance(done_event, threading.Event):
                done_event.set()

    @QtCore.Slot(str, object)
    def _on_worker_ok(self, task_name: str, result: object) -> None:
        self._active_retry_factory = None
        self._set_busy(False, self.tr("Done"))
        self.progress_bar.setValue(100)
        if task_name in {"login", "logout", "install_cli"}:
            self.refresh_auth_status()
            self._append_log(f"Authentication update complete ({task_name}).")
            if task_name == "login" and self._pending_after_cleanup is not None:
                pending = self._pending_after_cleanup
                self._pending_after_cleanup = None
                self._schedule_after_cleanup(pending)
            elif task_name == "login":
                self._schedule_after_cleanup(self._on_refresh_account_clicked)
            elif task_name == "logout":
                self._apply_account_info(
                    {
                        "signed_in": False,
                        "email": "",
                        "provider": "",
                        "detail": "Signed out. Click Sign In with Google to connect a personal account.",
                    }
                )
                if self._pending_after_cleanup is not None:
                    pending = self._pending_after_cleanup
                    self._pending_after_cleanup = None
                    self._schedule_after_cleanup(pending)
        elif task_name == "account_info" and isinstance(result, dict):
            self._apply_account_info(result)
        elif task_name == "sessions":
            self._populate_sessions_table(list(result or []))
        elif task_name == "specs":
            specs = list(result or [])
            if not specs:
                self._append_log("No Colab runtime specs returned.")
            for spec in specs:
                flag = "eligible" if getattr(spec, "eligible", False) else "ineligible"
                self._append_log(
                    f"  {spec.variant} | {spec.accelerator} | {spec.shape} [{flag}]"
                )
        elif task_name == "stop_session":
            self._append_log("Runtime stopped.")
            self._schedule_after_cleanup(self._on_refresh_sessions)
        elif task_name == "doctor" and isinstance(result, dict):
            self._append_log(
                f"Diagnostics ({'live' if result.get('live') else 'offline'}): "
                f"recommended={result.get('recommended_backend')} | ok={result.get('ok')}"
            )
            for check in list(result.get("checks") or []):
                badge = (
                    "OK"
                    if check.get("ok")
                    else str(check.get("severity") or "WARN").upper()
                )
                self._append_log(
                    f"  [{badge}] {check.get('name')}: {check.get('detail')}"
                )
        elif task_name in {"train", "predict", "pull"} and isinstance(
            result, ColabJobRecord
        ):
            self._elapsed_timer.stop()
            self.refresh_jobs()
            st = self._monitor_parser.state
            st.job_id = result.job_id
            st.stage = result.status or "completed"
            if result.session_name:
                st.session_name = result.session_name
            if result.colab_url:
                st.colab_url = result.colab_url
            if result.downloaded_files:
                st.artifact_count = len(result.downloaded_files)
            self._render_monitor_state()
            self._populate_monitor_outputs(result.local_output_dir)

            self._append_log(
                f"Job {result.job_id} finished with status '{result.status}'."
            )
            if result.colab_url:
                self._append_log(f"Colab URL: {result.colab_url}")
            if result.local_output_dir and result.downloaded_files:
                self._append_log(
                    f"Downloaded {len(result.downloaded_files)} file(s) -> {result.local_output_dir}"
                )
            if task_name == "train" and result.status == "completed":
                ckpt = self._find_best_checkpoint(result.downloaded_files)
                if ckpt:
                    self.predict_weights_combo.setCurrentText(ckpt)
                    self._append_log(f"Auto-filled Inference tab weights -> {ckpt}")
                if self.train_keep_rt_check.isChecked() and result.session_name:
                    self.predict_session_edit.setText(result.session_name)

    @QtCore.Slot(str, str)
    def _on_worker_auth_required(self, task_name: str, message: str) -> None:
        self._elapsed_timer.stop()
        retry_factory = self._active_retry_factory
        self._active_retry_factory = None
        self._set_busy(False, self.tr("Sign-in required"))
        self._append_log(f"[AUTH REQUIRED] ({task_name}) {message}")
        self.refresh_auth_status()
        choice = self._prompt_sign_in(
            reason=message,
            allow_notebook_fallback=(task_name in {"train", "predict"}),
        )
        if choice is None:
            return
        if choice == "notebook":
            if retry_factory is not None:
                self._schedule_after_cleanup(lambda: retry_factory("notebook"))
            return
        backend_for_method = {
            "personal": "cli",
            "colab-cli": "cli",
            "oauth2": "api",
            "adc": "cli" if self._current_service().auth.find_colab_cli() else "api",
            "drive": "notebook",
        }.get(choice, "auto")
        resume_cb = (
            (lambda b=backend_for_method: retry_factory(b))
            if retry_factory is not None
            else None
        )
        self._schedule_after_cleanup(
            lambda m=choice, cb=resume_cb: self._start_sign_in_and_resume(
                method=m, resume_fn=cb
            )
        )

    @QtCore.Slot(str, str)
    def _on_worker_error(self, task_name: str, message: str) -> None:
        self._elapsed_timer.stop()
        self._active_retry_factory = None
        self._pending_after_cleanup = None
        self._run_pending_after_cleanup = False
        if task_name in {"train", "predict"}:
            self._monitor_parser.state.stage = (
                "cancelled" if "cancelled" in message.lower() else "failed"
            )
            self._render_monitor_state()
        self._set_busy(False, self.tr("Error"))
        self._append_log(f"[ERROR] ({task_name}) {message}")
        self.refresh_jobs()
        QtWidgets.QMessageBox.warning(
            self,
            self.tr("Google Colab Error"),
            message,
        )

    @QtCore.Slot()
    def _on_thread_cleaned(self) -> None:
        thread = self._worker_thread
        self._worker = None
        self._worker_thread = None
        if thread is not None:
            thread.deleteLater()
        if self._run_pending_after_cleanup and self._pending_after_cleanup is not None:
            pending = self._pending_after_cleanup
            self._pending_after_cleanup = None
            self._run_pending_after_cleanup = False
            QtCore.QTimer.singleShot(0, pending)

    def _cancel_active_task(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._append_log("Cancellation requested…")
            self.status_label.setText(self.tr("Cancelling…"))

    # ------------------------------------------------------------------ #
    # Personal Account & Runtimes Actions
    # ------------------------------------------------------------------ #

    def _apply_account_info(self, info: dict[str, Any]) -> None:
        self._account_info = dict(info or {})
        signed_in = bool(self._account_info.get("signed_in"))
        email = str(self._account_info.get("email") or "").strip()
        provider = str(self._account_info.get("provider") or "").strip()
        balance = str(self._account_info.get("compute_balance") or "").strip()
        rate = str(self._account_info.get("usage_rate") or "").strip()
        expires = str(self._account_info.get("expires_in") or "").strip()
        active = self._account_info.get("active_assignments")

        if signed_in:
            self.header_account_dot.setStyleSheet("color: #16a34a;")
            who = email or self.tr("Connected Google Account")
            self.header_account_label.setText(self.tr("Signed in: %s") % who)
            usage_bits = [b for b in (balance, rate) if b]
            if active is not None:
                usage_bits.append(f"{active} active")
            self.header_usage_label.setText(
                " · ".join(usage_bits) if usage_bits else (provider or "")
            )
            self.account_email_label.setText(who)
            self.account_provider_label.setText(provider or "—")
            self.account_balance_label.setText(
                " · ".join(b for b in (balance, rate) if b) or "—"
            )
            self.account_expiry_label.setText(
                f"Expires in {expires}" if expires else "Active"
            )
        else:
            self.header_account_dot.setStyleSheet("color: #d97706;")
            self.header_account_label.setText(self.tr("Not signed in to Google Colab"))
            self.header_usage_label.setText(
                self.tr("Click Sign In with Google to use your personal Colab account")
            )
            self.account_email_label.setText(self.tr("Not signed in"))
            self.account_provider_label.setText(provider or "—")
            self.account_balance_label.setText("—")
            self.account_expiry_label.setText("—")

    def refresh_auth_status(self) -> dict[str, Any]:
        svc = self._current_service()
        status = svc.auth_status()
        self.auth_summary_label.setText(
            f"Recommended backend: {status['recommended_backend']} — {status['summary']}"
        )
        cli = status["colab_cli"]
        self._cli_installed = bool(cli.get("installed"))
        self.cli_status_label.setText(
            f"Installed ({cli.get('version') or cli.get('binary')})"
            if self._cli_installed
            else "Not installed (click Install Colab CLI or run `uv tool install google-colab-cli`)"
        )
        if hasattr(self, "install_cli_btn"):
            self.install_cli_btn.setText(
                self.tr("CLI Installed")
                if self._cli_installed
                else self.tr("Install Colab CLI")
            )
            self.install_cli_btn.setEnabled(not self._cli_installed)
        oc = status["oauth_colab"]
        self.oauth_colab_status_label.setText(
            "Ready"
            if oc.get("ready")
            else f"Not ready ({oc.get('reason') or 'sign in required'})"
        )
        od = status["oauth_drive"]
        self.oauth_drive_status_label.setText(
            "Ready"
            if od.get("ready")
            else f"Not ready ({od.get('reason') or 'sign in required'})"
        )
        adc = status["adc"]
        self.adc_status_label.setText(
            f"Ready ({adc.get('credential_type') or 'ADC'})"
            if adc.get("ready")
            else f"Not ready (gcloud={'installed' if adc.get('gcloud_installed') else 'missing'})"
        )

        # Update header banner from local state if we haven't queried whoami yet
        if not self._account_info:
            if status["cli_ready"] or status["api_ready"]:
                self.header_account_dot.setStyleSheet("color: #2563eb;")
                self.header_account_label.setText(
                    self.tr("Colab ready (%s)") % status["recommended_backend"]
                )
                self.header_usage_label.setText(
                    self.tr(
                        "Click Check Account to view your Google email & compute balance"
                    )
                )
                self.account_email_label.setText(
                    self.tr("Credentials detected — click Check Account")
                )
            else:
                self._apply_account_info({"signed_in": False})
        return status

    def _on_refresh_account_clicked(self) -> None:
        svc = self._current_service()

        def _task(**_kwargs: Any) -> dict[str, Any]:
            return svc.get_account_info()

        self._run_background(
            "account_info",
            _task,
            status_text=self.tr("Checking personal Google Colab account…"),
        )

    def _on_personal_sign_in_clicked(self) -> None:
        self._start_sign_in_and_resume(method="personal", resume_fn=None)

    def _on_switch_account_clicked(self) -> None:
        svc = self._current_service()

        def _task(log_callback=None, **_kwargs: Any) -> dict[str, Any]:
            res = svc.logout(include_colab_cli=True, clear_adc=True)
            if log_callback:
                log_callback(
                    "Cleared saved Google Colab credentials. Opening browser for new account sign-in…"
                )
            return res

        self._pending_after_cleanup = lambda: self._start_sign_in_and_resume(
            method="personal", resume_fn=None
        )
        self._run_pending_after_cleanup = False
        self._run_background(
            "logout",
            _task,
            status_text=self.tr("Signing out previous account…"),
        )

    def _prompt_sign_in(
        self,
        *,
        reason: str,
        allow_notebook_fallback: bool = True,
    ) -> Optional[str]:
        status = self._current_service().auth_status()
        dlg = ColabSignInDialog(
            reason=reason,
            auth_status=status,
            allow_notebook_fallback=allow_notebook_fallback,
            parent=self,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return dlg.selected_action()

    def _start_sign_in_and_resume(
        self,
        *,
        method: str,
        resume_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        svc = self._current_service()
        self._pending_after_cleanup = resume_fn
        self._run_pending_after_cleanup = False

        def _task(
            log_callback=None, auth_code_callback=None, **_kwargs: Any
        ) -> dict[str, Any]:
            if log_callback:
                log_callback(f"Starting Google Colab sign-in (method={method})...")
            return svc.login(
                method=method,
                include_drive=True,
                open_browser=True,
                install_cli_if_missing=(method in {"personal", "colab-cli"}),
                log_callback=log_callback,
                auth_code_callback=auth_code_callback,
            )

        self._run_background("login", _task, status_text=self.tr("Signing in…"))

    def _on_install_cli_clicked(self) -> None:
        svc = self._current_service()

        def _task(log_callback=None, **_kwargs: Any) -> str:
            return svc.install_colab_cli(log_callback=log_callback)

        self._run_background(
            "install_cli",
            _task,
            status_text=self.tr("Installing google-colab-cli…"),
        )

    def _on_login_clicked(self) -> None:
        method = str(self.auth_method_combo.currentData() or "personal")
        self._start_sign_in_and_resume(method=method, resume_fn=None)

    def _on_logout_clicked(self) -> None:
        svc = self._current_service()

        def _task(log_callback=None, **_kwargs: Any) -> dict[str, Any]:
            res = svc.logout(include_colab_cli=True, clear_adc=False)
            if log_callback:
                log_callback(
                    f"Removed token: {res['token_file']}"
                    if res["removed_token"]
                    else "Cleared Colab CLI / OAuth session state."
                )
            return res

        self._run_background("logout", _task, status_text=self.tr("Signing out…"))

    def _on_refresh_sessions(self, backend_override: Optional[str] = None) -> None:
        svc = self._current_service()
        backend = str(backend_override or self.backend_combo.currentData() or "auto")

        def _task(**_kwargs: Any) -> list[ColabRuntimeInfo]:
            return svc.list_sessions(backend=backend)

        self._run_background(
            "sessions",
            _task,
            status_text=self.tr("Listing runtimes…"),
            retry_factory=lambda b: self._on_refresh_sessions(backend_override=b),
        )

    def _on_open_selected_session_url(self) -> None:
        row = self.sessions_table.currentRow()
        if row < 0 or row >= len(self._sessions_cache):
            QtWidgets.QMessageBox.information(
                self, self.tr("Open Runtime"), self.tr("Select a runtime row first.")
            )
            return
        rt = self._sessions_cache[row]
        if rt.browser_url:
            webbrowser.open(rt.browser_url)
            return
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Open Runtime"),
            self.tr("Selected runtime does not expose a browser URL."),
        )

    def _on_check_specs(self) -> None:
        svc = self._current_service()

        def _task(**_kwargs: Any) -> Any:
            return svc.list_runtime_specs()

        self._run_background(
            "specs",
            _task,
            status_text=self.tr("Checking eligible GPUs…"),
            retry_factory=lambda _b: self._on_check_specs(),
        )

    def _on_run_diagnostics(self, *, live: bool = False) -> None:
        svc = self._current_service()

        def _task(**_kwargs: Any) -> dict[str, Any]:
            return svc.run_diagnostics(live=live)

        self._run_background(
            "doctor",
            _task,
            status_text=self.tr("Testing live Colab/Drive connections…")
            if live
            else self.tr("Running Colab diagnostics…"),
        )

    def _populate_sessions_table(self, sessions: list[ColabRuntimeInfo]) -> None:
        self._sessions_cache = list(sessions)
        self.sessions_table.setRowCount(len(sessions))
        for row_idx, rt in enumerate(sessions):
            vals = [
                rt.name or rt.session_name,
                rt.backend,
                rt.accelerator or "-",
                rt.state,
                rt.browser_url or rt.proxy_url or "",
            ]
            for col_idx, val in enumerate(vals):
                self.sessions_table.setItem(
                    row_idx, col_idx, QtWidgets.QTableWidgetItem(str(val))
                )

    def _on_stop_selected_session(self) -> None:
        row = self.sessions_table.currentRow()
        if row < 0 or row >= len(self._sessions_cache):
            QtWidgets.QMessageBox.information(
                self, self.tr("Stop Runtime"), self.tr("Select a runtime row first.")
            )
            return
        rt = self._sessions_cache[row]
        target = rt.name or rt.session_name
        svc = self._current_service()

        def _task(**_kwargs: Any) -> None:
            svc.stop_session(target, backend=rt.backend or "auto")

        self._run_background(
            "stop_session", _task, status_text=self.tr("Stopping runtime…")
        )

    # ------------------------------------------------------------------ #
    # Train & Predict Actions
    # ------------------------------------------------------------------ #

    def _ensure_direct_auth_or_prompt(
        self,
        *,
        backend: str,
        resume_with_backend: Callable[[str], None],
    ) -> Optional[str]:
        b = str(backend or "auto").strip().lower()
        if b == "notebook":
            return "notebook"
        status = self._current_service().auth_status()
        needs_auth = (
            (b == "auto" and not (status["cli_ready"] or status["api_ready"]))
            or (b == "cli" and not status["cli_ready"])
            or (b == "api" and not status["api_ready"])
        )
        if not needs_auth:
            return b
        choice = self._prompt_sign_in(
            reason=status["summary"],
            allow_notebook_fallback=True,
        )
        if choice is None:
            return None
        if choice == "notebook":
            return "notebook"
        backend_for_method = {
            "personal": "cli",
            "colab-cli": "cli",
            "oauth2": "api",
            "adc": "cli" if self._current_service().auth.find_colab_cli() else "api",
            "drive": "notebook",
        }.get(choice, "auto")
        self._start_sign_in_and_resume(
            method=choice,
            resume_fn=lambda target_b=backend_for_method: resume_with_backend(target_b),
        )
        return None

    def _launch_train_request(
        self,
        req: ColabTrainRequest,
        *,
        skip_preflight_prompt: bool = False,
    ) -> None:
        if not skip_preflight_prompt:
            resolved_backend = self._ensure_direct_auth_or_prompt(
                backend=req.backend,
                resume_with_backend=lambda b: self._launch_train_request(
                    dataclasses.replace(req, backend=b),
                    skip_preflight_prompt=True,
                ),
            )
            if resolved_backend is None:
                return
            if resolved_backend != req.backend:
                req = dataclasses.replace(req, backend=resolved_backend)

        self._reset_monitor(
            kind="train",
            model=req.model,
            session_name=req.session_name or "",
            total_steps=req.epochs or req.max_iterations,
        )
        if req.backend != "notebook":
            self.select_tab("monitor")

        svc = self._current_service()

        def _task(
            log_callback=None, progress_callback=None, cancel_event=None, **_kwargs: Any
        ):
            return svc.run_train(
                req,
                log_callback=log_callback,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

        self._run_background(
            "train",
            _task,
            status_text=self.tr("Generating Colab training notebook…")
            if req.backend == "notebook"
            else self.tr("Running Colab training…"),
            retry_factory=lambda b: self._launch_train_request(
                dataclasses.replace(req, backend=b or req.backend),
                skip_preflight_prompt=True,
            ),
        )

    def _start_train_job(self, *, force_notebook: bool) -> None:
        dataset_path = self.train_dataset_edit.text().strip()
        if not dataset_path:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Dataset Required"),
                self.tr(
                    "Please select a YOLO data.yaml, spec.yaml, COCO folder, or .zip dataset."
                ),
            )
            return
        extra_tokens = tuple(
            shlex.split(self.train_extra_args_edit.text().strip())
            if self.train_extra_args_edit.text().strip()
            else ()
        )
        model_name = str(self.train_model_combo.currentData() or "yolo")
        epochs_val = int(self.train_epochs_spin.value())
        req = ColabTrainRequest(
            model=model_name,
            dataset_path=dataset_path,
            weights=self.train_weights_combo.currentText().strip() or None,
            epochs=epochs_val
            if model_name not in {"maskrcnn_detectron2", "keypoint_rcnn"}
            else None,
            max_iterations=epochs_val
            if model_name in {"maskrcnn_detectron2", "keypoint_rcnn"}
            else None,
            batch=int(self.train_batch_spin.value()),
            imgsz=int(self.train_imgsz_spin.value()),
            preset=str(self.train_preset_combo.currentData() or "") or None,
            plots=bool(self.train_plots_check.isChecked()),
            extra_args=extra_tokens,
            accelerator=str(self.train_gpu_combo.currentData() or "T4"),
            high_mem=bool(self.train_high_mem_check.isChecked()),
            backend="notebook"
            if force_notebook
            else str(self.backend_combo.currentData() or "auto"),
            session_name=self.train_session_edit.text().strip() or None,
            keep_runtime=bool(self.train_keep_rt_check.isChecked()),
            output_dir=self.train_output_edit.text().strip() or None,
            drive_folder=self.train_drive_folder_edit.text().strip() or "Annolid/colab",
            upload_to_drive=bool(self.train_upload_drive_check.isChecked()),
            open_browser=bool(self.train_open_browser_check.isChecked()),
            annolid_install_spec=self.train_install_spec_combo.currentText().strip()
            or "annolid",
        )
        self._launch_train_request(req, skip_preflight_prompt=force_notebook)

    def _launch_predict_request(
        self,
        req: ColabPredictRequest,
        *,
        skip_preflight_prompt: bool = False,
    ) -> None:
        if not skip_preflight_prompt:
            resolved_backend = self._ensure_direct_auth_or_prompt(
                backend=req.backend,
                resume_with_backend=lambda b: self._launch_predict_request(
                    dataclasses.replace(req, backend=b),
                    skip_preflight_prompt=True,
                ),
            )
            if resolved_backend is None:
                return
            if resolved_backend != req.backend:
                req = dataclasses.replace(req, backend=resolved_backend)

        self._reset_monitor(
            kind="predict",
            model=req.model,
            session_name=req.session_name or "",
        )
        if req.backend != "notebook":
            self.select_tab("monitor")

        svc = self._current_service()

        def _task(
            log_callback=None, progress_callback=None, cancel_event=None, **_kwargs: Any
        ):
            return svc.run_predict(
                req,
                log_callback=log_callback,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

        self._run_background(
            "predict",
            _task,
            status_text=self.tr("Generating Colab inference notebook…")
            if req.backend == "notebook"
            else self.tr("Running Colab inference…"),
            retry_factory=lambda b: self._launch_predict_request(
                dataclasses.replace(req, backend=b or req.backend),
                skip_preflight_prompt=True,
            ),
        )

    def _start_predict_job(self, *, force_notebook: bool) -> None:
        source_path = self.predict_source_edit.text().strip()
        if not source_path:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Input Required"),
                self.tr("Please select a video file, image file, or image folder."),
            )
            return
        extra_tokens = tuple(
            shlex.split(self.predict_extra_args_edit.text().strip())
            if self.predict_extra_args_edit.text().strip()
            else ()
        )
        imgsz_val = int(self.predict_imgsz_spin.value())
        req = ColabPredictRequest(
            model=str(self.predict_model_combo.currentData() or "yolo_labelme"),
            source_path=source_path,
            weights=self.predict_weights_combo.currentText().strip() or None,
            conf=float(self.predict_conf_spin.value()),
            iou=float(self.predict_iou_spin.value()),
            imgsz=imgsz_val if imgsz_val > 0 else None,
            save_csv=bool(self.predict_save_csv_check.isChecked()),
            save_video=bool(self.predict_save_video_check.isChecked()),
            no_tracking=bool(self.predict_no_tracking_check.isChecked()),
            classes=self.predict_classes_edit.text().strip() or None,
            extra_args=extra_tokens,
            accelerator=str(self.predict_gpu_combo.currentData() or "T4"),
            high_mem=bool(self.predict_high_mem_check.isChecked()),
            backend="notebook"
            if force_notebook
            else str(self.backend_combo.currentData() or "auto"),
            session_name=self.predict_session_edit.text().strip() or None,
            keep_runtime=bool(self.predict_keep_rt_check.isChecked()),
            output_dir=self.predict_output_edit.text().strip() or None,
            drive_folder=self.predict_drive_folder_edit.text().strip()
            or "Annolid/colab",
            upload_to_drive=bool(self.predict_upload_drive_check.isChecked()),
            open_browser=bool(self.predict_open_browser_check.isChecked()),
            annolid_install_spec=self.predict_install_spec_combo.currentText().strip()
            or "annolid",
        )
        self._launch_predict_request(req, skip_preflight_prompt=force_notebook)

    # ------------------------------------------------------------------ #
    # Jobs & Templates Actions
    # ------------------------------------------------------------------ #

    def refresh_jobs(self) -> list[ColabJobRecord]:
        jobs = self._current_service().list_jobs(limit=50)
        self._jobs_cache = list(jobs)
        self.jobs_table.setRowCount(len(jobs))
        for row_idx, rec in enumerate(jobs):
            vals = [
                rec.job_id,
                rec.kind,
                rec.model,
                rec.backend,
                rec.status,
                rec.local_output_dir or rec.notebook_path,
            ]
            for col_idx, val in enumerate(vals):
                self.jobs_table.setItem(
                    row_idx, col_idx, QtWidgets.QTableWidgetItem(str(val))
                )
        return jobs

    def _selected_job(self) -> Optional[ColabJobRecord]:
        row = self.jobs_table.currentRow()
        if row < 0 or row >= len(self._jobs_cache):
            return None
        return self._jobs_cache[row]

    def _on_inspect_selected_job(self) -> None:
        rec = self._selected_job()
        if rec is None:
            QtWidgets.QMessageBox.information(
                self, self.tr("Inspect Job"), self.tr("Select a Colab job row first.")
            )
            return
        self.inspect_job_in_monitor(rec)

    def _on_pull_selected_job(self) -> None:
        rec = self._selected_job()
        if rec is None:
            QtWidgets.QMessageBox.information(
                self, self.tr("Pull Results"), self.tr("Select a Colab job row first.")
            )
            return
        svc = self._current_service()

        def _task(log_callback=None, **_kwargs: Any) -> ColabJobRecord:
            return svc.pull_results(rec.job_id, log_callback=log_callback)

        self._run_background(
            "pull",
            _task,
            status_text=self.tr("Pulling results for %s…") % rec.job_id,
        )

    def _on_open_selected_job_url(self) -> None:
        rec = self._selected_job()
        if rec is None or not rec.colab_url:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Open in Colab"),
                self.tr("Selected job does not have a Colab URL."),
            )
            return
        webbrowser.open(rec.colab_url)

    def _on_open_selected_job_folder(self) -> None:
        rec = self._selected_job()
        if rec is None:
            return
        target_str = rec.local_output_dir or rec.local_staging_dir
        if not target_str or not Path(target_str).exists():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Open Output Folder"),
                self.tr("Local output directory does not exist yet."),
            )
            return
        self._open_local_folder(Path(target_str))

    def _on_load_selected_predictions(self) -> None:
        rec = self._selected_job()
        if rec is None or not rec.local_output_dir:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Load Predictions"),
                self.tr("Select a completed job with downloaded local outputs first."),
            )
            return
        out_root = Path(rec.local_output_dir)
        preds_dir = out_root / "predictions"
        target = preds_dir if preds_dir.is_dir() else out_root
        if not target.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Folder Not Found"),
                self.tr("Output directory not found: %s") % str(target),
            )
            return
        self.openPredictionsRequested.emit(str(target))

    def _on_open_selected_template(self) -> None:
        item = self.templates_list.currentItem()
        if item is None:
            return
        key = str(item.data(QtCore.Qt.UserRole) or "")
        tpl = get_notebook_template(key)
        if tpl is not None:
            webbrowser.open(tpl.colab_url())

    # ------------------------------------------------------------------ #
    # Browse & OS helpers
    # ------------------------------------------------------------------ #

    def _copy_text(self, text: str) -> None:
        QtWidgets.QApplication.clipboard().setText(text)
        self._append_log(f"Copied to clipboard: {text}")

    def _copy_adc_command(self) -> None:
        status = self._current_service().auth_status()
        self._copy_text(str(status["adc"]["setup_command"]))

    def _browse_credentials_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Google OAuth Client JSON"),
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self.credentials_edit.setText(path)
            self.refresh_auth_status()

    def _browse_token_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Select Google OAuth Token File"),
            self.token_edit.text(),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self.token_edit.setText(path)
            self.refresh_auth_status()

    def _browse_train_dataset_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Dataset Config or Archive"),
            "",
            "Datasets (*.yaml *.yml *.zip);;All Files (*)",
        )
        if path:
            self.train_dataset_edit.setText(path)

    def _browse_train_dataset_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Dataset Folder")
        )
        if path:
            self.train_dataset_edit.setText(path)

    def _browse_train_weights(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Initial Weights"),
            "",
            "Checkpoints (*.pt *.pth *.ckpt);;All Files (*)",
        )
        if path:
            self.train_weights_combo.setCurrentText(path)

    def _browse_predict_source_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Video or Image"),
            "",
            "Media (*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self.predict_source_edit.setText(path)

    def _browse_predict_weights(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Trained Weights"),
            "",
            "Checkpoints (*.pt *.pth *.ckpt *.onnx);;All Files (*)",
        )
        if path:
            self.predict_weights_combo.setCurrentText(path)

    def _browse_dir_into(self, target_edit: QtWidgets.QLineEdit) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Folder")
        )
        if path:
            target_edit.setText(path)

    @staticmethod
    def _find_best_checkpoint(downloaded_files: list[str]) -> Optional[str]:
        candidates = [Path(p) for p in list(downloaded_files or []) if str(p).strip()]
        preferred_names = ("best.pt", "model_final.pth", "best.pth", "last.pt")
        for target_name in preferred_names:
            for cand in candidates:
                if cand.name.lower() == target_name:
                    return str(cand)
        for cand in candidates:
            if cand.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx"}:
                return str(cand)
        return None

    @staticmethod
    def _open_local_folder(path: Path) -> None:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _on_open_sim2real_studio_clicked(self) -> None:
        from annolid.gui.widgets.sim2real_dialog import Sim2RealStudioDialog

        initial_videos: list[str] = []
        src_edit = getattr(self, "predict_source_edit", None)
        if src_edit is not None:
            raw = src_edit.text().strip()
            if raw and Path(raw).exists():
                initial_videos.append(raw)
        dlg = getattr(self, "_sim2real_dialog", None)
        if dlg is None:
            dlg = Sim2RealStudioDialog(
                initial_video_paths=initial_videos or None,
                colab_service=self._service,
                parent=self,
            )
            dlg.finished.connect(lambda *_: setattr(self, "_sim2real_dialog", None))
            dlg.destroyed.connect(lambda *_: setattr(self, "_sim2real_dialog", None))
            self._sim2real_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def closeEvent(self, event: Any) -> None:
        self._elapsed_timer.stop()
        if self._worker_thread is not None and self._worker_thread.isRunning():
            reply = QtWidgets.QMessageBox.question(
                self,
                self.tr("Colab Task Running"),
                self.tr("A Colab task is still running. Cancel it and close?"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self._cancel_active_task()
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        super().closeEvent(event)
