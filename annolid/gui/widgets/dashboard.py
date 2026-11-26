"""Workflow Dashboard - Home screen with project overview and quick actions.

A theme-aware dashboard widget showing project status, recent activity,
quick-action buttons, and live 6-step workflow guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal

from annolid.gui.qt_compat import palette_color_role
from annolid.services.project_workspace import PROJECT_CONFIG_FILENAME


def _is_dark_palette(widget: QtWidgets.QWidget) -> bool:
    try:
        return bool(
            widget.palette().color(palette_color_role("Window")).lightness() < 128
        )
    except Exception:
        return False


def _dashboard_colors(is_dark: bool) -> Dict[str, str]:
    if is_dark:
        return {
            "card_bg": "#1e222a",
            "card_hover": "#262b35",
            "card_pressed": "#303643",
            "card_border": "#333945",
            "card_hover_border": "#4c8bf5",
            "fg": "#e8eaed",
            "muted": "#9aa0a6",
            "accent": "#4c8bf5",
            "success": "#34a853",
            "pending_bg": "#2e3440",
            "pending_fg": "#cbd5e1",
        }
    return {
        "card_bg": "#ffffff",
        "card_hover": "#f4f6f8",
        "card_pressed": "#e8ecf0",
        "card_border": "#dadce0",
        "card_hover_border": "#1a73e8",
        "fg": "#202124",
        "muted": "#5f6368",
        "accent": "#1a73e8",
        "success": "#2e7d32",
        "pending_bg": "#e8eaed",
        "pending_fg": "#3c4043",
    }


def _parse_iso_datetime(raw: object) -> datetime:
    text = str(raw or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
    return datetime.now()


@dataclass
class RecentProject:
    """Represents a recently opened project, video, or file."""

    name: str
    path: str
    last_opened: datetime
    thumbnail: Optional[str] = None
    annotations_count: int = 0
    frames_labeled: int = 0
    kind: str = "file"  # "project", "video", "file"


class QuickActionButton(QtWidgets.QPushButton):
    """A theme-aware card button for quick actions on the dashboard."""

    def __init__(
        self,
        title: str,
        description: str,
        icon_name: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        del icon_name
        self.setMinimumSize(156, 92)
        self.setCursor(Qt.PointingHandCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self._title_label = QtWidgets.QLabel(title, self)
        self._title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._title_label)

        self._desc_label = QtWidgets.QLabel(description, self)
        self._desc_label.setAlignment(Qt.AlignCenter)
        self._desc_label.setWordWrap(True)
        layout.addWidget(self._desc_label)

        self._title = title
        self._description = description
        self.apply_theme(_is_dark_palette(self))

    def apply_theme(self, is_dark: bool) -> None:
        c = _dashboard_colors(is_dark)
        self._title_label.setStyleSheet(
            f"font-weight: 700; font-size: 13px; color: {c['fg']}; background: transparent;"
        )
        self._desc_label.setStyleSheet(
            f"color: {c['muted']}; font-size: 11px; background: transparent;"
        )
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {c["card_bg"]};
                border: 1px solid {c["card_border"]};
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {c["card_hover"]};
                border-color: {c["card_hover_border"]};
            }}
            QPushButton:pressed {{
                background-color: {c["card_pressed"]};
            }}
            """
        )


class RecentItemWidget(QtWidgets.QFrame):
    """Widget displaying a recent project, video, or annotation item."""

    clicked = Signal(str)  # Emits the path when clicked

    def __init__(
        self,
        item: RecentProject,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._path = item.path
        self._kind = item.kind
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.setToolTip(str(item.path))

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        icon_glyph = {"project": "📁", "video": "🎬"}.get(item.kind, "📄")
        self._icon_label = QtWidgets.QLabel(icon_glyph, self)
        self._icon_label.setStyleSheet("font-size: 20px; background: transparent;")
        layout.addWidget(self._icon_label)

        info_layout = QtWidgets.QVBoxLayout()
        info_layout.setSpacing(2)

        self._name_label = QtWidgets.QLabel(item.name, self)
        info_layout.addWidget(self._name_label)

        self._path_label = QtWidgets.QLabel(str(Path(item.path).parent), self)
        info_layout.addWidget(self._path_label)

        self._stats_label: Optional[QtWidgets.QLabel] = None
        if item.annotations_count > 0:
            self._stats_label = QtWidgets.QLabel(
                f"{item.annotations_count} annotations • {item.frames_labeled} frames",
                self,
            )
            info_layout.addWidget(self._stats_label)

        layout.addLayout(info_layout, 1)

        self._time_label = QtWidgets.QLabel(
            self._format_time_ago(item.last_opened), self
        )
        layout.addWidget(self._time_label)
        self.apply_theme(_is_dark_palette(self))

    def apply_theme(self, is_dark: bool) -> None:
        c = _dashboard_colors(is_dark)
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {c["card_bg"]};
                border: 1px solid {c["card_border"]};
                border-radius: 6px;
            }}
            QFrame:hover {{
                background-color: {c["card_hover"]};
                border-color: {c["card_hover_border"]};
            }}
            """
        )
        self._name_label.setStyleSheet(
            f"font-weight: 700; color: {c['fg']}; background: transparent;"
        )
        self._path_label.setStyleSheet(
            f"color: {c['muted']}; font-size: 11px; background: transparent;"
        )
        if self._stats_label is not None:
            self._stats_label.setStyleSheet(
                f"color: {c['muted']}; font-size: 10px; background: transparent;"
            )
        self._time_label.setStyleSheet(
            f"color: {c['muted']}; font-size: 10px; background: transparent;"
        )

    def _format_time_ago(self, dt: datetime) -> str:
        now = datetime.now(tz=dt.tzinfo) if dt.tzinfo is not None else datetime.now()
        diff = now - dt
        if diff.total_seconds() < 0:
            return "Just now"
        if diff.days > 7:
            return dt.strftime("%b %d")
        if diff.days > 0:
            return f"{diff.days}d ago"
        if diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        if diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        return "Just now"

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._path)
        super().mousePressEvent(event)


class WorkflowStepWidget(QtWidgets.QFrame):
    """Widget showing a workflow step with live progress indicator."""

    step_clicked = Signal(str)  # Emits step id when clicked

    def __init__(
        self,
        step_id: str,
        number: int,
        title: str,
        description: str,
        status: str = "pending",  # pending, active, completed
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._step_id = step_id
        self._number = int(number)
        self._status = str(status or "pending")
        self._is_dark = _is_dark_palette(self)
        self.setCursor(Qt.PointingHandCursor)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self._number_label = QtWidgets.QLabel(str(self._number), self)
        self._number_label.setAlignment(Qt.AlignCenter)
        self._number_label.setFixedSize(28, 28)
        layout.addWidget(self._number_label)

        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(2)

        self._title_label = QtWidgets.QLabel(title, self)
        text_layout.addWidget(self._title_label)

        self._desc_label = QtWidgets.QLabel(description, self)
        self._desc_label.setWordWrap(True)
        text_layout.addWidget(self._desc_label)

        layout.addLayout(text_layout, 1)

        self._arrow_label = QtWidgets.QLabel("→", self)
        layout.addWidget(self._arrow_label)

        self.set_status(self._status)

    @property
    def status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        norm = str(status or "pending").strip().lower()
        if norm not in {"pending", "active", "completed"}:
            norm = "pending"
        self._status = norm
        self.apply_theme(self._is_dark)

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        c = _dashboard_colors(self._is_dark)
        border_color = c["accent"] if self._status == "active" else c["card_border"]
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {c["card_bg"]};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
            QFrame:hover {{
                background-color: {c["card_hover"]};
                border-color: {c["card_hover_border"]};
            }}
            """
        )
        if self._status == "completed":
            self._number_label.setText("✓")
            badge_bg, badge_fg = c["success"], "#ffffff"
        elif self._status == "active":
            self._number_label.setText(str(self._number))
            badge_bg, badge_fg = c["accent"], "#ffffff"
        else:
            self._number_label.setText(str(self._number))
            badge_bg, badge_fg = c["pending_bg"], c["pending_fg"]

        self._number_label.setStyleSheet(
            f"""
            background-color: {badge_bg};
            color: {badge_fg};
            border-radius: 14px;
            font-weight: 700;
            font-size: 12px;
            """
        )
        self._title_label.setStyleSheet(
            f"font-weight: 700; font-size: 13px; color: {c['fg']}; background: transparent;"
        )
        self._desc_label.setStyleSheet(
            f"color: {c['muted']}; font-size: 11px; background: transparent;"
        )
        self._arrow_label.setStyleSheet(
            f"color: {c['muted']}; font-size: 16px; background: transparent;"
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.step_clicked.emit(self._step_id)
        super().mousePressEvent(event)


class DashboardWidget(QtWidgets.QWidget):
    """Main dashboard widget showing project overview, workflow progress, and quick actions."""

    new_project_requested = Signal()
    open_project_requested = Signal()
    open_video_requested = Signal()
    open_recent_requested = Signal(str)  # path
    import_videos_requested = Signal()
    extract_frames_requested = Signal()
    train_model_requested = Signal()
    run_inference_requested = Signal()
    open_documentation_requested = Signal()
    workflow_step_requested = Signal(str)  # step_id

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._recent_items: List[RecentProject] = []
        self._quick_action_buttons: List[QuickActionButton] = []
        self._recent_widgets: List[RecentItemWidget] = []
        self._step_widgets: Dict[str, WorkflowStepWidget] = {}
        self._setup_ui()
        self._apply_theme_styles()

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(18)

        header = self._create_header()
        main_layout.addWidget(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(18)

        actions_section = self._create_quick_actions_section()
        content_layout.addWidget(actions_section)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(18)

        workflow_section = self._create_workflow_section()
        columns.addWidget(workflow_section, 1)

        recent_section = self._create_recent_section()
        columns.addWidget(recent_section, 1)

        content_layout.addLayout(columns)
        content_layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll, 1)

    def _create_header(self) -> QtWidgets.QWidget:
        header = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)

        title_layout = QtWidgets.QVBoxLayout()
        self._title_label = QtWidgets.QLabel("Welcome to Annolid")
        self._title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_layout.addWidget(self._title_label)

        self._subtitle_label = QtWidgets.QLabel(
            "AI-assisted annotation, tracking, and project workflow hub"
        )
        title_layout.addWidget(self._subtitle_label)

        layout.addLayout(title_layout)
        layout.addStretch()

        help_btn = QtWidgets.QPushButton("📖 Documentation")
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.clicked.connect(self.open_documentation_requested.emit)
        layout.addWidget(help_btn)

        return header

    def _create_quick_actions_section(self) -> QtWidgets.QWidget:
        section = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._quick_actions_title = QtWidgets.QLabel("Quick Actions")
        layout.addWidget(self._quick_actions_title)

        actions_layout = QtWidgets.QHBoxLayout()
        actions_layout.setSpacing(12)

        button_specs = [
            (
                "🆕 New Project",
                "Create a new annotation project",
                self.new_project_requested.emit,
            ),
            (
                "📂 Open Project",
                "Open an existing Annolid project",
                self.open_project_requested.emit,
            ),
            (
                "🎬 Open Video",
                "Open a video file for annotation",
                self.open_video_requested.emit,
            ),
            (
                "📥 Import Videos",
                "Batch import multiple videos",
                self.import_videos_requested.emit,
            ),
            (
                "🧠 Train Model",
                "Train detection or pose models",
                self.train_model_requested.emit,
            ),
        ]
        for title, desc, slot in button_specs:
            btn = QuickActionButton(title, desc, parent=section)
            btn.clicked.connect(slot)
            self._quick_action_buttons.append(btn)
            actions_layout.addWidget(btn)

        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        return section

    def _create_workflow_section(self) -> QtWidgets.QWidget:
        section = QtWidgets.QGroupBox("Workflow Guide")
        self._workflow_group = section

        layout = QtWidgets.QVBoxLayout(section)
        layout.setSpacing(8)

        self.workflow_summary_label = QtWidgets.QLabel(
            "Create or open a project to track progress across stages.",
            section,
        )
        self.workflow_summary_label.setWordWrap(True)
        layout.addWidget(self.workflow_summary_label)

        steps = [
            (
                "project",
                1,
                "Define Project",
                "Set up subjects, behaviors, and keypoints",
            ),
            ("collect", 2, "Collect Data", "Import videos and extract frames"),
            ("annotate", 3, "Annotate", "Label frames with AI assistance"),
            ("train", 4, "Train Model", "Train detection/pose models on your data"),
            ("inference", 5, "Run Inference", "Process videos with trained models"),
            ("analyze", 6, "Analyze", "Generate reports and export results"),
        ]

        for step_id, num, title, desc in steps:
            step_widget = WorkflowStepWidget(step_id, num, title, desc, parent=section)
            step_widget.step_clicked.connect(self.workflow_step_requested.emit)
            self._step_widgets[step_id] = step_widget
            layout.addWidget(step_widget)

        layout.addStretch()
        return section

    def _create_recent_section(self) -> QtWidgets.QWidget:
        section = QtWidgets.QGroupBox("Recent Projects & Files")
        self._recent_group = section

        layout = QtWidgets.QVBoxLayout(section)
        layout.setSpacing(8)

        self._recent_container = QtWidgets.QWidget()
        self._recent_layout = QtWidgets.QVBoxLayout(self._recent_container)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(8)

        self._empty_label = QtWidgets.QLabel(
            "No recent projects or files.\nCreate a new project or open a video to get started."
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._recent_layout.addWidget(self._empty_label)

        layout.addWidget(self._recent_container)
        layout.addStretch()

        return section

    def _apply_theme_styles(self) -> None:
        is_dark = _is_dark_palette(self)
        c = _dashboard_colors(is_dark)
        group_qss = f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                color: {c["fg"]};
                border: 1px solid {c["card_border"]};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
        """
        self._workflow_group.setStyleSheet(group_qss)
        self._recent_group.setStyleSheet(group_qss)
        self._subtitle_label.setStyleSheet(f"color: {c['muted']}; font-size: 13px;")
        self._quick_actions_title.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {c['fg']};"
        )
        self.workflow_summary_label.setStyleSheet(
            f"color: {c['muted']}; font-size: 12px;"
        )
        if getattr(self, "_empty_label", None) is not None:
            try:
                self._empty_label.setStyleSheet(f"color: {c['muted']}; padding: 20px;")
            except RuntimeError:
                pass
        for btn in self._quick_action_buttons:
            btn.apply_theme(is_dark)
        for step_widget in self._step_widgets.values():
            step_widget.apply_theme(is_dark)
        for item_widget in self._recent_widgets:
            try:
                item_widget.apply_theme(is_dark)
            except RuntimeError:
                pass

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
            QtCore.QEvent.StyleChange,
        ):
            self._apply_theme_styles()
        super().changeEvent(event)

    def set_recent_items(self, items: List[RecentProject]) -> None:
        """Update the list of recent projects/videos/files."""
        self._recent_items = list(items or [])
        self._recent_widgets = []

        while self._recent_layout.count():
            child = self._recent_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        is_dark = _is_dark_palette(self)
        c = _dashboard_colors(is_dark)
        if not self._recent_items:
            self._empty_label = QtWidgets.QLabel(
                "No recent projects or files.\nCreate a new project or open a video to get started."
            )
            self._empty_label.setStyleSheet(f"color: {c['muted']}; padding: 20px;")
            self._empty_label.setAlignment(Qt.AlignCenter)
            self._recent_layout.addWidget(self._empty_label)
        else:
            self._empty_label = None
            for item in self._recent_items[:10]:
                widget = RecentItemWidget(item, parent=self._recent_container)
                widget.apply_theme(is_dark)
                widget.clicked.connect(self.open_recent_requested.emit)
                self._recent_widgets.append(widget)
                self._recent_layout.addWidget(widget)

    def update_workflow_status(
        self,
        step_statuses: Mapping[str, str],
        *,
        summary: str = "",
    ) -> None:
        """Update the visual status of workflow steps.

        Args:
            step_statuses: Dict mapping step_id to status ('pending', 'active', 'completed')
            summary: Optional workflow summary line displayed above steps.
        """
        for step_id, widget in self._step_widgets.items():
            if step_id in step_statuses:
                widget.set_status(str(step_statuses[step_id]))
        if summary:
            self.workflow_summary_label.setText(str(summary))

    def add_recent_from_settings(self, settings: QtCore.QSettings) -> None:
        """Load recent projects and recent files from QSettings and populate the list."""
        items: List[RecentProject] = []
        seen: set[str] = set()

        raw_projects = settings.value("recentProjects", "[]")
        parsed_projects: list = []
        if isinstance(raw_projects, str):
            try:
                loaded = json.loads(raw_projects)
                if isinstance(loaded, list):
                    parsed_projects = loaded
            except json.JSONDecodeError:
                parsed_projects = []
        elif isinstance(raw_projects, list):
            parsed_projects = raw_projects

        for entry in parsed_projects:
            if isinstance(entry, dict):
                raw_path = str(entry.get("path") or "").strip()
                raw_name = str(entry.get("name") or "").strip()
                raw_ts = entry.get("last_opened")
            else:
                raw_path = str(entry or "").strip()
                raw_name = ""
                raw_ts = None
            if not raw_path or raw_path in seen:
                continue
            p = Path(raw_path)
            if not p.exists():
                continue
            seen.add(raw_path)
            items.append(
                RecentProject(
                    name=raw_name or p.name,
                    path=raw_path,
                    last_opened=_parse_iso_datetime(raw_ts),
                    kind="project",
                )
            )

        recent_files = settings.value("recentFiles", [], type=list) or []
        video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        for path in recent_files[:10]:
            text = str(path or "").strip()
            if not text or text in seen:
                continue
            p = Path(text)
            if not p.exists():
                continue
            seen.add(text)
            if p.is_dir() and (p / PROJECT_CONFIG_FILENAME).is_file():
                kind = "project"
            elif p.suffix.lower() in video_suffixes:
                kind = "video"
            else:
                kind = "file"
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except OSError:
                mtime = datetime.now()
            items.append(
                RecentProject(
                    name=p.name,
                    path=text,
                    last_opened=mtime,
                    kind=kind,
                )
            )
        self.set_recent_items(items)
