"""Workflow Dashboard - Home screen with project overview and quick actions.

A modern dashboard widget showing project status, recent activity,
quick-action buttons, and workflow guidance for new users.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal


@dataclass
class RecentProject:
    """Represents a recently opened project or video."""
    name: str
    path: str
    last_opened: datetime
    thumbnail: Optional[str] = None
    annotations_count: int = 0
    frames_labeled: int = 0


class QuickActionButton(QtWidgets.QPushButton):
    """A styled button for quick actions on the dashboard."""

    def __init__(
        self,
        title: str,
        description: str,
        icon_name: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 100)
        self.setCursor(Qt.PointingHandCursor)

        # Create layout for multi-line content
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        # Title
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QtWidgets.QLabel(description)
        desc_label.setStyleSheet("color: gray; font-size: 11px;")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Store for hover effects
        self._title = title
        self._description = description

        self.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #ccc;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
            }
        """)


class RecentItemWidget(QtWidgets.QFrame):
    """Widget displaying a recent project or video item."""

    clicked = Signal(str)  # Emits the path when clicked

    def __init__(
        self,
        item: RecentProject,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._path = item.path
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 8px;
            }
            QFrame:hover {
                background-color: #f8f8f8;
                border-color: #ccc;
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Icon/thumbnail placeholder
        icon_label = QtWidgets.QLabel("🎬")
        icon_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(icon_label)

        # Info
        info_layout = QtWidgets.QVBoxLayout()
        info_layout.setSpacing(2)

        name_label = QtWidgets.QLabel(item.name)
        name_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(name_label)

        path_label = QtWidgets.QLabel(str(Path(item.path).parent))
        path_label.setStyleSheet("color: gray; font-size: 11px;")
        info_layout.addWidget(path_label)

        if item.annotations_count > 0:
            stats_label = QtWidgets.QLabel(
                f"{item.annotations_count} annotations • {item.frames_labeled} frames"
            )
            stats_label.setStyleSheet("color: #666; font-size: 10px;")
            info_layout.addWidget(stats_label)

        layout.addLayout(info_layout, 1)

        # Time ago
        time_label = QtWidgets.QLabel(self._format_time_ago(item.last_opened))
        time_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(time_label)

    def _format_time_ago(self, dt: datetime) -> str:
        now = datetime.now()
        diff = now - dt
        if diff.days > 7:
            return dt.strftime("%b %d")
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._path)
        super().mousePressEvent(event)


class WorkflowStepWidget(QtWidgets.QFrame):
    """Widget showing a workflow step with progress indicator."""

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
        self.setCursor(Qt.PointingHandCursor)

        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 4px;
            }
            QFrame:hover {
                border-color: #2196F3;
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        # Step number indicator
        number_label = QtWidgets.QLabel(str(number))
        if status == "completed":
            number_label.setText("✓")
            number_label.setStyleSheet("""
                background-color: #4CAF50;
                color: white;
                border-radius: 14px;
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
            """)
        elif status == "active":
            number_label.setStyleSheet("""
                background-color: #2196F3;
                color: white;
                border-radius: 14px;
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
            """)
        else:
            number_label.setStyleSheet("""
                background-color: #e0e0e0;
                color: #666;
                border-radius: 14px;
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
            """)
        number_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(number_label)

        # Text content
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(2)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        text_layout.addWidget(title_label)

        desc_label = QtWidgets.QLabel(description)
        desc_label.setStyleSheet("color: gray; font-size: 11px;")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, 1)

        # Arrow indicator
        arrow = QtWidgets.QLabel("→")
        arrow.setStyleSheet("color: #ccc; font-size: 18px;")
        layout.addWidget(arrow)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.step_clicked.emit(self._step_id)
        super().mousePressEvent(event)


class DashboardWidget(QtWidgets.QWidget):
    """Main dashboard widget showing project overview and quick actions."""

    # Signals for main window to handle
    new_project_requested = Signal()
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
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(24)

        # Header
        header = self._create_header()
        main_layout.addWidget(header)

        # Content area with scroll
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(24)

        # Quick actions section
        actions_section = self._create_quick_actions_section()
        content_layout.addWidget(actions_section)

        # Two-column layout for workflow and recent
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(24)

        # Workflow guide
        workflow_section = self._create_workflow_section()
        columns.addWidget(workflow_section, 1)

        # Recent projects
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

        # Logo and title
        title_layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("Welcome to Annolid")
        title.setStyleSheet("font-size: 28px; font-weight: bold;")
        title_layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "AI-assisted annotation and tracking for behavior analysis"
        )
        subtitle.setStyleSheet("color: gray; font-size: 14px;")
        title_layout.addWidget(subtitle)

        layout.addLayout(title_layout)
        layout.addStretch()

        # Help button
        help_btn = QtWidgets.QPushButton("📖 Documentation")
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.clicked.connect(self.open_documentation_requested.emit)
        layout.addWidget(help_btn)

        return header

    def _create_quick_actions_section(self) -> QtWidgets.QWidget:
        section = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)

        # Section title
        title = QtWidgets.QLabel("Quick Actions")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        # Action buttons grid
        actions_layout = QtWidgets.QHBoxLayout()
        actions_layout.setSpacing(16)

        # New Project
        new_project_btn = QuickActionButton(
            "🆕 New Project",
            "Create a new annotation project",
        )
        new_project_btn.clicked.connect(self.new_project_requested.emit)
        actions_layout.addWidget(new_project_btn)

        # Open Video
        open_video_btn = QuickActionButton(
            "🎬 Open Video",
            "Open a video file for annotation",
        )
        open_video_btn.clicked.connect(self.open_video_requested.emit)
        actions_layout.addWidget(open_video_btn)

        # Import Videos
        import_btn = QuickActionButton(
            "📥 Import Videos",
            "Batch import multiple videos",
        )
        import_btn.clicked.connect(self.import_videos_requested.emit)
        actions_layout.addWidget(import_btn)

        # Train Model
        train_btn = QuickActionButton(
            "🧠 Train Model",
            "Train detection or pose models",
        )
        train_btn.clicked.connect(self.train_model_requested.emit)
        actions_layout.addWidget(train_btn)

        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        return section

    def _create_workflow_section(self) -> QtWidgets.QWidget:
        section = QtWidgets.QGroupBox("Workflow Guide")
        section.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(section)
        layout.setSpacing(8)

        # Workflow steps
        steps = [
            ("project", 1, "Define Project",
             "Set up subjects, behaviors, and keypoints"),
            ("collect", 2, "Collect Data", "Import videos and extract frames"),
            ("annotate", 3, "Annotate", "Label frames with AI assistance"),
            ("train", 4, "Train Model", "Train detection/pose models on your data"),
            ("inference", 5, "Run Inference", "Process videos with trained models"),
            ("analyze", 6, "Analyze", "Generate reports and export results"),
        ]

        for step_id, num, title, desc in steps:
            step_widget = WorkflowStepWidget(step_id, num, title, desc)
            step_widget.step_clicked.connect(self.workflow_step_requested.emit)
            layout.addWidget(step_widget)

        layout.addStretch()
        return section

    def _create_recent_section(self) -> QtWidgets.QWidget:
        section = QtWidgets.QGroupBox("Recent")
        section.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(section)
        layout.setSpacing(8)

        self._recent_container = QtWidgets.QWidget()
        self._recent_layout = QtWidgets.QVBoxLayout(self._recent_container)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(8)

        # Placeholder if no recent items
        self._empty_label = QtWidgets.QLabel(
            "No recent projects.\nCreate a new project or open a video to get started."
        )
        self._empty_label.setStyleSheet("color: gray; padding: 20px;")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._recent_layout.addWidget(self._empty_label)

        layout.addWidget(self._recent_container)
        layout.addStretch()

        return section

    def set_recent_items(self, items: List[RecentProject]) -> None:
        """Update the list of recent projects/videos."""
        self._recent_items = items

        # Clear existing items
        while self._recent_layout.count():
            child = self._recent_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not items:
            self._empty_label = QtWidgets.QLabel(
                "No recent projects.\nCreate a new project or open a video to get started."
            )
            self._empty_label.setStyleSheet("color: gray; padding: 20px;")
            self._empty_label.setAlignment(Qt.AlignCenter)
            self._recent_layout.addWidget(self._empty_label)
        else:
            for item in items[:8]:  # Show max 8 recent items
                widget = RecentItemWidget(item)
                widget.clicked.connect(self.open_recent_requested.emit)
                self._recent_layout.addWidget(widget)

    def update_workflow_status(self, step_statuses: dict) -> None:
        """Update the visual status of workflow steps.

        Args:
            step_statuses: Dict mapping step_id to status ('pending', 'active', 'completed')
        """
        # This would require storing references to WorkflowStepWidgets
        # For now, the initial creation sets all to pending
        pass

    def add_recent_from_settings(self, settings: QtCore.QSettings) -> None:
        """Load recent files from QSettings and populate the list."""
        recent_files = settings.value("recentFiles", [], type=list) or []
        items = []
        for path in recent_files[:8]:
            if Path(path).exists():
                items.append(RecentProject(
                    name=Path(path).name,
                    path=path,
                    last_opened=datetime.now(),  # Ideally track actual times
                ))
        self.set_recent_items(items)
