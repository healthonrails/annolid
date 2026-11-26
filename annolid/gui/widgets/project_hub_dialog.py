"""Project & Workflow Hub dialog for inspecting, managing, and navigating Annolid projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets.dashboard import (
    DashboardWidget,
    RecentProject,
    _parse_iso_datetime,
)
from annolid.services.project_workspace import PROJECT_CONFIG_FILENAME


class ProjectHubDialog(QtWidgets.QDialog):
    """Unified dialog showing active project metadata, live workflow progress, and quick actions."""

    newProjectRequested = QtCore.Signal()
    openProjectRequested = QtCore.Signal(str)  # empty string means prompt picker
    closeProjectRequested = QtCore.Signal()
    exportProjectRequested = QtCore.Signal()
    addVideosRequested = QtCore.Signal()
    editSchemaRequested = QtCore.Signal()
    auditAnnotationsRequested = QtCore.Signal()
    openVideoRequested = QtCore.Signal(str)
    workflowStepRequested = QtCore.Signal(str)

    def __init__(
        self,
        *,
        project_controller: Any = None,
        recent_projects_getter: Optional[Callable[[], List[Dict[str, str]]]] = None,
        recent_files_getter: Optional[Callable[[], List[str]]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project_controller = project_controller
        self._recent_projects_getter = recent_projects_getter
        self._recent_files_getter = recent_files_getter
        self.setWindowTitle("Project & Workflow Hub")
        self.resize(980, 720)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Active Project overview card
        self.project_group = QtWidgets.QGroupBox("Active Project", self)
        pg_layout = QtWidgets.QVBoxLayout(self.project_group)
        pg_layout.setSpacing(6)

        header_row = QtWidgets.QHBoxLayout()
        self.project_name_label = QtWidgets.QLabel("No project loaded", self)
        font = self.project_name_label.font()
        font.setPointSize(max(12, font.pointSize() + 2))
        font.setBold(True)
        self.project_name_label.setFont(font)
        header_row.addWidget(self.project_name_label, 1)

        self.refresh_btn = QtWidgets.QPushButton("Refresh", self)
        self.refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(self.refresh_btn)

        self.validate_btn = QtWidgets.QPushButton("Validate Structure", self)
        self.validate_btn.clicked.connect(self._validate_project)
        header_row.addWidget(self.validate_btn)

        self.open_folder_btn = QtWidgets.QPushButton("Open Folder", self)
        self.open_folder_btn.clicked.connect(self._open_project_folder)
        header_row.addWidget(self.open_folder_btn)

        self.close_project_btn = QtWidgets.QPushButton("Close Project", self)
        self.close_project_btn.clicked.connect(self._on_close_project_clicked)
        header_row.addWidget(self.close_project_btn)
        pg_layout.addLayout(header_row)

        self.project_path_label = QtWidgets.QLabel("", self)
        self.project_path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.project_path_label.setEnabled(False)
        pg_layout.addWidget(self.project_path_label)

        self.stats_label = QtWidgets.QLabel(
            "Open an existing Annolid project or create a new one below.", self
        )
        self.stats_label.setWordWrap(True)
        pg_layout.addWidget(self.stats_label)

        action_row = QtWidgets.QHBoxLayout()
        self.new_project_btn = QtWidgets.QPushButton("New Project…", self)
        self.new_project_btn.clicked.connect(self.newProjectRequested.emit)
        action_row.addWidget(self.new_project_btn)

        self.open_project_btn = QtWidgets.QPushButton("Open Project…", self)
        self.open_project_btn.clicked.connect(
            lambda: self.openProjectRequested.emit("")
        )
        action_row.addWidget(self.open_project_btn)

        self.add_videos_btn = QtWidgets.QPushButton("Add Videos…", self)
        self.add_videos_btn.clicked.connect(self.addVideosRequested.emit)
        action_row.addWidget(self.add_videos_btn)

        self.edit_schema_btn = QtWidgets.QPushButton("Edit Schema…", self)
        self.edit_schema_btn.clicked.connect(self.editSchemaRequested.emit)
        action_row.addWidget(self.edit_schema_btn)

        self.audit_annotations_btn = QtWidgets.QPushButton("Audit Annotations…", self)
        self.audit_annotations_btn.clicked.connect(self.auditAnnotationsRequested.emit)
        action_row.addWidget(self.audit_annotations_btn)

        self.export_project_btn = QtWidgets.QPushButton("Export Project…", self)
        self.export_project_btn.clicked.connect(self.exportProjectRequested.emit)
        action_row.addWidget(self.export_project_btn)
        action_row.addStretch(1)
        pg_layout.addLayout(action_row)

        root.addWidget(self.project_group)

        # Embedded DashboardWidget for 6-step workflow progress & recent activity
        self.dashboard = DashboardWidget(self)
        self.dashboard.new_project_requested.connect(self.newProjectRequested.emit)
        self.dashboard.open_project_requested.connect(
            lambda: self.openProjectRequested.emit("")
        )
        self.dashboard.open_video_requested.connect(
            lambda: self.openVideoRequested.emit("")
        )
        self.dashboard.import_videos_requested.connect(self.addVideosRequested.emit)
        self.dashboard.train_model_requested.connect(
            lambda: self.workflowStepRequested.emit("train")
        )
        self.dashboard.run_inference_requested.connect(
            lambda: self.workflowStepRequested.emit("inference")
        )
        self.dashboard.workflow_step_requested.connect(self.workflowStepRequested.emit)
        self.dashboard.open_recent_requested.connect(self._on_recent_clicked)
        root.addWidget(self.dashboard, 1)

        footer = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setEnabled(False)
        footer.addWidget(self.status_label, 1)
        self.close_btn = QtWidgets.QPushButton("Close", self)
        self.close_btn.clicked.connect(self.accept)
        footer.addWidget(self.close_btn)
        root.addLayout(footer)

    def refresh(self) -> None:
        info: Optional[Dict[str, Any]] = None
        if self._project_controller is not None:
            try:
                info = self._project_controller.get_project_info()
            except Exception:
                info = None

        has_project = bool(info and info.get("path"))
        for btn in (
            self.validate_btn,
            self.open_folder_btn,
            self.close_project_btn,
            self.add_videos_btn,
            self.export_project_btn,
        ):
            btn.setEnabled(has_project)

        if has_project and info is not None:
            name = str(info.get("name") or Path(str(info["path"])).name)
            self.project_name_label.setText(name)
            self.project_path_label.setText(str(info.get("path") or ""))
            stats = dict(info.get("statistics") or {})
            store_only = int(stats.get("store_only_annotations", 0) or 0)
            ann_label = str(int(stats.get("total_annotations", 0) or 0))
            if store_only > 0:
                ann_label += f" ({store_only} in NDJSON store)"
            missing_v = int(stats.get("missing_videos", 0) or 0)
            vid_label = str(int(stats.get("total_videos", 0) or 0))
            if missing_v > 0:
                vid_label += f" ({missing_v} missing)"
            parts = [
                f"Videos: {vid_label}",
                f"Images: {int(stats.get('total_images', 0) or 0)}",
                f"Annotations: {ann_label}",
                f"Models: {int(stats.get('total_models', 0) or 0)}",
                f"Exports: {int(stats.get('total_exports', 0) or 0)}",
            ]
            if stats.get("has_behavior_schema"):
                parts.append("Behavior schema ✓")
            if stats.get("has_pose_schema"):
                parts.append("Pose schema ✓")
            self.stats_label.setText("  |  ".join(parts))

            wf = dict(info.get("workflow_status") or {})
            self.dashboard.update_workflow_status(
                dict(wf.get("steps") or {}),
                summary=str(wf.get("summary") or ""),
            )
        else:
            self.project_name_label.setText("No project loaded")
            self.project_path_label.setText("")
            self.stats_label.setText(
                "Create a new project or open an existing project folder to track "
                "videos, annotations, models, and workflow progress."
            )
            self.dashboard.update_workflow_status(
                {
                    "project": "active",
                    "collect": "pending",
                    "annotate": "pending",
                    "train": "pending",
                    "inference": "pending",
                    "analyze": "pending",
                },
                summary="Create or open a project to start tracking workflow progress.",
            )

        self._refresh_recent_items()

    def _refresh_recent_items(self) -> None:
        items: List[RecentProject] = []
        seen: set[str] = set()

        if callable(self._recent_projects_getter):
            try:
                raw_projects = list(self._recent_projects_getter() or [])
            except Exception:
                raw_projects = []
            for entry in raw_projects:
                raw_path = str((entry or {}).get("path") or "").strip()
                if not raw_path or raw_path in seen or not Path(raw_path).exists():
                    continue
                seen.add(raw_path)
                items.append(
                    RecentProject(
                        name=str((entry or {}).get("name") or Path(raw_path).name),
                        path=raw_path,
                        last_opened=_parse_iso_datetime(
                            (entry or {}).get("last_opened")
                        ),
                        kind="project",
                    )
                )

        if callable(self._recent_files_getter):
            try:
                raw_files = list(self._recent_files_getter() or [])
            except Exception:
                raw_files = []
            video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
            for raw_path in raw_files:
                text = str(raw_path or "").strip()
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
                items.append(
                    RecentProject(
                        name=p.name,
                        path=text,
                        last_opened=_parse_iso_datetime(None),
                        kind=kind,
                    )
                )

        self.dashboard.set_recent_items(items)

    def _on_recent_clicked(self, path_str: str) -> None:
        p = Path(str(path_str or "")).expanduser()
        if p.is_dir() or p.name == PROJECT_CONFIG_FILENAME:
            self.openProjectRequested.emit(str(p))
            self.refresh()
        else:
            self.openVideoRequested.emit(str(p))

    def _validate_project(self) -> None:
        if self._project_controller is None:
            return
        project_path = self._project_controller.get_current_project_path()
        if project_path is None:
            self.status_label.setText("No active project to validate.")
            return
        ok, errors = self._project_controller.validate_project_structure(project_path)
        if ok:
            self.status_label.setText(
                f"Project structure is valid: {project_path.name}"
            )
        else:
            self.status_label.setText("Validation issues: " + "; ".join(errors))

    def _on_close_project_clicked(self) -> None:
        self.closeProjectRequested.emit()
        self.refresh()

    def _open_project_folder(self) -> None:
        if self._project_controller is None:
            return
        project_path = self._project_controller.get_current_project_path()
        if project_path is None or not Path(project_path).exists():
            self.status_label.setText("Project folder does not exist on disk.")
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(Path(project_path).resolve()))
        )
