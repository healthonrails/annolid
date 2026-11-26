"""Project Setup Wizard - Streamlined project creation workflow.

A modern QWizard-based interface for creating new Annolid projects,
guiding users through project definition, video import, schema setup,
and pose configuration in a step-by-step flow.
"""

from __future__ import annotations

import copy
import html
from pathlib import Path
from typing import List, Optional

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.domain import (
    BehaviorDefinition,
    ProjectSchema,
    SubjectDefinition,
    default_behavior_spec,
    load_behavior_spec,
    save_behavior_spec,
)
from annolid.annotation.pose_schema import (
    PoseSchema,
    dedup_undirected_pairs,
    get_pose_schema_preset,
    list_pose_schema_presets,
)
from annolid.gui.widgets.keypoint_sequencer import _SkeletonPreviewWidget
from annolid.gui.services.project_service import ProjectService
from annolid.services.project_workspace import (
    PROJECT_CONFIG_FILENAME,
    validate_project_name,
)


class ProjectInfoPage(QtWidgets.QWizardPage):
    """Page 1: Basic project information."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Project Information")
        self.setSubTitle(
            "Enter the basic details for your new Annolid project. "
            "The project folder will contain all annotations, models, and outputs."
        )

        layout = QtWidgets.QFormLayout(self)
        layout.setSpacing(12)

        # Project name
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g., MouseBehavior_Exp1")
        self.name_edit.textChanged.connect(self.completeChanged)
        layout.addRow("Project Name:", self.name_edit)

        # Project location
        location_layout = QtWidgets.QHBoxLayout()
        self.location_edit = QtWidgets.QLineEdit()
        self.location_edit.setPlaceholderText("Select project folder location")
        self.location_edit.textChanged.connect(self.completeChanged)
        location_layout.addWidget(self.location_edit)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_location)
        location_layout.addWidget(browse_btn)
        layout.addRow("Location:", location_layout)

        # Description
        self.description_edit = QtWidgets.QTextEdit()
        self.description_edit.setPlaceholderText(
            "Describe your project (optional): species, experimental setup, etc."
        )
        self.description_edit.setMaximumHeight(80)
        layout.addRow("Description:", self.description_edit)

        self.path_preview_label = QtWidgets.QLabel("")
        self.path_preview_label.setWordWrap(True)
        layout.addRow("Destination:", self.path_preview_label)

        self.name_edit.textChanged.connect(self._update_path_preview)
        self.location_edit.textChanged.connect(self._update_path_preview)

        # Register fields for wizard access
        self.registerField("projectName*", self.name_edit)
        self.registerField("projectLocation*", self.location_edit)

    def _browse_location(self) -> None:
        start_dir = self.location_edit.text() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Project Location", start_dir
        )
        if folder:
            self.location_edit.setText(folder)

    def _update_path_preview(self) -> None:
        name = self.name_edit.text().strip()
        location = self.location_edit.text().strip()
        if not name:
            self.path_preview_label.setText("Enter a project name.")
            return
        ok, reason = validate_project_name(name)
        if not ok:
            self.path_preview_label.setText(f"Invalid name: {reason}")
            return
        if not location:
            self.path_preview_label.setText("Select a parent folder.")
            return
        loc_path = Path(location).expanduser()
        if not loc_path.is_dir():
            self.path_preview_label.setText(f"Parent folder does not exist: {loc_path}")
            return
        self.path_preview_label.setText(str(loc_path / name))

    def isComplete(self) -> bool:
        name = self.name_edit.text().strip()
        location = self.location_edit.text().strip()
        ok, _reason = validate_project_name(name)
        return bool(ok and location and Path(location).expanduser().is_dir())

    def validatePage(self) -> bool:
        name = self.name_edit.text().strip()
        ok, reason = validate_project_name(name)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Invalid Project Name", reason)
            return False
        location = Path(self.location_edit.text().strip()).expanduser()
        project_dir = location / name

        if project_dir.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Project Exists",
                f"The folder '{project_dir}' already exists.\n\n"
                "Do you want to use this folder anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            return reply == QtWidgets.QMessageBox.Yes
        return True

    def project_path(self) -> Path:
        return (
            Path(self.location_edit.text().strip()).expanduser()
            / self.name_edit.text().strip()
        )

    def project_description(self) -> str:
        return self.description_edit.toPlainText().strip()


class VideoImportPage(QtWidgets.QWizardPage):
    """Page 2: Import videos for the project."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Import Videos")
        self.setSubTitle(
            "Add the video files you want to analyze. You can import more "
            "videos later from the Video Manager panel."
        )

        layout = QtWidgets.QVBoxLayout(self)

        # Video list
        self.video_list = QtWidgets.QListWidget()
        self.video_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.video_list.setMinimumHeight(200)
        layout.addWidget(self.video_list)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Videos…")
        add_btn.clicked.connect(self._add_videos)
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        add_folder_btn = QtWidgets.QPushButton("Add Folder…")
        add_folder_btn.clicked.connect(self._add_folder)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(add_folder_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Copy vs link option
        self.copy_videos_check = QtWidgets.QCheckBox(
            "Copy videos into the project's videos/ folder (leave unchecked to reference originals)"
        )
        self.copy_videos_check.setChecked(False)
        layout.addWidget(self.copy_videos_check)

        # Tip
        tip_label = QtWidgets.QLabel(
            "<i>Tip: You can skip this step and import videos later.</i>"
        )
        tip_label.setEnabled(False)
        layout.addWidget(tip_label)

    def _add_videos(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Videos",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        for f in files:
            if not self._has_video(f):
                self.video_list.addItem(f)

    def _add_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder with Videos", str(Path.home())
        )
        if folder:
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            for f in Path(folder).iterdir():
                if f.suffix.lower() in video_extensions and not self._has_video(str(f)):
                    self.video_list.addItem(str(f))

    def _remove_selected(self) -> None:
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))

    def _has_video(self, path: str) -> bool:
        for i in range(self.video_list.count()):
            if self.video_list.item(i).text() == path:
                return True
        return False

    def get_videos(self) -> List[str]:
        return [self.video_list.item(i).text() for i in range(self.video_list.count())]

    def should_copy_videos(self) -> bool:
        return bool(self.copy_videos_check.isChecked())


class SubjectBehaviorPage(QtWidgets.QWizardPage):
    """Page 3: Define subjects and behaviors (simplified schema editor)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Subjects & Behaviors")
        self.setSubTitle(
            "Define the subjects (animals/objects) you'll track and the "
            "behaviors you want to annotate. You can add more details later."
        )

        self._schema = default_behavior_spec()

        layout = QtWidgets.QVBoxLayout(self)

        # Use a splitter for subjects and behaviors side by side
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Subjects panel
        subjects_widget = QtWidgets.QWidget()
        subjects_layout = QtWidgets.QVBoxLayout(subjects_widget)
        subjects_layout.setContentsMargins(0, 0, 0, 0)
        subjects_label = QtWidgets.QLabel("<b>Subjects</b>")
        subjects_layout.addWidget(subjects_label)

        self.subjects_list = QtWidgets.QListWidget()
        self.subjects_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        subjects_layout.addWidget(self.subjects_list, 1)

        subjects_btn_layout = QtWidgets.QHBoxLayout()
        self.subject_add_edit = QtWidgets.QLineEdit()
        self.subject_add_edit.setPlaceholderText("Subject name")
        self.subject_add_edit.returnPressed.connect(self._add_subject)
        subjects_btn_layout.addWidget(self.subject_add_edit, 1)
        add_subject_btn = QtWidgets.QPushButton("+")
        add_subject_btn.setFixedWidth(30)
        add_subject_btn.clicked.connect(self._add_subject)
        subjects_btn_layout.addWidget(add_subject_btn)
        remove_subject_btn = QtWidgets.QPushButton("−")
        remove_subject_btn.setFixedWidth(30)
        remove_subject_btn.clicked.connect(self._remove_subject)
        subjects_btn_layout.addWidget(remove_subject_btn)
        subjects_layout.addLayout(subjects_btn_layout)

        splitter.addWidget(subjects_widget)

        # Behaviors panel
        behaviors_widget = QtWidgets.QWidget()
        behaviors_layout = QtWidgets.QVBoxLayout(behaviors_widget)
        behaviors_layout.setContentsMargins(0, 0, 0, 0)
        behaviors_label = QtWidgets.QLabel("<b>Behaviors</b>")
        behaviors_layout.addWidget(behaviors_label)

        self.behaviors_list = QtWidgets.QListWidget()
        self.behaviors_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        behaviors_layout.addWidget(self.behaviors_list, 1)

        behaviors_btn_layout = QtWidgets.QHBoxLayout()
        self.behavior_add_edit = QtWidgets.QLineEdit()
        self.behavior_add_edit.setPlaceholderText("Behavior name")
        self.behavior_add_edit.returnPressed.connect(self._add_behavior)
        behaviors_btn_layout.addWidget(self.behavior_add_edit, 1)
        add_behavior_btn = QtWidgets.QPushButton("+")
        add_behavior_btn.setFixedWidth(30)
        add_behavior_btn.clicked.connect(self._add_behavior)
        behaviors_btn_layout.addWidget(add_behavior_btn)
        remove_behavior_btn = QtWidgets.QPushButton("−")
        remove_behavior_btn.setFixedWidth(30)
        remove_behavior_btn.clicked.connect(self._remove_behavior)
        behaviors_btn_layout.addWidget(remove_behavior_btn)
        behaviors_layout.addLayout(behaviors_btn_layout)

        splitter.addWidget(behaviors_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # Import/Export buttons
        io_layout = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Import Schema…")
        import_btn.clicked.connect(self._import_schema)
        export_btn = QtWidgets.QPushButton("Export Schema…")
        export_btn.clicked.connect(self._export_schema)
        io_layout.addWidget(import_btn)
        io_layout.addWidget(export_btn)
        io_layout.addStretch()
        layout.addLayout(io_layout)

        self._populate_from_schema()

    def _populate_from_schema(self) -> None:
        self.subjects_list.clear()
        for subj in self._schema.subjects:
            self.subjects_list.addItem(subj.name)

        self.behaviors_list.clear()
        for beh in self._schema.behaviors:
            self.behaviors_list.addItem(beh.name)

    def _add_subject(self) -> None:
        name = self.subject_add_edit.text().strip()
        if not name:
            return
        # Check for duplicates
        for i in range(self.subjects_list.count()):
            if self.subjects_list.item(i).text().lower() == name.lower():
                self.subject_add_edit.clear()
                return
        self.subjects_list.addItem(name)
        self.subject_add_edit.clear()

    def _remove_subject(self) -> None:
        for item in self.subjects_list.selectedItems():
            self.subjects_list.takeItem(self.subjects_list.row(item))

    def _add_behavior(self) -> None:
        name = self.behavior_add_edit.text().strip()
        if not name:
            return
        for i in range(self.behaviors_list.count()):
            if self.behaviors_list.item(i).text().lower() == name.lower():
                self.behavior_add_edit.clear()
                return
        self.behaviors_list.addItem(name)
        self.behavior_add_edit.clear()

    def _remove_behavior(self) -> None:
        for item in self.behaviors_list.selectedItems():
            self.behaviors_list.takeItem(self.behaviors_list.row(item))

    def _import_schema(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Project Schema",
            "",
            "Schema Files (*.json *.yaml *.yml);;All Files (*)",
        )
        if not path_str:
            return
        try:
            schema, _ = load_behavior_spec(path=Path(path_str))
            self._schema = schema
            self._populate_from_schema()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Import Failed", f"Unable to import schema:\n{exc}"
            )

    def _export_schema(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Project Schema",
            "",
            "Schema Files (*.json *.yaml *.yml);;All Files (*)",
        )
        if not path_str:
            return
        schema = self.get_schema()
        try:
            save_behavior_spec(schema, Path(path_str))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Unable to export schema:\n{exc}"
            )

    def get_schema(self) -> ProjectSchema:
        """Collect current subjects and behaviors into a ProjectSchema."""
        subjects = []
        for i in range(self.subjects_list.count()):
            name = self.subjects_list.item(i).text().strip()
            if name:
                sid = name.lower().replace(" ", "_")
                subjects.append(SubjectDefinition(id=sid, name=name))

        behaviors = []
        for i in range(self.behaviors_list.count()):
            name = self.behaviors_list.item(i).text().strip()
            if name:
                code = name.lower().replace(" ", "_")
                behaviors.append(
                    BehaviorDefinition(
                        code=code,
                        name=name,
                        category_id="default",
                    )
                )

        # Preserve existing schema fields
        return ProjectSchema(
            subjects=subjects if subjects else self._schema.subjects,
            behaviors=behaviors if behaviors else self._schema.behaviors,
            categories=self._schema.categories,
            modifiers=self._schema.modifiers,
            pose_schema_path=self._schema.pose_schema_path,
            pose_schema=self._schema.pose_schema,
        )

    def set_schema(self, schema: ProjectSchema) -> None:
        self._schema = copy.deepcopy(schema)
        self._populate_from_schema()


class PoseSchemaPage(QtWidgets.QWizardPage):
    """Page 4: Define keypoints for pose estimation (optional)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Pose Schema (Optional)")
        self.setSubTitle(
            "Define body keypoints for pose estimation. Leave empty if you "
            "only need object detection or behavior annotation."
        )

        self._schema = PoseSchema()
        self._pending_edge_start: Optional[str] = None

        layout = QtWidgets.QVBoxLayout(self)

        # Enable pose checkbox
        self.enable_pose_check = QtWidgets.QCheckBox(
            "Enable pose estimation for this project"
        )
        self.enable_pose_check.toggled.connect(self._toggle_pose_ui)
        layout.addWidget(self.enable_pose_check)

        # Pose config container (hidden by default)
        self.pose_container = QtWidgets.QWidget()
        pose_layout = QtWidgets.QVBoxLayout(self.pose_container)
        pose_layout.setContentsMargins(0, 10, 0, 0)

        # Preset selector
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(["Custom", *list_pose_schema_presets()])
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        pose_layout.addLayout(preset_layout)

        # Splitter for keypoints, skeleton edges/preview, and symmetry pairs
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Keypoints panel
        kp_widget = QtWidgets.QWidget()
        kp_layout = QtWidgets.QVBoxLayout(kp_widget)
        kp_layout.setContentsMargins(0, 0, 0, 0)
        kp_layout.addWidget(QtWidgets.QLabel("<b>Keypoints</b> (order matters)"))

        self.kp_list = QtWidgets.QListWidget()
        self.kp_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.kp_list.model().rowsMoved.connect(lambda *_: self._sync_preview())
        kp_layout.addWidget(self.kp_list, 1)

        kp_btn_layout = QtWidgets.QHBoxLayout()
        self.kp_add_edit = QtWidgets.QLineEdit()
        self.kp_add_edit.setPlaceholderText("Keypoint name (e.g., nose)")
        self.kp_add_edit.returnPressed.connect(self._add_keypoint)
        kp_btn_layout.addWidget(self.kp_add_edit, 1)
        add_kp_btn = QtWidgets.QPushButton("+")
        add_kp_btn.setFixedWidth(30)
        add_kp_btn.clicked.connect(self._add_keypoint)
        kp_btn_layout.addWidget(add_kp_btn)
        remove_kp_btn = QtWidgets.QPushButton("−")
        remove_kp_btn.setFixedWidth(30)
        remove_kp_btn.clicked.connect(self._remove_keypoint)
        kp_btn_layout.addWidget(remove_kp_btn)
        kp_layout.addLayout(kp_btn_layout)

        splitter.addWidget(kp_widget)

        # Skeleton edges + interactive preview panel
        edges_widget = QtWidgets.QWidget()
        edges_layout = QtWidgets.QVBoxLayout(edges_widget)
        edges_layout.setContentsMargins(0, 0, 0, 0)

        edges_header = QtWidgets.QHBoxLayout()
        edges_header.addWidget(QtWidgets.QLabel("<b>Skeleton Edges</b>"))
        auto_edges_btn = QtWidgets.QPushButton("Auto-infer")
        auto_edges_btn.setToolTip(
            "Infer anatomical skeleton edges from current keypoints"
        )
        auto_edges_btn.clicked.connect(self._auto_edges)
        edges_header.addWidget(auto_edges_btn)
        edges_header.addStretch()
        edges_layout.addLayout(edges_header)

        self.skeleton_preview = _SkeletonPreviewWidget()
        self.skeleton_preview.keypointClicked.connect(self._on_preview_keypoint_clicked)
        edges_layout.addWidget(self.skeleton_preview, 1)

        self.edges_list = QtWidgets.QListWidget()
        self.edges_list.setMaximumHeight(96)
        edges_layout.addWidget(self.edges_list)

        edges_btn_layout = QtWidgets.QHBoxLayout()
        self.edge_left_combo = QtWidgets.QComboBox()
        self.edge_right_combo = QtWidgets.QComboBox()
        edges_btn_layout.addWidget(self.edge_left_combo, 1)
        edges_btn_layout.addWidget(QtWidgets.QLabel("—"))
        edges_btn_layout.addWidget(self.edge_right_combo, 1)
        add_edge_btn = QtWidgets.QPushButton("+")
        add_edge_btn.setFixedWidth(30)
        add_edge_btn.clicked.connect(self._add_edge)
        edges_btn_layout.addWidget(add_edge_btn)
        remove_edge_btn = QtWidgets.QPushButton("−")
        remove_edge_btn.setFixedWidth(30)
        remove_edge_btn.clicked.connect(self._remove_edge)
        edges_btn_layout.addWidget(remove_edge_btn)
        edges_layout.addLayout(edges_btn_layout)

        splitter.addWidget(edges_widget)

        # Symmetry pairs panel
        pairs_widget = QtWidgets.QWidget()
        pairs_layout = QtWidgets.QVBoxLayout(pairs_widget)
        pairs_layout.setContentsMargins(0, 0, 0, 0)

        pairs_header = QtWidgets.QHBoxLayout()
        pairs_header.addWidget(QtWidgets.QLabel("<b>Symmetry Pairs</b>"))
        auto_btn = QtWidgets.QPushButton("Auto-detect")
        auto_btn.clicked.connect(self._auto_symmetry)
        pairs_header.addWidget(auto_btn)
        pairs_header.addStretch()
        pairs_layout.addLayout(pairs_header)

        self.pairs_list = QtWidgets.QListWidget()
        pairs_layout.addWidget(self.pairs_list, 1)

        pairs_btn_layout = QtWidgets.QHBoxLayout()
        self.pair_left_combo = QtWidgets.QComboBox()
        self.pair_right_combo = QtWidgets.QComboBox()
        pairs_btn_layout.addWidget(self.pair_left_combo, 1)
        pairs_btn_layout.addWidget(QtWidgets.QLabel("↔"))
        pairs_btn_layout.addWidget(self.pair_right_combo, 1)
        add_pair_btn = QtWidgets.QPushButton("+")
        add_pair_btn.setFixedWidth(30)
        add_pair_btn.clicked.connect(self._add_pair)
        pairs_btn_layout.addWidget(add_pair_btn)
        remove_pair_btn = QtWidgets.QPushButton("−")
        remove_pair_btn.setFixedWidth(30)
        remove_pair_btn.clicked.connect(self._remove_pair)
        pairs_btn_layout.addWidget(remove_pair_btn)
        pairs_layout.addLayout(pairs_btn_layout)

        splitter.addWidget(pairs_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

        pose_layout.addWidget(splitter, 1)

        layout.addWidget(self.pose_container)

        # Import/Export
        io_layout = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Import Pose Schema…")
        import_btn.clicked.connect(self._import_pose_schema)
        export_btn = QtWidgets.QPushButton("Export Pose Schema…")
        export_btn.clicked.connect(self._export_pose_schema)
        io_layout.addWidget(import_btn)
        io_layout.addWidget(export_btn)
        io_layout.addStretch()
        layout.addLayout(io_layout)

        # Initially hide pose config
        self.pose_container.setVisible(False)

    def _toggle_pose_ui(self, state: bool | int) -> None:
        if isinstance(state, bool):
            visible = state
        else:
            visible = state == Qt.Checked or int(state) != 0
        self.pose_container.setVisible(visible)

    def _apply_preset(self, preset: str) -> None:
        if preset == "Custom":
            return
        try:
            preset_schema = get_pose_schema_preset(preset)
        except KeyError:
            return
        self._load_schema_into_ui(preset_schema)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(preset)
        self.preset_combo.blockSignals(False)

    def _load_schema_into_ui(self, schema: PoseSchema) -> None:
        self._schema = copy.deepcopy(schema)
        self._pending_edge_start = None
        self.skeleton_preview.set_pending_edge_start(None)
        self.kp_list.clear()
        for kp in schema.keypoints:
            self.kp_list.addItem(kp)
        self._update_combo_boxes()
        self._populate_edges_list(schema.base_edges())
        self._populate_pairs_list(schema.base_symmetry_pairs())
        self._sync_preview()

    def _populate_edges_list(self, edges: List[tuple[str, str]]) -> None:
        self.edges_list.clear()
        for left, right in dedup_undirected_pairs(edges):
            item = QtWidgets.QListWidgetItem(f"{left} — {right}")
            item.setData(Qt.UserRole, (left, right))
            self.edges_list.addItem(item)

    def _populate_pairs_list(self, pairs: List[tuple[str, str]]) -> None:
        self.pairs_list.clear()
        for left, right in dedup_undirected_pairs(pairs):
            item = QtWidgets.QListWidgetItem(f"{left} ↔ {right}")
            item.setData(Qt.UserRole, (left, right))
            self.pairs_list.addItem(item)

    def _add_keypoint(self) -> None:
        name = self.kp_add_edit.text().strip()
        if not name:
            return
        existing = {
            self.kp_list.item(i).text().lower() for i in range(self.kp_list.count())
        }
        if name.lower() in existing:
            self.kp_add_edit.clear()
            return
        self.kp_list.addItem(name)
        self.kp_add_edit.clear()
        self._update_combo_boxes()
        self._sync_preview()
        self.preset_combo.setCurrentText("Custom")

    def _remove_keypoint(self) -> None:
        for item in self.kp_list.selectedItems():
            self.kp_list.takeItem(self.kp_list.row(item))
        valid = set(self._get_keypoints())
        kept_edges = [(a, b) for a, b in self._get_edges() if a in valid and b in valid]
        kept_pairs = [(a, b) for a, b in self._get_pairs() if a in valid and b in valid]
        self._populate_edges_list(kept_edges)
        self._populate_pairs_list(kept_pairs)
        if self._pending_edge_start and self._pending_edge_start not in valid:
            self._pending_edge_start = None
            self.skeleton_preview.set_pending_edge_start(None)
        self._update_combo_boxes()
        self._sync_preview()
        self.preset_combo.setCurrentText("Custom")

    def _update_combo_boxes(self) -> None:
        keypoints = self._get_keypoints()
        for combo in (
            self.edge_left_combo,
            self.edge_right_combo,
            self.pair_left_combo,
            self.pair_right_combo,
        ):
            current = combo.currentText()
            combo.clear()
            combo.addItems(keypoints)
            if current in keypoints:
                combo.setCurrentText(current)

    def _get_keypoints(self) -> List[str]:
        return [
            self.kp_list.item(i).text().strip()
            for i in range(self.kp_list.count())
            if self.kp_list.item(i).text().strip()
        ]

    def _get_edges(self) -> List[tuple[str, str]]:
        edges: List[tuple[str, str]] = []
        for i in range(self.edges_list.count()):
            item = self.edges_list.item(i)
            stored = item.data(Qt.UserRole)
            if isinstance(stored, (tuple, list)) and len(stored) == 2:
                edges.append((str(stored[0]).strip(), str(stored[1]).strip()))
                continue
            text = item.text()
            for sep in (" — ", " - "):
                if sep in text:
                    left, right = text.split(sep, 1)
                    edges.append((left.strip(), right.strip()))
                    break
        return dedup_undirected_pairs(edges)

    def _get_pairs(self) -> List[tuple[str, str]]:
        pairs: List[tuple[str, str]] = []
        for i in range(self.pairs_list.count()):
            item = self.pairs_list.item(i)
            stored = item.data(Qt.UserRole)
            if isinstance(stored, (tuple, list)) and len(stored) == 2:
                pairs.append((str(stored[0]).strip(), str(stored[1]).strip()))
                continue
            text = item.text()
            if " ↔ " in text:
                left, right = text.split(" ↔ ", 1)
                pairs.append((left.strip(), right.strip()))
        return dedup_undirected_pairs(pairs)

    def _sync_preview(self) -> None:
        self.skeleton_preview.set_schema(self.get_pose_schema())

    def _on_preview_keypoint_clicked(self, name: str) -> None:
        if not name:
            return
        if self._pending_edge_start is None:
            self._pending_edge_start = name
            self.skeleton_preview.set_pending_edge_start(name)
            return
        first = self._pending_edge_start
        self._pending_edge_start = None
        self.skeleton_preview.set_pending_edge_start(None)
        if first == name:
            return
        current = self._get_edges()
        ukey = (first, name) if first < name else (name, first)
        filtered = [
            pair
            for pair in current
            if ((pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])) != ukey
        ]
        if len(filtered) == len(current):
            filtered.append((first, name))
        self._populate_edges_list(filtered)
        self._sync_preview()
        self.preset_combo.setCurrentText("Custom")

    def _add_edge(self) -> None:
        left = self.edge_left_combo.currentText().strip()
        right = self.edge_right_combo.currentText().strip()
        if not left or not right or left == right:
            return
        current = self._get_edges()
        current.append((left, right))
        self._populate_edges_list(current)
        self._sync_preview()
        self.preset_combo.setCurrentText("Custom")

    def _remove_edge(self) -> None:
        for item in self.edges_list.selectedItems():
            self.edges_list.takeItem(self.edges_list.row(item))
        self._sync_preview()
        self.preset_combo.setCurrentText("Custom")

    def _auto_edges(self) -> None:
        keypoints = self._get_keypoints()
        inferred = PoseSchema.infer_edges(keypoints, preset="auto")
        self._populate_edges_list(inferred)
        self._sync_preview()

    def _add_pair(self) -> None:
        left = self.pair_left_combo.currentText()
        right = self.pair_right_combo.currentText()
        if not left or not right or left == right:
            return
        current = self._get_pairs()
        current.append((left, right))
        self._populate_pairs_list(current)
        self._sync_preview()

    def _remove_pair(self) -> None:
        for item in self.pairs_list.selectedItems():
            self.pairs_list.takeItem(self.pairs_list.row(item))
        self._sync_preview()

    def _auto_symmetry(self) -> None:
        keypoints = self._get_keypoints()
        pairs = PoseSchema.infer_symmetry_pairs(keypoints)
        self._populate_pairs_list(pairs)
        self._sync_preview()

    def _import_pose_schema(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Pose Schema",
            "",
            "Pose Schema (*.json *.yaml *.yml);;All Files (*)",
        )
        if not path:
            return
        try:
            schema = PoseSchema.load(path)
            self._load_schema_into_ui(schema)
            self.enable_pose_check.setChecked(True)
            self.preset_combo.setCurrentText("Custom")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Import Failed", f"Unable to import pose schema:\n{exc}"
            )

    def _export_pose_schema(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Pose Schema",
            "",
            "Pose Schema (*.json *.yaml *.yml)",
        )
        if not path:
            return
        schema = self.get_pose_schema()
        try:
            schema.save(path)
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Pose schema saved to:\n{path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Unable to export pose schema:\n{exc}"
            )

    def is_pose_enabled(self) -> bool:
        return self.enable_pose_check.isChecked()

    def get_pose_schema(self) -> PoseSchema:
        keypoints = self._get_keypoints()
        edges = self._get_edges()
        pairs = self._get_pairs()
        base = self._schema
        return PoseSchema(
            version=str(getattr(base, "version", "1.0") or "1.0"),
            keypoints=keypoints,
            edges=edges,
            symmetry_pairs=pairs,
            flip_idx=(
                list(base.flip_idx)
                if base is not None
                and isinstance(base.flip_idx, list)
                and len(base.flip_idx) == len(keypoints)
                else None
            ),
            instances=list(getattr(base, "instances", []) or []),
            instance_separator=str(getattr(base, "instance_separator", "_") or "_"),
        )


class SummaryPage(QtWidgets.QWizardPage):
    """Page 5: Summary and confirmation."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Summary")
        self.setSubTitle(
            "Review your project configuration. Click Finish to create the project."
        )

        layout = QtWidgets.QVBoxLayout(self)

        self.summary_text = QtWidgets.QTextBrowser()
        self.summary_text.setOpenExternalLinks(False)
        layout.addWidget(self.summary_text)

        # Auto-open project checkbox
        self.open_project_check = QtWidgets.QCheckBox(
            "Open the first video after project creation"
        )
        self.open_project_check.setChecked(True)
        layout.addWidget(self.open_project_check)

    def initializePage(self) -> None:
        wizard = self.wizard()
        if not isinstance(wizard, ProjectWizard):
            return

        # Build summary HTML
        html_parts = []

        # Project info
        project_path = wizard.project_info_page.project_path()
        html_parts.append("<h3>📁 Project</h3>")
        html_parts.append(
            f"<p><b>Name:</b> {html.escape(project_path.name, quote=True)}</p>"
        )
        html_parts.append(
            f"<p><b>Location:</b> {html.escape(str(project_path.parent), quote=True)}</p>"
        )

        # Videos
        videos = wizard.video_import_page.get_videos()
        html_parts.append(f"<h3>🎬 Videos ({len(videos)})</h3>")
        if videos:
            html_parts.append("<ul>")
            for v in videos[:5]:
                html_parts.append(f"<li>{html.escape(Path(v).name, quote=True)}</li>")
            if len(videos) > 5:
                html_parts.append(f"<li><i>...and {len(videos) - 5} more</i></li>")
            html_parts.append("</ul>")
        else:
            html_parts.append("<p><i>No videos imported (can add later)</i></p>")

        # Schema
        schema = wizard.subject_behavior_page.get_schema()
        html_parts.append(f"<h3>🐭 Subjects ({len(schema.subjects)})</h3>")
        if schema.subjects:
            names = ", ".join(
                html.escape(str(s.name), quote=True) for s in schema.subjects[:5]
            )
            if len(schema.subjects) > 5:
                names += f", ...+{len(schema.subjects) - 5} more"
            html_parts.append(f"<p>{names}</p>")

        html_parts.append(f"<h3>📋 Behaviors ({len(schema.behaviors)})</h3>")
        if schema.behaviors:
            names = ", ".join(
                html.escape(str(b.name), quote=True) for b in schema.behaviors[:5]
            )
            if len(schema.behaviors) > 5:
                names += f", ...+{len(schema.behaviors) - 5} more"
            html_parts.append(f"<p>{names}</p>")

        # Pose schema
        if wizard.pose_schema_page.is_pose_enabled():
            pose = wizard.pose_schema_page.get_pose_schema()
            html_parts.append(
                f"<h3>📍 Pose Schema ({len(pose.keypoints)} keypoints, "
                f"{len(pose.edges)} skeleton edges, "
                f"{len(pose.symmetry_pairs)} symmetry pairs)</h3>"
            )
            if pose.keypoints:
                kps = ", ".join(
                    html.escape(str(kp), quote=True) for kp in pose.keypoints[:8]
                )
                if len(pose.keypoints) > 8:
                    kps += f", ...+{len(pose.keypoints) - 8} more"
                html_parts.append(f"<p>{kps}</p>")
        else:
            html_parts.append("<h3>📍 Pose Schema</h3>")
            html_parts.append("<p><i>Disabled</i></p>")

        self.summary_text.setHtml("".join(html_parts))

    def should_open_video(self) -> bool:
        return self.open_project_check.isChecked()


class ProjectWizard(QtWidgets.QWizard):
    """Main project setup wizard combining all pages."""

    # Signal emitted when project is created successfully
    project_created = QtCore.Signal(Path, ProjectSchema, list)  # path, schema, videos

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("New Annolid Project")
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)
        self.setMinimumSize(800, 600)

        # Set up pages
        self.project_info_page = ProjectInfoPage()
        self.video_import_page = VideoImportPage()
        self.subject_behavior_page = SubjectBehaviorPage()
        self.pose_schema_page = PoseSchemaPage()
        self.summary_page = SummaryPage()

        self.addPage(self.project_info_page)
        self.addPage(self.video_import_page)
        self.addPage(self.subject_behavior_page)
        self.addPage(self.pose_schema_page)
        self.addPage(self.summary_page)

        # Customize button text
        self.setButtonText(QtWidgets.QWizard.FinishButton, "Create Project")
        self.setButtonText(QtWidgets.QWizard.NextButton, "Next →")
        self.setButtonText(QtWidgets.QWizard.BackButton, "← Back")

        # Set window icon if available
        try:
            icon_path = Path(__file__).parent.parent / "icons" / "icon.png"
            if icon_path.exists():
                self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        except Exception:
            pass

    def accept(self) -> None:
        """Create the project workspace, schema, and video manifest when Finish is clicked."""
        try:
            project_path = self.project_info_page.project_path()
            project_name = project_path.name
            description = self.project_info_page.project_description()

            service = ProjectService()
            project_cfg = {"description": description} if description else None
            created, message = service.create_project(
                project_path.parent,
                project_name,
                project_cfg,
            )
            if not created:
                raise RuntimeError(message)

            # Save project behavior & pose schemas into the same project.annolid.json envelope.
            schema = self.subject_behavior_page.get_schema()

            if self.pose_schema_page.is_pose_enabled():
                pose_schema = self.pose_schema_page.get_pose_schema()
                pose_schema_path = project_path / "pose_schema.json"
                pose_schema.save(str(pose_schema_path))
                schema.pose_schema_path = pose_schema_path.name
                schema.pose_schema = pose_schema.to_dict()

            schema_path = project_path / PROJECT_CONFIG_FILENAME
            save_behavior_spec(schema, schema_path)

            # Register imported videos (optionally copying into videos/).
            raw_videos = self.video_import_page.get_videos()
            videos = list(raw_videos)
            if raw_videos:
                ok_vid, msg_vid, registered = service.register_videos(
                    raw_videos,
                    copy_into_project=self.video_import_page.should_copy_videos(),
                )
                if not ok_vid:
                    raise RuntimeError(msg_vid or "Failed to register project videos.")
                if registered:
                    videos = registered

            self.created_project_path = project_path
            self.created_videos = list(videos)

            # Emit signal for main window to handle
            self.project_created.emit(project_path, schema, videos)

            super().accept()

        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Project Creation Failed",
                f"Unable to create project:\n{exc}",
            )

    def should_open_video(self) -> bool:
        return self.summary_page.should_open_video()
