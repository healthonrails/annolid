"""Project Setup Wizard - Streamlined project creation workflow.

A modern QWizard-based interface for creating new Annolid projects,
guiding users through project definition, video import, schema setup,
and pose configuration in a step-by-step flow.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.behavior.project_schema import (
    BehaviorDefinition,
    CategoryDefinition,
    ModifierDefinition,
    ProjectSchema,
    SubjectDefinition,
    default_schema,
    save_schema as save_project_schema,
)
from annolid.annotation.pose_schema import PoseSchema


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

    def isComplete(self) -> bool:
        name = self.name_edit.text().strip()
        location = self.location_edit.text().strip()
        return bool(name and location and Path(location).is_dir())

    def validatePage(self) -> bool:
        name = self.name_edit.text().strip()
        location = Path(self.location_edit.text().strip())
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
        return Path(self.location_edit.text().strip()) / self.name_edit.text().strip()


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
        self.video_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
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

        # Tip
        tip_label = QtWidgets.QLabel(
            "<i>Tip: You can skip this step and import videos later.</i>"
        )
        tip_label.setStyleSheet("color: gray;")
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
        return [
            self.video_list.item(i).text()
            for i in range(self.video_list.count())
        ]


class SubjectBehaviorPage(QtWidgets.QWizardPage):
    """Page 3: Define subjects and behaviors (simplified schema editor)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Subjects & Behaviors")
        self.setSubTitle(
            "Define the subjects (animals/objects) you'll track and the "
            "behaviors you want to annotate. You can add more details later."
        )

        self._schema = default_schema()

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
        self.subjects_list.setDragDropMode(
            QtWidgets.QAbstractItemView.InternalMove)
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
        self.behaviors_list.setDragDropMode(
            QtWidgets.QAbstractItemView.InternalMove)
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
        from annolid.behavior.project_schema import load_schema as load_project_schema

        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Project Schema",
            "",
            "Schema Files (*.json *.yaml *.yml);;All Files (*)",
        )
        if not path_str:
            return
        try:
            schema = load_project_schema(Path(path_str))
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
            save_project_schema(schema, Path(path_str))
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

        layout = QtWidgets.QVBoxLayout(self)

        # Enable pose checkbox
        self.enable_pose_check = QtWidgets.QCheckBox(
            "Enable pose estimation for this project"
        )
        self.enable_pose_check.stateChanged.connect(self._toggle_pose_ui)
        layout.addWidget(self.enable_pose_check)

        # Pose config container (hidden by default)
        self.pose_container = QtWidgets.QWidget()
        pose_layout = QtWidgets.QVBoxLayout(self.pose_container)
        pose_layout.setContentsMargins(0, 10, 0, 0)

        # Preset selector
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Mouse (9 keypoints)",
            "Mouse (11 keypoints)",
            "Fly (5 keypoints)",
            "Fish (5 keypoints)",
            "Human COCO (17 keypoints)",
        ])
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        pose_layout.addLayout(preset_layout)

        # Splitter for keypoints and pairs
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Keypoints panel
        kp_widget = QtWidgets.QWidget()
        kp_layout = QtWidgets.QVBoxLayout(kp_widget)
        kp_layout.setContentsMargins(0, 0, 0, 0)
        kp_layout.addWidget(QtWidgets.QLabel(
            "<b>Keypoints</b> (order matters)"))

        self.kp_list = QtWidgets.QListWidget()
        self.kp_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
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
        splitter.setStretchFactor(1, 1)

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

    def _toggle_pose_ui(self, state: int) -> None:
        self.pose_container.setVisible(state == Qt.Checked)

    def _apply_preset(self, preset: str) -> None:
        presets = {
            "Mouse (9 keypoints)": [
                "nose", "left_ear", "right_ear", "left_forepaw", "right_forepaw",
                "left_hindpaw", "right_hindpaw", "tail_base", "tail_tip"
            ],
            "Mouse (11 keypoints)": [
                "nose", "left_ear", "right_ear", "neck", "left_forepaw",
                "right_forepaw", "spine", "left_hindpaw", "right_hindpaw",
                "tail_base", "tail_tip"
            ],
            "Fly (5 keypoints)": [
                "head", "thorax", "abdomen", "left_wing", "right_wing"
            ],
            "Fish (5 keypoints)": [
                "head", "dorsal_fin", "tail", "left_pectoral", "right_pectoral"
            ],
            "Human COCO (17 keypoints)": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
        }

        if preset in presets:
            self.kp_list.clear()
            for kp in presets[preset]:
                self.kp_list.addItem(kp)
            self._update_combo_boxes()
            self._auto_symmetry()
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(preset)
            self.preset_combo.blockSignals(False)

    def _add_keypoint(self) -> None:
        name = self.kp_add_edit.text().strip()
        if not name:
            return
        existing = {self.kp_list.item(i).text().lower()
                    for i in range(self.kp_list.count())}
        if name.lower() in existing:
            self.kp_add_edit.clear()
            return
        self.kp_list.addItem(name)
        self.kp_add_edit.clear()
        self._update_combo_boxes()
        self.preset_combo.setCurrentText("Custom")

    def _remove_keypoint(self) -> None:
        for item in self.kp_list.selectedItems():
            self.kp_list.takeItem(self.kp_list.row(item))
        self._update_combo_boxes()
        self.preset_combo.setCurrentText("Custom")

    def _update_combo_boxes(self) -> None:
        keypoints = self._get_keypoints()
        current_left = self.pair_left_combo.currentText()
        current_right = self.pair_right_combo.currentText()

        self.pair_left_combo.clear()
        self.pair_right_combo.clear()
        self.pair_left_combo.addItems(keypoints)
        self.pair_right_combo.addItems(keypoints)

        if current_left in keypoints:
            self.pair_left_combo.setCurrentText(current_left)
        if current_right in keypoints:
            self.pair_right_combo.setCurrentText(current_right)

    def _get_keypoints(self) -> List[str]:
        return [
            self.kp_list.item(i).text().strip()
            for i in range(self.kp_list.count())
            if self.kp_list.item(i).text().strip()
        ]

    def _add_pair(self) -> None:
        left = self.pair_left_combo.currentText()
        right = self.pair_right_combo.currentText()
        if not left or not right or left == right:
            return
        pair_text = f"{left} ↔ {right}"
        # Check for duplicates
        for i in range(self.pairs_list.count()):
            existing = self.pairs_list.item(i).text()
            if pair_text == existing or f"{right} ↔ {left}" == existing:
                return
        self.pairs_list.addItem(pair_text)

    def _remove_pair(self) -> None:
        for item in self.pairs_list.selectedItems():
            self.pairs_list.takeItem(self.pairs_list.row(item))

    def _auto_symmetry(self) -> None:
        keypoints = self._get_keypoints()
        pairs = PoseSchema.infer_symmetry_pairs(keypoints)
        self.pairs_list.clear()
        for left, right in pairs:
            self.pairs_list.addItem(f"{left} ↔ {right}")

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
            self._schema = schema
            self.kp_list.clear()
            for kp in schema.keypoints:
                self.kp_list.addItem(kp)
            self._update_combo_boxes()
            self.pairs_list.clear()
            for left, right in schema.symmetry_pairs:
                self.pairs_list.addItem(f"{left} ↔ {right}")
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
        pairs = []
        for i in range(self.pairs_list.count()):
            text = self.pairs_list.item(i).text()
            if " ↔ " in text:
                left, right = text.split(" ↔ ")
                pairs.append((left.strip(), right.strip()))
        return PoseSchema(keypoints=keypoints, symmetry_pairs=pairs)


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
        html_parts.append(f"<h3>📁 Project</h3>")
        html_parts.append(f"<p><b>Name:</b> {project_path.name}</p>")
        html_parts.append(f"<p><b>Location:</b> {project_path.parent}</p>")

        # Videos
        videos = wizard.video_import_page.get_videos()
        html_parts.append(f"<h3>🎬 Videos ({len(videos)})</h3>")
        if videos:
            html_parts.append("<ul>")
            for v in videos[:5]:
                html_parts.append(f"<li>{Path(v).name}</li>")
            if len(videos) > 5:
                html_parts.append(
                    f"<li><i>...and {len(videos) - 5} more</i></li>")
            html_parts.append("</ul>")
        else:
            html_parts.append(
                "<p><i>No videos imported (can add later)</i></p>")

        # Schema
        schema = wizard.subject_behavior_page.get_schema()
        html_parts.append(f"<h3>🐭 Subjects ({len(schema.subjects)})</h3>")
        if schema.subjects:
            names = ", ".join(s.name for s in schema.subjects[:5])
            if len(schema.subjects) > 5:
                names += f", ...+{len(schema.subjects) - 5} more"
            html_parts.append(f"<p>{names}</p>")

        html_parts.append(f"<h3>📋 Behaviors ({len(schema.behaviors)})</h3>")
        if schema.behaviors:
            names = ", ".join(b.name for b in schema.behaviors[:5])
            if len(schema.behaviors) > 5:
                names += f", ...+{len(schema.behaviors) - 5} more"
            html_parts.append(f"<p>{names}</p>")

        # Pose schema
        if wizard.pose_schema_page.is_pose_enabled():
            pose = wizard.pose_schema_page.get_pose_schema()
            html_parts.append(
                f"<h3>📍 Pose Schema ({len(pose.keypoints)} keypoints)</h3>")
            if pose.keypoints:
                kps = ", ".join(pose.keypoints[:8])
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
    project_created = QtCore.Signal(
        Path, ProjectSchema, list)  # path, schema, videos

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
            from annolid.gui.app import AnnolidWindow
            icon_path = Path(__file__).parent.parent / "icons" / "icon.png"
            if icon_path.exists():
                self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        except Exception:
            pass

    def accept(self) -> None:
        """Create the project when Finish is clicked."""
        try:
            project_path = self.project_info_page.project_path()
            project_path.mkdir(parents=True, exist_ok=True)

            # Save project schema
            schema = self.subject_behavior_page.get_schema()

            # Integrate pose schema if enabled
            if self.pose_schema_page.is_pose_enabled():
                pose_schema = self.pose_schema_page.get_pose_schema()
                pose_schema_path = project_path / "pose_schema.json"
                pose_schema.save(str(pose_schema_path))
                schema.pose_schema_path = str(pose_schema_path)

            # Save project schema
            schema_path = project_path / "project.annolid.json"
            save_project_schema(schema, schema_path)

            # Copy/link videos (optional - just track paths for now)
            videos = self.video_import_page.get_videos()

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
