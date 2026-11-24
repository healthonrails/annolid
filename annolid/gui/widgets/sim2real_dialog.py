"""Sim-to-Real Studio dialog: example-video matching -> 3D Mouse/Fly simulation -> Colab iterative fine-tuning -> evaluation & reports."""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Any, Callable, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.colab import (
    ColabAuthRequiredError,
    ColabLogMonitorParser,
    ColabWorkflowService,
    SUPPORTED_ACCELERATORS,
)
from annolid.gui.widgets.colab_dialog import (
    ColabSignInDialog,
    _ColabMetricsChartWidget,
)
from annolid.sim2real import (
    FLY_BEHAVIORS,
    FLY_CAMERA_VIEWS,
    MOUSE_ARENAS,
    MOUSE_CAMERA_VIEWS,
    MOUSE_COAT_PAIRS,
    MOUSE_HEADSTAGES,
    MOUSE_LIGHTING_PRESETS,
    MOUSE_OCCLUDERS,
    MOUSE_SCENARIOS,
    Sim2RealProjectState,
    Sim2RealWorkflowService,
    SimDatasetConfig,
    VideoEnvironmentProfile,
)


class _Sim2RealWorker(QtCore.QThread):
    logLine = QtCore.Signal(str)
    progressChanged = QtCore.Signal(int, str)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        cancel_event: Optional[threading.Event] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._fn = fn
        self._cancel_event = cancel_event or threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = self._fn(
                log_callback=self.logLine.emit,
                progress_callback=self.progressChanged.emit,
                cancel_event=self._cancel_event,
            )
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(exc)


class Sim2RealStudioDialog(QtWidgets.QDialog):
    """Guided 5-step studio for Example Videos -> 3D Mouse/Fly Sim -> Colab Train/Iterate -> Reports."""

    def __init__(
        self,
        *,
        project_dir: Optional[str | Path] = None,
        initial_video_paths: Optional[list[str]] = None,
        colab_service: Optional[ColabWorkflowService] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            self.tr("Annolid Sim-to-Real Studio (3D Simulation → Colab)")
        )
        self.resize(1120, 780)

        self._colab_service = colab_service or ColabWorkflowService()
        default_proj = (
            Path(project_dir).expanduser().resolve()
            if project_dir
            else (Path.home() / ".annolid" / "sim2real_projects" / "default_project")
        )
        self._service = Sim2RealWorkflowService(
            default_proj, colab_service=self._colab_service
        )
        self._worker: Optional[_Sim2RealWorker] = None
        self._monitor_parser = ColabLogMonitorParser()
        self._action_buttons: list[QtWidgets.QAbstractButton] = []

        self._build_ui()
        self.project_dir_edit.setText(str(default_proj))
        if initial_video_paths:
            for vp in initial_video_paths:
                if vp and Path(vp).exists():
                    self._add_example_video_item(str(Path(vp).resolve()))

        self._sync_ui_from_state()
        self._refresh_colab_badge()

    # ------------------------------------------------------------------ #
    # UI Layout
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)

        # Header + Project Dir + Personal Colab Badge
        header = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(self.tr("Sim-to-Real Studio"))
        tf = title.font()
        tf.setPointSize(max(13, tf.pointSize() + 3))
        tf.setBold(True)
        title.setFont(tf)
        subtitle = QtWidgets.QLabel(
            self.tr(
                "Add real example videos → match Mouse 3D or FlyBody 3D simulation → "
                "train/fine-tune on Google Colab → evaluate, iterate & generate reports."
            )
        )
        subtitle.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, 1)

        self.colab_status_label = QtWidgets.QLabel(self.tr("Colab: checking…"))
        self.colab_signin_btn = QtWidgets.QPushButton(self.tr("Colab Sign-In…"))
        self.colab_signin_btn.clicked.connect(self._on_colab_sign_in_clicked)
        header.addWidget(self.colab_status_label)
        header.addWidget(self.colab_signin_btn)
        root.addLayout(header)

        # Project workspace bar
        proj_row = QtWidgets.QHBoxLayout()
        proj_row.addWidget(QtWidgets.QLabel(self.tr("Project Folder:")))
        self.project_dir_edit = QtWidgets.QLineEdit()
        proj_browse_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        proj_browse_btn.clicked.connect(self._on_browse_project_dir)
        proj_load_btn = QtWidgets.QPushButton(self.tr("Load / Switch"))
        proj_load_btn.clicked.connect(self._on_load_project_clicked)
        proj_row.addWidget(self.project_dir_edit, 1)
        proj_row.addWidget(proj_browse_btn)
        proj_row.addWidget(proj_load_btn)
        root.addLayout(proj_row)

        # 5-Step Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(
            self._build_examples_tab(), self.tr("1. Example Videos && Match")
        )
        self.tabs.addTab(self._build_simulation_tab(), self.tr("2. 3D Sim Dataset"))
        self.tabs.addTab(
            self._build_train_iterate_tab(), self.tr("3. Train && Iterate")
        )
        self.tabs.addTab(self._build_results_tab(), self.tr("4. Iterations && Advisor"))
        self.tabs.addTab(self._build_report_tab(), self.tr("5. Report"))
        root.addWidget(self.tabs, 1)

        # Global progress bar + Cancel
        footer = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QtWidgets.QLabel(self.tr("Ready"))
        self.cancel_btn = QtWidgets.QPushButton(self.tr("Stop"))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        close_btn = QtWidgets.QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.close)
        footer.addWidget(self.progress_bar, 1)
        footer.addWidget(self.status_label, 1)
        footer.addWidget(self.cancel_btn)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def _build_examples_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        top_split = QtWidgets.QHBoxLayout()
        left_box = QtWidgets.QGroupBox(self.tr("Real Example Videos / Image Folders"))
        left_lay = QtWidgets.QVBoxLayout(left_box)
        self.videos_list = QtWidgets.QListWidget()
        self.videos_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        left_lay.addWidget(self.videos_list, 1)

        vid_btns = QtWidgets.QHBoxLayout()
        add_vid_btn = QtWidgets.QPushButton(self.tr("Add Video(s)…"))
        add_vid_btn.clicked.connect(self._on_add_videos_clicked)
        add_dir_btn = QtWidgets.QPushButton(self.tr("Add Image Folder…"))
        add_dir_btn.clicked.connect(self._on_add_image_folder_clicked)
        rem_vid_btn = QtWidgets.QPushButton(self.tr("Remove Selected"))
        rem_vid_btn.clicked.connect(self._on_remove_selected_videos)
        vid_btns.addWidget(add_vid_btn)
        vid_btns.addWidget(add_dir_btn)
        vid_btns.addWidget(rem_vid_btn)
        left_lay.addLayout(vid_btns)

        analyze_row = QtWidgets.QHBoxLayout()
        analyze_row.addWidget(QtWidgets.QLabel(self.tr("Species Hint:")))
        self.species_hint_combo = QtWidgets.QComboBox()
        self.species_hint_combo.addItem(self.tr("Auto-Detect (Mouse / Fly)"), "auto")
        self.species_hint_combo.addItem(self.tr("Mouse (3D Three.js)"), "mouse")
        self.species_hint_combo.addItem(self.tr("Fly (FlyBody 3D MuJoCo)"), "fly")
        self.analyze_btn = QtWidgets.QPushButton(
            self.tr("Analyze Environment && Match 3D Simulation")
        )
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        self._action_buttons.append(self.analyze_btn)
        analyze_row.addWidget(self.species_hint_combo)
        analyze_row.addWidget(self.analyze_btn, 1)
        left_lay.addLayout(analyze_row)

        right_box = QtWidgets.QGroupBox(self.tr("Inferred Environment Profile"))
        right_lay = QtWidgets.QVBoxLayout(right_box)
        self.profile_summary_text = QtWidgets.QPlainTextEdit()
        self.profile_summary_text.setReadOnly(True)
        self.profile_summary_text.setPlaceholderText(
            self.tr(
                "Add one or more example videos and click 'Analyze Environment' to extract "
                "arena geometry, camera perspective, lighting/IR profile, coat/species cues, "
                "and reference background."
            )
        )
        right_lay.addWidget(self.profile_summary_text, 1)

        preview_row = QtWidgets.QHBoxLayout()
        self.bg_preview_label = QtWidgets.QLabel(self.tr("Reference Background"))
        self.bg_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.bg_preview_label.setMinimumSize(220, 150)
        self.bg_preview_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.sample_preview_label = QtWidgets.QLabel(self.tr("Example Frame Samples"))
        self.sample_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sample_preview_label.setMinimumSize(220, 150)
        self.sample_preview_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        preview_row.addWidget(self.bg_preview_label, 1)
        preview_row.addWidget(self.sample_preview_label, 1)
        right_lay.addLayout(preview_row)

        top_split.addWidget(left_box, 1)
        top_split.addWidget(right_box, 1)
        layout.addLayout(top_split, 1)
        return page

    def _build_simulation_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        split = QtWidgets.QHBoxLayout()
        form_group = QtWidgets.QGroupBox(self.tr("3D Simulation & Domain Matching"))
        form = QtWidgets.QFormLayout(form_group)

        self.sim_species_combo = QtWidgets.QComboBox()
        self.sim_species_combo.addItem(self.tr("Mouse 3D (Bundled Three.js)"), "mouse")
        self.sim_species_combo.addItem(self.tr("Fly 3D (FlyBody MuJoCo)"), "fly")
        self.sim_species_combo.currentIndexChanged.connect(self._on_species_changed)
        form.addRow(self.tr("Simulation Engine:"), self.sim_species_combo)

        self.sim_task_combo = QtWidgets.QComboBox()
        self.sim_task_combo.addItem(
            self.tr("YOLO Instance / Part Segmentation"), "yolo_seg"
        )
        self.sim_task_combo.addItem(self.tr("YOLO Pose Keypoints"), "yolo_pose")
        self.sim_task_combo.addItem(
            self.tr("YOLO Bounding-Box Detection"), "yolo_detect"
        )
        form.addRow(self.tr("Dataset Task:"), self.sim_task_combo)

        self.sim_level_combo = QtWidgets.QComboBox()
        self.sim_level_combo.addItem(self.tr("Whole-Animal Instances"), "instances")
        self.sim_level_combo.addItem(self.tr("Anatomical Body Parts"), "parts")
        form.addRow(self.tr("Annotation Level:"), self.sim_level_combo)

        self.sim_arena_combo = QtWidgets.QComboBox()
        for a in MOUSE_ARENAS:
            self.sim_arena_combo.addItem(a, a)
        form.addRow(self.tr("Arena Geometry:"), self.sim_arena_combo)

        self.sim_camera_combo = QtWidgets.QComboBox()
        form.addRow(self.tr("Camera View:"), self.sim_camera_combo)

        self.sim_lighting_combo = QtWidgets.QComboBox()
        for lt in MOUSE_LIGHTING_PRESETS:
            self.sim_lighting_combo.addItem(lt, lt)
        form.addRow(self.tr("Lighting / IR:"), self.sim_lighting_combo)

        self.sim_coat_combo = QtWidgets.QComboBox()
        for cp in MOUSE_COAT_PAIRS:
            self.sim_coat_combo.addItem(cp, cp)
        form.addRow(self.tr("Mouse Coat Pair:"), self.sim_coat_combo)

        self.sim_occluder_combo = QtWidgets.QComboBox()
        for oc in MOUSE_OCCLUDERS:
            self.sim_occluder_combo.addItem(oc, oc)
        form.addRow(self.tr("Occluder:"), self.sim_occluder_combo)

        self.sim_headstage_combo = QtWidgets.QComboBox()
        for hs in MOUSE_HEADSTAGES:
            self.sim_headstage_combo.addItem(hs, hs)
        form.addRow(self.tr("Mouse Headstage:"), self.sim_headstage_combo)

        self.fly_count_spin = QtWidgets.QSpinBox()
        self.fly_count_spin.setRange(1, 6)
        self.fly_count_spin.setValue(1)
        form.addRow(self.tr("Fly Subject Count:"), self.fly_count_spin)

        self.frames_per_scen_spin = QtWidgets.QSpinBox()
        self.frames_per_scen_spin.setRange(4, 200)
        self.frames_per_scen_spin.setValue(24)
        form.addRow(self.tr("Frames / Scenario:"), self.frames_per_scen_spin)

        self.sim_seed_spin = QtWidgets.QSpinBox()
        self.sim_seed_spin.setRange(1, 999999)
        self.sim_seed_spin.setValue(42)
        form.addRow(self.tr("Random Seed:"), self.sim_seed_spin)

        self.match_photo_chk = QtWidgets.QCheckBox(
            self.tr("Match example video photometry (Lab color/IR/blur/noise/vignette)")
        )
        self.match_photo_chk.setChecked(True)
        form.addRow(self.match_photo_chk)

        self.photo_strength_spin = QtWidgets.QDoubleSpinBox()
        self.photo_strength_spin.setRange(0.0, 1.0)
        self.photo_strength_spin.setSingleStep(0.05)
        self.photo_strength_spin.setValue(0.75)
        form.addRow(self.tr("Photometry Strength:"), self.photo_strength_spin)

        self.dr_spin = QtWidgets.QDoubleSpinBox()
        self.dr_spin.setRange(0.0, 1.0)
        self.dr_spin.setSingleStep(0.05)
        self.dr_spin.setValue(0.25)
        form.addRow(self.tr("Domain Randomization:"), self.dr_spin)

        self.composite_bg_chk = QtWidgets.QCheckBox(
            self.tr("Composite simulated subjects onto extracted real arena background")
        )
        form.addRow(self.composite_bg_chk)

        real_lbl_row = QtWidgets.QHBoxLayout()
        self.real_labeled_edit = QtWidgets.QLineEdit()
        self.real_labeled_edit.setPlaceholderText(
            self.tr("Optional folder of real LabelMe JSON+images to mix in")
        )
        real_lbl_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        real_lbl_btn.clicked.connect(self._on_browse_real_labeled)
        real_lbl_row.addWidget(self.real_labeled_edit, 1)
        real_lbl_row.addWidget(real_lbl_btn)
        form.addRow(self.tr("Mix Real Labels:"), real_lbl_row)

        self.allow_fallback_chk = QtWidgets.QCheckBox(
            self.tr(
                "Allow procedural fallback if Chromium / FlyBody MuJoCo is unavailable"
            )
        )
        form.addRow(self.allow_fallback_chk)

        # Right side: Scenarios multi-select + Generate button + Preview
        right_col = QtWidgets.QVBoxLayout()
        scen_group = QtWidgets.QGroupBox(
            self.tr("Scenarios / Behaviors (Select 1 or more)")
        )
        scen_lay = QtWidgets.QVBoxLayout(scen_group)
        self.scenarios_list = QtWidgets.QListWidget()
        self.scenarios_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        scen_lay.addWidget(self.scenarios_list, 1)

        self.generate_ds_btn = QtWidgets.QPushButton(
            self.tr("Generate Environment-Matched Synthetic Dataset")
        )
        self.generate_ds_btn.clicked.connect(self._on_generate_dataset_clicked)
        self._action_buttons.append(self.generate_ds_btn)
        scen_lay.addWidget(self.generate_ds_btn)
        right_col.addWidget(scen_group, 1)

        prev_group = QtWidgets.QGroupBox(self.tr("Generated Dataset Preview"))
        prev_lay = QtWidgets.QVBoxLayout(prev_group)
        self.sim_dataset_info = QtWidgets.QLabel(
            self.tr("No synthetic dataset generated yet.")
        )
        self.sim_dataset_info.setWordWrap(True)
        self.sim_preview_label = QtWidgets.QLabel(
            self.tr("Preview appears here after generation")
        )
        self.sim_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sim_preview_label.setMinimumSize(320, 220)
        self.sim_preview_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        prev_lay.addWidget(self.sim_dataset_info)
        prev_lay.addWidget(self.sim_preview_label, 1)
        right_col.addWidget(prev_group, 1)

        split.addWidget(form_group, 1)
        split.addLayout(right_col, 1)
        layout.addLayout(split, 1)

        self._populate_species_specific_controls("mouse")
        return page

    def _build_train_iterate_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        ctrl_box = QtWidgets.QGroupBox(
            self.tr("Training, Fine-Tuning & Iterative Refinement")
        )
        grid = QtWidgets.QGridLayout(ctrl_box)

        self.train_backend_combo = QtWidgets.QComboBox()
        self.train_backend_combo.addItem(
            self.tr("Google Colab GPU (Recommended)"), "colab"
        )
        self.train_backend_combo.addItem(self.tr("Local Machine"), "local")

        self.colab_backend_combo = QtWidgets.QComboBox()
        self.colab_backend_combo.addItem(
            self.tr("Auto (Colab CLI / API / Notebook)"), "auto"
        )
        self.colab_backend_combo.addItem(self.tr("Colab CLI (google-colab-cli)"), "cli")
        self.colab_backend_combo.addItem(self.tr("Colab REST API"), "api")
        self.colab_backend_combo.addItem(
            self.tr("Drive / Notebook (.ipynb)"), "notebook"
        )

        self.gpu_combo = QtWidgets.QComboBox()
        for acc in SUPPORTED_ACCELERATORS:
            self.gpu_combo.addItem(acc, acc)

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 300)
        self.epochs_spin.setValue(20)

        self.imgsz_spin = QtWidgets.QSpinBox()
        self.imgsz_spin.setRange(128, 1280)
        self.imgsz_spin.setSingleStep(64)
        self.imgsz_spin.setValue(640)

        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)

        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1, 10)
        self.iterations_spin.setValue(2)

        self.weights_edit = QtWidgets.QLineEdit()
        self.weights_edit.setPlaceholderText(
            self.tr("Auto (task base weights on Iter 1, previous best.pt on Iter 2+)")
        )
        w_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        w_browse.clicked.connect(self._on_browse_weights)

        self.gt_eval_edit = QtWidgets.QLineEdit()
        self.gt_eval_edit.setPlaceholderText(
            self.tr("Optional labeled validation LabelMe folder for supervised mIoU/F1")
        )
        gt_browse = QtWidgets.QPushButton(self.tr("Browse…"))
        gt_browse.clicked.connect(self._on_browse_gt_dir)

        self.auto_adjust_chk = QtWidgets.QCheckBox(
            self.tr("Automatically apply advisor refinements between loop iterations")
        )
        self.auto_adjust_chk.setChecked(True)
        self.keep_runtime_chk = QtWidgets.QCheckBox(
            self.tr("Keep Colab runtime warm after finishing")
        )

        grid.addWidget(QtWidgets.QLabel(self.tr("Execution:")), 0, 0)
        grid.addWidget(self.train_backend_combo, 0, 1)
        grid.addWidget(QtWidgets.QLabel(self.tr("Colab Backend:")), 0, 2)
        grid.addWidget(self.colab_backend_combo, 0, 3)
        grid.addWidget(QtWidgets.QLabel(self.tr("Accelerator:")), 0, 4)
        grid.addWidget(self.gpu_combo, 0, 5)

        grid.addWidget(QtWidgets.QLabel(self.tr("Epochs:")), 1, 0)
        grid.addWidget(self.epochs_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel(self.tr("Image Size:")), 1, 2)
        grid.addWidget(self.imgsz_spin, 1, 3)
        grid.addWidget(QtWidgets.QLabel(self.tr("Eval Conf:")), 1, 4)
        grid.addWidget(self.conf_spin, 1, 5)

        grid.addWidget(QtWidgets.QLabel(self.tr("Init / Fine-Tune Weights:")), 2, 0)
        w_box = QtWidgets.QHBoxLayout()
        w_box.addWidget(self.weights_edit, 1)
        w_box.addWidget(w_browse)
        grid.addLayout(w_box, 2, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel(self.tr("Loop Iterations:")), 2, 4)
        grid.addWidget(self.iterations_spin, 2, 5)

        grid.addWidget(QtWidgets.QLabel(self.tr("Optional GT Val Folder:")), 3, 0)
        gt_box = QtWidgets.QHBoxLayout()
        gt_box.addWidget(self.gt_eval_edit, 1)
        gt_box.addWidget(gt_browse)
        grid.addLayout(gt_box, 3, 1, 1, 5)

        grid.addWidget(self.auto_adjust_chk, 4, 0, 1, 3)
        grid.addWidget(self.keep_runtime_chk, 4, 3, 1, 3)

        run_row = QtWidgets.QHBoxLayout()
        self.run_single_btn = QtWidgets.QPushButton(
            self.tr("Run 1 Iteration (Generate → Train → Test on Videos → Advise)")
        )
        self.run_single_btn.clicked.connect(lambda: self._on_run_loop_clicked(1))
        self.run_loop_btn = QtWidgets.QPushButton(
            self.tr("Run Multi-Iteration Closed Loop")
        )
        self.run_loop_btn.clicked.connect(
            lambda: self._on_run_loop_clicked(int(self.iterations_spin.value()))
        )
        self._action_buttons.extend([self.run_single_btn, self.run_loop_btn])
        run_row.addWidget(self.run_single_btn, 1)
        run_row.addWidget(self.run_loop_btn, 1)
        grid.addLayout(run_row, 5, 0, 1, 6)
        layout.addWidget(ctrl_box)

        # Live monitor chart + log split
        mon_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.live_chart = _ColabMetricsChartWidget()
        self.live_chart.setMinimumHeight(210)
        self.live_log = QtWidgets.QPlainTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setPlaceholderText(
            self.tr(
                "Live simulation, Colab GPU training, evaluation, and advisor logs appear here…"
            )
        )
        mon_split.addWidget(self.live_chart)
        mon_split.addWidget(self.live_log)
        mon_split.setSizes([480, 560])
        layout.addWidget(mon_split, 1)
        return page

    def _build_results_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        self.iter_table = QtWidgets.QTableWidget(0, 10)
        self.iter_table.setHorizontalHeaderLabels(
            [
                "Iter",
                "Status",
                "Renderer",
                "Frames",
                "Val mAP@50",
                "Real Coverage",
                "Mean Conf",
                "Temp IoU",
                "Sim2Real Score",
                "Delta",
            ]
        )
        self.iter_table.horizontalHeader().setStretchLastSection(True)
        self.iter_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.iter_table.itemSelectionChanged.connect(self._on_iteration_selected)
        layout.addWidget(self.iter_table, 1)

        advisor_box = QtWidgets.QGroupBox(
            self.tr("Automated Diagnosis & Next-Iteration Recommendations")
        )
        adv_lay = QtWidgets.QVBoxLayout(advisor_box)
        self.advisor_text = QtWidgets.QPlainTextEdit()
        self.advisor_text.setReadOnly(True)
        adv_lay.addWidget(self.advisor_text, 1)

        adv_btns = QtWidgets.QHBoxLayout()
        self.apply_rec_btn = QtWidgets.QPushButton(
            self.tr("Apply Recommended Adjustments to Next Iteration")
        )
        self.apply_rec_btn.clicked.connect(self._on_apply_recommendation_clicked)
        self._action_buttons.append(self.apply_rec_btn)
        adv_btns.addWidget(self.apply_rec_btn)
        adv_btns.addStretch(1)
        adv_lay.addLayout(adv_btns)
        layout.addWidget(advisor_box, 1)
        return page

    def _build_report_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        btn_row = QtWidgets.QHBoxLayout()
        self.gen_report_btn = QtWidgets.QPushButton(
            self.tr("Generate / Refresh Report (Markdown + JSON + HTML)")
        )
        self.gen_report_btn.clicked.connect(self._on_generate_report_clicked)
        self._action_buttons.append(self.gen_report_btn)
        self.open_html_btn = QtWidgets.QPushButton(
            self.tr("Open HTML Report in Browser")
        )
        self.open_html_btn.clicked.connect(self._on_open_html_report)
        btn_row.addWidget(self.gen_report_btn)
        btn_row.addWidget(self.open_html_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        rep_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.report_browser = QtWidgets.QTextBrowser()
        self.report_sheet_label = QtWidgets.QLabel(
            self.tr(
                "Environment-vs-Simulation & Prediction contact sheets appear here."
            )
        )
        self.report_sheet_label.setAlignment(QtCore.Qt.AlignCenter)
        self.report_sheet_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        rep_split.addWidget(self.report_browser)
        rep_split.addWidget(self.report_sheet_label)
        rep_split.setSizes([580, 440])
        layout.addWidget(rep_split, 1)
        return page

    # ------------------------------------------------------------------ #
    # Helpers & State Sync
    # ------------------------------------------------------------------ #

    def _set_busy(self, busy: bool) -> None:
        for btn in self._action_buttons:
            btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def _start_worker(
        self,
        fn: Callable[..., Any],
        *,
        on_success: Callable[[Any], None],
        label: str,
    ) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_busy(True)
        self.status_label.setText(label)
        self.progress_bar.setValue(5)
        self._monitor_parser.reset()
        self.live_chart.set_points([])

        worker = _Sim2RealWorker(fn, parent=self)
        worker.logLine.connect(self._on_worker_log)
        worker.progressChanged.connect(self._on_worker_progress)
        worker.succeeded.connect(lambda res: self._on_worker_finished(res, on_success))
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _on_worker_log(self, line: str) -> None:
        self.live_log.appendPlainText(line)
        if self._monitor_parser.feed_line(line):
            self.live_chart.set_points(list(self._monitor_parser.state.points))

    def _on_worker_progress(self, pct: int, label: str) -> None:
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(label)

    def _on_worker_finished(
        self, result: Any, on_success: Callable[[Any], None]
    ) -> None:
        self._set_busy(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(self.tr("Done"))
        on_success(result)

    def _on_worker_failed(self, exc: object) -> None:
        self._set_busy(False)
        self.status_label.setText(self.tr("Failed"))
        if isinstance(exc, ColabAuthRequiredError):
            self._on_colab_sign_in_clicked()
            return
        QtWidgets.QMessageBox.warning(
            self,
            self.tr("Sim-to-Real Workflow Error"),
            str(exc),
        )

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self.status_label.setText(self.tr("Cancelling…"))

    def _refresh_colab_badge(self) -> None:
        try:
            st = self._colab_service.auth_status()
            if (st.get("colab_cli") or {}).get("installed") or (
                st.get("oauth_colab") or {}
            ).get("ready"):
                self.colab_status_label.setText(self.tr("Colab: Ready"))
            else:
                self.colab_status_label.setText(self.tr("Colab: Sign-in needed"))
        except Exception:
            self.colab_status_label.setText(self.tr("Colab: Offline"))

    def _on_colab_sign_in_clicked(self) -> None:
        dlg = ColabSignInDialog(
            auth_status=self._colab_service.auth_status(),
            parent=self,
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            method = dlg.selected_method()
            try:
                self._colab_service.auth.login(
                    method=method,
                    install_cli=dlg.should_install_cli(),
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, self.tr("Colab Sign-In"), str(exc))
            self._refresh_colab_badge()

    def _add_example_video_item(self, path_str: str) -> None:
        for i in range(self.videos_list.count()):
            if self.videos_list.item(i).text() == path_str:
                return
        self.videos_list.addItem(path_str)

    def _example_video_paths(self) -> list[str]:
        return [
            self.videos_list.item(i).text().strip()
            for i in range(self.videos_list.count())
            if self.videos_list.item(i).text().strip()
        ]

    def _set_pixmap(self, label: QtWidgets.QLabel, path_str: str) -> None:
        if path_str and Path(path_str).is_file():
            pm = QtGui.QPixmap(path_str)
            if not pm.isNull():
                label.setPixmap(
                    pm.scaled(
                        max(220, label.width() - 8),
                        max(150, label.height() - 8),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )
                return

    def _populate_species_specific_controls(
        self, species: str, selected_scenarios: Optional[list[str]] = None
    ) -> None:
        sp = str(species or "mouse").strip().lower()
        is_fly = sp == "fly"

        self.sim_camera_combo.clear()
        cams = FLY_CAMERA_VIEWS if is_fly else MOUSE_CAMERA_VIEWS
        for cam in cams:
            self.sim_camera_combo.addItem(cam, cam)

        self.sim_coat_combo.setEnabled(not is_fly)
        self.sim_occluder_combo.setEnabled(not is_fly)
        self.sim_headstage_combo.setEnabled(not is_fly)
        self.fly_count_spin.setEnabled(is_fly)

        self.scenarios_list.clear()
        choices = FLY_BEHAVIORS if is_fly else MOUSE_SCENARIOS
        picked = set(
            selected_scenarios
            if selected_scenarios
            else (
                ["walk_imitation", "forage_walk"]
                if is_fly
                else ["arena_explore", "social_huddle"]
            )
        )
        for item_name in choices:
            item = QtWidgets.QListWidgetItem(item_name)
            self.scenarios_list.addItem(item)
            if item_name in picked:
                item.setSelected(True)

    def _on_species_changed(self) -> None:
        sp = str(self.sim_species_combo.currentData() or "mouse")
        self._populate_species_specific_controls(sp)

    def build_sim_config_from_ui(self) -> SimDatasetConfig:
        species = str(self.sim_species_combo.currentData() or "mouse")
        selected_scens = [item.text() for item in self.scenarios_list.selectedItems()]
        if not selected_scens:
            selected_scens = (
                ["walk_imitation"] if species == "fly" else ["arena_explore"]
            )
        state = self._service.load_state()
        width = 640
        height = 480
        if isinstance(state.current_sim_config, dict):
            width = int(state.current_sim_config.get("width") or 640)
            height = int(state.current_sim_config.get("height") or 480)
        return SimDatasetConfig(
            species=species,
            task=str(self.sim_task_combo.currentData() or "yolo_seg"),
            export_level=str(self.sim_level_combo.currentData() or "instances"),
            arena=str(self.sim_arena_combo.currentData() or "open_field"),
            camera_view=str(self.sim_camera_combo.currentData() or "top-close"),
            lighting=str(self.sim_lighting_combo.currentData() or "default"),
            coat_pair=str(self.sim_coat_combo.currentData() or "default"),
            occluder=str(self.sim_occluder_combo.currentData() or "none"),
            headstage=str(self.sim_headstage_combo.currentData() or "none"),
            scenarios=selected_scens,
            frames_per_scenario=int(self.frames_per_scen_spin.value()),
            width=width,
            height=height,
            seed=int(self.sim_seed_spin.value()),
            match_example_photometry=bool(self.match_photo_chk.isChecked()),
            photometry_strength=float(self.photo_strength_spin.value()),
            composite_on_real_background=bool(self.composite_bg_chk.isChecked()),
            domain_randomization=float(self.dr_spin.value()),
            real_labeled_dir=self.real_labeled_edit.text().strip() or None,
            fly_subject_count=int(self.fly_count_spin.value()),
            allow_fallback=bool(self.allow_fallback_chk.isChecked()),
        )

    def _apply_sim_config_to_ui(self, cfg: SimDatasetConfig) -> None:
        idx = self.sim_species_combo.findData(cfg.species)
        if idx >= 0:
            self.sim_species_combo.blockSignals(True)
            self.sim_species_combo.setCurrentIndex(idx)
            self.sim_species_combo.blockSignals(False)
        self._populate_species_specific_controls(cfg.species, list(cfg.scenarios))

        for combo, val in (
            (self.sim_task_combo, cfg.task),
            (self.sim_level_combo, cfg.export_level),
            (self.sim_arena_combo, cfg.arena),
            (self.sim_camera_combo, cfg.camera_view),
            (self.sim_lighting_combo, cfg.lighting),
            (self.sim_coat_combo, cfg.coat_pair),
            (self.sim_occluder_combo, cfg.occluder),
            (self.sim_headstage_combo, cfg.headstage),
        ):
            c_idx = combo.findData(val)
            if c_idx >= 0:
                combo.setCurrentIndex(c_idx)

        self.frames_per_scen_spin.setValue(int(cfg.frames_per_scenario))
        self.sim_seed_spin.setValue(int(cfg.seed))
        self.match_photo_chk.setChecked(bool(cfg.match_example_photometry))
        self.photo_strength_spin.setValue(float(cfg.photometry_strength))
        self.dr_spin.setValue(float(cfg.domain_randomization))
        self.composite_bg_chk.setChecked(bool(cfg.composite_on_real_background))
        self.fly_count_spin.setValue(int(cfg.fly_subject_count))
        self.real_labeled_edit.setText(str(cfg.real_labeled_dir or ""))
        self.allow_fallback_chk.setChecked(bool(cfg.allow_fallback))

    def _sync_ui_from_state(self) -> None:
        state: Sim2RealProjectState = self._service.load_state()
        for vp in state.example_videos:
            self._add_example_video_item(vp)

        if isinstance(state.profile, dict):
            prof = VideoEnvironmentProfile.from_dict(state.profile)
            lines = [
                f"Species: {prof.species} (confidence {prof.species_confidence:.2f})",
                f"Resolution / FPS: {prof.width}x{prof.height} @ {prof.fps:.1f} fps ({prof.total_frames} frames)",
                f"Arena: {prof.recommended_arena} | Camera: {prof.recommended_camera_view}",
                f"Lighting: {prof.recommended_lighting} (luma={prof.mean_luminance:.1f}, grayscale={prof.is_grayscale})",
                f"Subject Coat: {prof.recommended_coat_pair} | Est. Subjects: {prof.estimated_subject_count}",
                f"Occluder: {prof.recommended_occluder} | Headstage: {prof.recommended_headstage}",
                f"Scenarios: {', '.join(prof.recommended_scenarios)}",
                "",
                "Matching Rationale:",
            ]
            for k, v in prof.match_rationale.items():
                lines.append(f"  - {k}: {v}")
            self.profile_summary_text.setPlainText("\n".join(lines))
            self._set_pixmap(self.bg_preview_label, prof.reference_background_path)
            self._set_pixmap(self.sample_preview_label, prof.sample_montage_path)

        if isinstance(state.current_sim_config, dict):
            self._apply_sim_config_to_ui(
                SimDatasetConfig.from_dict(state.current_sim_config)
            )

        if state.best_weights_path and not self.weights_edit.text().strip():
            self.weights_edit.setText(state.best_weights_path)

        # Populate Iterations table
        iters = list(state.iterations or [])
        self.iter_table.setRowCount(len(iters))
        for r_idx, it in enumerate(iters):
            ds = it.get("dataset") or {}
            ev = it.get("evaluation") or {}
            val_m = ev.get("val_mask_map50")
            if val_m is None:
                val_m = ev.get("val_box_map50")
            delta = it.get("score_delta_vs_prev")
            vals = [
                str(it.get("iteration", r_idx + 1)),
                str(it.get("status", "")),
                str(ds.get("renderer", "—")),
                str(ds.get("total_frames", "—")),
                f"{val_m:.3f}" if val_m is not None else "—",
                f"{float(ev.get('detection_coverage', 0.0)) * 100:.1f}%",
                f"{float(ev.get('mean_confidence', 0.0)):.3f}",
                f"{float(ev.get('temporal_iou_stability', 0.0)):.3f}",
                f"{float(ev.get('sim2real_score', 0.0)):.3f}",
                f"{float(delta):+.3f}" if delta is not None else "—",
            ]
            for c_idx, txt in enumerate(vals):
                self.iter_table.setItem(r_idx, c_idx, QtWidgets.QTableWidgetItem(txt))
        if iters:
            self.iter_table.selectRow(len(iters) - 1)
            self._render_advisor_for_iteration(iters[-1])

        # Load report preview if available
        rep_paths = state.report_paths or {}
        md_path = str(rep_paths.get("markdown") or "")
        if md_path and Path(md_path).is_file():
            self.report_browser.setMarkdown(Path(md_path).read_text(encoding="utf-8"))
        sheet = str(
            rep_paths.get("latest_prediction_overlay")
            or rep_paths.get("env_vs_sim_contact_sheet")
            or ""
        )
        self._set_pixmap(self.report_sheet_label, sheet)

    def _render_advisor_for_iteration(self, it: dict[str, Any]) -> None:
        adj = it.get("adjustment") or {}
        lines = [
            f"Iteration {it.get('iteration')} | Status: {it.get('status')}",
            f"Checkpoint: {it.get('best_weights_path') or '—'}",
            "",
            "Diagnosis:",
        ]
        for d in adj.get("diagnosis") or ["(none)"]:
            lines.append(f"  - {d}")
        lines.append("")
        lines.append("Recommended Adjustments for Next Iteration:")
        for r in adj.get("recommendations") or ["(none)"]:
            lines.append(f"  - {r}")
        if adj.get("suggested_train_overrides"):
            lines.append("")
            lines.append(
                f"Suggested Train Overrides: {adj['suggested_train_overrides']}"
            )
        self.advisor_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Button Slots
    # ------------------------------------------------------------------ #

    def _on_browse_project_dir(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Sim-to-Real Project Directory"),
            self.project_dir_edit.text(),
        )
        if chosen:
            self.project_dir_edit.setText(chosen)
            self._on_load_project_clicked()

    def _on_load_project_clicked(self) -> None:
        raw = self.project_dir_edit.text().strip()
        if not raw:
            return
        self._service = Sim2RealWorkflowService(
            Path(raw), colab_service=self._colab_service
        )
        self.videos_list.clear()
        self._sync_ui_from_state()
        self.status_label.setText(self.tr(f"Loaded project: {Path(raw).name}"))

    def _on_add_videos_clicked(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            self.tr("Select Example Video(s)"),
            "",
            self.tr("Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"),
        )
        for f in files:
            self._add_example_video_item(f)

    def _on_add_image_folder_clicked(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Example Image Folder")
        )
        if chosen:
            self._add_example_video_item(chosen)

    def _on_remove_selected_videos(self) -> None:
        for item in self.videos_list.selectedItems():
            self.videos_list.takeItem(self.videos_list.row(item))

    def _on_browse_real_labeled(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Real LabelMe Annotation Directory")
        )
        if chosen:
            self.real_labeled_edit.setText(chosen)

    def _on_browse_weights(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Base or Fine-Tune Weights"),
            "",
            self.tr("PyTorch Weights (*.pt *.pth);;All Files (*)"),
        )
        if chosen:
            self.weights_edit.setText(chosen)

    def _on_browse_gt_dir(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Ground-Truth LabelMe Folder")
        )
        if chosen:
            self.gt_eval_edit.setText(chosen)

    def _on_analyze_clicked(self) -> None:
        videos = self._example_video_paths()
        if not videos:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Example Videos"),
                self.tr("Please add at least one example video or image folder first."),
            )
            return
        hint = str(self.species_hint_combo.currentData() or "auto")
        task = str(self.sim_task_combo.currentData() or "yolo_seg")
        level = str(self.sim_level_combo.currentData() or "instances")
        frames = int(self.frames_per_scen_spin.value())

        def _work(log_callback, progress_callback, cancel_event):
            progress_callback(25, "Profiling example videos…")
            return self._service.analyze_videos(
                videos,
                species_hint=hint,
                task=task,
                export_level=level,
                frames_per_scenario=frames,
                log_callback=log_callback,
            )

        self._start_worker(
            _work,
            on_success=lambda _res: self._sync_ui_from_state(),
            label=self.tr("Analyzing example video environment…"),
        )

    def _on_generate_dataset_clicked(self) -> None:
        cfg = self.build_sim_config_from_ui()

        def _work(log_callback, progress_callback, cancel_event):
            progress_callback(20, f"Generating {cfg.species} 3D synthetic dataset…")
            return self._service.generate_dataset(cfg, log_callback=log_callback)

        def _done(artifact: Any) -> None:
            self.sim_dataset_info.setText(
                f"Renderer: {artifact.renderer} | Frames: {artifact.total_frames} "
                f"({artifact.train_frames} train / {artifact.val_frames} val) | "
                f"Classes: {artifact.class_names}\nYAML: {artifact.data_yaml_path}"
            )
            if artifact.preview_images:
                self._set_pixmap(self.sim_preview_label, artifact.preview_images[0])

        self._start_worker(
            _work,
            on_success=_done,
            label=self.tr("Generating 3D synthetic dataset…"),
        )

    def _on_run_loop_clicked(self, iterations: int) -> None:
        cfg = self.build_sim_config_from_ui()
        self._service.update_sim_config(cfg)
        train_backend = str(self.train_backend_combo.currentData() or "colab")
        colab_backend = str(self.colab_backend_combo.currentData() or "auto")
        gpu = str(self.gpu_combo.currentData() or "T4")
        epochs = int(self.epochs_spin.value())
        imgsz = int(self.imgsz_spin.value())
        conf = float(self.conf_spin.value())
        weights = self.weights_edit.text().strip() or None
        gt_dir = self.gt_eval_edit.text().strip() or None
        auto_adj = bool(self.auto_adjust_chk.isChecked())
        keep_rt = bool(self.keep_runtime_chk.isChecked())

        def _work(log_callback, progress_callback, cancel_event):
            return self._service.run_loop(
                iterations=iterations,
                auto_apply_recommendations=auto_adj,
                sim_config=cfg,
                train_backend=train_backend,
                colab_backend=colab_backend,
                accelerator=gpu,
                keep_runtime=keep_rt,
                weights=weights,
                epochs=epochs,
                imgsz=imgsz,
                conf=conf,
                ground_truth_labelme_dir=gt_dir,
                log_callback=log_callback,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

        def _done(_records: Any) -> None:
            self._sync_ui_from_state()
            self.tabs.setCurrentIndex(3)

        self._start_worker(
            _work,
            on_success=_done,
            label=self.tr(f"Running {iterations} Sim-to-Real iteration(s)…"),
        )

    def _on_iteration_selected(self) -> None:
        row = self.iter_table.currentRow()
        state = self._service.load_state()
        if 0 <= row < len(state.iterations):
            self._render_advisor_for_iteration(state.iterations[row])

    def _on_apply_recommendation_clicked(self) -> None:
        try:
            cfg, overrides = self._service.apply_latest_recommendation()
            self._apply_sim_config_to_ui(cfg)
            if overrides.get("weights"):
                self.weights_edit.setText(str(overrides["weights"]))
            if overrides.get("epochs"):
                self.epochs_spin.setValue(int(overrides["epochs"]))
            if overrides.get("imgsz"):
                self.imgsz_spin.setValue(int(overrides["imgsz"]))
            if overrides.get("conf"):
                self.conf_spin.setValue(float(overrides["conf"]))
            self.status_label.setText(
                self.tr("Applied recommended adjustments for the next iteration.")
            )
            self.tabs.setCurrentIndex(1)
        except Exception as exc:
            QtWidgets.QMessageBox.information(
                self, self.tr("Advisor Recommendations"), str(exc)
            )

    def _on_generate_report_clicked(self) -> None:
        paths = self._service.write_report()
        self._sync_ui_from_state()
        self.status_label.setText(
            self.tr(f"Report written: {Path(paths['markdown']).name}")
        )

    def _on_open_html_report(self) -> None:
        state = self._service.load_state()
        html_path = str((state.report_paths or {}).get("html") or "")
        if not html_path or not Path(html_path).is_file():
            paths = self._service.write_report()
            html_path = paths.get("html", "")
        if html_path and Path(html_path).is_file():
            webbrowser.open(Path(html_path).resolve().as_uri())
