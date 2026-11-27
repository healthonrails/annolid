"""Non-modal Pose Estimation Workbench dialog (SLEAP / DeepLabCut workflow parity).

Provides a five-tab background-threaded workbench integrated with the main Annolid
canvas (including live canvas pose rendering controls and double-click frame seeking):

1. Evaluate Pose (DLC RMSE/p-cutoff, SLEAP p50/p90/p95, PCK, OKS mAP, L/R swaps, worst frames)
2. Filter & Clean Pose (p-cutoff masking, jump/bone outlier cleaning, gap interpolation, smoothing)
3. Active Learning Outliers (extract diverse high-risk frames + editable LabelMe JSONs)
4. Pose Kinematics (per-keypoint speeds, bone lengths, joint angles, egocentric pose, dyadic pairs)
5. Labeled Video & Swap IDs (offline skeleton video renderer + swap identities from frame t forward)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from qtpy import QtCore, QtWidgets

from annolid.annotation.pose_schema import find_pose_schema_in_dir


class _PoseWorkbenchWorker(QtCore.QThread):
    """Background worker running pose computations off the GUI thread."""

    finishedResult = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)

    def __init__(
        self,
        task_name: str,
        fn: Callable[[], Any],
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._task_name = str(task_name)
        self._fn = fn

    def run(self) -> None:  # noqa: N802
        try:
            result = self._fn()
        except Exception as exc:
            self.failed.emit(self._task_name, str(exc))
            return
        self.finishedResult.emit(self._task_name, result)


class PoseWorkbenchWidget(QtWidgets.QWidget):
    """Tabbed pose estimation workbench embeddable in a dialog or dock."""

    jumpToFrameRequested = QtCore.Signal(int)
    openDirectoryRequested = QtCore.Signal(str)
    queueFramesRequested = QtCore.Signal(list, str, dict)

    _TAB_KEYS = ("eval", "filter", "outliers", "kinematics", "video")

    def __init__(
        self,
        owner_window: Optional[Any] = None,
        *,
        initial_tab: str = "eval",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._owner_window = owner_window
        self._active_worker: Optional[_PoseWorkbenchWorker] = None
        self._last_eval_report = None
        self._last_kinematics_report = None
        self._last_outlier_output_dir: Optional[str] = None
        self._build_ui()
        self.apply_owner_context()
        self.select_tab(initial_tab)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Shared Canvas Display Controls bar (live SLEAP/DLC canvas visualization settings).
        canvas_box = QtWidgets.QGroupBox("Canvas Pose Display Controls")
        canvas_row = QtWidgets.QHBoxLayout(canvas_box)
        canvas_row.setSpacing(8)
        canvas_row.addWidget(QtWidgets.QLabel("Color mode:"))
        self.canvas_color_combo = QtWidgets.QComboBox()
        self.canvas_color_combo.addItem("By Instance (Animal)", "instance")
        self.canvas_color_combo.addItem("By Bodypart (DLC)", "bodypart")
        self.canvas_color_combo.addItem("By Confidence (Score)", "confidence")
        self.canvas_color_combo.currentIndexChanged.connect(
            self._apply_canvas_display_settings
        )
        canvas_row.addWidget(self.canvas_color_combo)

        canvas_row.addWidget(QtWidgets.QLabel("Canvas p-cutoff:"))
        self.canvas_pcutoff_spin = QtWidgets.QDoubleSpinBox()
        self.canvas_pcutoff_spin.setRange(0.0, 1.0)
        self.canvas_pcutoff_spin.setSingleStep(0.05)
        self.canvas_pcutoff_spin.setDecimals(2)
        self.canvas_pcutoff_spin.setValue(0.0)
        self.canvas_pcutoff_spin.setToolTip(
            "Hide predicted keypoints below this confidence score on the canvas (0.00 = show all)"
        )
        self.canvas_pcutoff_spin.valueChanged.connect(
            self._apply_canvas_display_settings
        )
        canvas_row.addWidget(self.canvas_pcutoff_spin)

        self.canvas_show_labels_cb = QtWidgets.QCheckBox(
            "Show keypoint labels on canvas"
        )
        self.canvas_show_labels_cb.setChecked(False)
        self.canvas_show_labels_cb.toggled.connect(self._apply_canvas_display_settings)
        canvas_row.addWidget(self.canvas_show_labels_cb)
        canvas_row.addStretch(1)
        root.addWidget(canvas_box)

        # Shared PoseSchema selector
        schema_row = QtWidgets.QHBoxLayout()
        schema_row.addWidget(QtWidgets.QLabel("Pose Schema (optional):"))
        self.pose_schema_edit = QtWidgets.QLineEdit()
        self.pose_schema_edit.setPlaceholderText(
            "Auto-discovered pose_schema.json / .yaml"
        )
        schema_btn = QtWidgets.QPushButton("Browse…")
        schema_btn.clicked.connect(
            lambda: self._browse_file(
                self.pose_schema_edit,
                "Select Pose Schema",
                "Pose Schema (*.json *.yaml *.yml);;All Files (*)",
            )
        )
        schema_row.addWidget(self.pose_schema_edit, 1)
        schema_row.addWidget(schema_btn)
        root.addLayout(schema_row)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_eval_tab(), "1. Evaluate Pose")
        self.tabs.addTab(self._build_filter_tab(), "2. Filter && Clean")
        self.tabs.addTab(self._build_outliers_tab(), "3. Active Learning Outliers")
        self.tabs.addTab(self._build_kinematics_tab(), "4. Pose Kinematics")
        self.tabs.addTab(
            self._build_video_and_swap_tab(), "5. Labeled Video && Swap IDs"
        )
        root.addWidget(self.tabs, 1)

        status_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setVisible(False)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress_bar)
        root.addLayout(status_row)

    def _build_path_row(
        self,
        label: str,
        placeholder: str,
        *,
        allow_dir: bool = True,
        save_mode: bool = False,
        file_filter: str = "All Files (*)",
    ) -> tuple[QtWidgets.QHBoxLayout, QtWidgets.QLineEdit]:
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        edit = QtWidgets.QLineEdit()
        edit.setPlaceholderText(placeholder)
        row.addWidget(edit, 1)
        if save_mode:
            btn = QtWidgets.QPushButton("Save As…")
            btn.clicked.connect(
                lambda: self._browse_save_file(edit, f"Select {label}", file_filter)
            )
            row.addWidget(btn)
        else:
            btn_file = QtWidgets.QPushButton("File…")
            btn_file.clicked.connect(
                lambda: self._browse_file(edit, f"Select {label}", file_filter)
            )
            row.addWidget(btn_file)
            if allow_dir:
                btn_dir = QtWidgets.QPushButton("Folder…")
                btn_dir.clicked.connect(
                    lambda: self._browse_dir(edit, f"Select {label}")
                )
                row.addWidget(btn_dir)
        return row, edit

    def _build_eval_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)

        r_gt, self.eval_gt_edit = self._build_path_row(
            "Ground Truth:",
            "LabelMe folder, SLEAP .slp/.h5, DLC CSV/H5/project, or COCO JSON",
        )
        r_pr, self.eval_pred_edit = self._build_path_row(
            "Predictions:",
            "Predicted LabelMe folder, SLEAP, DLC CSV/H5, COCO JSON, or tracking CSV",
        )
        layout.addLayout(r_gt)
        layout.addLayout(r_pr)

        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("p-cutoff:"))
        self.eval_pcutoff_spin = QtWidgets.QDoubleSpinBox()
        self.eval_pcutoff_spin.setRange(0.0, 1.0)
        self.eval_pcutoff_spin.setSingleStep(0.05)
        self.eval_pcutoff_spin.setValue(0.60)
        opts.addWidget(self.eval_pcutoff_spin)

        opts.addWidget(QtWidgets.QLabel("OKS sigma:"))
        self.eval_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.eval_sigma_spin.setRange(0.005, 0.5)
        self.eval_sigma_spin.setSingleStep(0.01)
        self.eval_sigma_spin.setDecimals(3)
        self.eval_sigma_spin.setValue(0.05)
        opts.addWidget(self.eval_sigma_spin)

        self.eval_run_btn = QtWidgets.QPushButton("Run Evaluation")
        self.eval_run_btn.clicked.connect(self._run_eval)
        self.eval_export_btn = QtWidgets.QPushButton("Export HTML/CSV/JSON Report…")
        self.eval_export_btn.setEnabled(False)
        self.eval_export_btn.clicked.connect(self._export_eval_report)
        opts.addStretch(1)
        opts.addWidget(self.eval_run_btn)
        opts.addWidget(self.eval_export_btn)
        layout.addLayout(opts)

        self.eval_summary_label = QtWidgets.QLabel(
            "Run evaluation to inspect RMSE, p50/p90/p95, PCK, OKS mAP, and L/R swap rate."
        )
        self.eval_summary_label.setWordWrap(True)
        layout.addWidget(self.eval_summary_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.eval_kp_table = QtWidgets.QTableWidget(0, 8)
        self.eval_kp_table.setHorizontalHeaderLabels(
            [
                "Keypoint",
                "Matched",
                "Recall",
                "RMSE (px)",
                "RMSE@pcut",
                "p50",
                "p90",
                "p95",
            ]
        )
        self.eval_kp_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.eval_kp_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.eval_kp_table)

        self.eval_worst_table = QtWidgets.QTableWidget(0, 6)
        self.eval_worst_table.setHorizontalHeaderLabels(
            [
                "Frame",
                "Image",
                "Matched Inst",
                "Missing KPs",
                "Mean Err (px)",
                "L/R Swaps",
            ]
        )
        self.eval_worst_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.eval_worst_table.horizontalHeader().setStretchLastSection(True)
        self.eval_worst_table.setToolTip(
            "Double-click any row to seek the main canvas to that frame"
        )
        self.eval_worst_table.cellDoubleClicked.connect(
            lambda row, _col: self._emit_jump_from_table(self.eval_worst_table, row)
        )
        splitter.addWidget(self.eval_worst_table)
        layout.addWidget(splitter, 1)
        return page

    def _build_filter_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)

        r_src, self.filter_src_edit = self._build_path_row(
            "Input Pose:", "LabelMe folder, SLEAP .slp/.h5, or DLC CSV/H5"
        )
        r_out, self.filter_out_edit = self._build_path_row(
            "Output Path:", "Optional output folder/file (default: <source>_filtered)"
        )
        layout.addLayout(r_src)
        layout.addLayout(r_out)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Smoothing filter:"), 0, 0)
        self.filter_method_combo = QtWidgets.QComboBox()
        for label, key in (
            ("Savitzky-Golay (zero-phase)", "savgol"),
            ("Median (spike removal)", "median"),
            ("One-Euro (adaptive)", "one_euro"),
            ("Moving Average", "moving_average"),
            ("None (clean/interpolate only)", "none"),
        ):
            self.filter_method_combo.addItem(label, key)
        grid.addWidget(self.filter_method_combo, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Window (odd frames):"), 0, 2)
        self.filter_window_spin = QtWidgets.QSpinBox()
        self.filter_window_spin.setRange(3, 51)
        self.filter_window_spin.setSingleStep(2)
        self.filter_window_spin.setValue(5)
        grid.addWidget(self.filter_window_spin, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Gap interpolation:"), 1, 0)
        self.filter_interp_combo = QtWidgets.QComboBox()
        for label, key in (
            ("Linear", "linear"),
            ("Cubic Spline", "cubic"),
            ("Nearest", "nearest"),
            ("None", "none"),
        ):
            self.filter_interp_combo.addItem(label, key)
        grid.addWidget(self.filter_interp_combo, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Max gap (frames):"), 1, 2)
        self.filter_max_gap_spin = QtWidgets.QSpinBox()
        self.filter_max_gap_spin.setRange(0, 300)
        self.filter_max_gap_spin.setValue(5)
        grid.addWidget(self.filter_max_gap_spin, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Min confidence (p-cutoff):"), 2, 0)
        self.filter_min_conf_spin = QtWidgets.QDoubleSpinBox()
        self.filter_min_conf_spin.setRange(0.0, 1.0)
        self.filter_min_conf_spin.setSingleStep(0.05)
        self.filter_min_conf_spin.setValue(0.0)
        self.filter_min_conf_spin.setSpecialValueText("Disabled (0.00)")
        grid.addWidget(self.filter_min_conf_spin, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Max jump (px/frame):"), 2, 2)
        self.filter_max_jump_spin = QtWidgets.QDoubleSpinBox()
        self.filter_max_jump_spin.setRange(0.0, 2000.0)
        self.filter_max_jump_spin.setSingleStep(5.0)
        self.filter_max_jump_spin.setValue(0.0)
        self.filter_max_jump_spin.setSpecialValueText("Disabled (0)")
        grid.addWidget(self.filter_max_jump_spin, 2, 3)

        self.filter_bone_cb = QtWidgets.QCheckBox(
            "Mask anatomical bone-length outliers (stretch/collapse)"
        )
        self.filter_bone_cb.setChecked(True)
        grid.addWidget(self.filter_bone_cb, 3, 0, 1, 2)

        self.filter_run_btn = QtWidgets.QPushButton("Run Pose Filtering")
        self.filter_run_btn.clicked.connect(self._run_filter)
        grid.addWidget(self.filter_run_btn, 3, 3)
        layout.addLayout(grid)

        self.filter_log_text = QtWidgets.QPlainTextEdit()
        self.filter_log_text.setReadOnly(True)
        self.filter_log_text.setPlaceholderText(
            "Filter report summary will appear here (non-point polygons and zones are always preserved)."
        )
        layout.addWidget(self.filter_log_text, 1)
        return page

    def _build_outliers_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)

        r_src, self.out_src_edit = self._build_path_row(
            "Pose Predictions:", "LabelMe folder, SLEAP, DLC CSV/H5, or tracking CSV"
        )
        r_vid, self.out_vid_edit = self._build_path_row(
            "Source Video:",
            "Video file to extract outlier frames from",
            allow_dir=False,
            file_filter="Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        r_dst, self.out_dst_edit = self._build_path_row(
            "Output Folder:", "Folder for outlier PNGs + editable LabelMe JSONs"
        )
        layout.addLayout(r_src)
        layout.addLayout(r_vid)
        layout.addLayout(r_dst)

        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("Frames:"))
        self.out_num_spin = QtWidgets.QSpinBox()
        self.out_num_spin.setRange(1, 500)
        self.out_num_spin.setValue(20)
        opts.addWidget(self.out_num_spin)

        opts.addWidget(QtWidgets.QLabel("Min gap:"))
        self.out_gap_spin = QtWidgets.QSpinBox()
        self.out_gap_spin.setRange(0, 1000)
        self.out_gap_spin.setValue(10)
        opts.addWidget(self.out_gap_spin)

        opts.addWidget(QtWidgets.QLabel("Min conf:"))
        self.out_conf_spin = QtWidgets.QDoubleSpinBox()
        self.out_conf_spin.setRange(0.0, 1.0)
        self.out_conf_spin.setSingleStep(0.05)
        self.out_conf_spin.setValue(0.60)
        opts.addWidget(self.out_conf_spin)

        self.out_queue_only_cb = QtWidgets.QCheckBox("Queue in video (no extraction)")
        self.out_queue_only_cb.setToolTip(
            "Send selected outlier frames directly to the Label Queue without writing PNG/JSON copies"
        )
        self.out_queue_only_cb.toggled.connect(self._on_outlier_queue_mode_toggled)
        opts.addWidget(self.out_queue_only_cb)

        self.out_run_btn = QtWidgets.QPushButton("Extract Outlier Frames")
        self.out_run_btn.clicked.connect(self._run_outliers)
        self.out_open_btn = QtWidgets.QPushButton("Open Output Folder")
        self.out_open_btn.setEnabled(False)
        self.out_open_btn.clicked.connect(self._open_outlier_output)
        opts.addStretch(1)
        opts.addWidget(self.out_run_btn)
        opts.addWidget(self.out_open_btn)
        layout.addLayout(opts)

        self.out_table = QtWidgets.QTableWidget(0, 6)
        self.out_table.setHorizontalHeaderLabels(
            ["Frame", "Score", "Low Conf", "Jumps", "Bone Outliers", "Reasons"]
        )
        self.out_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.out_table.horizontalHeader().setStretchLastSection(True)
        self.out_table.setToolTip(
            "Double-click any row to seek the main canvas to that frame"
        )
        self.out_table.cellDoubleClicked.connect(
            lambda row, _col: self._emit_jump_from_table(self.out_table, row)
        )
        layout.addWidget(self.out_table, 1)
        return page

    def _build_kinematics_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)

        r_src, self.kin_src_edit = self._build_path_row(
            "Pose Source:", "LabelMe folder, SLEAP, DLC CSV/H5, or COCO JSON"
        )
        layout.addLayout(r_src)

        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("FPS:"))
        self.kin_fps_spin = QtWidgets.QDoubleSpinBox()
        self.kin_fps_spin.setRange(0.1, 1000.0)
        self.kin_fps_spin.setValue(30.0)
        opts.addWidget(self.kin_fps_spin)

        opts.addWidget(QtWidgets.QLabel("px/cm:"))
        self.kin_ppc_spin = QtWidgets.QDoubleSpinBox()
        self.kin_ppc_spin.setRange(0.0, 5000.0)
        self.kin_ppc_spin.setValue(0.0)
        self.kin_ppc_spin.setSpecialValueText("Pixels only")
        opts.addWidget(self.kin_ppc_spin)

        opts.addWidget(QtWidgets.QLabel("Min conf:"))
        self.kin_conf_spin = QtWidgets.QDoubleSpinBox()
        self.kin_conf_spin.setRange(0.0, 1.0)
        self.kin_conf_spin.setSingleStep(0.05)
        self.kin_conf_spin.setValue(0.0)
        self.kin_conf_spin.setSpecialValueText("All")
        opts.addWidget(self.kin_conf_spin)

        self.kin_run_btn = QtWidgets.QPushButton("Analyze Pose Kinematics")
        self.kin_run_btn.clicked.connect(self._run_kinematics)
        self.kin_export_btn = QtWidgets.QPushButton("Export CSV / Egocentric Bundle…")
        self.kin_export_btn.setEnabled(False)
        self.kin_export_btn.clicked.connect(self._export_kinematics_bundle)
        opts.addStretch(1)
        opts.addWidget(self.kin_run_btn)
        opts.addWidget(self.kin_export_btn)
        layout.addLayout(opts)

        subtabs = QtWidgets.QTabWidget()
        self.kin_kp_table = QtWidgets.QTableWidget(0, 6)
        self.kin_kp_table.setHorizontalHeaderLabels(
            [
                "Instance",
                "Keypoint",
                "Frames",
                "Mean Speed (px/s)",
                "p95 Speed",
                "Distance (px)",
            ]
        )
        self.kin_kp_table.horizontalHeader().setStretchLastSection(True)
        subtabs.addTab(self.kin_kp_table, "Keypoint Speeds")

        self.kin_bone_table = QtWidgets.QTableWidget(0, 5)
        self.kin_bone_table.setHorizontalHeaderLabels(
            ["Instance", "Bone", "Count", "Median (px)", "Std (px)"]
        )
        self.kin_bone_table.horizontalHeader().setStretchLastSection(True)
        subtabs.addTab(self.kin_bone_table, "Bone Lengths")

        self.kin_angle_table = QtWidgets.QTableWidget(0, 6)
        self.kin_angle_table.setHorizontalHeaderLabels(
            ["Instance", "Joint", "Count", "Mean (deg)", "Std (deg)", "AngVel (deg/s)"]
        )
        self.kin_angle_table.horizontalHeader().setStretchLastSection(True)
        subtabs.addTab(self.kin_angle_table, "Joint Angles")

        self.kin_dyadic_table = QtWidgets.QTableWidget(0, 6)
        self.kin_dyadic_table.setHorizontalHeaderLabels(
            [
                "Animal A",
                "KP A",
                "Animal B",
                "KP B",
                "Mean Dist (px)",
                "Close Contact (s)",
            ]
        )
        self.kin_dyadic_table.horizontalHeader().setStretchLastSection(True)
        subtabs.addTab(self.kin_dyadic_table, "Dyadic Pairs")
        layout.addWidget(subtabs, 1)
        return page

    def _build_video_and_swap_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(8)

        # Section 1: Render Labeled Skeleton Video
        vid_box = QtWidgets.QGroupBox("Render Labeled Skeleton Video")
        vl = QtWidgets.QVBoxLayout(vid_box)
        r_v, self.ren_vid_edit = self._build_path_row(
            "Source Video:",
            "Input video (.mp4 / .avi / .mov)",
            allow_dir=False,
            file_filter="Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        r_p, self.ren_pose_edit = self._build_path_row(
            "Pose Source:", "LabelMe folder, SLEAP, DLC CSV/H5, or tracking CSV"
        )
        r_o, self.ren_out_edit = self._build_path_row(
            "Output Video:",
            "labeled_pose.mp4",
            save_mode=True,
            file_filter="MP4 Video (*.mp4);;AVI Video (*.avi)",
        )
        vl.addLayout(r_v)
        vl.addLayout(r_p)
        vl.addLayout(r_o)

        vopts = QtWidgets.QHBoxLayout()
        vopts.addWidget(QtWidgets.QLabel("Color:"))
        self.ren_color_combo = QtWidgets.QComboBox()
        self.ren_color_combo.addItem("Instance", "instance")
        self.ren_color_combo.addItem("Bodypart", "bodypart")
        self.ren_color_combo.addItem("Confidence", "confidence")
        vopts.addWidget(self.ren_color_combo)

        vopts.addWidget(QtWidgets.QLabel("p-cutoff:"))
        self.ren_conf_spin = QtWidgets.QDoubleSpinBox()
        self.ren_conf_spin.setRange(0.0, 1.0)
        self.ren_conf_spin.setSingleStep(0.05)
        self.ren_conf_spin.setValue(0.0)
        vopts.addWidget(self.ren_conf_spin)

        vopts.addWidget(QtWidgets.QLabel("Trail frames:"))
        self.ren_trail_spin = QtWidgets.QSpinBox()
        self.ren_trail_spin.setRange(0, 120)
        self.ren_trail_spin.setValue(10)
        vopts.addWidget(self.ren_trail_spin)

        self.ren_labels_cb = QtWidgets.QCheckBox("KP labels")
        vopts.addWidget(self.ren_labels_cb)
        self.ren_run_btn = QtWidgets.QPushButton("Render Labeled Video")
        self.ren_run_btn.clicked.connect(self._run_render_video)
        vopts.addStretch(1)
        vopts.addWidget(self.ren_run_btn)
        vl.addLayout(vopts)
        layout.addWidget(vid_box)

        # Section 2: Swap Pose Identities From Frame t Forward
        swap_box = QtWidgets.QGroupBox(
            "Swap Pose Instances From Frame Forward (Multi-Animal Proofreading)"
        )
        sl = QtWidgets.QVBoxLayout(swap_box)
        r_sd, self.swap_dir_edit = self._build_path_row(
            "LabelMe Folder:",
            "Directory containing frame JSONs or AnnotationStore",
        )
        sl.addLayout(r_sd)

        sopts = QtWidgets.QHBoxLayout()
        sopts.addWidget(QtWidgets.QLabel("Start frame:"))
        self.swap_start_spin = QtWidgets.QSpinBox()
        self.swap_start_spin.setRange(0, 10_000_000)
        sopts.addWidget(self.swap_start_spin)

        sopts.addWidget(QtWidgets.QLabel("End frame (-1=end):"))
        self.swap_end_spin = QtWidgets.QSpinBox()
        self.swap_end_spin.setRange(-1, 10_000_000)
        self.swap_end_spin.setValue(-1)
        sopts.addWidget(self.swap_end_spin)

        sopts.addWidget(QtWidgets.QLabel("Instance A:"))
        self.swap_a_edit = QtWidgets.QLineEdit()
        self.swap_a_edit.setPlaceholderText("e.g. mouse_1")
        sopts.addWidget(self.swap_a_edit)

        sopts.addWidget(QtWidgets.QLabel("Instance B:"))
        self.swap_b_edit = QtWidgets.QLineEdit()
        self.swap_b_edit.setPlaceholderText("e.g. mouse_2")
        sopts.addWidget(self.swap_b_edit)
        sl.addLayout(sopts)

        sbtns = QtWidgets.QHBoxLayout()
        self.swap_preview_btn = QtWidgets.QPushButton("Preview Swap (Dry Run)")
        self.swap_preview_btn.clicked.connect(lambda: self._run_swap(dry_run=True))
        self.swap_apply_btn = QtWidgets.QPushButton("Apply Swap Forward")
        self.swap_apply_btn.clicked.connect(lambda: self._run_swap(dry_run=False))
        sbtns.addStretch(1)
        sbtns.addWidget(self.swap_preview_btn)
        sbtns.addWidget(self.swap_apply_btn)
        sl.addLayout(sbtns)
        layout.addWidget(swap_box)

        self.video_swap_log = QtWidgets.QPlainTextEdit()
        self.video_swap_log.setReadOnly(True)
        layout.addWidget(self.video_swap_log, 1)
        return page

    # ------------------------------------------------------------------
    # Owner Context & Canvas Integration
    # ------------------------------------------------------------------
    def select_tab(self, tab_key: str) -> None:
        key = str(tab_key or "eval").strip().lower()
        if key in self._TAB_KEYS:
            self.tabs.setCurrentIndex(self._TAB_KEYS.index(key))

    def apply_owner_context(self) -> None:
        owner = self._owner_window
        if owner is None:
            return

        video_file = getattr(owner, "video_file", None)
        if video_file and Path(str(video_file)).is_file():
            vp = str(Path(str(video_file)).resolve())
            for edit in (self.out_vid_edit, self.ren_vid_edit):
                if not edit.text().strip():
                    edit.setText(vp)
            cand_dir = Path(vp).with_suffix("")
            if cand_dir.is_dir():
                c_str = str(cand_dir)
                for edit in (
                    self.eval_pred_edit,
                    self.filter_src_edit,
                    self.out_src_edit,
                    self.kin_src_edit,
                    self.ren_pose_edit,
                    self.swap_dir_edit,
                ):
                    if not edit.text().strip():
                        edit.setText(c_str)

        cur_frame = getattr(owner, "current_frame_number", None)
        if isinstance(cur_frame, int) and cur_frame >= 0:
            self.swap_start_spin.setValue(cur_frame)

        # Sync PoseSchema path
        ps_path = getattr(owner, "pose_schema_path", None)
        if not ps_path:
            seq = getattr(owner, "keypoint_sequencer", None)
            if seq is not None and callable(getattr(seq, "schema_path", None)):
                ps_path = seq.schema_path()
        if not ps_path and self.filter_src_edit.text().strip():
            found = find_pose_schema_in_dir(
                self.filter_src_edit.text().strip(), search_parent=True
            )
            if found is not None:
                ps_path = str(found)
        if ps_path and not self.pose_schema_edit.text().strip():
            self.pose_schema_edit.setText(str(ps_path))

        # Sync live Canvas display controls
        canvas = getattr(owner, "canvas", None)
        if canvas is not None:
            mode = getattr(canvas, "poseColorMode", lambda: "instance")()
            idx = self.canvas_color_combo.findData(mode)
            if idx >= 0:
                self.canvas_color_combo.blockSignals(True)
                self.canvas_color_combo.setCurrentIndex(idx)
                self.canvas_color_combo.blockSignals(False)
            thr = float(getattr(canvas, "poseConfidenceThreshold", lambda: 0.0)())
            self.canvas_pcutoff_spin.blockSignals(True)
            self.canvas_pcutoff_spin.setValue(thr)
            self.canvas_pcutoff_spin.blockSignals(False)
            show_lbl = bool(getattr(canvas, "showPoseLabels", lambda: False)())
            self.canvas_show_labels_cb.blockSignals(True)
            self.canvas_show_labels_cb.setChecked(show_lbl)
            self.canvas_show_labels_cb.blockSignals(False)

    def _apply_canvas_display_settings(self) -> None:
        owner = self._owner_window
        canvas = getattr(owner, "canvas", None) if owner is not None else None
        if canvas is None:
            return
        mode = str(self.canvas_color_combo.currentData() or "instance")
        if hasattr(canvas, "setPoseColorMode"):
            canvas.setPoseColorMode(mode)
        if hasattr(canvas, "setPoseConfidenceThreshold"):
            canvas.setPoseConfidenceThreshold(float(self.canvas_pcutoff_spin.value()))
        if hasattr(canvas, "setShowPoseLabels"):
            canvas.setShowPoseLabels(bool(self.canvas_show_labels_cb.isChecked()))

    # ------------------------------------------------------------------
    # File Browsers & Helpers
    # ------------------------------------------------------------------
    def _browse_file(self, edit: QtWidgets.QLineEdit, title: str, filt: str) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, edit.text().strip() or str(Path.home()), filt
        )
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QtWidgets.QLineEdit, title: str) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, title, edit.text().strip() or str(Path.home())
        )
        if path:
            edit.setText(path)

    def _browse_save_file(
        self, edit: QtWidgets.QLineEdit, title: str, filt: str
    ) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, title, edit.text().strip() or str(Path.home()), filt
        )
        if path:
            edit.setText(path)

    def _pose_schema_arg(self) -> Optional[str]:
        raw = self.pose_schema_edit.text().strip()
        return raw if raw else None

    def _emit_jump_from_table(self, table: QtWidgets.QTableWidget, row: int) -> None:
        cell = table.item(row, 0)
        if cell is None:
            return
        try:
            self.jumpToFrameRequested.emit(max(0, int(float(cell.text().strip()))))
        except ValueError:
            pass

    def _start_job(self, task_name: str, fn: Callable[[], Any]) -> None:
        if self._active_worker is not None and self._active_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Task Running",
                "Another Pose Workbench task is still running. Please wait for it to finish.",
            )
            return
        self.status_label.setText(f"Running {task_name}…")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        worker = _PoseWorkbenchWorker(task_name, fn, parent=self)
        worker.finishedResult.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_worker_cleanup)
        self._active_worker = worker
        worker.start()

    def _on_worker_cleanup(self) -> None:
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        if self._active_worker is not None:
            self._active_worker.deleteLater()
            self._active_worker = None

    def _on_worker_failed(self, task_name: str, error_msg: str) -> None:
        self.status_label.setText(f"{task_name} failed: {error_msg}")
        QtWidgets.QMessageBox.warning(self, f"{task_name} Failed", error_msg)

    def _on_worker_finished(self, task_name: str, result: Any) -> None:
        if task_name == "eval":
            self._populate_eval_result(result)
        elif task_name == "filter":
            self._populate_filter_result(result)
        elif task_name == "outliers":
            self._populate_outliers_result(result)
        elif task_name == "kinematics":
            self._populate_kinematics_result(result)
        elif task_name == "render_video":
            self.video_swap_log.appendPlainText(
                f"Rendered labeled video: {result.output_video_path} "
                f"({result.frames_rendered} frames, {result.frames_with_pose} with pose, mode={result.color_mode})"
            )
            self.status_label.setText(
                f"Labeled video written to {result.output_video_path}"
            )
        elif task_name == "swap_pose":
            tag = "DRY-RUN" if result.dry_run else "APPLIED"
            self.video_swap_log.appendPlainText(
                f"[{tag}] Swapped '{result.instance_a}' <-> '{result.instance_b}' from frame {result.start_frame}: "
                f"{result.frames_modified}/{result.frames_scanned} frames, "
                f"{result.points_swapped} keypoints + {result.polygons_swapped} polygons."
            )
            self.status_label.setText(
                f"Swap {tag.lower()} complete ({result.frames_modified} frames)."
            )

    # ------------------------------------------------------------------
    # Tab Job Launchers & Result Populators
    # ------------------------------------------------------------------
    def _run_eval(self) -> None:
        gt = self.eval_gt_edit.text().strip()
        pred = self.eval_pred_edit.text().strip()
        if not gt or not pred:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please specify both Ground Truth and Predictions paths.",
            )
            return
        ps = self._pose_schema_arg()
        pcut = float(self.eval_pcutoff_spin.value())
        sigma = float(self.eval_sigma_spin.value())

        def _job():
            from annolid.postprocessing.pose_evaluation import evaluate_pose_predictions

            return evaluate_pose_predictions(
                gt, pred, pose_schema=ps, pcutoff=pcut, oks_sigma=sigma
            )

        self._start_job("eval", _job)

    def _populate_eval_result(self, report) -> None:
        self._last_eval_report = report
        self.eval_export_btn.setEnabled(True)

        def _f(v):
            return "n/a" if v is None else f"{v:.2f}"

        swap_txt = (
            "n/a" if report.swap_rate is None else f"{report.swap_rate * 100:.1f}%"
        )
        self.eval_summary_label.setText(
            f"RMSE: {_f(report.rmse_px)} px (at pcut={report.pcutoff:.2f}: {_f(report.rmse_pcutoff_px)} px) | "
            f"Median/p90/p95: {_f(report.p50_px)} / {_f(report.p90_px)} / {_f(report.p95_px)} px | "
            f"OKS mAP: {report.oks_map:.3f} (AP50={report.oks_ap50:.3f}) | "
            f"KP F1: {report.keypoint_f1:.3f} | L/R Swap: {swap_txt}"
        )

        self.eval_kp_table.setRowCount(len(report.per_keypoint))
        for r_i, row in enumerate(report.per_keypoint):
            vals = [
                row.keypoint,
                f"{row.matched_count}/{row.gt_count}",
                f"{row.recall * 100:.1f}%",
                _f(row.rmse_px),
                _f(row.rmse_pcutoff_px),
                _f(row.p50_px),
                _f(row.p90_px),
                _f(row.p95_px),
            ]
            for c_i, txt in enumerate(vals):
                self.eval_kp_table.setItem(r_i, c_i, QtWidgets.QTableWidgetItem(txt))

        self.eval_worst_table.setRowCount(len(report.worst_frames))
        for r_i, wf in enumerate(report.worst_frames):
            vals = [
                str(wf.frame_index),
                wf.image_path or "",
                f"{wf.matched_instances}/{wf.gt_instances}",
                str(wf.missing_keypoints),
                _f(wf.mean_error_px),
                str(wf.swaps),
            ]
            for c_i, txt in enumerate(vals):
                self.eval_worst_table.setItem(r_i, c_i, QtWidgets.QTableWidgetItem(txt))

        self.status_label.setText(
            f"Evaluated {report.frames_evaluated}/{report.frames_gt} frames ({report.matched_keypoints_total} keypoints)."
        )

    def _export_eval_report(self) -> None:
        if self._last_eval_report is None:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Pose Evaluation Report", str(Path.home())
        )
        if not out_dir:
            return
        from annolid.postprocessing.pose_evaluation import write_pose_eval_report_bundle

        artifacts = write_pose_eval_report_bundle(self._last_eval_report, out_dir)
        self.status_label.setText(f"Evaluation report exported to {artifacts['html']}")

    def _run_filter(self) -> None:
        src = self.filter_src_edit.text().strip()
        if not src:
            QtWidgets.QMessageBox.warning(
                self, "Missing Input", "Please specify an Input Pose source."
            )
            return
        out = self.filter_out_edit.text().strip() or None
        ps = self._pose_schema_arg()
        method = str(self.filter_method_combo.currentData() or "savgol")
        interp = str(self.filter_interp_combo.currentData() or "linear")
        window = int(self.filter_window_spin.value())
        max_gap = int(self.filter_max_gap_spin.value())
        min_conf = float(self.filter_min_conf_spin.value())
        max_jump = float(self.filter_max_jump_spin.value())
        enforce_bones = bool(self.filter_bone_cb.isChecked())

        def _job():
            from annolid.postprocessing.pose_filtering import (
                PoseFilterConfig,
                filter_pose_source,
            )

            cfg = PoseFilterConfig(
                min_confidence=min_conf if min_conf > 0.0 else None,
                max_jump_px=max_jump if max_jump > 0.0 else None,
                enforce_bone_bounds=enforce_bones,
                interpolate=interp,
                max_gap_frames=max_gap,
                method=method,
                window_length=window,
            )
            return filter_pose_source(src, output_path=out, config=cfg, pose_schema=ps)

        self._start_job("filter", _job)

    def _populate_filter_result(self, report) -> None:
        msg = (
            f"Filtered {report.frames_processed} frames ({report.method}, interp={report.interpolate})\n"
            f"Output: {report.output_path}\n"
            f"Masked: low_conf={report.masked_low_confidence}, jumps={report.masked_jump_outliers}, "
            f"bones={report.masked_bone_outliers}\n"
            f"Interpolated gaps: {report.gaps_interpolated} | Smoothed points: {report.points_smoothed}\n"
            f"Mean displacement: {report.mean_displacement_px:.2f} px (max {report.max_displacement_px:.2f} px)\n"
        )
        self.filter_log_text.appendPlainText(msg)
        self.status_label.setText(f"Filtered pose saved to {report.output_path}")

    def _on_outlier_queue_mode_toggled(self, _checked: bool) -> None:
        queue_only = bool(self.out_queue_only_cb.isChecked())
        self.out_dst_edit.setEnabled(not queue_only)
        self.out_run_btn.setText(
            "Queue Outlier Frames" if queue_only else "Extract Outlier Frames"
        )

    def _run_outliers(self) -> None:
        src = self.out_src_edit.text().strip()
        vid = self.out_vid_edit.text().strip()
        dst = self.out_dst_edit.text().strip()
        queue_only = bool(self.out_queue_only_cb.isChecked())
        if not src or not vid or (not queue_only and not dst):
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Inputs",
                (
                    "Please specify Pose Predictions and Source Video."
                    if queue_only
                    else "Please specify Pose Predictions, Source Video, and Output Folder."
                ),
            )
            return
        ps = self._pose_schema_arg()
        n_frames = int(self.out_num_spin.value())
        min_gap = int(self.out_gap_spin.value())
        min_conf = float(self.out_conf_spin.value())

        def _job():
            from annolid.data.frame_sampling import extract_pose_outlier_frames

            return extract_pose_outlier_frames(
                src,
                vid,
                dst if not queue_only else None,
                pose_schema=ps,
                num_frames=n_frames,
                min_frame_gap=min_gap,
                min_confidence=min_conf,
                extract_images=not queue_only,
            )

        self._start_job("outliers", _job)

    def _populate_outliers_result(self, res) -> None:
        extracted = bool(res.frames_extracted > 0 and res.saved_images)
        self._last_outlier_output_dir = res.output_dir if extracted else None
        self.out_open_btn.setEnabled(extracted)
        self.out_table.setRowCount(len(res.selected_frames))
        reasons_map = {}
        for r_i, item in enumerate(res.selected_frames):
            reason_text = (
                "; ".join(item.reasons) or f"outlier score={item.total_score:.2f}"
            )
            reasons_map[int(item.frame_index)] = reason_text
            vals = [
                str(item.frame_index),
                f"{item.total_score:.2f}",
                str(item.low_conf_count),
                str(item.jump_count),
                str(item.bone_outlier_count),
                reason_text,
            ]
            for c_i, txt in enumerate(vals):
                self.out_table.setItem(r_i, c_i, QtWidgets.QTableWidgetItem(txt))
        if extracted:
            self.status_label.setText(
                f"Extracted {res.frames_extracted} outlier frame(s) -> {res.output_dir}"
            )
        else:
            indices = [int(item.frame_index) for item in res.selected_frames]
            self.status_label.setText(
                f"Queued {len(indices)} pose outlier frame(s) directly in video (no PNGs extracted)."
            )
            if indices:
                self.queueFramesRequested.emit(
                    indices, str(res.video_path), reasons_map
                )

    def _open_outlier_output(self) -> None:
        if self._last_outlier_output_dir:
            self.openDirectoryRequested.emit(self._last_outlier_output_dir)

    def _run_kinematics(self) -> None:
        src = self.kin_src_edit.text().strip()
        if not src:
            QtWidgets.QMessageBox.warning(
                self, "Missing Input", "Please specify a Pose Source."
            )
            return
        ps = self._pose_schema_arg()
        fps = float(self.kin_fps_spin.value())
        ppc = float(self.kin_ppc_spin.value())
        conf = float(self.kin_conf_spin.value())

        def _job():
            from annolid.postprocessing.pose_kinematics import analyze_pose_kinematics

            return analyze_pose_kinematics(
                src,
                pose_schema=ps,
                fps=fps,
                px_per_cm=ppc if ppc > 0.0 else None,
                min_confidence=conf if conf > 0.0 else None,
            )

        self._start_job("kinematics", _job)

    def _populate_kinematics_result(self, report) -> None:
        self._last_kinematics_report = report
        self.kin_export_btn.setEnabled(True)

        self.kin_kp_table.setRowCount(len(report.keypoint_rows))
        for r_i, r in enumerate(report.keypoint_rows):
            for c_i, txt in enumerate(
                [
                    r.instance,
                    r.keypoint,
                    str(r.observed_frames),
                    f"{r.mean_speed_px_s:.2f}",
                    f"{r.p95_speed_px_s:.2f}",
                    f"{r.total_distance_px:.1f}",
                ]
            ):
                self.kin_kp_table.setItem(r_i, c_i, QtWidgets.QTableWidgetItem(txt))

        self.kin_bone_table.setRowCount(len(report.bone_rows))
        for r_i, b in enumerate(report.bone_rows):
            for c_i, txt in enumerate(
                [
                    b.instance,
                    f"{b.kp_a}-{b.kp_b}",
                    str(b.count),
                    f"{b.median_px:.2f}",
                    f"{b.std_px:.2f}",
                ]
            ):
                self.kin_bone_table.setItem(r_i, c_i, QtWidgets.QTableWidgetItem(txt))

        self.kin_angle_table.setRowCount(len(report.joint_angle_rows))
        for r_i, j in enumerate(report.joint_angle_rows):
            for c_i, txt in enumerate(
                [
                    j.instance,
                    f"{j.kp_a}-{j.vertex}-{j.kp_b}",
                    str(j.count),
                    f"{j.mean_deg:.1f}",
                    f"{j.std_deg:.1f}",
                    f"{j.mean_angular_velocity_deg_s:.1f}",
                ]
            ):
                self.kin_angle_table.setItem(r_i, c_i, QtWidgets.QTableWidgetItem(txt))

        self.kin_dyadic_table.setRowCount(len(report.dyadic_rows))
        for r_i, d in enumerate(report.dyadic_rows):
            for c_i, txt in enumerate(
                [
                    d.instance_a,
                    d.kp_a,
                    d.instance_b,
                    d.kp_b,
                    f"{d.mean_distance_px:.1f}",
                    f"{d.close_contact_seconds:.2f}",
                ]
            ):
                self.kin_dyadic_table.setItem(r_i, c_i, QtWidgets.QTableWidgetItem(txt))

        self.status_label.setText(
            f"Analyzed {report.frames_analyzed} frames ({len(report.keypoint_rows)} keypoint tracks, "
            f"{len(report.joint_angle_rows)} joint angles)."
        )

    def _export_kinematics_bundle(self) -> None:
        if self._last_kinematics_report is None:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Pose Kinematics Bundle", str(Path.home())
        )
        if not out_dir:
            return
        from annolid.postprocessing.pose_kinematics import write_pose_kinematics_bundle

        artifacts = write_pose_kinematics_bundle(self._last_kinematics_report, out_dir)
        self.status_label.setText(
            f"Exported {len(artifacts)} pose kinematics files to {out_dir}"
        )

    def _run_render_video(self) -> None:
        vid = self.ren_vid_edit.text().strip()
        pose = self.ren_pose_edit.text().strip()
        out = self.ren_out_edit.text().strip()
        if not vid or not pose or not out:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please specify Source Video, Pose Source, and Output Video path.",
            )
            return
        ps = self._pose_schema_arg()
        color_mode = str(self.ren_color_combo.currentData() or "instance")
        min_conf = float(self.ren_conf_spin.value())
        trail = int(self.ren_trail_spin.value())
        draw_lbl = bool(self.ren_labels_cb.isChecked())

        def _job():
            from annolid.postprocessing.pose_video_renderer import (
                PoseVideoRenderConfig,
                render_pose_video,
            )

            cfg = PoseVideoRenderConfig(
                color_mode=color_mode,
                min_confidence=min_conf if min_conf > 0.0 else None,
                trail_length=trail,
                draw_labels=draw_lbl,
            )
            return render_pose_video(vid, pose, out, pose_schema=ps, config=cfg)

        self._start_job("render_video", _job)

    def _run_swap(self, *, dry_run: bool) -> None:
        folder = self.swap_dir_edit.text().strip()
        inst_a = self.swap_a_edit.text().strip()
        inst_b = self.swap_b_edit.text().strip()
        if not folder or not inst_a or not inst_b:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please specify LabelMe Folder, Instance A, and Instance B.",
            )
            return
        start_f = int(self.swap_start_spin.value())
        end_raw = int(self.swap_end_spin.value())
        end_f = end_raw if end_raw >= 0 else None
        ps = self._pose_schema_arg()

        def _job():
            from annolid.postprocessing.pose_filtering import (
                swap_pose_instances_from_frame,
            )

            return swap_pose_instances_from_frame(
                folder,
                start_frame=start_f,
                end_frame=end_f,
                instance_a=inst_a,
                instance_b=inst_b,
                pose_schema=ps,
                dry_run=dry_run,
            )

        self._start_job("swap_pose", _job)

    def shutdown_workers(self, timeout_ms: int = 3000) -> None:
        if self._active_worker is not None and self._active_worker.isRunning():
            self._active_worker.wait(timeout_ms)


class PoseWorkbenchDialog(QtWidgets.QDialog):
    """Non-modal top-level dialog wrapping ``PoseWorkbenchWidget``."""

    jumpToFrameRequested = QtCore.Signal(int)
    openDirectoryRequested = QtCore.Signal(str)
    queueFramesRequested = QtCore.Signal(list, str, dict)

    def __init__(
        self,
        owner_window: Optional[Any] = None,
        *,
        initial_tab: str = "eval",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pose Estimation Workbench")
        self.resize(960, 720)
        self.setModal(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.workbench = PoseWorkbenchWidget(
            owner_window=owner_window,
            initial_tab=initial_tab,
            parent=self,
        )
        self.workbench.jumpToFrameRequested.connect(self.jumpToFrameRequested.emit)
        self.workbench.openDirectoryRequested.connect(self.openDirectoryRequested.emit)
        self.workbench.queueFramesRequested.connect(self.queueFramesRequested.emit)
        layout.addWidget(self.workbench)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.workbench.shutdown_workers()
        super().closeEvent(event)
