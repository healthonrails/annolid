"""Task-aware, embedding-powered frame selection & optional extraction dialog.

Supports both:
- **Queue in Video (No Extraction)**: selects diverse frames for pose or
  segmentation directly in the open video timeline without duplicating PNGs.
- **Extract Images to Folder**: writes selected PNG/JPG frames + manifest to disk.

Retains full backward compatibility with legacy ``ExtractFrameDialog`` attributes
(``slider``, ``framesLineEdit``, ``radio_btn_kmeans``, ``radio_btn_uniform``,
``radio_btn1``..``radio_btn3``, ``startSecondsLineEdit``, ``endSecondsLineEdit``,
``start_sconds``, ``algo``, ``num_frames``, ``video_file``, ``out_dir``).
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from annolid.data.frame_sampling import discover_labeled_frame_indices
from annolid.features.dino_models import iter_dino_feature_models
from annolid.gui.qt_compat import configure_compact_combo
from annolid.services.frame_selection import (
    FrameSelectionConfig,
    TASK_PROFILES,
    find_cached_video_embedding_indices,
    get_task_profile,
    list_available_feature_backends,
    select_frames_for_labeling,
)


class FrameSelectionWorker(QtCore.QThread):
    """Cancellable background worker running frame selection / extraction off the UI thread."""

    progress = QtCore.Signal(int, str)
    finishedResult = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(
        self,
        video_path: str | Path,
        config: FrameSelectionConfig,
        *,
        backend: Optional[Any] = None,
        precomputed_index: Optional[Any] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = Path(video_path)
        self._config = config
        self._backend = backend
        self._precomputed_index = precomputed_index
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:  # noqa: N802
        try:
            result = select_frames_for_labeling(
                self._video_path,
                config=self._config,
                backend=self._backend,
                precomputed_index=self._precomputed_index,
                progress_callback=lambda pct, msg: self.progress.emit(
                    max(0, min(100, int(round(float(pct))))), str(msg)
                ),
                is_stopped=self._stop_event.is_set,
            )
            if self._stop_event.is_set() or result.cancelled:
                self.cancelled.emit()
                return
            self.finishedResult.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExtractFrameDialog(QtWidgets.QDialog):
    """Unified dialog for selecting or extracting diverse frames for labeling."""

    _FEATURE_ALGOS = frozenset({"hybrid", "kmeans", "farthest"})

    def __init__(
        self,
        video_file=None,
        *args,
        annotation_dir: Optional[str | Path] = None,
        owner_window: Optional[Any] = None,
        total_frames: Optional[int] = None,
        fps: Optional[float] = None,
        slider_selection: Optional[Tuple[int, int]] = None,
        default_extract_images: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Select / Extract Frames for Labeling")
        self.setMinimumWidth(640)

        self._owner_window = owner_window
        self._annotation_dir: Optional[str] = (
            str(annotation_dir) if annotation_dir else None
        )
        self._total_frames: Optional[int] = (
            int(total_frames) if total_frames is not None else None
        )
        self._fps: Optional[float] = float(fps) if fps and float(fps) > 0 else None
        self._slider_selection: Optional[Tuple[int, int]] = slider_selection
        self._syncing_count = False
        self._applying_task_preset = False

        # Public attributes (preserving legacy names & defaults)
        self.num_frames: int = 100
        self.algo: str = "hybrid"
        self.task: str = "pose"
        self.video_file: Optional[str] = str(video_file) if video_file else None
        self.out_dir: Optional[str] = None
        self._start_seconds: Optional[float] = None
        self.end_seconds: Optional[float] = None
        self.sub_clip: bool = False

        if default_extract_images is None:
            # Prefer zero-extraction in-video labeling whenever an open video is passed.
            self._default_extract_images = not bool(self.video_file)
        else:
            self._default_extract_images = bool(default_extract_images)

        self._build_ui()
        self._refresh_video_context()
        self._update_dynamic_visibility()

    # ------------------------------------------------------------------
    # Backward-compatible property alias for legacy typo `start_sconds`
    # ------------------------------------------------------------------
    @property
    def start_seconds(self) -> Optional[float]:
        return self._start_seconds

    @start_seconds.setter
    def start_seconds(self, value: Optional[float]) -> None:
        self._start_seconds = value

    @property
    def start_sconds(self) -> Optional[float]:
        return self._start_seconds

    @start_sconds.setter
    def start_sconds(self, value: Optional[float]) -> None:
        self._start_seconds = value

    @property
    def extract_images(self) -> bool:
        return bool(self.mode_extract_radio.isChecked())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 1. Source Video row
        self.groupBoxFiles = QtWidgets.QGroupBox("Source Video")
        file_layout = QtWidgets.QVBoxLayout(self.groupBoxFiles)
        file_layout.setSpacing(4)
        file_row = QtWidgets.QHBoxLayout()
        self.inputFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputFileLineEdit.setPlaceholderText(
            "Choose a video file (.mp4, .avi, .mov, .mkv)…"
        )
        if self.video_file:
            self.inputFileLineEdit.setText(self.video_file)
        self.inputFileLineEdit.textChanged.connect(self._on_video_path_edited)
        self.inputFileButton = QtWidgets.QPushButton("Browse…", self)
        self.inputFileButton.clicked.connect(self.onInputFileButtonClicked)
        file_row.addWidget(self.inputFileLineEdit, 1)
        file_row.addWidget(self.inputFileButton)
        file_layout.addLayout(file_row)

        self.video_meta_label = QtWidgets.QLabel("")
        self.video_meta_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        self.video_meta_label.setWordWrap(True)
        file_layout.addWidget(self.video_meta_label)
        root.addWidget(self.groupBoxFiles)

        # 2. Workflow Mode (No Extraction vs Extract to Folder) + Task Preset
        top_split = QtWidgets.QHBoxLayout()
        top_split.setSpacing(8)

        # Task profile box
        self.task_group = QtWidgets.QGroupBox("1. Labeling Task")
        task_v = QtWidgets.QVBoxLayout(self.task_group)
        task_v.setSpacing(4)
        task_row = QtWidgets.QHBoxLayout()
        self.task_pose_radio = QtWidgets.QRadioButton("Pose (Keypoints)")
        self.task_seg_radio = QtWidgets.QRadioButton("Segmentation (Masks)")
        self.task_general_radio = QtWidgets.QRadioButton("General")
        self.task_pose_radio.setChecked(True)
        self._task_btn_group = QtWidgets.QButtonGroup(self)
        for btn, key in (
            (self.task_pose_radio, "pose"),
            (self.task_seg_radio, "segmentation"),
            (self.task_general_radio, "general"),
        ):
            btn.setProperty("task_key", key)
            btn.toggled.connect(self._on_task_toggled)
            self._task_btn_group.addButton(btn)
            task_row.addWidget(btn)
        task_v.addLayout(task_row)
        self.task_hint_label = QtWidgets.QLabel(TASK_PROFILES["pose"].description)
        self.task_hint_label.setWordWrap(True)
        self.task_hint_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        task_v.addWidget(self.task_hint_label)
        top_split.addWidget(self.task_group, 3)

        # Output mode box ("extract or not extract")
        self.mode_group = QtWidgets.QGroupBox("2. Output Mode")
        mode_v = QtWidgets.QVBoxLayout(self.mode_group)
        mode_v.setSpacing(4)
        self.mode_queue_radio = QtWidgets.QRadioButton("Queue in Video (No Extraction)")
        self.mode_queue_radio.setToolTip(
            "Select frames directly on the video timeline and open the Label Queue "
            "without writing extra PNG files to disk. You can still extract later."
        )
        self.mode_extract_radio = QtWidgets.QRadioButton("Extract Images to Folder")
        self.mode_extract_radio.setToolTip(
            "Save selected frames as PNG/JPG images (+ manifest JSON) into a folder."
        )
        self._mode_btn_group = QtWidgets.QButtonGroup(self)
        self._mode_btn_group.addButton(self.mode_queue_radio)
        self._mode_btn_group.addButton(self.mode_extract_radio)
        if self._default_extract_images:
            self.mode_extract_radio.setChecked(True)
        else:
            self.mode_queue_radio.setChecked(True)
        self.mode_queue_radio.toggled.connect(self._update_dynamic_visibility)
        self.mode_extract_radio.toggled.connect(self._update_dynamic_visibility)
        mode_v.addWidget(self.mode_queue_radio)
        mode_v.addWidget(self.mode_extract_radio)
        top_split.addWidget(self.mode_group, 2)
        root.addLayout(top_split)

        # 3. Algorithm + Frame Embedding Features
        self.raidoButtons()
        root.addWidget(self.groupBox)

        self.features_group = QtWidgets.QGroupBox(
            "4. Frame Embedding / Feature Representation"
        )
        feat_v = QtWidgets.QVBoxLayout(self.features_group)
        feat_v.setSpacing(6)

        feat_row = QtWidgets.QHBoxLayout()
        feat_row.addWidget(QtWidgets.QLabel("Features:"))
        self.feature_backend_combo = QtWidgets.QComboBox(self)
        configure_compact_combo(
            self.feature_backend_combo, min_chars=18, popup_min_width=360
        )
        self.feature_backend_combo.currentIndexChanged.connect(
            self._on_feature_backend_changed
        )
        feat_row.addWidget(self.feature_backend_combo, 1)
        feat_v.addLayout(feat_row)

        self.feature_desc_label = QtWidgets.QLabel("")
        self.feature_desc_label.setWordWrap(True)
        self.feature_desc_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        feat_v.addWidget(self.feature_desc_label)

        # Contextual ONNX EmbeddingGemma 2 options (preset + optional text focus prompt)
        self.onnx_options_widget = QtWidgets.QWidget(self)
        onnx_grid = QtWidgets.QGridLayout(self.onnx_options_widget)
        onnx_grid.setContentsMargins(0, 0, 0, 0)
        onnx_grid.setSpacing(6)
        onnx_grid.addWidget(QtWidgets.QLabel("ONNX Preset:"), 0, 0)
        self.onnx_preset_combo = QtWidgets.QComboBox(self)
        self.onnx_preset_combo.addItem(
            "Q4 · 70t · 256d (Fast scan — recommended)", ("q4", 256, 70)
        )
        self.onnx_preset_combo.addItem("Q4 · 140t · 768d (Balanced)", ("q4", 768, 140))
        self.onnx_preset_combo.addItem(
            "Q4 · 280t · 768d (High detail)", ("q4", 768, 280)
        )
        self.onnx_preset_combo.addItem("INT8 · 140t · 768d", ("quantized", 768, 140))
        onnx_grid.addWidget(self.onnx_preset_combo, 0, 1)

        onnx_grid.addWidget(QtWidgets.QLabel("Focus Prompt (optional):"), 1, 0)
        self.query_text_edit = QtWidgets.QLineEdit(self)
        self.query_text_edit.setPlaceholderText(
            "e.g. rearing, grooming, two mice interacting (blank = whole-video diversity)"
        )
        onnx_grid.addWidget(self.query_text_edit, 1, 1)
        feat_v.addWidget(self.onnx_options_widget)

        # Contextual DINOv3 / Vision backbone picker
        self.dino_options_widget = QtWidgets.QWidget(self)
        dino_row = QtWidgets.QHBoxLayout(self.dino_options_widget)
        dino_row.setContentsMargins(0, 0, 0, 0)
        dino_row.addWidget(QtWidgets.QLabel("Vision Model:"))
        self.dino_model_combo = QtWidgets.QComboBox(self)
        for info in iter_dino_feature_models():
            self.dino_model_combo.addItem(info.display_name, info.model_id)
        configure_compact_combo(
            self.dino_model_combo, min_chars=16, popup_min_width=320
        )
        dino_row.addWidget(self.dino_model_combo, 1)
        dino_row.addWidget(QtWidgets.QLabel("Short side:"))
        self.dino_short_side_spin = QtWidgets.QSpinBox(self)
        self.dino_short_side_spin.setRange(160, 768)
        self.dino_short_side_spin.setSingleStep(32)
        self.dino_short_side_spin.setValue(384)
        dino_row.addWidget(self.dino_short_side_spin)
        feat_v.addWidget(self.dino_options_widget)

        # Contextual cached .npz index picker
        self.cached_index_widget = QtWidgets.QWidget(self)
        cached_row = QtWidgets.QHBoxLayout(self.cached_index_widget)
        cached_row.setContentsMargins(0, 0, 0, 0)
        cached_row.addWidget(QtWidgets.QLabel("Index (.npz):"))
        self.cached_index_edit = QtWidgets.QLineEdit(self)
        self.cached_index_edit.setPlaceholderText("Auto-detected or browse .npz file…")
        self.cached_index_browse_btn = QtWidgets.QPushButton("Browse…", self)
        self.cached_index_browse_btn.clicked.connect(self._browse_cached_index)
        cached_row.addWidget(self.cached_index_edit, 1)
        cached_row.addWidget(self.cached_index_browse_btn)
        feat_v.addWidget(self.cached_index_widget)

        self.cache_status_label = QtWidgets.QLabel("")
        self.cache_status_label.setWordWrap(True)
        self.cache_status_label.setStyleSheet("color: #0f766e; font-size: 11px;")
        feat_v.addWidget(self.cache_status_label)
        root.addWidget(self.features_group)

        # 4. Frame Count & Smart Filters
        self.count_group = QtWidgets.QGroupBox("5. Frame Count, Quality && Range")
        count_v = QtWidgets.QVBoxLayout(self.count_group)
        count_v.setSpacing(6)

        self.label1 = QtWidgets.QLabel(
            f"Number of frames to select (default={self.num_frames}, or -1 for all frames):"
        )
        count_v.addWidget(self.label1)

        count_row = QtWidgets.QHBoxLayout()
        self.framesLineEdit = QtWidgets.QLineEdit(self)
        self.framesLineEdit.setFixedWidth(78)
        self.framesLineEdit.setText(str(self.num_frames))
        self.framesLineEdit.setValidator(QtGui.QIntValidator(-1, 2_147_483_647, self))
        self.framesLineEdit.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.framesLineEdit.textChanged.connect(self._on_frames_text_changed)
        count_row.addWidget(self.framesLineEdit)

        self.slider()
        count_row.addWidget(self.slider, 1)

        for preset_n in (20, 40, 100):
            pbtn = QtWidgets.QToolButton(self)
            pbtn.setText(str(preset_n))
            pbtn.setToolTip(f"Set count to {preset_n} frames")
            pbtn.clicked.connect(
                lambda _checked=False, n=preset_n: self._set_frame_count(n)
            )
            count_row.addWidget(pbtn)
        count_v.addLayout(count_row)

        # Smart filter controls
        filters_row = QtWidgets.QHBoxLayout()
        filters_row.setSpacing(10)
        self.exclude_labeled_cb = QtWidgets.QCheckBox(
            "Skip already-labeled frames (seed diversity)"
        )
        self.exclude_labeled_cb.setChecked(True)
        self.exclude_labeled_cb.setToolTip(
            "Excludes frames with existing annotations (disk JSON or AnnotationStore) "
            "and seeds diversity from them so new picks fill coverage gaps."
        )
        filters_row.addWidget(self.exclude_labeled_cb)

        self.filter_blur_cb = QtWidgets.QCheckBox("Filter blurry / dark")
        self.filter_blur_cb.setChecked(True)
        filters_row.addWidget(self.filter_blur_cb)

        self.include_boundaries_cb = QtWidgets.QCheckBox("Include start anchor (VOS)")
        self.include_boundaries_cb.setChecked(False)
        self.include_boundaries_cb.setToolTip(
            "Pin a sharp frame near the start of the range (helpful for Cutie / SAM video propagation)."
        )
        filters_row.addWidget(self.include_boundaries_cb)
        filters_row.addStretch(1)
        count_v.addLayout(filters_row)

        # Spacing + candidate scan pool
        tuning_row = QtWidgets.QHBoxLayout()
        tuning_row.setSpacing(8)
        tuning_row.addWidget(QtWidgets.QLabel("Min frame gap:"))
        self.min_gap_spin = QtWidgets.QSpinBox(self)
        self.min_gap_spin.setRange(-1, 5000)
        self.min_gap_spin.setValue(-1)
        self.min_gap_spin.setSpecialValueText("Auto")
        self.min_gap_spin.setToolTip(
            "Minimum number of frames between picks (-1 = Auto scaled to video length)."
        )
        tuning_row.addWidget(self.min_gap_spin)

        tuning_row.addWidget(QtWidgets.QLabel("Candidate pool:"))
        self.max_candidates_spin = QtWidgets.QSpinBox(self)
        self.max_candidates_spin.setRange(20, 20000)
        self.max_candidates_spin.setSingleStep(100)
        self.max_candidates_spin.setValue(600)
        self.max_candidates_spin.setToolTip(
            "Maximum candidate frames evaluated during visual/embedding clustering."
        )
        tuning_row.addWidget(self.max_candidates_spin)

        tuning_row.addWidget(QtWidgets.QLabel("Seed:"))
        self.seed_spin = QtWidgets.QSpinBox(self)
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        tuning_row.addWidget(self.seed_spin)
        tuning_row.addStretch(1)
        count_v.addLayout(tuning_row)

        # Optional time range (supports floats + slider selection fill)
        self.groupBoxSubClip = QtWidgets.QGroupBox("Time Range in Seconds (Optional)")
        range_row = QtWidgets.QHBoxLayout(self.groupBoxSubClip)
        self.startSecondsLabel = QtWidgets.QLabel("Start (s):", self)
        self.startSecondsLineEdit = QtWidgets.QLineEdit(self)
        self.startSecondsLineEdit.setPlaceholderText("e.g. 0")
        start_val = QtGui.QDoubleValidator(0.0, 864000.0, 3, self)
        start_val.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.startSecondsLineEdit.setValidator(start_val)
        self.startSecondsLineEdit.textChanged.connect(self.onCutClipStartTimeChanged)

        self.endSecondsLabel = QtWidgets.QLabel("End (s):", self)
        self.endSecondsLineEdit = QtWidgets.QLineEdit(self)
        self.endSecondsLineEdit.setPlaceholderText("e.g. 60")
        end_val = QtGui.QDoubleValidator(0.0, 864000.0, 3, self)
        end_val.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.endSecondsLineEdit.setValidator(end_val)
        self.endSecondsLineEdit.textChanged.connect(self.onCutClipEndTimeChanged)

        self.use_slider_range_btn = QtWidgets.QPushButton(
            "Use Timeline Selection", self
        )
        self.use_slider_range_btn.setToolTip(
            "Fill start/end seconds from the current Shift-drag selection on the video slider"
        )
        self.use_slider_range_btn.clicked.connect(self._fill_from_slider_selection)
        self.use_slider_range_btn.setVisible(bool(self._slider_selection and self._fps))

        range_row.addWidget(self.startSecondsLabel)
        range_row.addWidget(self.startSecondsLineEdit)
        range_row.addWidget(self.endSecondsLabel)
        range_row.addWidget(self.endSecondsLineEdit)
        range_row.addWidget(self.use_slider_range_btn)
        count_v.addWidget(self.groupBoxSubClip)
        root.addWidget(self.count_group)

        # 5. Output Directory (only shown/active when Extract Images mode is selected)
        self.groupBoxOutDir = QtWidgets.QGroupBox("Extract Output Settings")
        out_v = QtWidgets.QVBoxLayout(self.groupBoxOutDir)
        out_v.setSpacing(6)
        out_row = QtWidgets.QHBoxLayout()
        self.outFileDirEdit = QtWidgets.QLineEdit(self)
        self.outFileDirEdit.setPlaceholderText(
            "Optional folder (default: <video_stem>_<algo> next to video)"
        )
        self.outFileDirEdit.textChanged.connect(self._on_out_dir_edited)
        self.outDirButton = QtWidgets.QPushButton("Select…", self)
        self.outDirButton.clicked.connect(self.onOutDirButtonClicked)
        out_row.addWidget(self.outFileDirEdit, 1)
        out_row.addWidget(self.outDirButton)
        out_v.addLayout(out_row)

        out_opts = QtWidgets.QHBoxLayout()
        out_opts.addWidget(QtWidgets.QLabel("Format:"))
        self.image_format_combo = QtWidgets.QComboBox(self)
        self.image_format_combo.addItem("PNG (lossless)", "png")
        self.image_format_combo.addItem("JPEG (compact)", "jpg")
        out_opts.addWidget(self.image_format_combo)
        self.open_folder_cb = QtWidgets.QCheckBox(
            "Open extracted folder in Files dock when done"
        )
        self.open_folder_cb.setChecked(True)
        out_opts.addWidget(self.open_folder_cb)
        out_opts.addStretch(1)
        out_v.addLayout(out_opts)
        root.addWidget(self.groupBoxOutDir)

        # Dialog Buttons
        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn, self)
        self.buttonbox.accepted.connect(self._on_accept)
        self.buttonbox.rejected.connect(self.reject)
        root.addWidget(self.buttonbox)

    def raidoButtons(self) -> None:
        """Build algorithm selector radios (legacy method name retained for compat)."""
        self.groupBox = QtWidgets.QGroupBox("3. Selection Strategy")
        grid = QtWidgets.QGridLayout(self.groupBox)
        grid.setSpacing(6)

        self.radio_btn_hybrid = QtWidgets.QRadioButton("hybrid")
        self.radio_btn_hybrid.setToolTip(
            "Recommended: K-Means++ cluster medoids + Farthest-Point coreset "
            "(covers both common and rare postures/appearances)"
        )
        self.radio_btn_hybrid.setChecked(True)

        self.radio_btn_kmeans = QtWidgets.QRadioButton("kmeans")
        self.radio_btn_kmeans.setToolTip(
            "K-means++ clustering medoids (representative visual diversity)"
        )

        self.radio_btn_farthest = QtWidgets.QRadioButton("farthest")
        self.radio_btn_farthest.setToolTip(
            "Farthest-Point / K-Center coreset (maximizes distance from already-labeled & selected frames)"
        )

        self.radio_btn_uniform = QtWidgets.QRadioButton("uniform")
        self.radio_btn_uniform.setToolTip("Evenly spaced frames across the range")

        self.radio_btn1 = QtWidgets.QRadioButton("random")
        self.radio_btn1.setToolTip("Deterministic random sample across the range")

        self.radio_btn2 = QtWidgets.QRadioButton("keyframes")
        self.radio_btn2.setToolTip("Video codec I-frames / keyframes")

        self.radio_btn3 = QtWidgets.QRadioButton("flow")
        self.radio_btn3.setToolTip("Highest optical-flow motion peak frames")

        self._algo_btn_group = QtWidgets.QButtonGroup(self)
        ordered = (
            self.radio_btn_hybrid,
            self.radio_btn_kmeans,
            self.radio_btn_farthest,
            self.radio_btn_uniform,
            self.radio_btn1,
            self.radio_btn2,
            self.radio_btn3,
        )
        for idx, rb in enumerate(ordered):
            rb.toggled.connect(self.onRadioButtonChecked)
            self._algo_btn_group.addButton(rb)
            grid.addWidget(rb, idx // 4, idx % 4)

    def slider(self) -> None:
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(-1)
        self.slider.setMaximum(1000)
        self.slider.setValue(self.num_frames)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(50)
        self.slider.valueChanged.connect(self.onSliderChange)

    # ------------------------------------------------------------------
    # Handlers & live context refresh
    # ------------------------------------------------------------------
    def _refresh_video_context(self) -> None:
        vpath_text = self.inputFileLineEdit.text().strip()
        vpath = Path(vpath_text).expanduser() if vpath_text else None
        valid_video = vpath if (vpath is not None and vpath.is_file()) else None

        # Populate feature backends with live availability + cached index discovery.
        prev_key = self.feature_backend_combo.currentData()
        options = list_available_feature_backends(valid_video)
        self.feature_backend_combo.blockSignals(True)
        self.feature_backend_combo.clear()
        self._feature_options_by_key: Dict[str, Any] = {}
        for opt in options:
            self._feature_options_by_key[opt.key] = opt
            self.feature_backend_combo.addItem(opt.label, opt.key)
        if prev_key is not None:
            idx = self.feature_backend_combo.findData(prev_key)
            if idx >= 0:
                self.feature_backend_combo.setCurrentIndex(idx)
        self.feature_backend_combo.blockSignals(False)

        # Auto-fill discovered cached .npz index path
        cached_list = (
            find_cached_video_embedding_indices(valid_video)
            if valid_video is not None
            else []
        )
        if cached_list:
            best = cached_list[0]
            if not self.cached_index_edit.text().strip():
                self.cached_index_edit.setText(best.path)
            self.cache_status_label.setText(
                f"Cached embeddings ready: {best.display_label} — near-instant selection!"
            )
            self.cache_status_label.setVisible(True)
        else:
            self.cache_status_label.setText("")
            self.cache_status_label.setVisible(False)

        # Discover existing annotations count for user awareness
        meta_parts = []
        if valid_video is not None:
            if self._total_frames and self._fps:
                dur = self._total_frames / self._fps
                meta_parts.append(
                    f"{self._total_frames:,} frames · {self._fps:.1f} fps · {dur:.1f}s"
                )
            ann_dir = (
                Path(self._annotation_dir)
                if self._annotation_dir
                else valid_video.with_suffix("")
            )
            labeled = discover_labeled_frame_indices(ann_dir, require_shapes=True)
            if labeled:
                meta_parts.append(f"{len(labeled)} frame(s) already labeled")
                self.exclude_labeled_cb.setText(
                    f"Skip {len(labeled)} already-labeled frames (seed diversity)"
                )
            else:
                self.exclude_labeled_cb.setText(
                    "Skip already-labeled frames (seed diversity)"
                )
        self.video_meta_label.setText(" · ".join(meta_parts))
        self.video_meta_label.setVisible(bool(meta_parts))
        self._on_feature_backend_changed()

    def _on_video_path_edited(self, text: str) -> None:
        clean = str(text or "").strip()
        self.video_file = clean or None
        self._refresh_video_context()

    def _on_out_dir_edited(self, text: str) -> None:
        clean = str(text or "").strip()
        self.out_dir = clean or None

    def _on_task_toggled(self, checked: bool) -> None:
        if not checked:
            return
        sender = self.sender()
        key = (
            str(sender.property("task_key") or "pose") if sender is not None else "pose"
        )
        self.task = key
        profile = get_task_profile(key)
        self.task_hint_label.setText(profile.description)
        self._applying_task_preset = True
        try:
            self.include_boundaries_cb.setChecked(profile.include_boundaries)
            self.exclude_labeled_cb.setChecked(profile.exclude_labeled)
            self.filter_blur_cb.setChecked(profile.filter_blur)
            if key == "pose":
                self.radio_btn_hybrid.setChecked(True)
            elif key == "segmentation":
                self.radio_btn_hybrid.setChecked(True)
        finally:
            self._applying_task_preset = False

    def _on_feature_backend_changed(self) -> None:
        key = str(self.feature_backend_combo.currentData() or "auto")
        opt = getattr(self, "_feature_options_by_key", {}).get(key)
        if opt is not None:
            desc = opt.description
            if not opt.available and opt.install_hint:
                desc = f"{desc}  (Install: {opt.install_hint})"
            self.feature_desc_label.setText(desc)
        else:
            self.feature_desc_label.setText("")

        uses_features = self.algo in self._FEATURE_ALGOS
        self.onnx_options_widget.setVisible(
            uses_features and key == "embeddinggemma2_onnx"
        )
        self.dino_options_widget.setVisible(uses_features and key == "dinov3")
        self.cached_index_widget.setVisible(uses_features and key == "cached_index")

    def _update_dynamic_visibility(self) -> None:
        extract_mode = bool(self.mode_extract_radio.isChecked())
        self.groupBoxOutDir.setVisible(extract_mode)

        ok_btn = self.buttonbox.button(QtWidgets.QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText(
                "Select && Extract Frames"
                if extract_mode
                else "Select && Open Label Queue"
            )

        uses_features = self.algo in self._FEATURE_ALGOS
        self.features_group.setEnabled(uses_features)
        self.filter_blur_cb.setEnabled(uses_features)
        self.include_boundaries_cb.setEnabled(uses_features)
        self.max_candidates_spin.setEnabled(uses_features or self.algo == "flow")
        self._on_feature_backend_changed()

    def _browse_cached_index(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Cached Embedding Index (.npz)",
            self.cached_index_edit.text().strip() or str(Path.home()),
            "NumPy Embedding Index (*.npz);;All Files (*)",
        )
        if path:
            self.cached_index_edit.setText(path)

    def _fill_from_slider_selection(self) -> None:
        if not self._slider_selection or not self._fps or self._fps <= 0:
            return
        start_f, end_f = self._slider_selection
        self.startSecondsLineEdit.setText(f"{start_f / self._fps:.2f}")
        self.endSecondsLineEdit.setText(f"{end_f / self._fps:.2f}")

    def onInputFileButtonClicked(self) -> None:
        start_dir = (
            str(Path(self.video_file).parent)
            if self.video_file and Path(self.video_file).parent.exists()
            else str(Path.home())
        )
        chosen, _flt = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open video file",
            directory=start_dir,
            filter="Videos (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg);;All Files (*)",
        )
        if chosen:
            self.video_file = chosen
            self.inputFileLineEdit.setText(chosen)

    def onOutDirButtonClicked(self) -> None:
        start_dir = self.out_dir or (
            str(Path(self.video_file).parent) if self.video_file else str(Path.home())
        )
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start_dir
        )
        if chosen:
            self.out_dir = chosen
            self.outFileDirEdit.setText(chosen)

    def onRadioButtonChecked(self) -> None:
        radio_btn = self.sender()
        if radio_btn is None or not radio_btn.isChecked():
            return
        self.algo = radio_btn.text()
        is_keyframes = self.algo == "keyframes"
        self.slider.setEnabled(not is_keyframes)
        self.slider.setVisible(not is_keyframes)
        self.label1.setVisible(not is_keyframes)
        self.framesLineEdit.setVisible(not is_keyframes)
        self._update_dynamic_visibility()

    def _set_frame_count(self, value: int) -> None:
        val = int(value)
        self.num_frames = val
        self._syncing_count = True
        try:
            self.framesLineEdit.setText(str(val))
            self.slider.setValue(
                max(self.slider.minimum(), min(val, self.slider.maximum()))
            )
        finally:
            self._syncing_count = False
        self._update_count_label()

    def onSliderChange(self, position) -> None:
        if self._syncing_count:
            return
        try:
            val = int(position)
        except (TypeError, ValueError):
            val = -1
        self.num_frames = val
        self._syncing_count = True
        try:
            self.framesLineEdit.setText(str(val))
        finally:
            self._syncing_count = False
        self._update_count_label()

    def _on_frames_text_changed(self, text: str) -> None:
        if self._syncing_count:
            return
        raw = str(text or "").strip()
        if not raw or raw == "-":
            return
        try:
            val = int(raw)
        except ValueError:
            return
        self.num_frames = val
        self._syncing_count = True
        try:
            clamped = max(self.slider.minimum(), min(val, self.slider.maximum()))
            self.slider.setValue(clamped)
        finally:
            self._syncing_count = False
        self._update_count_label()

    def _update_count_label(self) -> None:
        if self.num_frames == -1:
            self.label1.setText("Selected: all frames in range.")
        else:
            self.label1.setText(f"Selected: {self.num_frames} frames.")

    @staticmethod
    def _parse_seconds(raw: str) -> Optional[float]:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            val = float(text)
        except ValueError:
            return None
        if val < 0:
            return None
        return int(val) if val.is_integer() else val

    def onCutClipStartTimeChanged(self) -> None:
        self._start_seconds = self._parse_seconds(self.startSecondsLineEdit.text())
        self.sub_clip = self._start_seconds is not None and self.end_seconds is not None

    def onCutClipEndTimeChanged(self) -> None:
        self.end_seconds = self._parse_seconds(self.endSecondsLineEdit.text())
        self.sub_clip = self._start_seconds is not None and self.end_seconds is not None

    def _on_accept(self) -> None:
        self.video_file = self.inputFileLineEdit.text().strip() or None
        self.out_dir = self.outFileDirEdit.text().strip() or None
        if not self.video_file:
            QtWidgets.QMessageBox.warning(
                self, "Missing Video", "Please select a video file first."
            )
            return
        if not Path(self.video_file).expanduser().is_file():
            QtWidgets.QMessageBox.warning(
                self,
                "Video Not Found",
                f"Video file does not exist:\n{self.video_file}",
            )
            return
        if (
            self._start_seconds is not None
            and self.end_seconds is not None
            and float(self.end_seconds) < float(self._start_seconds)
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Time Range",
                "End seconds must be greater than or equal to Start seconds.",
            )
            return

        if self.algo in self._FEATURE_ALGOS:
            feat_key = str(self.feature_backend_combo.currentData() or "auto")
            opt = getattr(self, "_feature_options_by_key", {}).get(feat_key)
            if feat_key == "cached_index":
                idx_text = self.cached_index_edit.text().strip()
                if not idx_text or not Path(idx_text).expanduser().is_file():
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Missing Embedding Index",
                        "Please browse to an existing cached embedding .npz file, "
                        "or choose another feature representation.",
                    )
                    return
            elif opt is not None and not opt.available:
                has_usable_cache = opt.cached_index is not None and not (
                    feat_key == "embeddinggemma2_onnx"
                    and self.query_text_edit.text().strip()
                )
                if not has_usable_cache:
                    hint = (
                        f"\n\nInstall hint:\n{opt.install_hint}"
                        if opt.install_hint
                        else ""
                    )
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Feature Backend Not Installed",
                        f"'{opt.label}' requires optional dependencies.{hint}\n\n"
                        "Switch to built-in 'Foreground + Motion Visual' instead?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.Yes,
                    )
                    if reply == QtWidgets.QMessageBox.Yes:
                        fallback_idx = self.feature_backend_combo.findData("visual_fg")
                        if fallback_idx >= 0:
                            self.feature_backend_combo.setCurrentIndex(fallback_idx)
                    else:
                        return
        self.accept()

    def build_config(self) -> FrameSelectionConfig:
        """Build a ``FrameSelectionConfig`` reflecting all current dialog inputs."""
        feat_key = str(self.feature_backend_combo.currentData() or "auto")
        backend_params: Dict[str, Any] = {}
        query_text: Optional[str] = None
        embedding_index_path: Optional[str] = None

        if feat_key == "embeddinggemma2_onnx":
            preset = self.onnx_preset_combo.currentData()
            if isinstance(preset, tuple) and len(preset) >= 3:
                quant, dim, soft_tokens = preset[0], preset[1], preset[2]
                backend_params = {
                    "quantization": str(quant),
                    "matryoshka_dim": int(dim),
                    "image_max_soft_tokens": int(soft_tokens),
                }
            qt = self.query_text_edit.text().strip()
            query_text = qt or None
        elif feat_key == "dinov3":
            model_id = str(self.dino_model_combo.currentData() or "").strip()
            if model_id:
                backend_params["model_name"] = model_id
            backend_params["short_side"] = int(self.dino_short_side_spin.value())
        elif feat_key == "cached_index":
            raw_idx = self.cached_index_edit.text().strip()
            embedding_index_path = raw_idx or None

        min_gap_val = int(self.min_gap_spin.value())
        out_dir_text = self.outFileDirEdit.text().strip() or None
        if out_dir_text is not None and self.video_file:
            # Place extracted images inside a cleanly named subfolder of chosen dir
            # unless the user already pointed to an empty/dedicated leaf folder.
            chosen_out = Path(out_dir_text).expanduser()
            vstem = Path(self.video_file).stem.replace(" ", "_")
            if chosen_out.is_dir() and any(chosen_out.iterdir()):
                out_dir_text = str(chosen_out / f"{vstem}_{self.algo}")

        return FrameSelectionConfig(
            task=self.task,
            algo=self.algo,
            num_frames=int(self.num_frames),
            feature_backend=feat_key,
            backend_params=backend_params,
            extract_images=bool(self.mode_extract_radio.isChecked()),
            output_dir=out_dir_text,
            annotation_dir=self._annotation_dir,
            image_format=str(self.image_format_combo.currentData() or "png"),
            start_seconds=self._start_seconds,
            end_seconds=self.end_seconds,
            max_candidates=int(self.max_candidates_spin.value()),
            min_frame_gap=None if min_gap_val < 0 else min_gap_val,
            exclude_labeled=bool(self.exclude_labeled_cb.isChecked()),
            seed_from_labeled=bool(self.exclude_labeled_cb.isChecked()),
            filter_blur=bool(self.filter_blur_cb.isChecked()),
            include_boundaries=bool(self.include_boundaries_cb.isChecked()),
            query_text=query_text,
            embedding_index_path=embedding_index_path,
            open_output_in_files=bool(self.open_folder_cb.isChecked()),
            seed=int(self.seed_spin.value()),
        )


__all__ = ["ExtractFrameDialog", "FrameSelectionWorker"]
