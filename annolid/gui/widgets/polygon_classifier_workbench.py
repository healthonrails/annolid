"""GUI workbench for polygon-based behavior classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.utils.runs import shared_runs_root


class _PathPicker(QtWidgets.QWidget):
    changed = QtCore.Signal(str)

    def __init__(
        self,
        label: str,
        placeholder: str,
        *,
        mode: str = "file",
        file_filter: str = "All Files (*)",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._file_filter = file_filter

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(label)
        title.setProperty("muted", True)
        top.addWidget(title)
        top.addStretch()
        self.status = QtWidgets.QLabel("-")
        self.status.setProperty("muted", True)
        top.addWidget(self.status)
        layout.addLayout(top)

        row = QtWidgets.QHBoxLayout()
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        self.edit.textChanged.connect(lambda text: self.changed.emit(text.strip()))
        self.edit.setAcceptDrops(True)
        self.edit.installEventFilter(self)
        row.addWidget(self.edit, 1)

        browse = QtWidgets.QPushButton("Browse...")
        browse.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        browse.setToolTip("Browse")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)

    def path(self) -> str:
        return self.edit.text().strip()

    def setPath(self, value: str | Path) -> None:
        self.edit.setText(str(value))

    def setStatus(self, text: str, ok: Optional[bool]) -> None:
        self.status.setText(text)
        self.status.setProperty("good", ok is True)
        self.status.setProperty("bad", ok is False)
        self.status.setProperty("muted", ok is None)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.status.update()

    def _browse(self) -> None:
        if self._mode == "dir":
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        elif self._mode == "save":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Select Output File", "", self._file_filter
            )
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select File", "", self._file_filter
            )
        if path:
            self.setPath(path)

    def eventFilter(self, obj: object, event: QtCore.QEvent) -> bool:
        if obj is self.edit:
            if event.type() == QtCore.QEvent.DragEnter and event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
            if event.type() == QtCore.QEvent.Drop:
                urls = event.mimeData().urls()
                if urls:
                    self.setPath(urls[0].toLocalFile())
                event.acceptProposedAction()
                return True
        return super().eventFilter(obj, event)


def _section_title(title: str, subtitle: str) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 4)
    layout.setSpacing(4)
    heading = QtWidgets.QLabel(title)
    heading.setProperty("role", "title")
    body = QtWidgets.QLabel(subtitle)
    body.setProperty("role", "subtitle")
    body.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(body)
    return widget


def _page_container() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(12)
    return page, layout


def _scroll_tab(page: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setWidget(page)
    return scroll


def _card(parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QFrame:
    frame = QtWidgets.QFrame(parent)
    frame.setProperty("card", True)
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    return frame


def _summary_card(
    title: str, value: str = "-"
) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
    frame = _card()
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(4)
    heading = QtWidgets.QLabel(title)
    heading.setProperty("muted", True)
    label = QtWidgets.QLabel(value)
    label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(heading)
    layout.addWidget(label)
    return frame, label


def _action_row(button: QtWidgets.QPushButton) -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    row.addStretch()
    button.setProperty("primary", True)
    row.addWidget(button)
    return row


class PolygonClassifierWorkbench(QtWidgets.QDialog):
    """Guided dataset, training, and inference UI for polygon classifiers."""

    def __init__(
        self,
        *,
        default_source_dir: str | None = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Polygon Classifier Workbench")
        self.setMinimumSize(900, 680)
        self._threads: list[tuple[QtCore.QThread, FlexibleWorker]] = []
        self._last_train_csv = ""
        self._last_test_csv = ""
        self._last_checkpoint = ""
        self._busy = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        hero = _card(self)
        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(16, 14, 16, 14)
        hero_layout.setSpacing(10)
        hero_layout.addWidget(
            _section_title(
                "Polygon Classifier",
                "Generate polygon-point tables from Annolid shapes, train a temporal classifier, and run frame-level inference.",
            )
        )
        workflow = QtWidgets.QLabel("Polygon CSV  ->  Train  ->  Inference")
        workflow.setProperty("muted", True)
        hero_layout.addWidget(workflow)
        root.addWidget(hero)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)
        self._build_points_tab(default_source_dir)
        self._build_dataset_tab(default_source_dir)
        self._build_training_tab()
        self._build_inference_tab()

        footer = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        footer.addWidget(self.progress, 1)
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setProperty("muted", True)
        footer.addWidget(self.status_label)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)
        root.addLayout(footer)

        self._refresh_validation()

    def _build_points_tab(self, default_source_dir: str | None) -> None:
        tab, layout = _page_container()
        layout.addWidget(
            _section_title(
                "Polygon Points CSV",
                "Merge Annolid predicted polygon JSON or NDJSON shapes with a manual one-hot label CSV. This is the recommended path for predicted-shape workflows.",
            )
        )

        paths = QtWidgets.QGroupBox("Inputs")
        form = QtWidgets.QVBoxLayout(paths)
        self.points_annotation_dir = _PathPicker(
            "Predicted shapes folder",
            "/path/to/video_folder_with_json_or_annotations_ndjson",
            mode="dir",
        )
        self.manual_label_csv = _PathPicker(
            "Manual labels CSV",
            "/path/to/manual_labels.csv",
            file_filter="CSV Files (*.csv);;All Files (*)",
        )
        default_output = (
            shared_runs_root() / "polygon_classifier" / "polygon_points.csv"
        )
        self.points_output_csv = _PathPicker(
            "Output polygon points CSV",
            str(default_output),
            mode="save",
            file_filter="CSV Files (*.csv);;All Files (*)",
        )
        self.points_output_csv.setPath(default_output)
        for picker in (
            self.points_annotation_dir,
            self.manual_label_csv,
            self.points_output_csv,
        ):
            picker.changed.connect(lambda _text: self._refresh_validation())
            form.addWidget(picker)
        if default_source_dir:
            self.points_annotation_dir.setPath(default_source_dir)
        layout.addWidget(paths)

        options = QtWidgets.QGroupBox("Export Options")
        grid = QtWidgets.QGridLayout(options)
        self.points_num_points = QtWidgets.QSpinBox()
        self.points_num_points.setRange(3, 256)
        self.points_num_points.setValue(50)
        self.include_unlabeled = QtWidgets.QCheckBox("Keep frames with no manual label")
        grid.addWidget(QtWidgets.QLabel("Resampled points per polygon"), 0, 0)
        grid.addWidget(self.points_num_points, 0, 1)
        grid.addWidget(self.include_unlabeled, 1, 0, 1, 2)
        layout.addWidget(options)

        self.generate_points_btn = QtWidgets.QPushButton("Generate Polygon Points CSV")
        self.generate_points_btn.clicked.connect(self._generate_points_csv)
        layout.addLayout(_action_row(self.generate_points_btn))

        summary = QtWidgets.QGroupBox("Export Summary")
        summary_layout = QtWidgets.QVBoxLayout(summary)
        self.points_summary = QtWidgets.QLabel("No polygon CSV generated yet.")
        self.points_summary.setWordWrap(True)
        self.points_summary.setProperty("muted", True)
        summary_layout.addWidget(self.points_summary)
        layout.addWidget(summary)
        layout.addStretch()
        self.tabs.addTab(_scroll_tab(tab), "Polygon CSV")

    def _build_dataset_tab(self, default_source_dir: str | None) -> None:
        tab, layout = _page_container()

        layout.addWidget(
            _section_title(
                "Create Dataset",
                "Use train/test folders containing video subfolders of LabelMe JSON frames. JSON files must include behavior flags and intruder/resident polygons.",
            )
        )

        paths = QtWidgets.QGroupBox("Folders")
        form = QtWidgets.QVBoxLayout(paths)
        self.train_folder = _PathPicker(
            "Training folder",
            "/path/to/train",
            mode="dir",
        )
        self.test_folder = _PathPicker("Test folder", "/path/to/test", mode="dir")
        default_output = shared_runs_root() / "polygon_classifier" / "datasets"
        self.dataset_output = _PathPicker(
            "Output folder",
            str(default_output),
            mode="dir",
        )
        self.dataset_output.setPath(default_output)
        for picker in (self.train_folder, self.test_folder, self.dataset_output):
            picker.changed.connect(lambda _text: self._refresh_validation())
            form.addWidget(picker)
        if default_source_dir:
            self.train_folder.setPath(default_source_dir)

        layout.addWidget(paths)

        options = QtWidgets.QGroupBox("Feature Options")
        grid = QtWidgets.QGridLayout(options)
        self.num_points = QtWidgets.QSpinBox()
        self.num_points.setRange(3, 128)
        self.num_points.setValue(10)
        self.normalize_polygons = QtWidgets.QCheckBox("Normalize polygon coordinates")
        grid.addWidget(QtWidgets.QLabel("Points per polygon"), 0, 0)
        grid.addWidget(self.num_points, 0, 1)
        grid.addWidget(self.normalize_polygons, 1, 0, 1, 2)
        layout.addWidget(options)

        self.create_dataset_btn = QtWidgets.QPushButton("Create Feature CSVs")
        self.create_dataset_btn.clicked.connect(self._create_dataset)
        layout.addLayout(_action_row(self.create_dataset_btn))

        summary = QtWidgets.QGroupBox("Dataset Summary")
        summary_layout = QtWidgets.QVBoxLayout(summary)
        self.dataset_summary = QtWidgets.QLabel("No legacy dataset generated yet.")
        self.dataset_summary.setWordWrap(True)
        self.dataset_summary.setProperty("muted", True)
        summary_layout.addWidget(self.dataset_summary)
        layout.addWidget(summary)
        layout.addStretch()
        self.tabs.addTab(_scroll_tab(tab), "Legacy Dataset")

    def _build_training_tab(self) -> None:
        tab, layout = _page_container()
        layout.addWidget(
            _section_title(
                "Train",
                "Train the temporal polygon classifier from feature CSV files. Defaults are conservative for GUI runs.",
            )
        )

        files = QtWidgets.QGroupBox("Dataset CSVs")
        file_layout = QtWidgets.QVBoxLayout(files)
        self.train_csv = _PathPicker(
            "Training CSV",
            "/path/to/train_dataset.csv",
            file_filter="CSV Files (*.csv);;All Files (*)",
        )
        self.test_csv = _PathPicker(
            "Test CSV",
            "/path/to/test_dataset.csv",
            file_filter="CSV Files (*.csv);;All Files (*)",
        )
        self.training_output = _PathPicker(
            "Run output folder",
            str(shared_runs_root() / "polygon_classifier" / "train"),
            mode="dir",
        )
        self.training_output.setPath(
            shared_runs_root() / "polygon_classifier" / "train"
        )
        for picker in (self.train_csv, self.test_csv, self.training_output):
            picker.changed.connect(lambda _text: self._refresh_validation())
            file_layout.addWidget(picker)
        layout.addWidget(files)

        params = QtWidgets.QGroupBox("Training Parameters")
        grid = QtWidgets.QGridLayout(params)
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(30)
        self.batch_size = QtWidgets.QSpinBox()
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(64)
        self.window_size = QtWidgets.QSpinBox()
        self.window_size.setRange(3, 101)
        self.window_size.setSingleStep(2)
        self.window_size.setValue(11)
        self.learning_rate = QtWidgets.QDoubleSpinBox()
        self.learning_rate.setRange(0.000001, 1.0)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setValue(0.004)
        self.hidden_dim = QtWidgets.QSpinBox()
        self.hidden_dim.setRange(16, 2048)
        self.hidden_dim.setValue(128)
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["auto", "cpu", "mps", "cuda"])
        for row, (label, widget) in enumerate(
            (
                ("Epochs", self.epochs),
                ("Batch size", self.batch_size),
                ("Window size", self.window_size),
                ("Learning rate", self.learning_rate),
                ("Hidden dim", self.hidden_dim),
                ("Device", self.device),
            )
        ):
            grid.addWidget(QtWidgets.QLabel(label), row // 2, (row % 2) * 2)
            grid.addWidget(widget, row // 2, (row % 2) * 2 + 1)
        layout.addWidget(params)

        self.train_btn = QtWidgets.QPushButton("Start Training")
        self.train_btn.clicked.connect(self._train)
        layout.addLayout(_action_row(self.train_btn))

        summary = QtWidgets.QGroupBox("Latest Run")
        summary_layout = QtWidgets.QGridLayout(summary)
        run_card, self.train_run_value = _summary_card("Run", "-")
        ckpt_card, self.train_checkpoint_value = _summary_card("Checkpoint", "-")
        metrics_card, self.train_metrics_value = _summary_card("Metrics", "-")
        summary_layout.addWidget(run_card, 0, 0)
        summary_layout.addWidget(ckpt_card, 0, 1)
        summary_layout.addWidget(metrics_card, 0, 2)
        layout.addWidget(summary)
        layout.addStretch()
        self.tabs.addTab(_scroll_tab(tab), "Train")

    def _build_inference_tab(self) -> None:
        tab, layout = _page_container()
        layout.addWidget(
            _section_title(
                "Inference",
                "Run a trained checkpoint on polygon feature CSVs and save per-frame labels plus confidence scores.",
            )
        )

        files = QtWidgets.QGroupBox("Inputs")
        file_layout = QtWidgets.QVBoxLayout(files)
        self.infer_csv = _PathPicker(
            "Feature CSV",
            "/path/to/features.csv",
            file_filter="CSV Files (*.csv);;All Files (*)",
        )
        self.checkpoint = _PathPicker(
            "Checkpoint",
            "/path/to/polygon_frame_classifier_best.pt",
            file_filter="Model Files (*.pt *.pth);;All Files (*)",
        )
        self.infer_output = _PathPicker(
            "Predictions CSV",
            str(shared_runs_root() / "polygon_classifier" / "predictions.csv"),
            mode="save",
            file_filter="CSV Files (*.csv);;All Files (*)",
        )
        self.infer_output.setPath(
            shared_runs_root() / "polygon_classifier" / "predictions.csv"
        )
        for picker in (self.infer_csv, self.checkpoint, self.infer_output):
            picker.changed.connect(lambda _text: self._refresh_validation())
            file_layout.addWidget(picker)
        layout.addWidget(files)

        self.infer_btn = QtWidgets.QPushButton("Run Inference")
        self.infer_btn.clicked.connect(self._infer)
        layout.addLayout(_action_row(self.infer_btn))

        summary = QtWidgets.QGroupBox("Inference Summary")
        summary_layout = QtWidgets.QVBoxLayout(summary)
        self.inference_summary = QtWidgets.QLabel("No inference run yet.")
        self.inference_summary.setProperty("muted", True)
        self.inference_summary.setWordWrap(True)
        summary_layout.addWidget(self.inference_summary)
        layout.addWidget(summary)
        layout.addStretch()
        self.tabs.addTab(_scroll_tab(tab), "Inference")

    def _run_worker(
        self,
        label: str,
        task: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        self.progress.show()
        self._set_busy(True)
        self.status_label.setText(f"{label} running...")
        self._append_log(f"{label} started.")
        worker = FlexibleWorker(task)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._threads.append((thread, worker))

        def finished(result: Any) -> None:
            try:
                if isinstance(result, Exception):
                    self._append_log(f"{label} failed: {result}")
                    self.status_label.setText(f"{label} failed")
                    QtWidgets.QMessageBox.critical(self, label, str(result))
                else:
                    on_success(result)
                    self.status_label.setText(f"{label} completed")
                    self._append_log(f"{label} completed.")
            finally:
                self.progress.hide()
                self._set_busy(False)
                if (thread, worker) in self._threads:
                    self._threads.remove((thread, worker))
                thread.quit()
                worker.deleteLater()

        worker.finished_signal.connect(finished, QtCore.Qt.QueuedConnection)
        thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater, QtCore.Qt.QueuedConnection)
        thread.start()

    def _create_dataset(self) -> None:
        from annolid.behavior.polygon_classifier_workflow import (
            build_polygon_feature_dataset,
        )

        def task():
            return build_polygon_feature_dataset(
                train_folder=self.train_folder.path(),
                test_folder=self.test_folder.path(),
                output_folder=self.dataset_output.path(),
                num_points=int(self.num_points.value()),
                normalize=bool(self.normalize_polygons.isChecked()),
            )

        def success(outcome: Any) -> None:
            self._last_train_csv = outcome.train_csv
            self._last_test_csv = outcome.test_csv
            self.train_csv.setPath(outcome.train_csv)
            self.test_csv.setPath(outcome.test_csv)
            self.infer_csv.setPath(outcome.test_csv)
            self.dataset_summary.setText(
                f"Created {outcome.train_rows} train rows and {outcome.test_rows} test rows. "
                f"Labels: {', '.join(outcome.labels)}"
            )
            self.tabs.setCurrentIndex(2)
            self._refresh_validation()

        self._run_worker("Dataset creation", task, success)

    def _generate_points_csv(self) -> None:
        from annolid.behavior.polygon_classifier_workflow import (
            generate_polygon_points_csv,
        )

        def task():
            return generate_polygon_points_csv(
                annotation_dir=self.points_annotation_dir.path(),
                label_csv=self.manual_label_csv.path(),
                output_csv=self.points_output_csv.path(),
                num_points=int(self.points_num_points.value()),
                include_unlabeled=bool(self.include_unlabeled.isChecked()),
            )

        def success(outcome: Any) -> None:
            self.train_csv.setPath(outcome.output_csv)
            self.test_csv.setPath(outcome.output_csv)
            self.infer_csv.setPath(outcome.output_csv)
            self.points_summary.setText(
                f"Created {outcome.rows} rows in {outcome.output_csv}. "
                f"Labels: {', '.join(outcome.labels) or '-'}; "
                f"polygon columns: {', '.join(outcome.polygon_columns)}; "
                f"skipped frames: {outcome.skipped_frames}."
            )
            self.tabs.setCurrentIndex(3)
            self._refresh_validation()

        self._run_worker("Polygon CSV export", task, success)

    def _train(self) -> None:
        from annolid.behavior.polygon_classifier_workflow import (
            train_polygon_classifier,
        )

        def task():
            return train_polygon_classifier(
                train_csv=self.train_csv.path(),
                test_csv=self.test_csv.path(),
                output_dir=self.training_output.path(),
                num_epochs=int(self.epochs.value()),
                batch_size=int(self.batch_size.value()),
                learning_rate=float(self.learning_rate.value()),
                window_size=int(self.window_size.value()),
                hidden_dim=int(self.hidden_dim.value()),
                device=""
                if self.device.currentText() == "auto"
                else self.device.currentText(),
            )

        def success(outcome: Any) -> None:
            self._last_checkpoint = outcome.checkpoint_path
            self.checkpoint.setPath(outcome.checkpoint_path)
            if self.test_csv.path():
                self.infer_csv.setPath(self.test_csv.path())
            self._append_log(f"Run: {outcome.run_dir}")
            self._append_log(f"Checkpoint: {outcome.checkpoint_path}")
            self._append_log(f"Metrics: {outcome.metrics_path}")
            self.train_run_value.setText(outcome.run_dir)
            self.train_checkpoint_value.setText(outcome.checkpoint_path)
            self.train_metrics_value.setText(outcome.metrics_path)
            self.tabs.setCurrentIndex(3)
            self._refresh_validation()

        self._run_worker("Training", task, success)

    def _infer(self) -> None:
        from annolid.behavior.polygon_classifier_workflow import (
            predict_polygon_classifier_csv,
        )

        def task():
            return predict_polygon_classifier_csv(
                feature_csv=self.infer_csv.path(),
                checkpoint_path=self.checkpoint.path(),
                output_csv=self.infer_output.path(),
                device=""
                if self.device.currentText() == "auto"
                else self.device.currentText(),
            )

        def success(outcome: Any) -> None:
            self._append_log(
                f"Predictions saved: {outcome.output_csv} ({outcome.rows} rows)"
            )
            self.inference_summary.setText(
                f"Predictions saved to {outcome.output_csv}. "
                f"Rows: {outcome.rows}; labels: {', '.join(outcome.labels) or '-'}."
            )
            self._refresh_validation()

        self._run_worker("Inference", task, success)

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        for button in (
            self.generate_points_btn,
            self.create_dataset_btn,
            self.train_btn,
            self.infer_btn,
        ):
            button.setEnabled(not busy)

    def _refresh_validation(self) -> None:
        if getattr(self, "_busy", False):
            self._set_busy(True)
            return

        if hasattr(self, "points_annotation_dir"):
            self._validate_picker(self.points_annotation_dir, want_dir=True)
            self._validate_picker(self.manual_label_csv, want_file=True)
            self._validate_picker(
                self.points_output_csv, want_file=False, allow_missing=True
            )
            if hasattr(self, "generate_points_btn"):
                self.generate_points_btn.setEnabled(
                    Path(self.points_annotation_dir.path()).is_dir()
                    and Path(self.manual_label_csv.path()).is_file()
                    and bool(self.points_output_csv.path())
                )

        if hasattr(self, "train_folder"):
            self._validate_picker(self.train_folder, want_dir=True)
            self._validate_picker(self.test_folder, want_dir=True)
            self._validate_picker(
                self.dataset_output, want_dir=False, allow_missing=True
            )
            if hasattr(self, "create_dataset_btn"):
                self.create_dataset_btn.setEnabled(
                    Path(self.train_folder.path()).is_dir()
                    and Path(self.test_folder.path()).is_dir()
                    and bool(self.dataset_output.path())
                )

        if hasattr(self, "train_csv"):
            self._validate_picker(self.train_csv, want_file=True)
            self._validate_picker(self.test_csv, want_file=True)
            self._validate_picker(
                self.training_output, want_dir=False, allow_missing=True
            )
            if hasattr(self, "train_btn"):
                self.train_btn.setEnabled(
                    Path(self.train_csv.path()).is_file()
                    and Path(self.test_csv.path()).is_file()
                    and bool(self.training_output.path())
                )

        if hasattr(self, "infer_csv"):
            self._validate_picker(self.infer_csv, want_file=True)
            self._validate_picker(self.checkpoint, want_file=True)
            self._validate_picker(
                self.infer_output, want_file=False, allow_missing=True
            )
            if hasattr(self, "infer_btn"):
                self.infer_btn.setEnabled(
                    Path(self.infer_csv.path()).is_file()
                    and Path(self.checkpoint.path()).is_file()
                    and bool(self.infer_output.path())
                )

    def _validate_picker(
        self,
        picker: _PathPicker,
        *,
        want_dir: bool = False,
        want_file: bool = False,
        allow_missing: bool = False,
    ) -> None:
        raw = picker.path()
        if not raw:
            picker.setStatus("-", None)
            return
        path = Path(raw).expanduser()
        if want_dir and path.is_dir():
            picker.setStatus("OK", True)
        elif want_file and path.is_file():
            picker.setStatus("OK", True)
        elif allow_missing:
            picker.setStatus("Will create", None)
        else:
            picker.setStatus("Missing", False)

    def _append_log(self, text: str) -> None:
        if not hasattr(self, "log"):
            self.log = QtWidgets.QTextBrowser()
            self.log.setReadOnly(True)
            self.tabs.addTab(self.log, "Log")
        self.log.append(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        running = [(t, w) for t, w in self._threads if t.isRunning()]
        if running:
            QtWidgets.QMessageBox.information(
                self,
                "Polygon Classifier",
                "A polygon classifier job is still running. Wait for it to finish before closing this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)
