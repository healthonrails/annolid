from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
from PIL import Image
from qtpy import QtCore, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.utils.logger import logger
from annolid.vision.florence_2 import (
    Florence2Predictor,
    Florence2Result,
    create_shapes_from_mask_dict,
    process_nth_frame_from_video,
)

FlorenceResultPayload = Tuple["Florence2Request", Optional[Florence2Result]]


@dataclass(frozen=True)
class Florence2Request:
    """Configuration payload emitted by the Florence2Widget."""

    target: Literal["frame", "video"]
    model_name: str
    text_input: Optional[str]
    segmentation_task: Optional[str]
    include_caption: bool
    caption_task: Optional[str]
    every_n: Optional[int] = None
    description: str = "florence"
    replace_existing: bool = False


class Florence2Widget(QtWidgets.QWidget):
    """
    Toolbar widget that exposes common Florence-2 actions.
    """

    runFrameRequested = QtCore.Signal(Florence2Request)
    runVideoRequested = QtCore.Signal(Florence2Request)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._default_model = "microsoft/Florence-2-large"
        self._build_ui()

    def _build_ui(self) -> None:
        self.model_edit = QtWidgets.QLineEdit(self._default_model)
        self.model_edit.setPlaceholderText("HuggingFace model id")
        self.model_edit.setMinimumWidth(200)

        self.prompt_edit = QtWidgets.QLineEdit()
        self.prompt_edit.setPlaceholderText("e.g., a black mouse")

        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.addItem(
            "Caption + Segmentation",
            {
                "segmentation_task": "<REFERRING_EXPRESSION_SEGMENTATION>",
                "include_caption": True,
                "caption_task": "<MORE_DETAILED_CAPTION>",
            },
        )
        self.task_combo.addItem(
            "Segmentation only",
            {
                "segmentation_task": "<REFERRING_EXPRESSION_SEGMENTATION>",
                "include_caption": False,
                "caption_task": None,
            },
        )
        self.task_combo.addItem(
            "Caption only",
            {
                "segmentation_task": None,
                "include_caption": True,
                "caption_task": "<MORE_DETAILED_CAPTION>",
            },
        )

        self.every_n_spin = QtWidgets.QSpinBox()
        self.every_n_spin.setRange(1, 9999)
        self.every_n_spin.setValue(10)
        self.every_n_spin.setSuffix(" frame(s)")
        self.every_n_spin.setToolTip(
            "Number of frames to skip between Florence-2 runs when processing the entire video."
        )

        self.replace_checkbox = QtWidgets.QCheckBox(
            "Replace existing Florence shapes"
        )
        self.replace_checkbox.setChecked(True)
        self.replace_checkbox.setToolTip(
            "Remove shapes created by Florence-2 before adding new ones."
        )

        self.run_frame_button = QtWidgets.QPushButton("Run on Frame")
        self.run_frame_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.run_frame_button.clicked.connect(
            lambda: self._emit_request(target="frame")
        )

        self.run_video_button = QtWidgets.QPushButton("Process Video")
        self.run_video_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        )
        self.run_video_button.clicked.connect(
            lambda: self._emit_request(target="video")
        )

        form = QtWidgets.QFormLayout()
        form.addRow("Model", self.model_edit)
        form.addRow("Task", self.task_combo)
        form.addRow("Text Prompt", self.prompt_edit)
        form.addRow("Every", self.every_n_spin)
        form.addRow("", self.replace_checkbox)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.run_frame_button)
        button_layout.addWidget(self.run_video_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(button_layout)
        layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(layout)

        self.widget_action = QtWidgets.QWidgetAction(self)
        self.widget_action.setDefaultWidget(self)

    def action(self) -> QtWidgets.QWidgetAction:
        return self.widget_action

    def _collect_task_options(self) -> dict:
        task_data = self.task_combo.currentData()
        if isinstance(task_data, dict):
            return task_data
        return {
            "segmentation_task": "<REFERRING_EXPRESSION_SEGMENTATION>",
            "include_caption": True,
            "caption_task": "<MORE_DETAILED_CAPTION>",
        }

    def _emit_request(self, *, target: Literal["frame", "video"]) -> None:
        model_name = self.model_edit.text().strip() or self._default_model
        prompt = self.prompt_edit.text().strip()
        options = self._collect_task_options()
        segmentation_task = options.get("segmentation_task")
        include_caption = bool(options.get("include_caption", True))
        caption_task = options.get("caption_task")

        if segmentation_task and not prompt:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing prompt",
                "Please provide a text prompt describing the object for Florence-2.",
            )
            return

        request = Florence2Request(
            target=target,
            model_name=model_name,
            text_input=prompt if prompt else None,
            segmentation_task=segmentation_task,
            include_caption=include_caption,
            caption_task=caption_task,
            every_n=self.every_n_spin.value() if target == "video" else None,
            replace_existing=self.replace_checkbox.isChecked(),
        )

        if target == "frame":
            self.runFrameRequested.emit(request)
        else:
            self.runVideoRequested.emit(request)


class Florence2DockWidget(QtWidgets.QDockWidget):
    """
    Dockable container that owns the Florence-2 UI and background execution flow.

    The parent window is expected to provide:
      - _get_pil_image_from_state()
      - video_file, canvas, loadShapes(), setDirty()
      - openCaption(), caption_widget, filename, statusBar()
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window.tr("Florence-2"), window)
        self.window = window
        self.setObjectName("Florence2Dock")

        self._florence_worker: Optional[FlexibleWorker] = None
        self._florence_thread: Optional[QtCore.QThread] = None
        self._florence_predictors: Dict[str, Florence2Predictor] = {}
        self._running_florence_request: Optional[Florence2Request] = None

        self.florence_widget = Florence2Widget(self)
        self.florence_widget.runFrameRequested.connect(
            self._handle_florence_frame_request
        )
        self.florence_widget.runVideoRequested.connect(
            self._handle_florence_video_request
        )

        self.setWidget(self.florence_widget)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.destroyed.connect(lambda *_: self._clear_florence_worker())

    def show_or_raise(self) -> None:
        """Show the dock if hidden and bring it to the front."""
        if self.isHidden():
            self.show()
        self.raise_()

    def _get_florence_predictor(self, model_name: str) -> Florence2Predictor:
        predictor = self._florence_predictors.get(model_name)
        if predictor is None:
            predictor = Florence2Predictor(model_name=model_name)
            self._florence_predictors[model_name] = predictor
        return predictor

    def _handle_florence_frame_request(self, request: Florence2Request) -> None:
        getter = getattr(self.window, "_get_pil_image_from_state", None)
        image = getter() if callable(getter) else None
        if image is None:
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("No frame available"),
                self.window.tr("Load a frame before running Florence-2."),
            )
            return
        self._start_florence_job(request, image=image)

    def _handle_florence_video_request(self, request: Florence2Request) -> None:
        video_path = getattr(self.window, "video_file", None)
        if not video_path:
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("No video loaded"),
                self.window.tr(
                    "Open a video before processing it with Florence-2."),
            )
            return
        self._start_florence_job(request, video_path=video_path)

    def _start_florence_job(
        self,
        request: Florence2Request,
        *,
        image: Optional[Image.Image] = None,
        video_path: Optional[str] = None,
    ) -> None:
        if self._florence_thread and self._florence_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self.window,
                self.window.tr("Florence-2 busy"),
                self.window.tr(
                    "Please wait for the current Florence-2 task to finish."),
            )
            return

        predictor = self._get_florence_predictor(request.model_name)
        self._florence_worker = FlexibleWorker(
            task_function=self._execute_florence_job,
            predictor=predictor,
            request=request,
            image=image.copy() if image is not None else None,
            video_path=video_path,
        )
        self._florence_thread = QtCore.QThread()
        self._florence_worker.moveToThread(self._florence_thread)
        self._florence_worker.start_signal.connect(self._florence_worker.run)
        self._florence_worker.result_signal.connect(
            self._handle_florence_result)
        self._florence_worker.finished_signal.connect(
            self._handle_florence_finished
        )
        self._florence_worker.finished_signal.connect(
            self._florence_thread.quit)
        self._florence_worker.finished_signal.connect(
            self._florence_worker.deleteLater
        )
        self._florence_thread.finished.connect(self._clear_florence_worker)
        self._florence_thread.finished.connect(
            self._florence_thread.deleteLater)

        self._running_florence_request = request
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.window.statusBar().showMessage(self.window.tr("Running Florence-2…"))
        self._florence_thread.start()
        QtCore.QTimer.singleShot(
            0, lambda: self._florence_worker.start_signal.emit()
        )

    @staticmethod
    def _execute_florence_job(
        *,
        predictor: Florence2Predictor,
        request: Florence2Request,
        image: Optional[Image.Image] = None,
        video_path: Optional[str] = None,
    ) -> FlorenceResultPayload:
        if request.target == "frame":
            if image is None:
                raise ValueError(
                    "Florence-2 frame request missing image data.")
            result = predictor.predict(
                image,
                text_input=request.text_input,
                segmentation_task=request.segmentation_task,
                include_caption=request.include_caption,
                caption_task=request.caption_task,
            )
            return request, result

        if video_path is None:
            raise ValueError(
                "Florence-2 video request requires an open video file.")

        process_nth_frame_from_video(
            video_path,
            request.every_n or 1,
            predictor,
            segmentation_task=request.segmentation_task,
            text_input=request.text_input,
            caption_task=request.caption_task,
            description=request.description,
        )
        return request, None

    def _handle_florence_result(self, payload: FlorenceResultPayload) -> None:
        request, result = payload
        if request.target != "frame" or not isinstance(result, Florence2Result):
            return

        shapes = create_shapes_from_mask_dict(
            result.mask_dict, description=request.description
        )

        if request.replace_existing:
            preserved_shapes = [
                shape.copy()
                for shape in self.window.canvas.shapes
                if getattr(shape, "description", None) != request.description
            ]
            shapes_to_load = preserved_shapes + shapes
            self.window.loadShapes(shapes_to_load, replace=True)
        else:
            self.window.loadShapes(shapes, replace=False)

        if shapes:
            self.window.setDirty()
            self.window.statusBar().showMessage(
                self.window.tr("Florence-2 added %d shape(s).") % len(shapes),
                5000,
            )
        elif not result.caption:
            self.window.statusBar().showMessage(
                self.window.tr("Florence-2 did not return any shapes."), 5000
            )

        if result.caption:
            self._apply_florence_caption(result.caption)
            if not shapes:
                self.window.statusBar().showMessage(
                    self.window.tr("Florence-2 caption updated."), 5000
                )

    def _handle_florence_finished(self, outcome: object) -> None:
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass

        request = self._running_florence_request
        self._running_florence_request = None

        if isinstance(outcome, Exception):
            logger.error("Florence-2 job failed: %s", outcome, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self.window,
                self.window.tr("Florence-2 Error"),
                self.window.tr("Failed to run Florence-2:\n%s") % str(outcome),
            )
            self.window.statusBar().showMessage(
                self.window.tr("Florence-2 failed."), 5000
            )
            return

        if request and request.target == "video":
            self.window.statusBar().showMessage(
                self.window.tr("Florence-2 video processing complete."), 5000
            )
        elif request and request.target == "frame":
            return
        else:
            self.window.statusBar().showMessage(
                self.window.tr("Florence-2 finished."), 3000
            )

    def _clear_florence_worker(self) -> None:
        self._florence_thread = None
        self._florence_worker = None

    def _apply_florence_caption(self, caption: str) -> None:
        if not caption:
            return

        self.window.canvas.setCaption(caption)
        if getattr(self.window, "caption_widget", None) is None:
            self.window.openCaption()
        if getattr(self.window, "caption_widget", None) is not None:
            self.window.caption_widget.set_caption(caption)
            if getattr(self.window, "filename", None):
                self.window.caption_widget.set_image_path(self.window.filename)
        self.window.setDirty()
