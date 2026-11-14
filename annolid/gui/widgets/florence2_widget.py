from __future__ import annotations

from dataclasses import dataclass
import inspect
import threading
from typing import Any, Dict, Literal, Optional, Tuple
from PIL import Image
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.cursor_utils import set_widget_busy_cursor
from annolid.gui.workers import FlexibleWorker
from annolid.utils.cancellation import (
    InferenceCancelledError,
    ObservableStopEvent,
    is_stop_requested,
)
from annolid.utils.logger import logger
from annolid.vision.florence_2 import (
    Florence2Predictor,
    Florence2Result,
    create_shapes_from_mask_dict,
    process_nth_frame_from_video,
)

FlorenceResultPayload = Tuple["Florence2Request", Optional[Florence2Result]]

_ANCHORED_FLORENCE_WORKERS: set[tuple[QtCore.QThread, QtCore.QObject]] = set()


def _callable_accepts_stop_event(func: Any) -> bool:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "stop_event" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


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
    stopRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._default_model = "microsoft/Florence-2-large"
        self._is_running = False
        self._widget_action: Optional[QtWidgets.QWidgetAction] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.model_edit = QtWidgets.QLineEdit(self._default_model)
        self.model_edit.setPlaceholderText("HuggingFace model id")
        self.model_edit.setMinimumWidth(200)

        self.prompt_edit = QtWidgets.QLineEdit()
        self.prompt_edit.setPlaceholderText(
            "e.g., black mouse -> mouse_1, white mouse -> mouse_2"
        )
        self.prompt_edit.setClearButtonEnabled(True)
        self.prompt_edit.returnPressed.connect(
            lambda: self._emit_request(target="frame")
        )

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

        self.replace_checkbox = QtWidgets.QCheckBox("Replace existing Florence shapes")
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

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserStop)
        )
        self.stop_button.setToolTip("Cancel the active Florence-2 frame or video run.")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stopRequested.emit)

        form = QtWidgets.QFormLayout()
        form.addRow("Model", self.model_edit)
        form.addRow("Task", self.task_combo)
        form.addRow("Text Prompt", self.prompt_edit)
        form.addRow("Every", self.every_n_spin)
        form.addRow("", self.replace_checkbox)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.run_frame_button)
        button_layout.addWidget(self.run_video_button)
        button_layout.addWidget(self.stop_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(button_layout)
        layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(layout)

    @property
    def widget_action(self) -> QtWidgets.QWidgetAction:
        return self.action()

    def action(self) -> QtWidgets.QWidgetAction:
        if self._widget_action is None:
            # Parent QWidgetAction to parentWidget() rather than self:
            # QWidgetAction.setDefaultWidget(self) takes C++ ownership of self,
            # so parenting the action to self would cause a double-delete on teardown.
            self._widget_action = QtWidgets.QWidgetAction(self.parentWidget())
            self._widget_action.setDefaultWidget(self)
        return self._widget_action

    def is_running(self) -> bool:
        return bool(self._is_running)

    def set_running(self, running: bool) -> None:
        active = bool(running)
        self._is_running = active
        self.run_frame_button.setEnabled(not active)
        self.run_video_button.setEnabled(not active)
        self.stop_button.setEnabled(active)
        self.stop_button.setStyleSheet(
            "background-color: #c62828; color: white;" if active else ""
        )

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
        if self._is_running:
            return
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
        self._florence_stop_event: Optional[ObservableStopEvent] = None
        self._stop_holder: list[Optional[ObservableStopEvent]] = [None]
        self._florence_predictors: Dict[str, Florence2Predictor] = {}
        self._running_florence_request: Optional[Florence2Request] = None
        self._running_frame_token: Optional[tuple] = None
        self._active_run_id: int = 0
        self._dock_closing: bool = False

        self.florence_widget = Florence2Widget(self)
        self.florence_widget.runFrameRequested.connect(
            self._handle_florence_frame_request
        )
        self.florence_widget.runVideoRequested.connect(
            self._handle_florence_video_request
        )
        self.florence_widget.stopRequested.connect(
            lambda: self.cancel_florence_job(wait_ms=0, reason="user_cancelled")
        )

        self.setWidget(self.florence_widget)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        predictors_ref = self._florence_predictors
        stop_holder = self._stop_holder

        def _on_qt_destroyed(*_args: object) -> None:
            ev = stop_holder[0]
            stop_holder[0] = None
            if ev is not None:
                try:
                    ev.set()
                except Exception:
                    pass
            items = list(predictors_ref.values())
            predictors_ref.clear()
            for pred in items:
                close_fn = getattr(pred, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass

        self.destroyed.connect(_on_qt_destroyed)

    def show_or_raise(self) -> None:
        """Show the dock if hidden and bring it to the front."""
        self._dock_closing = False
        if self.isHidden():
            self.show()
        self.raise_()

    def is_job_running(self) -> bool:
        thread = self._florence_thread
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except RuntimeError:
            self._florence_thread = None
            self._florence_worker = None
            return False

    def _capture_window_frame_token(self) -> tuple:
        filename = getattr(self.window, "filename", None)
        frame_number = getattr(self.window, "frame_number", None)
        canvas = getattr(self.window, "canvas", None)
        canvas_token = None
        if canvas is not None:
            capture_fn = getattr(canvas, "_capture_prompt_frame_token", None)
            if callable(capture_fn):
                try:
                    canvas_token = capture_fn()
                except Exception:
                    canvas_token = None
            else:
                pixmap = getattr(canvas, "pixmap", None)
                if pixmap is not None and hasattr(pixmap, "cacheKey"):
                    try:
                        canvas_token = int(pixmap.cacheKey())
                    except Exception:
                        canvas_token = None
        return (filename, frame_number, canvas_token)

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
                self.window.tr("Open a video before processing it with Florence-2."),
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
        if self._dock_closing:
            return
        if self.is_job_running():
            QtWidgets.QMessageBox.information(
                self.window,
                self.window.tr("Florence-2 busy"),
                self.window.tr(
                    "Please wait for the current Florence-2 task to finish."
                ),
            )
            return

        predictor = self._get_florence_predictor(request.model_name)
        stop_event = ObservableStopEvent()
        self._active_run_id += 1
        run_id = self._active_run_id

        worker = FlexibleWorker(
            task_function=self._execute_florence_job,
            predictor=predictor,
            request=request,
            image=image.copy() if image is not None else None,
            video_path=video_path,
            stop_event=stop_event,
        )
        thread = QtCore.QThread()
        thread.setStackSize(8 * 1024 * 1024)
        anchor_pair = (thread, worker)
        _ANCHORED_FLORENCE_WORKERS.add(anchor_pair)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result_signal.connect(
            lambda payload, rid=run_id: self._handle_florence_result(
                payload, run_id=rid
            )
        )
        worker.finished_signal.connect(
            lambda outcome, rid=run_id: self._handle_florence_finished(
                outcome, run_id=rid
            )
        )
        worker.finished_signal.connect(thread.quit)
        worker.finished_signal.connect(worker.deleteLater)
        thread.finished.connect(
            lambda rid=run_id: self._clear_florence_worker(run_id=rid)
        )
        thread.finished.connect(
            lambda pair=anchor_pair: _ANCHORED_FLORENCE_WORKERS.discard(pair)
        )
        thread.finished.connect(thread.deleteLater)

        self._florence_worker = worker
        self._florence_thread = thread
        self._florence_stop_event = stop_event
        self._stop_holder[0] = stop_event
        self._running_florence_request = request
        self._running_frame_token = (
            self._capture_window_frame_token() if request.target == "frame" else None
        )
        try:
            self.florence_widget.set_running(True)
            set_widget_busy_cursor(self.florence_widget, True)
            self.window.statusBar().showMessage(self.window.tr("Running Florence-2…"))
            thread.start()
        except Exception:
            _ANCHORED_FLORENCE_WORKERS.discard(anchor_pair)
            self.florence_widget.set_running(False)
            self._running_florence_request = None
            self._running_frame_token = None
            self._florence_stop_event = None
            self._stop_holder[0] = None
            self._clear_florence_worker(run_id=run_id)
            try:
                set_widget_busy_cursor(self.florence_widget, False)
            except Exception:
                pass
            raise

    @staticmethod
    def _execute_florence_job(
        *,
        predictor: Florence2Predictor,
        request: Florence2Request,
        image: Optional[Image.Image] = None,
        video_path: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> FlorenceResultPayload:
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("Florence-2 job cancelled.")
        if request.target == "frame":
            if image is None:
                raise ValueError("Florence-2 frame request missing image data.")
            predict_kwargs: Dict[str, Any] = {
                "text_input": request.text_input,
                "segmentation_task": request.segmentation_task,
                "include_caption": request.include_caption,
                "caption_task": request.caption_task,
            }
            if stop_event is not None and _callable_accepts_stop_event(
                getattr(predictor, "predict", None)
            ):
                predict_kwargs["stop_event"] = stop_event
            result = predictor.predict(image, **predict_kwargs)
            if is_stop_requested(stop_event):
                raise InferenceCancelledError("Florence-2 job cancelled.")
            return request, result

        if video_path is None:
            raise ValueError("Florence-2 video request requires an open video file.")

        video_kwargs: Dict[str, Any] = {
            "segmentation_task": request.segmentation_task,
            "text_input": request.text_input,
            "caption_task": request.caption_task,
            "description": request.description,
        }
        if stop_event is not None and _callable_accepts_stop_event(
            process_nth_frame_from_video
        ):
            video_kwargs["stop_event"] = stop_event
        process_nth_frame_from_video(
            video_path,
            request.every_n or 1,
            predictor,
            **video_kwargs,
        )
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("Florence-2 job cancelled.")
        return request, None

    def _handle_florence_result(
        self,
        payload: FlorenceResultPayload,
        *,
        run_id: Optional[int] = None,
    ) -> None:
        if self._dock_closing:
            return
        if run_id is not None and run_id != self._active_run_id:
            return
        if is_stop_requested(self._florence_stop_event):
            return
        request, result = payload
        if request.target != "frame" or not isinstance(result, Florence2Result):
            return

        expected_token = self._running_frame_token
        current_token = self._capture_window_frame_token()
        if expected_token is not None and current_token != expected_token:
            logger.info(
                "Discarding stale Florence-2 frame result (expected=%s, current=%s).",
                expected_token,
                current_token,
            )
            self.window.statusBar().showMessage(
                self.window.tr(
                    "Discarded Florence-2 result because the active frame changed."
                ),
                5000,
            )
            return

        shapes = create_shapes_from_mask_dict(
            result.mask_dict, description=request.description
        )

        replaced_existing = False
        if request.replace_existing:
            preserved_shapes = [
                shape.copy()
                for shape in self.window.canvas.shapes
                if getattr(shape, "description", None) != request.description
            ]
            replaced_existing = len(preserved_shapes) != len(self.window.canvas.shapes)
            shapes_to_load = preserved_shapes + shapes
            if shapes or replaced_existing:
                self.window.loadShapes(shapes_to_load, replace=True)
        elif shapes:
            self.window.loadShapes(shapes, replace=False)

        if shapes or replaced_existing:
            self.window.setDirty()
        if shapes:
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

    def _handle_florence_finished(
        self,
        outcome: object,
        *,
        run_id: Optional[int] = None,
    ) -> None:
        if run_id is not None and run_id != self._active_run_id:
            return
        try:
            set_widget_busy_cursor(self.florence_widget, False)
        except Exception:
            pass
        self.florence_widget.set_running(False)

        request = self._running_florence_request
        stop_event = self._florence_stop_event
        self._running_florence_request = None
        self._running_frame_token = None
        self._florence_stop_event = None
        self._stop_holder[0] = None

        if self._dock_closing:
            return

        if isinstance(outcome, InferenceCancelledError) or is_stop_requested(
            stop_event
        ):
            self.window.statusBar().showMessage(
                self.window.tr("Florence-2 cancelled."), 4000
            )
            return

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

    def cancel_florence_job(
        self,
        *,
        wait_ms: int = 0,
        reason: str = "cancelled",
    ) -> bool:
        """Cooperatively cancel an in-flight Florence-2 frame or video job."""
        stop_event = self._florence_stop_event
        worker = self._florence_worker
        thread = self._florence_thread
        was_active = bool(
            stop_event is not None or worker is not None or self.is_job_running()
        )

        self._active_run_id += 1
        self._running_florence_request = None
        self._running_frame_token = None
        self._florence_stop_event = None
        self._stop_holder[0] = None

        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass
        if worker is not None:
            try:
                worker.request_stop()
            except Exception:
                pass
        if thread is not None:
            try:
                if thread.isRunning():
                    thread.requestInterruption()
                    thread.quit()
                    if wait_ms > 0 and QtCore.QThread.currentThread() is not thread:
                        thread.wait(int(wait_ms))
            except RuntimeError:
                pass

        try:
            set_widget_busy_cursor(self.florence_widget, False)
        except Exception:
            pass
        try:
            self.florence_widget.set_running(False)
        except RuntimeError:
            pass

        if was_active and not self._dock_closing and reason == "user_cancelled":
            try:
                self.window.statusBar().showMessage(
                    self.window.tr("Florence-2 cancelled."), 4000
                )
            except Exception:
                pass
        return was_active

    def close_predictors(self) -> None:
        """Release cached Florence-2 predictor instances safely."""
        predictors = list(self._florence_predictors.values())
        self._florence_predictors.clear()
        if not predictors:
            return

        def _release_all() -> None:
            for predictor in predictors:
                close_fn = getattr(predictor, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        logger.debug(
                            "Failed closing Florence-2 predictor.", exc_info=True
                        )

        thread = self._florence_thread
        try:
            if thread is not None and thread.isRunning():
                thread.finished.connect(_release_all)
                return
        except RuntimeError:
            pass
        _release_all()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._dock_closing = True
        self.cancel_florence_job(wait_ms=250, reason="dock_closed")
        super().closeEvent(event)

    def _clear_florence_worker(self, *, run_id: Optional[int] = None) -> None:
        if run_id is not None and run_id != self._active_run_id:
            return
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
