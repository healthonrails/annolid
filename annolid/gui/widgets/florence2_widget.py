from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from qtpy import QtCore, QtWidgets


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
