from __future__ import annotations

from qtpy import QtCore, QtWidgets

from annolid.gui.window_base import format_tool_button_text, utils
from annolid.utils.logger import logger


class PlaybackDrawMixin:
    """Playback controls, draw-mode actions, and tool panel population."""

    def segmentAnything(
        self,
    ):
        self.toggleDrawMode(False, createMode="polygonSAM")
        self.canvas.loadSamPredictor()
        if not getattr(self.canvas, "sam_predictor", None):
            error = getattr(self.canvas, "_sam_last_load_error", None)
            if error == "missing_dependency":
                QtWidgets.QMessageBox.information(
                    self,
                    "Segment Anything not installed",
                    "Install Segment Anything first:\n\n"
                    "  pip install git+https://github.com/facebookresearch/segment-anything.git",
                )
            elif error == "no_pixmap":
                QtWidgets.QMessageBox.information(
                    self,
                    "No image loaded",
                    "Open an image first, then enable Segment Anything.",
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Segment Anything unavailable",
                    "SAM predictor was not initialized.",
                )
            self.toggleDrawMode(True)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        self.tools.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tools.setIconSize(QtCore.QSize(32, 32))
        for item in tool:
            if item is None:
                self.tools.addSeparator()
                continue
            if isinstance(item, QtWidgets.QWidgetAction):
                self.tools.addAction(item)
                continue
            if hasattr(item, "menuAction") and not isinstance(item, QtWidgets.QAction):
                self.tools.addAction(item.menuAction())
                continue
            if isinstance(item, QtWidgets.QAction):
                stacked = format_tool_button_text(item.text())
                try:
                    item.setIconText(stacked)
                except Exception:
                    pass
                self.tools.add_stacked_action(
                    item,
                    stacked,
                    width=58,
                    min_height=68,
                    icon_size=QtCore.QSize(32, 32),
                )
                continue
            self.tools.addAction(item)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def playVideo(self, isPlaying=False):
        self.isPlaying = isPlaying
        timer = getattr(self, "timer", None)
        if self.video_loader is None:
            if timer is not None and timer.isActive():
                timer.stop()
            return

        if self.isPlaying:
            if timer is None:
                timer = QtCore.QTimer(self)
                timer.timeout.connect(self.openNextImg)
                self.timer = timer
            if timer.isActive():
                return
            audio_loader = self._active_audio_loader()
            if audio_loader:
                audio_loader.play(start_frame=self.frame_number)
            if self.fps is not None and self.fps > 0:
                timer.start(int(1000 / self.fps))
            else:
                timer.start(20)
        else:
            if timer is not None and timer.isActive():
                timer.stop()
            audio_loader = self._active_audio_loader()
            if audio_loader:
                audio_loader.stop()

    def startPlaying(self):
        self.playVideo(isPlaying=True)

    def stopPlaying(self):
        self.playVideo(isPlaying=False)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": getattr(self.actions, "createMode", None),
            "rectangle": getattr(self.actions, "createRectangleMode", None),
            "circle": getattr(self.actions, "createCircleMode", None),
            "point": getattr(self.actions, "createPointMode", None),
            "line": getattr(self.actions, "createLineMode", None),
            "linestrip": getattr(self.actions, "createLineStripMode", None),
            "ai_polygon": getattr(self.actions, "createAiPolygonMode", None),
            "ai_mask": getattr(self.actions, "createAiMaskMode", None),
            "grounding_sam": getattr(self.actions, "createGroundingSAMMode", None),
            "polygonSAM": getattr(self, "createPolygonSAMMode", None),
        }

        self.canvas.setEditing(bool(edit))
        if not edit:
            self.canvas.createMode = createMode

        if edit:
            for action in draw_actions.values():
                if action is not None:
                    action.setEnabled(True)
            if getattr(self.actions, "editMode", None) is not None:
                self.actions.editMode.setEnabled(False)
        else:
            for mode, action in draw_actions.items():
                if action is not None:
                    action.setEnabled(createMode != mode)
            if getattr(self.actions, "editMode", None) is not None:
                self.actions.editMode.setEnabled(True)

        self._sync_draw_mode_action_checks(edit=bool(edit), createMode=str(createMode))

    def _setup_drawing_mode_actions(self) -> None:
        if getattr(self, "_draw_mode_actions_initialized", False):
            return
        self._draw_mode_actions_initialized = True

        group = QtWidgets.QActionGroup(self)
        group.setExclusive(True)
        self._draw_mode_action_group = group

        def _safe_disconnect(action: QtWidgets.QAction) -> None:
            for sig in ("triggered", "toggled"):
                try:
                    getattr(action, sig).disconnect()
                except Exception:
                    pass

        def _wire(action: QtWidgets.QAction, *, edit: bool, mode: str) -> None:
            if action is None:
                return
            action.setCheckable(True)
            group.addAction(action)
            _safe_disconnect(action)

            def _on_toggled(checked: bool) -> None:
                if not checked:
                    return
                if edit:
                    self.toggleDrawMode(True, createMode="polygon")
                    return
                if mode == "polygonSAM":
                    self.segmentAnything()
                    return
                if mode in ("ai_polygon", "ai_mask"):
                    try:
                        self.canvas.initializeAiModel(
                            name=self._selectAiModelComboBox.currentText(),
                            _custom_ai_models=self.ai_model_manager.custom_model_names,
                        )
                    except Exception:
                        logger.debug("Failed to initialize AI model.", exc_info=True)
                self.toggleDrawMode(False, createMode=mode)

            action.toggled.connect(_on_toggled)

        _wire(getattr(self.actions, "editMode", None), edit=True, mode="edit")
        _wire(getattr(self.actions, "createMode", None), edit=False, mode="polygon")
        _wire(
            getattr(self.actions, "createRectangleMode", None),
            edit=False,
            mode="rectangle",
        )
        _wire(
            getattr(self.actions, "createCircleMode", None), edit=False, mode="circle"
        )
        _wire(getattr(self.actions, "createLineMode", None), edit=False, mode="line")
        _wire(getattr(self.actions, "createPointMode", None), edit=False, mode="point")
        _wire(
            getattr(self.actions, "createLineStripMode", None),
            edit=False,
            mode="linestrip",
        )
        _wire(
            getattr(self.actions, "createAiPolygonMode", None),
            edit=False,
            mode="ai_polygon",
        )
        _wire(
            getattr(self.actions, "createAiMaskMode", None),
            edit=False,
            mode="ai_mask",
        )
        _wire(
            getattr(self.actions, "createGroundingSAMMode", None),
            edit=False,
            mode="grounding_sam",
        )
        _wire(
            getattr(self, "createPolygonSAMMode", None), edit=False, mode="polygonSAM"
        )

        try:
            if getattr(self.actions, "editMode", None) is not None:
                self.actions.editMode.setChecked(True)
        except Exception:
            pass

    def _sync_draw_mode_action_checks(self, *, edit: bool, createMode: str) -> None:
        actions = {
            "edit": getattr(self.actions, "editMode", None),
            "polygon": getattr(self.actions, "createMode", None),
            "rectangle": getattr(self.actions, "createRectangleMode", None),
            "circle": getattr(self.actions, "createCircleMode", None),
            "line": getattr(self.actions, "createLineMode", None),
            "point": getattr(self.actions, "createPointMode", None),
            "linestrip": getattr(self.actions, "createLineStripMode", None),
            "ai_polygon": getattr(self.actions, "createAiPolygonMode", None),
            "ai_mask": getattr(self.actions, "createAiMaskMode", None),
            "grounding_sam": getattr(self.actions, "createGroundingSAMMode", None),
            "polygonSAM": getattr(self, "createPolygonSAMMode", None),
        }

        target = "edit" if edit else createMode
        for key, action in actions.items():
            if action is None or not action.isCheckable():
                continue
            should = key == target
            if action.isChecked() == should:
                continue
            action.blockSignals(True)
            try:
                action.setChecked(should)
            finally:
                action.blockSignals(False)
