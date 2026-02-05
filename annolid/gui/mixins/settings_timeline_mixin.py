from __future__ import annotations

from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets


class SettingsTimelineMixin:
    """Shared settings/timeline behaviors extracted from the main app window."""

    def _timeline_user_enabled(self) -> bool:
        """Return True if the user wants the timeline dock visible when a video is open."""
        try:
            return bool(self.settings.value("timeline/show_dock", False, type=bool))
        except Exception:
            return False

    def _setup_timeline_view_toggle(self) -> None:
        """Add a checkable View menu action for the timeline dock (off by default)."""
        if getattr(self, "timeline_dock", None) is None:
            return

        self._toggle_timeline_action = QtWidgets.QAction("Timeline", self)
        self._toggle_timeline_action.setCheckable(True)
        self._toggle_timeline_action.setChecked(self._timeline_user_enabled())
        self._toggle_timeline_action.setEnabled(False)
        self._toggle_timeline_action.toggled.connect(self._on_toggle_timeline_requested)

        view_menu = None
        try:
            view_menu = getattr(getattr(self, "menus", None), "view", None)
        except Exception:
            view_menu = None
        if view_menu is None:
            view_menu = self.menuBar().addMenu(self.tr("&View"))

        view_menu.addAction(self._toggle_timeline_action)

        try:
            self.timeline_dock.visibilityChanged.connect(
                self._on_timeline_dock_visibility_changed
            )
        except Exception:
            pass

    def _apply_timeline_dock_visibility(self, *, video_open: bool) -> None:
        """Apply timeline dock visibility based on video state and user preference."""
        if getattr(self, "timeline_dock", None) is None:
            return
        was_normal = bool(self.windowState() == QtCore.Qt.WindowNoState)
        previous_size = self.size() if was_normal else None
        action = getattr(self, "_toggle_timeline_action", None)
        user_enabled = self._timeline_user_enabled()
        should_show = bool(video_open and user_enabled)

        if action is not None:
            action.blockSignals(True)
            try:
                action.setChecked(user_enabled)
                action.setEnabled(bool(video_open))
            finally:
                action.blockSignals(False)

        blocker = QtCore.QSignalBlocker(self.timeline_dock)
        try:
            if should_show:
                self.timeline_panel.setEnabled(True)
                self.timeline_dock.show()
                self.timeline_dock.raise_()
            else:
                self.timeline_panel.setEnabled(False)
                self.timeline_dock.hide()
        finally:
            del blocker

        if was_normal and previous_size is not None:
            QtCore.QTimer.singleShot(
                0,
                lambda sz=QtCore.QSize(previous_size): self.resize(sz),
            )
        QtCore.QTimer.singleShot(0, self._apply_fixed_dock_sizes)

    def _apply_fixed_dock_sizes(self) -> None:
        """Set sensible initial dock sizes while keeping docks user-resizable."""
        right_docks = [
            getattr(self, "file_dock", None),
            getattr(self, "label_dock", None),
            getattr(self, "shape_dock", None),
            getattr(self, "flag_dock", None),
            getattr(self, "video_dock", None),
            getattr(self, "behavior_log_dock", None),
            getattr(self, "behavior_controls_dock", None),
            getattr(self, "embedding_search_dock", None),
        ]
        bottom_docks = [
            getattr(self, "timeline_dock", None),
            getattr(self, "audio_dock", None),
            getattr(self, "caption_dock", None),
        ]

        def _tight_hint(
            dock: QtWidgets.QDockWidget, *, vertical: bool
        ) -> Optional[int]:
            widget = dock.widget()
            hint = widget.sizeHint() if widget is not None else dock.sizeHint()
            if not hint.isValid():
                return None
            if vertical:
                return int(max(220, min(380, hint.width() + 18)))
            return int(max(130, min(340, hint.height() + 18)))

        width_candidates: List[int] = []
        for dock in right_docks:
            if dock is None:
                continue
            width = _tight_hint(dock, vertical=True)
            if width is not None:
                width_candidates.append(width)
        target_width = max(width_candidates) if width_candidates else None
        if target_width is not None:
            visible_right_docks: List[QtWidgets.QDockWidget] = []
            for dock in right_docks:
                if dock is None:
                    continue
                dock.setMinimumWidth(160)
                dock.setMaximumWidth(16777215)
                if dock.isVisible():
                    visible_right_docks.append(dock)
            if visible_right_docks:
                self.resizeDocks(
                    visible_right_docks,
                    [target_width] * len(visible_right_docks),
                    QtCore.Qt.Horizontal,
                )

        visible_bottom_docks: List[QtWidgets.QDockWidget] = []
        bottom_sizes: List[int] = []
        for dock in bottom_docks:
            if dock is None:
                continue
            height = _tight_hint(dock, vertical=False)
            if height is None:
                continue
            dock.setMinimumHeight(90)
            dock.setMaximumHeight(16777215)
            if dock.isVisible():
                visible_bottom_docks.append(dock)
                bottom_sizes.append(height)
        if visible_bottom_docks and bottom_sizes:
            self.resizeDocks(
                visible_bottom_docks,
                bottom_sizes,
                QtCore.Qt.Vertical,
            )

    def _on_toggle_timeline_requested(self, checked: bool) -> None:
        try:
            self.settings.setValue("timeline/show_dock", bool(checked))
        except Exception:
            pass
        video_open = bool(
            getattr(self, "video_loader", None) is not None
            and getattr(self, "video_file", None)
        )
        self._apply_timeline_dock_visibility(video_open=video_open)
        if checked and video_open and getattr(self, "timeline_panel", None) is not None:
            try:
                self.timeline_panel.refresh_behavior_catalog()
            except Exception:
                pass

    def _on_timeline_dock_visibility_changed(self, visible: bool) -> None:
        """Keep the View menu action in sync when the user closes/floats the dock."""
        action = getattr(self, "_toggle_timeline_action", None)
        if action is None or not action.isEnabled():
            return
        if action.isChecked() == bool(visible):
            return
        action.blockSignals(True)
        try:
            action.setChecked(bool(visible))
        finally:
            action.blockSignals(False)
        try:
            self.settings.setValue("timeline/show_dock", bool(visible))
        except Exception:
            pass

    def _load_agent_run_settings(self) -> Dict[str, Any]:
        settings = self.settings
        return {
            "schema_path": settings.value("agent_run/schema_path", "", type=str),
            "vision_adapter": settings.value(
                "agent_run/vision_adapter", "none", type=str
            ),
            "vision_weights": settings.value("agent_run/vision_weights", "", type=str),
            "vision_pretrained": settings.value(
                "agent_run/vision_pretrained", False, type=bool
            ),
            "vision_score_threshold": settings.value(
                "agent_run/vision_score_threshold", 0.5, type=float
            ),
            "vision_device": settings.value("agent_run/vision_device", "", type=str),
            "llm_adapter": settings.value("agent_run/llm_adapter", "none", type=str),
            "llm_profile": settings.value("agent_run/llm_profile", "", type=str),
            "llm_provider": settings.value("agent_run/llm_provider", "", type=str),
            "llm_model": settings.value("agent_run/llm_model", "", type=str),
            "llm_persist": settings.value("agent_run/llm_persist", False, type=bool),
            "include_llm_summary": settings.value(
                "agent_run/include_llm_summary", False, type=bool
            ),
            "llm_summary_prompt": settings.value(
                "agent_run/llm_summary_prompt",
                "Summarize the behaviors defined in this behavior spec.",
                type=str,
            ),
            "stride": settings.value("agent_run/stride", 1, type=int),
            "max_frames": settings.value("agent_run/max_frames", -1, type=int),
            "streaming": settings.value("agent_run/streaming", False, type=bool),
            "anchor_rerun": settings.value("agent_run/anchor_rerun", False, type=bool),
        }

    def _save_agent_run_settings(self, values: Dict[str, Any]) -> None:
        settings = self.settings
        settings.setValue("agent_run/schema_path", values.get("schema_path") or "")
        settings.setValue(
            "agent_run/vision_adapter", values.get("vision_adapter", "none")
        )
        settings.setValue(
            "agent_run/vision_weights", values.get("vision_weights") or ""
        )
        settings.setValue(
            "agent_run/vision_pretrained",
            bool(values.get("vision_pretrained", False)),
        )
        settings.setValue(
            "agent_run/vision_score_threshold",
            float(values.get("vision_score_threshold", 0.5)),
        )
        settings.setValue("agent_run/vision_device", values.get("vision_device") or "")
        settings.setValue("agent_run/llm_adapter", values.get("llm_adapter", "none"))
        settings.setValue("agent_run/llm_profile", values.get("llm_profile") or "")
        settings.setValue("agent_run/llm_provider", values.get("llm_provider") or "")
        settings.setValue("agent_run/llm_model", values.get("llm_model") or "")
        settings.setValue(
            "agent_run/llm_persist", bool(values.get("llm_persist", False))
        )
        settings.setValue(
            "agent_run/include_llm_summary",
            bool(values.get("include_llm_summary", False)),
        )
        settings.setValue(
            "agent_run/llm_summary_prompt",
            values.get("llm_summary_prompt")
            or "Summarize the behaviors defined in this behavior spec.",
        )
        settings.setValue("agent_run/stride", int(values.get("stride", 1)))
        max_frames = values.get("max_frames")
        settings.setValue(
            "agent_run/max_frames", -1 if max_frames is None else int(max_frames)
        )
        settings.setValue("agent_run/streaming", bool(values.get("streaming", False)))
        settings.setValue(
            "agent_run/anchor_rerun", bool(values.get("anchor_rerun", False))
        )

    def toggle_agent_mode(self, checked: bool = False) -> None:
        enabled = bool(checked)
        self._agent_mode_enabled = enabled
        self.settings.setValue("ui/agent_mode", enabled)
        self._apply_agent_mode(enabled)
        self.statusBar().showMessage(
            self.tr("Agent mode enabled")
            if enabled
            else self.tr("Agent mode disabled"),
            3000,
        )

    def toggle_embedding_search(self, checked: bool = False) -> None:
        enabled = bool(checked)
        self._show_embedding_search = enabled
        self.settings.setValue("ui/show_embedding_search", enabled)
        dock = getattr(self, "embedding_search_dock", None)
        if dock is None:
            return
        dock.setVisible(bool(self._agent_mode_enabled) and bool(enabled))
        if enabled:
            try:
                dock.raise_()
            except Exception:
                pass

    def _apply_agent_mode(self, enabled: bool) -> None:
        behavior_log = getattr(self, "behavior_log_dock", None)
        if behavior_log is not None:
            behavior_log.setVisible(bool(enabled))

        behavior_controls = getattr(self, "behavior_controls_dock", None)
        if behavior_controls is not None:
            behavior_controls.setVisible(bool(enabled))

        embedding_search = getattr(self, "embedding_search_dock", None)
        if embedding_search is not None:
            embedding_search.setVisible(
                bool(enabled) and bool(getattr(self, "_show_embedding_search", False))
            )

        try:
            action = self.menu_controller._actions.get("run_agent")
            if action is not None:
                action.setEnabled(bool(enabled))
        except Exception:
            pass

    def toggle_pose_edges_display(self, checked: bool = False) -> None:
        """Toggle skeleton edge overlay for pose keypoints."""
        self._show_pose_edges = bool(checked)
        try:
            self.settings.setValue("pose/show_edges", self._show_pose_edges)
        except Exception:
            pass
        try:
            self.canvas.setShowPoseEdges(self._show_pose_edges)
        except Exception:
            pass

    def toggle_pose_bbox_display(self, checked: bool = False) -> None:
        """Toggle pose bounding box visibility on the canvas."""
        self._show_pose_bboxes = bool(checked)
        try:
            self.settings.setValue("pose/show_bbox", self._show_pose_bboxes)
        except Exception:
            pass
        try:
            self.canvas.setShowPoseBBoxes(self._show_pose_bboxes)
        except Exception:
            pass

    def toggle_pose_bbox_saving(self, checked: bool = False) -> None:
        """Toggle saving pose bounding boxes for YOLO pose inference."""
        self._save_pose_bbox = bool(checked)
        try:
            self.settings.setValue("pose/save_bbox", self._save_pose_bbox)
        except Exception:
            pass
