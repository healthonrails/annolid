from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Optional

from qtpy import QtCore, QtWidgets

from annolid.features.dino_models import set_dino_model_on_runtime
from annolid.gui.cursor_utils import set_widget_busy_cursor
from annolid.gui.widgets.advanced_parameters_dialog import AdvancedParametersDialog
from annolid.utils.logger import logger


class CoreInteractionMixin:
    """Core canvas interaction and advanced-parameter workflows."""

    def paintCanvas(self):
        if getattr(self, "_active_image_view", "canvas") == "tiled":
            tiled = getattr(self, "large_image_view", None)
            if tiled is None or getattr(self, "large_image_backend", None) is None:
                return
            if self.zoomMode == self.FIT_WINDOW:
                tiled.fit_to_window()
            elif self.zoomMode == self.FIT_WIDTH:
                tiled.fit_to_width()
            else:
                tiled.set_zoom_percent(self.zoomWidget.value())
            return
        if self.image.isNull():
            return
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.updateGeometry()
        if getattr(self, "_viewer_stack", None) is not None:
            self._viewer_stack.updateGeometry()
            if (
                not getattr(self, "isPlaying", False)
                or getattr(self, "video_loader", None) is not None
            ):
                self._viewer_stack.adjustSize()
        else:
            if (
                not getattr(self, "isPlaying", False)
                or getattr(self, "video_loader", None) is not None
            ):
                self.canvas.adjustSize()
        self.canvas.update()

    @QtCore.Slot()
    def _open_segment_editor_dialog(self):
        if not self.video_file or self.fps is None or self.num_frames is None:
            QtWidgets.QMessageBox.information(
                self, "No Video Loaded", "Please load a video first."
            )
            return

        initial_segment_dicts = [
            s.to_dict() for s in self._current_video_defined_segments
        ]

        from annolid.gui.widgets.segment_editor import SegmentEditorDialog

        dialog = SegmentEditorDialog(
            active_video_path=Path(self.video_file),
            active_video_fps=self.fps,
            active_video_total_frames=self.num_frames,
            current_annolid_frame=self.frame_number,
            initial_segments_data=initial_segment_dicts,
            annolid_config=self.config,
            parent=self,
        )

        dialog.tracking_initiated.connect(self.tracking_controller.start_tracking)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self._current_video_defined_segments = dialog.get_defined_segments()
            logger.info(
                f"Segment Editor OK. {len(self._current_video_defined_segments)} segments stored."
            )
            self._save_segments_for_active_video()
            if self.caption_widget is not None:
                self.caption_widget.set_video_segments(
                    self._current_video_defined_segments
                )
        else:
            logger.info("Segment Editor Cancelled/Closed.")

        dialog.deleteLater()

    def is_tracking_busy(self) -> bool:
        return self.tracking_controller.is_tracking_busy()

    def _show_prompt_status_message(self, message: str, timeout_ms: int = 5000) -> None:
        try:
            status_fn = getattr(self, "status", None)
            if callable(status_fn):
                status_fn(message, timeout_ms)
                return
            bar_fn = getattr(self, "statusBar", None)
            if callable(bar_fn):
                bar = bar_fn()
                if bar is not None:
                    bar.showMessage(message, timeout_ms)
        except Exception:
            pass

    def _sync_ai_prompt_completions(self) -> None:
        widget = getattr(self, "aiRectangle", None)
        setter = getattr(widget, "set_completion_suggestions", None)
        uniq_list = getattr(self, "uniqLabelList", None)
        if not callable(setter) or uniq_list is None:
            return
        try:
            labels = []
            for row in range(uniq_list.count()):
                item = uniq_list.item(row)
                if item is None:
                    continue
                val = item.data(QtCore.Qt.UserRole) or item.text()
                if val:
                    labels.append(str(val))
            setter(labels)
        except Exception:
            pass

    def _set_ai_rectangle_running(self, running: bool) -> None:
        widget = getattr(self, "aiRectangle", None)
        setter = getattr(widget, "set_running", None)
        if callable(setter):
            try:
                setter(bool(running))
            except Exception:
                pass

    def _read_ai_rectangle_options(self) -> tuple[bool, float | None, str | None, bool]:
        widget = getattr(self, "aiRectangle", None)
        use_countgd = False
        box_threshold = None
        output_mode = None
        replace_existing = False
        try:
            if widget is not None:
                if hasattr(widget, "use_countgd") and callable(widget.use_countgd):
                    use_countgd = bool(widget.use_countgd())
                elif hasattr(widget, "_useCountGDCheckbox"):
                    use_countgd = bool(widget._useCountGDCheckbox.isChecked())
                if hasattr(widget, "box_threshold") and callable(widget.box_threshold):
                    box_threshold = float(widget.box_threshold())
                if hasattr(widget, "output_mode") and callable(widget.output_mode):
                    output_mode = str(widget.output_mode())
                if hasattr(widget, "replace_existing") and callable(
                    widget.replace_existing
                ):
                    replace_existing = bool(widget.replace_existing())
        except Exception:
            pass
        return use_countgd, box_threshold, output_mode, replace_existing

    def _prepare_text_prompt_invocation(self) -> tuple[str | None, bool]:
        """Validate prompt input and execute immediate `flags:` commands.

        Returns `(raw_prompt, should_segment)`.
        """
        self.toggleDrawMode(False, createMode="grounding_sam")
        raw_prompt = ""
        try:
            if hasattr(self, "aiRectangle") and hasattr(
                self.aiRectangle, "_aiRectanglePrompt"
            ):
                raw_prompt = str(
                    self.aiRectangle._aiRectanglePrompt.text() or ""
                ).strip()
        except Exception:
            raw_prompt = ""

        if len(raw_prompt) < 1:
            logger.info("Empty text prompt; skipping segmentation.")
            self._show_prompt_status_message(
                self.tr(
                    "Enter a text prompt (e.g., mouse, pup=baby) before running segmentation."
                )
            )
            return None, False

        self._sync_ai_prompt_completions()
        recorder = getattr(getattr(self, "aiRectangle", None), "record_prompt", None)
        if callable(recorder):
            try:
                recorder(raw_prompt)
            except Exception:
                pass

        if raw_prompt.lower().startswith("flags:"):
            flags_body = raw_prompt[len("flags:") :]
            flags = {
                k.strip(): False for k in flags_body.split(",") if len(k.strip()) > 0
            }
            if len(flags) > 0:
                self.flags_controller.apply_prompt_flags(flags)
                self._show_prompt_status_message(
                    self.tr("Applied %d frame flag(s).") % len(flags)
                )
            else:
                self.flags_controller.clear_flags()
                self._show_prompt_status_message(self.tr("Cleared frame flags."))
            return raw_prompt, False

        return raw_prompt, True

    def _report_prompt_completion(
        self,
        raw_prompt: str,
        *,
        status_kind: str,
        count: int,
        elapsed: float,
        box_threshold: float | None = None,
        detail: str | None = None,
    ) -> None:
        canvas = getattr(self, "canvas", None)
        still_running = bool(
            callable(getattr(canvas, "is_text_prompt_running", None))
            and canvas.is_text_prompt_running()
        )
        self._set_ai_rectangle_running(still_running)

        if status_kind == "completed":
            if count > 0:
                self._show_prompt_status_message(
                    self.tr("Segmented %d shape(s) for '%s' in %.2fs.")
                    % (count, raw_prompt, elapsed)
                )
            elif detail:
                self._show_prompt_status_message(
                    self.tr("Text-prompt segmentation unavailable: %s") % detail,
                    timeout_ms=8000,
                )
            else:
                conf_hint = (
                    f" (conf ≥ {box_threshold:.2f})"
                    if box_threshold is not None
                    else ""
                )
                self._show_prompt_status_message(
                    self.tr(
                        "No objects segmented for '%s'%s. Try lowering confidence or enabling CountGD."
                    )
                    % (raw_prompt, conf_hint)
                )
        elif status_kind == "cancelled":
            if detail == "frame_changed":
                self._show_prompt_status_message(
                    self.tr(
                        "Cancelled text-prompt segmentation because the active frame changed."
                    )
                )
            elif detail not in {"superseded", "closing", "file_closed"}:
                self._show_prompt_status_message(
                    self.tr("Cancelled text-prompt segmentation for '%s'.") % raw_prompt
                )
        elif status_kind == "stale":
            self._show_prompt_status_message(
                self.tr(
                    "Discarded stale text-prompt result for '%s' after frame change."
                )
                % raw_prompt
            )
        elif status_kind == "error":
            err_text = str(detail or "Unknown error")
            self._show_prompt_status_message(
                self.tr("Text-prompt segmentation failed: %s") % err_text,
                timeout_ms=8000,
            )

    def _grounding_sam(self):
        """Run text-prompt segmentation synchronously (programmatic & bot contract)."""
        raw_prompt, should_segment = self._prepare_text_prompt_invocation()
        if not should_segment or raw_prompt is None:
            return 0

        use_countgd, box_threshold, output_mode, replace_existing = (
            self._read_ai_rectangle_options()
        )
        canvas = getattr(self, "canvas", None)
        canvas_widget = canvas if isinstance(canvas, QtWidgets.QWidget) else None

        started = time.perf_counter()
        set_widget_busy_cursor(canvas_widget, True)
        try:
            extra_kwargs = {}
            if box_threshold is not None:
                extra_kwargs["box_threshold"] = box_threshold
            if output_mode is not None:
                extra_kwargs["output_mode"] = output_mode
            if replace_existing:
                extra_kwargs["replace_existing"] = True
            try:
                created = self.canvas.predictAiRectangle(
                    raw_prompt,
                    use_countgd=use_countgd,
                    **extra_kwargs,
                )
            except TypeError:
                # Preserve compatibility with test doubles / custom Canvas subclasses
                # that only accept legacy positional/use_countgd parameters.
                created = self.canvas.predictAiRectangle(
                    raw_prompt,
                    use_countgd=use_countgd,
                )
        finally:
            set_widget_busy_cursor(canvas_widget, False)

        elapsed = time.perf_counter() - started
        count = int(created) if isinstance(created, int) else 0
        self._report_prompt_completion(
            raw_prompt,
            status_kind="completed",
            count=count,
            elapsed=elapsed,
            box_threshold=box_threshold,
        )
        return count

    def _grounding_sam_interactive(self) -> bool:
        """Run toolbar text-prompt segmentation asynchronously off the Qt UI thread."""
        raw_prompt, should_segment = self._prepare_text_prompt_invocation()
        if not should_segment or raw_prompt is None:
            return False

        canvas = getattr(self, "canvas", None)
        async_fn = getattr(canvas, "start_async_text_prompt", None)
        if not callable(async_fn):
            self._grounding_sam()
            return True

        pixmap = getattr(canvas, "pixmap", None)
        if pixmap is None or pixmap.isNull():
            self._show_prompt_status_message(
                self.tr(
                    "Load an image or video frame before running text-prompt segmentation."
                )
            )
            return False

        use_countgd, box_threshold, output_mode, replace_existing = (
            self._read_ai_rectangle_options()
        )
        self._set_ai_rectangle_running(True)
        self._show_prompt_status_message(
            self.tr("Segmenting '%s'… (click Stop to cancel)") % raw_prompt,
            timeout_ms=0,
        )

        launched = bool(
            async_fn(
                raw_prompt,
                use_countgd=use_countgd,
                box_threshold=box_threshold,
                output_mode=output_mode,
                replace_existing=replace_existing,
                on_finished=lambda kind, cnt, elapsed, detail: (
                    self._report_prompt_completion(
                        raw_prompt,
                        status_kind=kind,
                        count=cnt,
                        elapsed=elapsed,
                        box_threshold=box_threshold,
                        detail=detail,
                    )
                ),
            )
        )
        if not launched:
            self._set_ai_rectangle_running(False)
        return launched

    def _start_exemplar_countgd_async(
        self, prompt: str, rectangle_shapes: list
    ) -> bool:
        """Run CountGD exemplar-box segmentation asynchronously from Canvas.finalise."""
        canvas = getattr(self, "canvas", None)
        async_fn = getattr(canvas, "start_async_text_prompt", None)
        if not callable(async_fn):
            return False

        _use_countgd, box_threshold, output_mode, _replace_existing = (
            self._read_ai_rectangle_options()
        )
        label_text = str(prompt or "object").strip() or "object"
        self._set_ai_rectangle_running(True)
        self._show_prompt_status_message(
            self.tr("Counting & segmenting '%s'… (click Stop to cancel)") % label_text,
            timeout_ms=0,
        )
        launched = bool(
            async_fn(
                label_text,
                rectangle_shapes=rectangle_shapes,
                is_polygon_output=(output_mode != "mask"),
                use_countgd=True,
                box_threshold=box_threshold,
                output_mode=output_mode,
                replace_existing=False,
                on_finished=lambda kind, cnt, elapsed, detail: (
                    self._report_prompt_completion(
                        label_text,
                        status_kind=kind,
                        count=cnt,
                        elapsed=elapsed,
                        box_threshold=box_threshold,
                        detail=detail,
                    )
                ),
            )
        )
        if not launched:
            self._set_ai_rectangle_running(False)
        return launched

    def _cancel_grounding_sam(self) -> bool:
        """Cancel an active background text-prompt segmentation run."""
        canvas = getattr(self, "canvas", None)
        cancel_fn = getattr(canvas, "cancel_async_text_prompt", None)
        cancelled = False
        if callable(cancel_fn):
            cancelled = bool(cancel_fn(wait_ms=0, reason="user_cancelled"))
        self._set_ai_rectangle_running(False)
        if cancelled:
            self._show_prompt_status_message(
                self.tr("Cancelled text-prompt segmentation.")
            )
        return cancelled

    @staticmethod
    def _is_reserved_toolbar_command(prompt: str) -> bool:
        stripped = (prompt or "").strip()
        if not stripped:
            return False
        if stripped.lower().startswith("flags:"):
            return True
        return bool(re.match(r"^every\s+\d+(?:\s+|$)", stripped, flags=re.IGNORECASE))

    def _current_text_prompt(self) -> Optional[str]:
        prompt = None
        try:
            if hasattr(self, "aiRectangle") and hasattr(
                self.aiRectangle, "_aiRectanglePrompt"
            ):
                widget = self.aiRectangle._aiRectanglePrompt
                if widget:
                    prompt = widget.text().strip() or None
        except Exception:
            prompt = None
        if prompt and self._is_reserved_toolbar_command(prompt):
            return None
        return prompt

    def update_step_size(self, value):
        self.step_size = value
        self.stepSizeWidget.set_value(self.step_size)

    def set_advanced_params(self):
        sam3_defaults = self.sam3_manager.dialog_defaults(self._config)
        advanced_params_dialog = AdvancedParametersDialog(
            self,
            tracker_config=self.tracker_runtime_config,
            sam3_runtime=sam3_defaults,
        )
        advanced_params_dialog.load_window_state(
            self,
            optical_flow_manager=getattr(self, "optical_flow_manager", None),
        )
        of_manager = getattr(self, "optical_flow_manager", None)
        if advanced_params_dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.epsilon_for_polygon = advanced_params_dialog.get_epsilon_value()
        self.automatic_pause_enabled = (
            advanced_params_dialog.is_automatic_pause_enabled()
        )
        self.t_max_value = advanced_params_dialog.get_t_max_value()
        self.use_cpu_only = advanced_params_dialog.is_cpu_only_enabled()
        self.save_video_with_color_mask = (
            advanced_params_dialog.is_save_video_with_color_mask_enabled()
        )
        self.auto_recovery_missing_instances = (
            advanced_params_dialog.is_auto_recovery_missing_instances_enabled()
        )
        if of_manager is not None:
            of_manager.set_compute_optical_flow(
                advanced_params_dialog.is_compute_optiocal_flow_enabled()
            )
            of_manager.set_backend(advanced_params_dialog.get_optical_flow_backend())
        self._follow_prediction_progress = (
            advanced_params_dialog.is_follow_prediction_progress_enabled()
        )
        videomt_settings = advanced_params_dialog.get_videomt_settings()
        self.videomt_mask_threshold = float(
            videomt_settings.get("videomt_mask_threshold", 0.5)
        )
        self.videomt_logit_threshold = float(
            videomt_settings.get("videomt_logit_threshold", -2.0)
        )
        self.videomt_seed_iou_threshold = float(
            videomt_settings.get("videomt_seed_iou_threshold", 0.01)
        )
        self.videomt_window = int(videomt_settings.get("videomt_window", 8))
        self.videomt_input_height = int(videomt_settings.get("videomt_input_height", 0))
        self.videomt_input_width = int(videomt_settings.get("videomt_input_width", 0))

        tracker_settings = advanced_params_dialog.get_tracker_settings()
        for key, value in tracker_settings.items():
            setattr(self.tracker_runtime_config, key, value)
        selected_dino_model = set_dino_model_on_runtime(
            self.tracker_runtime_config,
            tracker_settings,
        )
        if selected_dino_model:
            self.patch_similarity_model = selected_dino_model
            try:
                self.settings.setValue("patch_similarity/model", selected_dino_model)
            except Exception:
                logger.debug(
                    "Unable to persist DINO feature model selection.",
                    exc_info=True,
                )
            logger.info(
                "Set DINO feature model for tracking to: %s",
                selected_dino_model,
            )

        self.sam3_manager.apply_dialog_results(advanced_params_dialog, self._config)

        of_manager = getattr(self, "optical_flow_manager", None)
        logger.info(
            "Computing optical flow is %s .",
            getattr(of_manager, "compute_optical_flow", True),
        )
        logger.info("Set epsilon for polygon to : %s", self.epsilon_for_polygon)
