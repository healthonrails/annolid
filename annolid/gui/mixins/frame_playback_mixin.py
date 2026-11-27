from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy import QtCore, QtGui

from annolid.gui.shape import Shape
from annolid.utils.logger import logger


class FramePlaybackMixin:
    """Frame playback, loading, and DeepLabCut frame parsing helpers."""

    _PLAYBACK_EMBEDDING_SYNC_STRIDE = 5

    def togglePlay(self):
        if self.isPlaying:
            self.stopPlaying()
            self.update_step_size(1)
            if hasattr(self, "_set_play_button_state"):
                self._set_play_button_state(False)
        else:
            self.startPlaying()
            if hasattr(self, "_set_play_button_state"):
                self._set_play_button_state(True)

    @staticmethod
    def _control_matches_int_value(control, target: int) -> bool:
        value_attr = getattr(control, "value", None)
        if not callable(value_attr):
            return False
        try:
            return int(value_attr()) == int(target)
        except Exception:
            return False

    def _sync_frame_jump_input(self, frame_number: int) -> None:
        frame_jump = getattr(self, "frameJumpInput", None)
        if frame_jump is None:
            return
        target = int(frame_number)
        if self._control_matches_int_value(frame_jump, target):
            return
        try:
            with QtCore.QSignalBlocker(frame_jump):
                frame_jump.setValue(target)
        except Exception:
            try:
                previous = frame_jump.blockSignals(True)
                frame_jump.setValue(target)
                frame_jump.blockSignals(previous)
            except Exception:
                pass

    def _sync_seekbar_value(self, frame_number: int) -> None:
        seekbar = getattr(self, "seekbar", None)
        if seekbar is None:
            return
        target = int(frame_number)
        if self._control_matches_int_value(seekbar, target):
            return
        try:
            with QtCore.QSignalBlocker(seekbar):
                seekbar.setValue(target)
        except Exception:
            try:
                previous = seekbar.blockSignals(True)
                seekbar.setValue(target)
                seekbar.blockSignals(previous)
            except Exception:
                try:
                    seekbar.setValue(target)
                except Exception:
                    pass

    def _resolve_frame_timestamp_ms(self, frame_number: int) -> Optional[float]:
        frame_loader = getattr(self, "frame_loader", None)
        get_cached_ts = getattr(frame_loader, "get_cached_timestamp", None)
        if callable(get_cached_ts):
            try:
                cached = get_cached_ts(int(frame_number))
                if cached is not None:
                    return cached
            except Exception:
                pass
        fps = getattr(self, "fps", None)
        try:
            fps_val = float(fps) if fps is not None else 0.0
        except (TypeError, ValueError):
            fps_val = 0.0
        if fps_val > 0.0:
            return (float(frame_number) / fps_val) * 1000.0
        video_loader = getattr(self, "video_loader", None)
        if video_loader is not None and hasattr(video_loader, "get_time_stamp"):
            try:
                return video_loader.get_time_stamp()
            except Exception:
                return None
        return None

    def _on_frame_load_error(self, frame_idx: int, message: str) -> None:
        """Surface frame decode errors without blocking playback with a modal dialog."""
        current = getattr(self, "frame_number", None)
        if current is not None:
            try:
                if int(frame_idx) != int(current):
                    return
            except Exception:
                pass
        status_bar_fn = getattr(self, "statusBar", None)
        if callable(status_bar_fn):
            try:
                status_bar = status_bar_fn()
                if status_bar is not None:
                    status_bar.showMessage(
                        f"Frame {int(frame_idx)} decode warning: {message}",
                        4000,
                    )
            except Exception:
                pass

    def _on_frame_loaded(self, frame_idx: int, qimage: QtGui.QImage) -> None:
        """Render frame updates with a bounded-lag tolerance during playback.

        In active playback/tracking runs, decoder throughput can lag behind
        timer-driven frame requests. Strict matching can drop emitted frames and
        make per-frame predictions appear missing. We keep strict matching when
        paused/seeking, and allow any non-future decoded frame while playing.
        """
        current = getattr(self, "frame_number", None)
        if current is not None:
            try:
                current_idx = int(current)
                loaded_idx = int(frame_idx)
            except Exception:
                current_idx = current
                loaded_idx = frame_idx
            if loaded_idx != current_idx:
                is_playing = bool(getattr(self, "isPlaying", False))
                if not is_playing:
                    logger.debug(
                        "Dropping stale frame %s while paused (current=%s)",
                        loaded_idx,
                        current_idx,
                    )
                    return
                if int(loaded_idx) > int(current_idx):
                    logger.debug(
                        "Dropping future frame %s while playing (current=%s)",
                        loaded_idx,
                        current_idx,
                    )
                    return
        if qimage is None or (hasattr(qimage, "isNull") and qimage.isNull()):
            logger.debug("Skipping null QImage for frame %s.", frame_idx)
            return
        resolved_ts = self._resolve_frame_timestamp_ms(int(frame_idx))
        if resolved_ts is not None:
            self.current_frame_time_stamp = resolved_ts
        frame_path = self._frame_image_path(frame_idx)
        self.image_to_canvas(qimage, frame_path, frame_idx)

    def _frame_image_path(self, frame_number: int) -> Path:
        if self.video_results_folder:
            return (
                self.video_results_folder
                / f"{str(self.video_results_folder.name)}_{frame_number:09}.png"
            )
        if getattr(self, "filename", None):
            try:
                return Path(self.filename)
            except Exception:
                pass
        return Path()

    def _confirm_unsaved_before_frame_change(self, target_frame: int) -> bool:
        if not bool(getattr(self, "dirty", False)):
            return True
        current_frame = getattr(self, "frame_number", None)
        if current_frame is not None and int(current_frame) == int(target_frame):
            return True
        auto_save_check = getattr(self, "_is_auto_save_enabled", None)
        run_auto_save = getattr(self, "_run_auto_save", None)
        if callable(auto_save_check) and auto_save_check() and callable(run_auto_save):
            run_auto_save()
            if not bool(getattr(self, "dirty", False)):
                return True
        may_continue = getattr(self, "mayContinue", None)
        if callable(may_continue) and not may_continue():
            if current_frame is not None:
                self._sync_frame_jump_input(int(current_frame))
                self._sync_seekbar_value(int(current_frame))
            return False
        return True

    def set_frame_number(self, frame_number):
        total_frames = getattr(self, "num_frames", None)
        if total_frames is None:
            return
        frame_number = int(frame_number)
        if frame_number >= int(total_frames) or frame_number < 0:
            return
        if not self._confirm_unsaved_before_frame_change(frame_number):
            return
        self.frame_number = frame_number
        self._sync_frame_jump_input(frame_number)
        self._sync_seekbar_value(frame_number)
        if getattr(self, "timeline_panel", None) is not None:
            self.timeline_panel.set_current_frame(frame_number)
        if hasattr(self, "_refresh_ethogram_active_badges"):
            try:
                self._refresh_ethogram_active_badges(frame_number)
            except Exception:
                pass
        if hasattr(self, "_sync_boris_media_and_streams"):
            try:
                self._sync_boris_media_and_streams(frame_number)
            except Exception:
                pass
        self._update_audio_playhead(frame_number)
        if self.isPlaying and not self._suppress_audio_seek:
            if hasattr(self, "_play_synchronized_video_audio"):
                self._play_synchronized_video_audio(frame_number)
            else:
                audio_loader = self._active_audio_loader()
                if audio_loader:
                    audio_loader.play(start_frame=frame_number)
        self.filename = str(self._frame_image_path(frame_number))
        self.current_frame_time_stamp = self._resolve_frame_timestamp_ms(frame_number)
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(self.filename)
        should_refresh_embedding = True
        if bool(getattr(self, "isPlaying", False)):
            stride = int(getattr(self, "_PLAYBACK_EMBEDDING_SYNC_STRIDE", 5) or 5)
            stride = max(1, stride)
            should_refresh_embedding = (int(frame_number) % stride) == 0
        if should_refresh_embedding:
            self._update_embedding_query_frame()
        queue_widget = getattr(self, "frame_selection_queue_widget", None)
        if queue_widget is not None and not bool(getattr(self, "isPlaying", False)):
            try:
                queue_widget.set_current_frame(frame_number)
            except Exception:
                pass
        if self.frame_loader is not None:
            self.frame_loader.request(frame_number)

    def load_tracking_results(
        self, cur_video_folder, video_filename, *, non_blocking=True
    ):
        controller = getattr(self, "tracking_data_controller", None)
        if controller is None:
            return
        cur_video_folder_path = Path(cur_video_folder)
        if non_blocking and hasattr(controller, "load_tracking_results_async"):
            controller.load_tracking_results_async(
                cur_video_folder_path, video_filename
            )
            return
        controller.load_tracking_results(cur_video_folder_path, video_filename)

    def is_behavior_active(self, frame_number, behavior):
        """Checks if a behavior is active at a given frame."""
        return self.behavior_controller.is_behavior_active(frame_number, behavior)

    def _load_deeplabcut_results(
        self,
        frame_number: int,
        is_multi_animal: Optional[bool] = None,
    ) -> None:
        """Load DeepLabCut / Lightning Pose tracking results for a frame into ``Shape`` objects."""
        if self._df_deeplabcut is None or self._df_deeplabcut.empty:
            return

        import math
        import pandas as pd

        if is_multi_animal is None:
            is_multi_animal = bool(
                getattr(self, "_df_deeplabcut_multi_animal", False)
                or self._df_deeplabcut.columns.nlevels == 4
            )

        try:
            row = self._df_deeplabcut.loc[frame_number]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        except KeyError:
            if 0 <= frame_number < len(self._df_deeplabcut.index):
                row = self._df_deeplabcut.iloc[frame_number]
            else:
                logger.debug(
                    "Frame %s is outside the DeepLabCut table bounds (%s rows).",
                    frame_number,
                    len(self._df_deeplabcut.index),
                )
                return
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Unexpected error accessing DeepLabCut frame %s: %s",
                frame_number,
                exc,
            )
            return

        shapes = []
        try:
            if self._df_deeplabcut_scorer is None:
                self._df_deeplabcut_scorer = (
                    self._df_deeplabcut.columns.get_level_values(0)[0]
                )

            if self._df_deeplabcut_animal_ids is None:
                if is_multi_animal:
                    level_names = list(self._df_deeplabcut.columns.names or [])
                    if "individuals" in level_names:
                        ind_level = "individuals"
                    elif "animal" in level_names:
                        ind_level = "animal"
                    else:
                        ind_level = 1
                    self._df_deeplabcut_animal_ids = list(
                        self._df_deeplabcut.columns.get_level_values(ind_level).unique()
                    )
                else:
                    self._df_deeplabcut_animal_ids = [None]

            if self._df_deeplabcut_bodyparts is None:
                level_names = list(self._df_deeplabcut.columns.names or [])
                bp_level = (
                    "bodyparts"
                    if "bodyparts" in level_names
                    else (2 if is_multi_animal else 1)
                )
                self._df_deeplabcut_bodyparts = list(
                    self._df_deeplabcut.columns.get_level_values(bp_level).unique()
                )

            for animal_idx, animal_id in enumerate(self._df_deeplabcut_animal_ids):
                for bodypart in self._df_deeplabcut_bodyparts:
                    if is_multi_animal:
                        x_col = (self._df_deeplabcut_scorer, animal_id, bodypart, "x")
                        y_col = (self._df_deeplabcut_scorer, animal_id, bodypart, "y")
                        l_col = (
                            self._df_deeplabcut_scorer,
                            animal_id,
                            bodypart,
                            "likelihood",
                        )
                    else:
                        x_col = (self._df_deeplabcut_scorer, bodypart, "x")
                        y_col = (self._df_deeplabcut_scorer, bodypart, "y")
                        l_col = (self._df_deeplabcut_scorer, bodypart, "likelihood")

                    x = row.get(x_col, None)
                    y = row.get(y_col, None)
                    if pd.notna(x) and pd.notna(y):
                        xf, yf = float(x), float(y)
                        if not (math.isfinite(xf) and math.isfinite(yf)):
                            continue
                        label = (
                            f"{animal_id}_{bodypart}"
                            if is_multi_animal and animal_id is not None
                            else str(bodypart)
                        )
                        shape = Shape(
                            label=label,
                            group_id=animal_idx if is_multi_animal else None,
                            shape_type="point",
                            visible=True,
                        )
                        shape.addPoint(QtCore.QPointF(xf, yf))
                        other_data = getattr(shape, "other_data", None)
                        if not isinstance(other_data, dict):
                            other_data = {}
                            shape.other_data = other_data
                        other_data["display_label"] = str(bodypart)
                        if is_multi_animal and animal_id is not None:
                            other_data["instance_label"] = str(animal_id)
                        like = row.get(l_col, None)
                        if pd.notna(like):
                            try:
                                lf = float(like)
                                if math.isfinite(lf):
                                    other_data["score"] = lf
                            except (TypeError, ValueError):
                                pass
                        shapes.append(shape)

        except KeyError as e:
            logger.warning(f"Missing columns in DeepLabCut results: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading DeepLabCut results: {e}")

        self.loadShapes(shapes)
