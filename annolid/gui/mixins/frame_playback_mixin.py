from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from qtpy import QtGui, QtWidgets

from annolid.gui.shape import Shape
from annolid.utils.logger import logger


class FramePlaybackMixin:
    """Frame playback, loading, and DeepLabCut frame parsing helpers."""

    def togglePlay(self):
        if self.isPlaying:
            self.stopPlaying()
            self.update_step_size(1)
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(
                    QtWidgets.QStyle.SP_MediaPlay
                )
            )
            self.playButton.setText("Play")
        else:
            self.startPlaying()
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(
                    QtWidgets.QStyle.SP_MediaStop
                )
            )
            self.playButton.setText("Pause")

    def _on_frame_loaded(self, frame_idx: int, qimage: QtGui.QImage) -> None:
        """Render a frame only if it matches the latest requested index."""
        current = getattr(self, "frame_number", None)
        if current is not None and frame_idx != current:
            logger.debug("Dropping stale frame %s (current=%s)", frame_idx, current)
            return
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

    def set_frame_number(self, frame_number):
        if frame_number >= self.num_frames or frame_number < 0:
            return
        self.frame_number = frame_number
        if getattr(self, "timeline_panel", None) is not None:
            self.timeline_panel.set_current_frame(frame_number)
        self._update_audio_playhead(frame_number)
        if self.isPlaying and not self._suppress_audio_seek:
            audio_loader = self._active_audio_loader()
            if audio_loader:
                audio_loader.play(start_frame=frame_number)
        self.filename = str(self._frame_image_path(frame_number))
        self.current_frame_time_stamp = self.video_loader.get_time_stamp()
        if self.frame_loader is not None:
            self.frame_loader.request(frame_number)
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(self.filename)
        self._update_embedding_query_frame()

    def load_tracking_results(self, cur_video_folder, video_filename):
        self.tracking_data_controller.load_tracking_results(
            Path(cur_video_folder), video_filename
        )

    def is_behavior_active(self, frame_number, behavior):
        """Checks if a behavior is active at a given frame."""
        return self.behavior_controller.is_behavior_active(frame_number, behavior)

    def _load_deeplabcut_results(
        self,
        frame_number: int,
        is_multi_animal: Optional[bool] = None,
    ) -> None:
        """
        Load DeepLabCut tracking results for a given frame and convert them into shape objects.

        This method extracts x, y coordinates for each body part and, if applicable, for each animal.
        It then creates shape objects for visualization.

        Args:
            frame_number (int): The index of the frame to extract tracking data from.
            is_multi_animal (bool, optional): Force multi-animal parsing. Auto-detected when None.

        Notes:
            - Assumes self._df_deeplabcut is a multi-index Pandas DataFrame.
            - Multi-animal mode expects an 'animal' level in the column index.
            - Logs warnings for missing data instead of failing.

        Raises:
            KeyError: If expected columns are missing.
            Exception: For unexpected errors during shape extraction.
        """
        if self._df_deeplabcut is None or self._df_deeplabcut.empty:
            return

        if is_multi_animal is None:
            is_multi_animal = getattr(self, "_df_deeplabcut_multi_animal", False)

        try:
            row = self._df_deeplabcut.loc[frame_number]
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
                    self._df_deeplabcut_animal_ids = (
                        self._df_deeplabcut.columns.get_level_values("animal").unique()
                    )
                else:
                    self._df_deeplabcut_animal_ids = [None]

            if self._df_deeplabcut_bodyparts is None:
                self._df_deeplabcut_bodyparts = (
                    self._df_deeplabcut.columns.get_level_values("bodyparts").unique()
                )

            for animal_id in self._df_deeplabcut_animal_ids:
                for bodypart in self._df_deeplabcut_bodyparts:
                    x_col = (
                        (self._df_deeplabcut_scorer, animal_id, bodypart, "x")
                        if is_multi_animal
                        else (self._df_deeplabcut_scorer, bodypart, "x")
                    )
                    y_col = (
                        (self._df_deeplabcut_scorer, animal_id, bodypart, "y")
                        if is_multi_animal
                        else (self._df_deeplabcut_scorer, bodypart, "y")
                    )

                    x, y = row.get(x_col, None), row.get(y_col, None)
                    if pd.notna(x) and pd.notna(y):
                        shape = Shape(label=bodypart, shape_type="point", visible=True)
                        shape.addPoint((x, y))
                        shapes.append(shape)

        except KeyError as e:
            logger.warning(f"Missing columns in DeepLabCut results: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading DeepLabCut results: {e}")

        self.loadShapes(shapes)
