from __future__ import annotations

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from annolid.annotation.timestamps import convert_frame_number_to_time
from annolid.utils.logger import logger


class NavigationWorkflowMixin:
    """Frame and file navigation helpers."""

    def jump_to_frame(self):
        """Jump to the specified frame number."""
        try:
            input_frame_number = int(self.seekbar.input_value.text())
            if 0 <= input_frame_number < self.num_frames:
                self.set_frame_number(input_frame_number)
            else:
                logger.info(f"Frame number {input_frame_number} is out of range.")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Frame Number",
                    f"{input_frame_number} is out of range.",
                )
        except ValueError:
            logger.info(
                f"Invalid input: {self.seekbar.input_value.text()} is not a valid frame number."
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                f"'{self.seekbar.input_value.text()}' is not a valid frame number.",
            )
        except Exception as e:
            logger.error(f"Error while jumping to frame: {e}")

    def tooltip_callable(self, val):
        if (
            self.behavior_controller.highlighted_mark is not None
            and self.frame_number == val
        ):
            return f"Frame:{val},Time:{convert_frame_number_to_time(val)}"
        return ""

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config["keep_prev"] = True

        self._set_active_view("canvas")

        if not self.mayContinue():
            return

        if self.video_loader is not None:
            if self.frame_number >= self.num_frames:
                self.frame_number = 0
                self.set_frame_number(self.frame_number)

            if self.frame_number < self.num_frames:
                if self.step_size + self.frame_number <= self.num_frames:
                    self.frame_number += self.step_size
                else:
                    self.frame_number += 1
            else:
                self.frame_number = self.num_frames
                self.togglePlay()
            self._suppress_audio_seek = True
            try:
                self.set_frame_number(self.frame_number)
                self.seekbar.setValue(self.frame_number)
            finally:
                self._suppress_audio_seek = False
            self.uniqLabelList.itemSelectionChanged.connect(
                self.handle_uniq_label_list_selection_change
            )

        else:
            visible_files = self._checked_file_paths()
            if len(visible_files) <= 0:
                self._config["keep_prev"] = keep_prev
                self._update_frame_display_and_emit_update()
                return

            filename = None
            if self.filename is None or self.filename not in visible_files:
                filename = visible_files[0]
            else:
                currIndex = visible_files.index(self.filename)
                if currIndex + 1 < len(visible_files):
                    filename = visible_files[currIndex + 1]
                else:
                    filename = visible_files[-1]
            self.filename = filename

            if self.filename and load:
                self.loadFile(self.filename)
                self._set_current_file_item(self.filename)
                if self.caption_widget is not None:
                    self.caption_widget.set_image_path(self.filename)

        self._config["keep_prev"] = keep_prev
        self._update_frame_display_and_emit_update()

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config["keep_prev"] = True

        self._set_active_view("canvas")

        if not self.mayContinue():
            return

        if self.video_loader is not None:
            if self.frame_number > 1:
                if self.frame_number - self.step_size >= 1:
                    self.frame_number -= self.step_size
                else:
                    self.frame_number -= 1
            else:
                self.frame_number = 0
            self.set_frame_number(self.frame_number)
            self.seekbar.setValue(self.frame_number)

        else:
            visible_files = self._checked_file_paths()
            if len(visible_files) <= 0:
                return
            if self.filename is None or self.filename not in visible_files:
                filename = visible_files[0]
                self.loadFile(filename)
                self._set_current_file_item(filename)
                if self.caption_widget is not None:
                    self.caption_widget.set_image_path(self.filename)
                self._config["keep_prev"] = keep_prev
                self._update_frame_display_and_emit_update()
                return

            currIndex = visible_files.index(self.filename)
            if currIndex - 1 >= 0:
                filename = visible_files[currIndex - 1]
                if filename:
                    self.loadFile(filename)
                    self._set_current_file_item(filename)
                    if self.caption_widget is not None:
                        self.caption_widget.set_image_path(self.filename)

        self._config["keep_prev"] = keep_prev
        self._update_frame_display_and_emit_update()

    def _emit_live_frame_update(self):
        if (
            self.filename
            and self.frame_number is not None
            and hasattr(self, "_time_stamp")
        ):
            self.live_annolid_frame_updated.emit(
                self.frame_number, self._time_stamp or ""
            )
