from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Optional

import imgviz
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.data.audios import AudioLoader
from annolid.gui.widgets.caption import CaptionWidget
from annolid.gui.widgets.florence2_widget import Florence2DockWidget
from annolid.gui.widgets.image_editing_widget import ImageEditingDockWidget
from annolid.utils.logger import logger


class MediaWorkflowMixin:
    """Audio, caption, and media dock workflows."""

    def openAudio(self):
        from annolid.gui.widgets.audio import AudioWidget

        if not self.video_file:
            start_dir = getattr(self, "lastOpenDir", None) or str(Path.home())
            audio_widget, audio_filename = AudioWidget.create_from_dialog(
                parent=self,
                start_dir=start_dir,
                caption=self.tr("Annolid - Choose Audio"),
                error_title=self.tr("Audio"),
                error_message=self.tr("Unable to load the selected audio file."),
            )
            if not audio_widget or not audio_filename:
                return

            self.lastOpenDir = str(Path(audio_filename).parent)
            if self.audio_dock:
                self.audio_dock.close()
            self.audio_dock = None
            if self.audio_widget:
                self.audio_widget.close()
            self.audio_widget = None
            self._release_audio_loader()

            self.audio_widget = audio_widget

            self.audio_dock = QtWidgets.QDockWidget(self.tr("Audio"), self)
            self.audio_dock.setObjectName("Audio")
            self.audio_dock.setWidget(self.audio_widget)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.audio_dock)
            self.audio_dock.visibilityChanged.connect(
                self._on_audio_dock_visibility_changed
            )
            self._apply_fixed_dock_sizes()
            return

        if self.audio_dock:
            self.audio_dock.close()
        self.audio_dock = None
        if self.audio_widget:
            self.audio_widget.close()
        self.audio_widget = None

        if self._audio_loader is None:
            self._configure_audio_for_video(self.video_file, self.fps)

        if self._audio_loader is None:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Audio"),
                self.tr("No audio track available for this video."),
            )
            return

        self.audio_widget = AudioWidget(
            self.video_file, audio_loader=self._audio_loader
        )
        self.audio_dock = QtWidgets.QDockWidget(self.tr("Audio"), self)
        self.audio_dock.setObjectName("Audio")
        self.audio_dock.setWidget(self.audio_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.audio_dock)
        self.audio_dock.visibilityChanged.connect(
            self._on_audio_dock_visibility_changed
        )
        self._apply_fixed_dock_sizes()

    def _on_audio_dock_visibility_changed(self, visible: bool) -> None:
        if visible:
            return
        QtCore.QTimer.singleShot(0, self._cleanup_audio_ui)

    def _cleanup_audio_ui(self) -> None:
        """Close the audio dock/widget and release any associated audio loader."""
        if getattr(self, "_cleaning_audio_ui", False):
            return
        self._cleaning_audio_ui = True
        try:
            if self.audio_widget and getattr(self.audio_widget, "audio_loader", None):
                with contextlib.suppress(Exception):
                    self.audio_widget.audio_loader.stop()
            if self.audio_widget:
                with contextlib.suppress(Exception):
                    self.audio_widget.close()
            self.audio_widget = None
            if self.audio_dock:
                with contextlib.suppress(Exception):
                    self.audio_dock.close()
                with contextlib.suppress(Exception):
                    self.audio_dock.deleteLater()
            self.audio_dock = None
            self._release_audio_loader()
        finally:
            self._cleaning_audio_ui = False

    def _configure_audio_for_video(
        self, video_path: Optional[str], fps: Optional[float]
    ) -> None:
        """Prepare audio playback for the active video if an audio track exists."""
        self._release_audio_loader()

        if not video_path:
            return

        effective_fps = fps if fps and fps > 0 else 29.97
        try:
            self._audio_loader = AudioLoader(video_path, fps=effective_fps)
        except Exception as exc:
            logger.debug(
                "Skipping audio playback for %s: %s",
                video_path,
                exc,
            )
            self._audio_loader = None

    def _release_audio_loader(self) -> None:
        """Stop and discard any cached audio loader."""
        if self._audio_loader is None:
            return

        with contextlib.suppress(Exception):
            self._audio_loader.stop()
        self._audio_loader = None

    def _active_audio_loader(self) -> Optional[AudioLoader]:
        """Return the audio loader currently associated with playback."""
        if self.audio_widget and self.audio_widget.audio_loader:
            return self.audio_widget.audio_loader
        return self._audio_loader

    def _update_audio_playhead(self, frame_number: int) -> None:
        """Align cached audio playback position with the given frame number."""
        audio_loader = self._active_audio_loader()
        if not audio_loader:
            return

        set_playhead = getattr(audio_loader, "set_playhead_frame", None)
        if callable(set_playhead):
            try:
                set_playhead(frame_number)
            except Exception as exc:
                logger.debug(
                    "Failed to align audio playhead for frame %s: %s",
                    frame_number,
                    exc,
                )
            return

        frame_to_sample = getattr(audio_loader, "_frame_to_sample_index", None)
        if callable(frame_to_sample) and hasattr(audio_loader, "_playhead_sample"):
            try:
                audio_loader._playhead_sample = frame_to_sample(frame_number)
            except Exception as exc:
                logger.debug(
                    "Failed fallback audio playhead update for frame %s: %s",
                    frame_number,
                    exc,
                )

    def openCaption(self):
        dock = getattr(self, "caption_dock", None)
        widget = getattr(self, "caption_widget", None)
        if dock is None or widget is None:
            self.caption_dock = QtWidgets.QDockWidget(self.tr("Caption"), self)
            self.caption_dock.setObjectName("Caption")
            self.caption_widget = CaptionWidget()
            self.caption_dock.setWidget(self.caption_widget)
            self.caption_dock.installEventFilter(self.caption_widget)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.caption_dock)
            self._apply_fixed_dock_sizes()

            self.caption_widget.charInserted.connect(self.setDirty)
            self.caption_widget.charDeleted.connect(self.setDirty)
            self.caption_widget.captionChanged.connect(self.canvas.setCaption)
            self.caption_widget.imageGenerated.connect(self.display_generated_image)

            dock = self.caption_dock
            widget = self.caption_widget

        if dock is not None:
            dock.show()
            dock.raise_()

        if widget is None:
            return

        widget.set_canvas(self.canvas)
        widget.set_host_window(self)
        widget.set_default_visual_share_mode(attach_canvas=True, attach_window=False)

        if self.video_file and self.fps and self.num_frames:
            widget.set_video_context(
                self.video_file,
                self.fps,
                self.num_frames,
            )
            try:
                widget.set_video_segments(self._current_video_defined_segments)
            except Exception:
                pass

        if self.filename:
            widget.set_image_path(self.filename)
        if self.video_loader is not None:
            widget.set_video_context(
                self.video_file,
                self.fps,
                self.num_frames,
            )
            widget.set_video_segments(
                getattr(self, "_current_video_defined_segments", [])
            )
            if getattr(widget, "behavior_widget", None) is not None:
                try:
                    widget.behavior_widget.set_current_frame(self.frame_number)
                except Exception:
                    pass

    def open_mini_cpm_chat_dock(self) -> None:
        """Open the dedicated AI chat dock preconfigured for MiniCPM-o."""
        manager = getattr(self, "ai_chat_manager", None)
        if manager is None:
            self.openCaption()
            widget = getattr(self, "caption_widget", None)
            if widget is None:
                return
            widget.set_provider_and_model("ollama", "MiniCPM-o-4_5-gguf")
            widget.set_default_visual_share_mode(
                attach_canvas=True, attach_window=False
            )
            widget.prompt_text_edit.setPlaceholderText(
                "Ask MiniCPM-o about this canvas/window snapshot..."
            )
            return

        manager.show_chat_dock(provider="ollama", model="MiniCPM-o-4_5-gguf")
        widget = getattr(manager, "ai_chat_widget", None)
        if widget is not None:
            widget.prompt_text_edit.setPlaceholderText(
                "Ask MiniCPM-o about this canvas/window snapshot..."
            )

    def open_ai_chat_dock(self) -> None:
        """Open the dedicated right-side AI chat dock."""
        manager = getattr(self, "ai_chat_manager", None)
        if manager is not None:
            manager.show_chat_dock()
            widget = getattr(manager, "ai_chat_widget", None)
            if widget is not None:
                widget.prompt_text_edit.setPlaceholderText("Type a message…")
            return
        self.openCaption()

    def _apply_timeline_caption_if_available(
        self,
        frame_number: Optional[int],
        *,
        only_if_empty: bool,
    ) -> bool:
        widget = getattr(self, "caption_widget", None)
        if widget is None:
            return False
        behavior_widget = getattr(widget, "behavior_widget", None)
        if behavior_widget is None:
            return False
        try:
            return bool(
                behavior_widget.apply_timeline_description(
                    frame_number, only_if_empty=only_if_empty
                )
            )
        except Exception:
            return False

    def openFlorence2(self):
        """Open or show the Florence-2 dock widget."""
        dock = getattr(self, "florence_dock", None)
        if dock is None:
            dock = Florence2DockWidget(self)
            dock.destroyed.connect(lambda *_: setattr(self, "florence_dock", None))
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.florence_dock = dock

        if isinstance(dock, Florence2DockWidget):
            dock.show_or_raise()
        else:
            if dock.isHidden():
                dock.show()
            dock.raise_()

    def openImageEditing(self):
        """Open or show the Image Editing dock widget."""
        dock = getattr(self, "image_editing_dock", None)
        if dock is None:
            dock = ImageEditingDockWidget(self)
            dock.destroyed.connect(lambda *_: setattr(self, "image_editing_dock", None))
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.image_editing_dock = dock

        if isinstance(dock, ImageEditingDockWidget):
            dock.show_or_raise()
        else:
            if dock.isHidden():
                dock.show()
            dock.raise_()

    @QtCore.Slot(str)
    def display_generated_image(self, image_path: str) -> None:
        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Image generation failed"),
                self.tr("Could not load generated image:\n%s") % image_path,
            )
            return

        self.canvas.loadPixmap(pixmap, clear_shapes=True)
        try:
            self.imageData = imgviz.io.imread(image_path)
        except Exception:
            self.imageData = None

        self.imagePath = image_path
        self.filename = os.path.basename(image_path)
        self.statusBar().showMessage(self.tr("Generated image loaded: %s") % image_path)
