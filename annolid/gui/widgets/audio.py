import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from qtpy import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from qtpy.QtCore import Qt

from annolid.utils.audio_playback import play_audio_buffer

from annolid.data.audios import AudioLoader


class AudioWidget(QtWidgets.QWidget):
    """A QWidget that displays audio waveform and spectrum using Matplotlib."""

    AUDIO_EXTENSIONS = (
        "*.mp3",
        "*.wav",
        "*.flac",
        "*.ogg",
        "*.m4a",
        "*.aac",
        "*.opus",
        "*.wma",
        "*.aiff",
        "*.aif",
    )

    CLICK_DRAG_THRESHOLD_SEC = 0.05

    def __init__(
        self,
        audio_path: str,
        audio_loader: Optional[AudioLoader] = None,
        parent=None,
    ):
        """
        Constructor for the AudioWidget class.

        Args:
            audio_path (str): Path to the audio file.
            audio_loader (AudioLoader | None): Optional preloaded audio loader.
            parent (QWidget): Optional parent widget.
        """
        super().__init__(parent)
        self.audio_path = audio_path
        self.figure_waveform = Figure()
        self.figure_spectrum = Figure()
        self.canvas_waveform = FigureCanvas(self.figure_waveform)
        self.canvas_spectrum = FigureCanvas(self.figure_spectrum)

        self.controls_widget = QtWidgets.QWidget(self)
        self._build_controls(self.controls_widget)

        self._build_segments_panel()

        plots_widget = QtWidgets.QWidget(self)
        plots_layout = QtWidgets.QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.addWidget(self.canvas_waveform)
        plots_layout.addWidget(self.canvas_spectrum)

        splitter = QtWidgets.QSplitter(Qt.Horizontal, self)
        splitter.addWidget(plots_widget)
        splitter.addWidget(self.segments_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.controls_widget)
        self.layout.addWidget(splitter)
        self.setLayout(self.layout)

        self.audio_loader: Optional[AudioLoader] = audio_loader
        self._duration_sec = 0.0
        self._selection: Optional[Tuple[float, float]] = None
        self._updating_selection_controls = False
        self._span_selector: Optional[SpanSelector] = None
        self._suppress_span_callbacks_flag = False
        self._mouse_down_xdata: Optional[float] = None

        if self.audio_loader is None and audio_path:
            try:
                self.audio_loader = AudioLoader(audio_path)
            except Exception:
                self.audio_loader = None
        for canvas in (self.canvas_waveform, self.canvas_spectrum):
            canvas.mpl_connect("button_press_event", self._on_canvas_press)
            canvas.mpl_connect("button_release_event", self._on_canvas_release)

        self._refresh_plots()

    @classmethod
    def file_dialog_filter(cls) -> str:
        formats = " ".join(cls.AUDIO_EXTENSIONS)
        return f"Audio Files ({formats});;All Files (*.*)"

    @classmethod
    def select_audio_file(
        cls,
        parent=None,
        start_dir: Optional[str] = None,
        caption: str = "Choose Audio",
    ) -> Optional[str]:
        start_dir = start_dir or str(Path.home())
        selection = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            caption,
            start_dir,
            cls.file_dialog_filter(),
        )
        if isinstance(selection, tuple):
            selection = selection[0]
        selection = str(selection)
        return selection or None

    @classmethod
    def create_from_dialog(
        cls,
        parent=None,
        start_dir: Optional[str] = None,
        caption: str = "Choose Audio",
        *,
        show_error: bool = True,
        error_title: str = "Audio",
        error_message: str = "Unable to load the selected audio file.",
    ):
        audio_path = cls.select_audio_file(
            parent=parent, start_dir=start_dir, caption=caption
        )
        if not audio_path:
            return None, None

        widget = cls(audio_path, parent=parent)
        if widget.audio_loader is None:
            if show_error:
                QtWidgets.QMessageBox.information(
                    parent,
                    error_title,
                    error_message,
                )
            widget.close()
            return None, None
        return widget, audio_path

    def _refresh_plots(self):
        """Refresh the plotted data to reflect the current audio loader."""
        if self.audio_loader:
            self.plot()
        else:
            self.clear()
        self._update_controls_state()

    def plot(self):
        """Plot the audio waveform and spectrum."""
        self.figure_waveform.clear()
        # Waveform plot
        ax_waveform = self.figure_waveform.add_subplot(111)
        y = self.audio_loader.audio_data
        sample_rate = float(self.audio_loader.sample_rate)
        times = np.arange(0, len(y)) / sample_rate
        self._duration_sec = float(times[-1]) if len(times) else 0.0
        ax_waveform.plot(times, y)
        ax_waveform.set_xlabel('Time (s)')
        ax_waveform.set_ylabel('Amplitude')
        ax_waveform.set_title('Audio Waveform')
        self._sync_selection_limits()
        self._attach_span_selector(ax_waveform)

        # Spectrum plot
        self.figure_spectrum.clear()
        ax_spectrum = self.figure_spectrum.add_subplot(111)
        spectrum, frequencies, times, _ = ax_spectrum.specgram(
            y, Fs=self.audio_loader.sample_rate)
        ax_spectrum.set_xlabel('Time (s)')
        ax_spectrum.set_ylabel('Frequency (Hz)')
        ax_spectrum.set_title('Audio Spectrum')
        ax_spectrum.set_xlim(0, times[-1])  # Set xlim based on waveform times
        colorbar = self.figure_spectrum.colorbar(_)
        colorbar.set_label('Intensity')

        self.canvas_waveform.draw()
        self.canvas_spectrum.draw()

    def _attach_span_selector(self, ax_waveform) -> None:
        if not self.audio_loader:
            self._span_selector = None
            return

        self._span_selector = SpanSelector(
            ax_waveform,
            self._on_span_selected,
            "horizontal",
            useblit=True,
            props={"facecolor": "tab:orange", "alpha": 0.2},
            interactive=True,
            handle_props={"color": "tab:orange",
                          "alpha": 0.8, "linewidth": 2.0},
            grab_range=12,
            drag_from_anywhere=False,
            onmove_callback=self._on_span_moved,
        )
        if self._selection:
            with self._suppress_span_callbacks():
                self._span_selector.set_visible(True)
                self._span_selector.extents = self._selection

    def _on_span_selected(self, xmin: float, xmax: float) -> None:
        if self._suppress_span_callbacks_flag:
            return
        self._set_selection(float(xmin), float(xmax), update_span=False)

    def _on_span_moved(self, xmin: float, xmax: float) -> None:
        if self._suppress_span_callbacks_flag:
            return
        self._set_selection(float(xmin), float(xmax), update_span=False)

    def clear(self):
        """Clear any plotted data."""
        self.figure_waveform.clear()
        self.figure_spectrum.clear()
        self.canvas_waveform.draw()
        self.canvas_spectrum.draw()
        self._duration_sec = 0.0
        self._selection = None
        self._span_selector = None
        self._sync_selection_limits()
        self._update_controls_state()

    def set_audio_loader(self, audio_loader: Optional[AudioLoader]):
        """
        Update the widget with a new audio loader and redraw the plots.

        Args:
            audio_loader (AudioLoader | None): New loader to visualize.
        """
        self.audio_loader = audio_loader
        self._refresh_plots()

    def _build_controls(self, container: QtWidgets.QWidget) -> None:
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)

        self.play_selection_button = QtWidgets.QPushButton(
            "Play Selection", container)
        self.play_all_button = QtWidgets.QPushButton("Play All", container)
        self.stop_button = QtWidgets.QPushButton("Stop", container)

        self.selection_start_spin = QtWidgets.QDoubleSpinBox(container)
        self.selection_start_spin.setDecimals(3)
        self.selection_start_spin.setSingleStep(0.1)
        self.selection_start_spin.setPrefix("Start: ")

        self.selection_end_spin = QtWidgets.QDoubleSpinBox(container)
        self.selection_end_spin.setDecimals(3)
        self.selection_end_spin.setSingleStep(0.1)
        self.selection_end_spin.setPrefix("End: ")

        self.add_segment_button = QtWidgets.QPushButton(
            "Add Segment", container)
        self.save_segment_button = QtWidgets.QPushButton("Save…", container)

        self.status_label = QtWidgets.QLabel("", container)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setMinimumWidth(180)

        layout.addWidget(self.play_selection_button)
        layout.addWidget(self.play_all_button)
        layout.addWidget(self.stop_button)
        layout.addSpacing(12)
        layout.addWidget(self.selection_start_spin)
        layout.addWidget(self.selection_end_spin)
        layout.addSpacing(12)
        layout.addWidget(self.add_segment_button)
        layout.addWidget(self.save_segment_button)
        layout.addStretch(1)
        layout.addWidget(self.status_label)

        self.play_selection_button.clicked.connect(self.play_selection)
        self.play_all_button.clicked.connect(self.play_all)
        self.stop_button.clicked.connect(self.stop_playback)
        self.add_segment_button.clicked.connect(
            self.add_segment_from_selection)
        self.save_segment_button.clicked.connect(
            self.save_selected_or_current_segment)
        self.selection_start_spin.valueChanged.connect(
            self._on_selection_spin_changed)
        self.selection_end_spin.valueChanged.connect(
            self._on_selection_spin_changed)

    def _build_segments_panel(self) -> None:
        self.segments_panel = QtWidgets.QWidget(self)
        panel_layout = QtWidgets.QVBoxLayout(self.segments_panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)

        panel_layout.addWidget(QtWidgets.QLabel(
            "Segments", self.segments_panel))
        self.segments_list = QtWidgets.QListWidget(self.segments_panel)
        self.segments_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        panel_layout.addWidget(self.segments_list, 1)

        buttons_row = QtWidgets.QHBoxLayout()
        self.remove_segment_button = QtWidgets.QPushButton(
            "Remove", self.segments_panel)
        self.clear_segments_button = QtWidgets.QPushButton(
            "Clear", self.segments_panel)
        buttons_row.addWidget(self.remove_segment_button)
        buttons_row.addWidget(self.clear_segments_button)
        panel_layout.addLayout(buttons_row)

        self.segments_list.itemSelectionChanged.connect(
            self._on_segment_selected)
        self.remove_segment_button.clicked.connect(
            self.remove_selected_segment)
        self.clear_segments_button.clicked.connect(self.clear_segments)

    def _update_controls_state(self) -> None:
        has_audio = bool(self.audio_loader and getattr(
            self.audio_loader, "audio_data", None) is not None)
        has_selection = bool(
            self._selection and self._selection[1] > self._selection[0])
        has_segment_selected = bool(self.segments_list.selectedItems())

        self.play_selection_button.setEnabled(has_audio and has_selection)
        self.play_all_button.setEnabled(has_audio)
        self.stop_button.setEnabled(has_audio)
        self.add_segment_button.setEnabled(has_audio and has_selection)
        self.save_segment_button.setEnabled(
            has_audio and (has_selection or has_segment_selected))
        self.remove_segment_button.setEnabled(has_segment_selected)
        self.clear_segments_button.setEnabled(self.segments_list.count() > 0)

        if not has_audio:
            self.status_label.setText("No audio loaded.")
        else:
            if has_selection:
                self.status_label.setText("")
            else:
                self.status_label.setText(
                    "Drag to select; drag handles to adjust; Shift-click plays 1s."
                )

    def _sync_selection_limits(self) -> None:
        self._updating_selection_controls = True
        try:
            max_value = max(0.0, float(self._duration_sec))
            for spin in (self.selection_start_spin, self.selection_end_spin):
                spin.setRange(0.0, max_value)
        finally:
            self._updating_selection_controls = False

    class _SpanCallbackGuard:
        def __init__(self, widget: "AudioWidget"):
            self.widget = widget

        def __enter__(self):
            self.widget._suppress_span_callbacks_flag = True

        def __exit__(self, exc_type, exc, tb):
            self.widget._suppress_span_callbacks_flag = False
            return False

    def _suppress_span_callbacks(self):
        return AudioWidget._SpanCallbackGuard(self)

    def _set_selection(
        self,
        start_sec: float,
        end_sec: float,
        *,
        update_span: bool = True,
    ) -> None:
        start = float(min(start_sec, end_sec))
        end = float(max(start_sec, end_sec))
        if self._duration_sec:
            start = max(0.0, min(start, self._duration_sec))
            end = max(0.0, min(end, self._duration_sec))
        if end <= start:
            self._selection = None
        else:
            self._selection = (start, end)

        self._updating_selection_controls = True
        try:
            self.selection_start_spin.setValue(start)
            self.selection_end_spin.setValue(end)
        finally:
            self._updating_selection_controls = False

        if update_span and self._span_selector is not None:
            with self._suppress_span_callbacks():
                if self._selection is None:
                    self._span_selector.set_visible(False)
                else:
                    self._span_selector.set_visible(True)
                    self._span_selector.extents = self._selection
            self.canvas_waveform.draw_idle()

        self._update_controls_state()

    def _on_selection_spin_changed(self) -> None:
        if self._updating_selection_controls:
            return
        start = float(self.selection_start_spin.value())
        end = float(self.selection_end_spin.value())
        self._set_selection(start, end)

    def _on_segment_selected(self) -> None:
        items = self.segments_list.selectedItems()
        if not items:
            self._update_controls_state()
            return
        segment = items[0].data(Qt.UserRole)
        if not segment:
            self._update_controls_state()
            return
        start, end = segment
        self._set_selection(float(start), float(end))

    def _format_segment(self, start: float, end: float) -> str:
        return f"{start:.3f}s – {end:.3f}s"

    def _current_audio_slice(self, start_sec: float, end_sec: float) -> np.ndarray:
        if not self.audio_loader:
            return np.asarray([])
        sr = float(self.audio_loader.sample_rate)
        start = int(round(max(0.0, start_sec) * sr))
        end = int(round(max(0.0, end_sec) * sr))
        end = max(start, end)
        return self.audio_loader.audio_data[start:end]

    def play_selection(self) -> None:
        if not self.audio_loader or not self._selection:
            return
        start, end = self._selection
        samples = self._current_audio_slice(start, end)
        if samples.size == 0:
            return
        self.stop_playback()
        started = play_audio_buffer(
            samples, int(self.audio_loader.sample_rate), blocking=False
        )
        if not started:
            self.status_label.setText("Audio playback unavailable.")

    def play_all(self) -> None:
        if not self.audio_loader:
            return
        self.stop_playback()
        play_method = getattr(self.audio_loader, "play", None)
        if callable(play_method):
            play_method()
            return
        started = play_audio_buffer(
            self.audio_loader.audio_data,
            int(self.audio_loader.sample_rate),
            blocking=False,
        )
        if not started:
            self.status_label.setText("Audio playback unavailable.")

    def stop_playback(self) -> None:
        if not self.audio_loader:
            return
        stop_method = getattr(self.audio_loader, "stop", None)
        if callable(stop_method):
            stop_method()
        self.status_label.setText("")

    def add_segment_from_selection(self) -> None:
        if not self._selection:
            return
        start, end = self._selection
        if end <= start:
            return
        item = QtWidgets.QListWidgetItem(self._format_segment(start, end))
        item.setData(Qt.UserRole, (float(start), float(end)))
        self.segments_list.addItem(item)
        self._update_controls_state()

    def remove_selected_segment(self) -> None:
        row = self.segments_list.currentRow()
        if row < 0:
            return
        self.segments_list.takeItem(row)
        self._update_controls_state()

    def clear_segments(self) -> None:
        self.segments_list.clear()
        self._update_controls_state()

    def save_selected_or_current_segment(self) -> None:
        segment = None
        items = self.segments_list.selectedItems()
        if items:
            segment = items[0].data(Qt.UserRole)
        if not segment and self._selection:
            segment = self._selection
        if not segment or not self.audio_loader:
            return

        start, end = map(float, segment)
        if end <= start:
            return

        default_base = Path(self.audio_path).with_suffix("")
        default_name = f"{default_base.name}_segment_{int(start * 1000)}_{int(end * 1000)}.wav"
        default_path = str(default_base.with_name(default_name))
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Audio Segment",
            default_path,
            "WAV (*.wav);;FLAC (*.flac);;OGG (*.ogg);;All Files (*.*)",
        )
        save_path = str(save_path)
        if not save_path:
            return

        samples = self._current_audio_slice(start, end)
        if samples.size == 0:
            return

        try:
            import soundfile as sf
        except Exception:
            QtWidgets.QMessageBox.warning(
                self,
                "Save Segment",
                "Saving audio requires the 'soundfile' package.",
            )
            return

        try:
            sf.write(save_path, samples, int(self.audio_loader.sample_rate))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Save Segment",
                f"Failed to save segment:\n{exc}",
            )
            return

        self.status_label.setText(f"Saved: {Path(save_path).name}")

    def _on_canvas_press(self, event) -> None:
        if event.button != Qt.LeftButton:
            return
        if not self.audio_loader:
            return
        if event.xdata is None:
            return
        self._mouse_down_xdata = float(event.xdata)

    def _on_canvas_release(self, event) -> None:
        if event.button != Qt.LeftButton:
            return
        if not self.audio_loader:
            return
        if event.xdata is None or self._mouse_down_xdata is None:
            self._mouse_down_xdata = None
            return

        x_up = float(event.xdata)
        x_down = float(self._mouse_down_xdata)
        self._mouse_down_xdata = None

        if abs(x_up - x_down) > self.CLICK_DRAG_THRESHOLD_SEC:
            return

        start_sec = x_up
        end_sec = start_sec + 1.0
        if self._duration_sec:
            end_sec = max(start_sec, end_sec)
        self._set_selection(start_sec, end_sec)
        self.play_selection()


if __name__ == '__main__':
    # Example usage
    app = QtWidgets.QApplication(sys.argv)
    widget = AudioWidget('/path/to/ants.mp4')
    widget.show()
    sys.exit(app.exec_())
