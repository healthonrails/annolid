import sys
from typing import Optional

import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt

from annolid.data.audios import AudioLoader


class AudioWidget(QtWidgets.QWidget):
    """A QWidget that displays audio waveform and spectrum using Matplotlib."""

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
        self.figure_waveform = Figure()
        self.figure_spectrum = Figure()
        self.canvas_waveform = FigureCanvas(self.figure_waveform)
        self.canvas_spectrum = FigureCanvas(self.figure_spectrum)

        self.layout = QGridLayout(self)
        self.layout.addWidget(self.canvas_waveform, 0, 0)
        self.layout.addWidget(self.canvas_spectrum, 1, 0)
        self.setLayout(self.layout)
        self.audio_loader: Optional[AudioLoader] = audio_loader
        if self.audio_loader is None and audio_path:
            try:
                self.audio_loader = AudioLoader(audio_path)
            except Exception:
                self.audio_loader = None
        self.canvas_waveform.mpl_connect(
            'button_press_event', self.on_canvas_clicked)
        self.canvas_spectrum.mpl_connect(
            'button_press_event', self.on_canvas_clicked)

        self._refresh_plots()

    def _refresh_plots(self):
        """Refresh the plotted data to reflect the current audio loader."""
        if self.audio_loader:
            self.plot()
        else:
            self.clear()

    def plot(self):
        """Plot the audio waveform and spectrum."""
        self.figure_waveform.clear()
        # Waveform plot
        ax_waveform = self.figure_waveform.add_subplot(111)
        y = self.audio_loader.audio_data
        times = np.arange(0, len(y)) / self.audio_loader.sample_rate
        ax_waveform.plot(times, y)
        ax_waveform.set_xlabel('Time (s)')
        ax_waveform.set_ylabel('Amplitude')
        ax_waveform.set_title('Audio Waveform')

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

    def clear(self):
        """Clear any plotted data."""
        self.figure_waveform.clear()
        self.figure_spectrum.clear()
        self.canvas_waveform.draw()
        self.canvas_spectrum.draw()

    def set_audio_loader(self, audio_loader: Optional[AudioLoader]):
        """
        Update the widget with a new audio loader and redraw the plots.

        Args:
            audio_loader (AudioLoader | None): New loader to visualize.
        """
        self.audio_loader = audio_loader
        self._refresh_plots()

    def on_canvas_clicked(self, event):
        """
        Handle the mouse press event on the canvas.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Mouse event object.
        """
        if event.button == Qt.LeftButton and self.audio_loader:
            x = event.xdata
            if x is not None:
                x_start = x
                x_end = x + 1
                self.audio_loader.play_selected_part(x_start, x_end)


if __name__ == '__main__':
    # Example usage
    app = QtWidgets.QApplication(sys.argv)
    widget = AudioWidget('/path/to/ants.mp4')
    widget.show()
    sys.exit(app.exec_())
