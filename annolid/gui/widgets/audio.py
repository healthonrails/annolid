import sys
import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QVBoxLayout, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from annolid.data.audios import AudioLoader
from qtpy.QtCore import Qt


class AudioWidget(QtWidgets.QWidget):
    """A QWidget that displays audio waveform and spectrum using Matplotlib."""

    def __init__(self, audio_path, parent=None):
        """
        Constructor for the AudioWidget class.

        Args:
            audio_path (str): Path to the audio file.
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
        try:
            self.audio_loader = AudioLoader(audio_path)
        except Exception as e:
            self.audio_loader = None
        self.canvas_waveform.mpl_connect(
            'button_press_event', self.on_canvas_clicked)
        self.canvas_spectrum.mpl_connect(
            'button_press_event', self.on_canvas_clicked)

        if self.audio_loader:
            self.plot()

    def plot(self):
        """Plot the audio waveform and spectrum."""
        # Waveform plot
        ax_waveform = self.figure_waveform.add_subplot(111)
        y = self.audio_loader.audio_data
        times = np.arange(0, len(y)) / self.audio_loader.sample_rate
        ax_waveform.plot(times, y)
        ax_waveform.set_xlabel('Time (s)')
        ax_waveform.set_ylabel('Amplitude')
        ax_waveform.set_title('Audio Waveform')

        # Spectrum plot
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

    def on_canvas_clicked(self, event):
        """
        Handle the mouse press event on the canvas.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Mouse event object.
        """
        if event.button == Qt.LeftButton:
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
