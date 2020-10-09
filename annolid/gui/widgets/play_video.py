import sys
import time
import decord as de
import numpy as np
from qtpy import QtGui
from qtpy.QtWidgets import QWidget, QPushButton, QGroupBox
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout
from qtpy.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot


class VideoPlayerThread(QThread):
    """
    Use decord to read video frames and

    """

    pixmap_updated_signal = pyqtSignal(np.ndarray)

    def __init__(self,
                 video_url=None):
        super().__init__()
        self.video_url = video_url
        self.run_flag = True
        self.back_flag = False
        self.pause_flag = False
        self.vr = de.VideoReader(self.video_url, ctx=de.cpu(0))
        self.frame_numbers = range(len(self.vr))

    def run(self):
        de.bridge.set_bridge('native')
        for fn in self.frame_numbers:
            if not self.run_flag:
                # faster when close the video player window
                return
            if self.back_flag:
                if fn - 10 >= 0:
                    fn = fn - 10
                self.back_flag = False
            rgb_img = self.vr[fn].asnumpy()
            self.pixmap_updated_signal.emit(rgb_img)
            while self.pause_flag:
                time.sleep(1)

    def stop(self):
        self.run_flag = False
        self.wait()


class VideoPlayerWindow(QWidget):
    def __init__(self, video_url=None):
        super().__init__()
        self.setWindowTitle("Annolid Video Player")
        self.video_url = video_url
        assert self.video_url is not None
        # Use a label that holds the image
        self.image_label = QLabel(self)

        self.group_btn = QGroupBox(self)
        self.back_btn = QPushButton(self)
        self.back_btn.setText("Back 10 frames")
        self.back_btn.pressed.connect(self.on_back_btn_pressed)

        self.play_btn = QPushButton(self)
        self.play_btn.setText("Play")
        self.play_btn.pressed.connect(self.on_play_btn_pressed)

        self.pause_btn = QPushButton(self)
        self.pause_btn.setText("Pause")
        self.pause_btn.pressed.connect(self.on_pause_btn_pressed)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        vbox.addWidget(self.image_label)
        hbox.addWidget(self.back_btn)
        hbox.addWidget(self.play_btn)
        hbox.addWidget(self.pause_btn)
        self.group_btn.setLayout(hbox)
        vbox.addWidget(self.group_btn)

        self.setLayout(vbox)
        self.thread = VideoPlayerThread(self.video_url)
        self.thread.pixmap_updated_signal.connect(self.update_image)
        self.thread.start()

    def rgb_to_qt(self, rgb_image):
        """convert an rgb image to QPixmap"""
        h, w, c = rgb_image.shape
        bytes_per_line = c * w
        img_qt = QtGui.QImage(
            rgb_image.data,
            w, h,
            bytes_per_line,
            QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(img_qt)

    @pyqtSlot(np.ndarray)
    def update_image(self, rgb_img):
        """update the image_label with a new rgb image"""
        qt_img = self.rgb_to_qt(rgb_img)
        self.image_label.setPixmap(qt_img)

    def on_back_btn_pressed(self):
        self.thread.back_flag = True

    def on_pause_btn_pressed(self):
        self.thread.pause_flag = True
        self.thread.back_flag = False

    def on_play_btn_pressed(self):
        self.thread.pause_flag = False
        self.thread.back_flag = False

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    vp = VideoPlayerWindow("/Users/chenyang/Downloads/novelctrl.mkv")
    vp.show()
    sys.exit(app.exec_())
