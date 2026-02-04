import sys
import time
import cv2
import numpy as np
from qtpy import QtGui
from qtpy.QtWidgets import QWidget, QPushButton, QGroupBox
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout
from qtpy.QtGui import QPixmap
from qtpy.QtCore import QThread, Signal, Slot


class VideoPlayerThread(QThread):
    """
    Use OpenCV to read video frames and feed them back to the GUI thread.
    """

    pixmap_updated_signal = Signal(np.ndarray)

    def __init__(self, video_url=None):
        super().__init__()
        self.video_url = video_url
        self.run_flag = True
        self.back_flag = False
        self.pause_flag = False
        self.cap = cv2.VideoCapture(self.video_url)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video: {self.video_url}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) or 0

    def run(self):
        try:
            while self.run_flag and self.cap.isOpened():
                if self.back_flag:
                    # jump back 10 frames while clamping to start
                    self.frame_idx = max(self.frame_idx - 10, 0)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
                    self.back_flag = False

                if self.pause_flag:
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break

                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.pixmap_updated_signal.emit(rgb_img)

                # Track frame index after read to keep UI in sync
                self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                while self.pause_flag and self.run_flag:
                    time.sleep(0.1)

            if not self.run_flag:
                return
        finally:
            self._release()

    def stop(self):
        self.run_flag = False
        self.pause_flag = False
        self.back_flag = False
        self.wait()
        self._release()

    def _release(self):
        if getattr(self, "cap", None) is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None


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
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return QPixmap.fromImage(img_qt)

    @Slot(np.ndarray)
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
    vp = VideoPlayerWindow("my_video.mkv")
    vp.show()
    sys.exit(app.exec_())
