import sys
import decord as de
import numpy as np
from qtpy import QtGui
from qtpy.QtWidgets import QWidget
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import QLabel, QVBoxLayout
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

    def run(self):
        de.bridge.set_bridge('native')
        vr = de.VideoReader(self.video_url, ctx=de.cpu(0))
        kidxs = range(len(vr))
        for i in kidxs:
            if not self.run_flag:
                # faster when close the video player window
                return
            rgb_img = vr[i].asnumpy()
            self.pixmap_updated_signal.emit(rgb_img)

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

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)

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

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    vp = VideoPlayerWindow("my_video.mp4")
    vp.show()
    sys.exit(app.exec_())
