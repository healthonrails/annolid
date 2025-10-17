"""TensorBoard launch utilities used by the Annolid GUI."""

import subprocess
import time
from pathlib import Path
from typing import Optional

import requests
from qtpy import QtCore, QtWidgets, QtGui


def start_tensorboard(
    log_dir: Optional[Path] = None,
    tensorboard_url: str = "http://localhost:6006",
) -> Optional[subprocess.Popen]:
    """
    Ensure a TensorBoard server is available and return the launched process.

    If TensorBoard is already responding at ``tensorboard_url`` the existing
    server is reused and ``None`` is returned. Otherwise a new instance is
    spawned pointing at ``log_dir``.
    """
    process = None
    if log_dir is None:
        here = Path(__file__).parent
        log_dir = here.parent.resolve() / "runs" / "logs"
    try:
        requests.get(tensorboard_url)
    except requests.exceptions.ConnectionError:
        process = subprocess.Popen(
            ["tensorboard", f"--logdir={str(log_dir)}"]
        )
        time.sleep(8)
    return process


class VisualizationWindow(QtWidgets.QDialog):
    """Embedded TensorBoard viewer dialog."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        log_dir: Optional[Path] = None,
        tensorboard_url: str = "http://localhost:6006",
    ) -> None:
        super().__init__(parent=parent)
        from qtpy.QtWebEngineWidgets import QWebEngineView

        self.setWindowTitle("Visualization Tensorboard")
        self.tensorboard_url = tensorboard_url
        self.process = start_tensorboard(log_dir=log_dir,
                                         tensorboard_url=tensorboard_url)
        self.browser = QWebEngineView()
        self.browser.setUrl(QtCore.QUrl(self.tensorboard_url))
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.browser)
        self.setLayout(vbox)
        self.show()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.process is not None:
            time.sleep(3)
            self.process.kill()
        event.accept()
