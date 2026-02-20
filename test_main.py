import sys
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication
from annolid.utils.macos_fixes import apply_macos_webengine_sandbox_patch
from annolid.gui.widgets.web_viewer import WebViewerWidget

apply_macos_webengine_sandbox_patch()


def main():
    app = QApplication(sys.argv)
    w = WebViewerWidget()
    w.show()

    def check():
        w.load_url("https://example.com")
        QTimer.singleShot(2000, lambda: sys.exit(0))

    QTimer.singleShot(100, check)
    sys.exit(app.exec_())


main()
