import sys
import os
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

# Run the user's setup
from annolid.gui.widgets.web_viewer import WebViewerWidget
from annolid.utils.macos_fixes import apply_macos_webengine_sandbox_patch

apply_macos_webengine_sandbox_patch()

print("QTWEBENGINEPROCESS_PATH:", os.environ.get("QTWEBENGINEPROCESS_PATH"))
print("QTWEBENGINE_ICU_DATA_PATH:", os.environ.get("QTWEBENGINE_ICU_DATA_PATH"))
print("QTWEBENGINE_RESOURCES_PATH:", os.environ.get("QTWEBENGINE_RESOURCES_PATH"))
print("QTWEBENGINE_CHROMIUM_FLAGS:", os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS"))

app = QApplication(sys.argv)
w = WebViewerWidget()
w.show()


def check():
    w.load_url("https://example.com")
    QTimer.singleShot(2000, lambda: sys.exit(0))


QTimer.singleShot(100, check)
sys.exit(app.exec_())
