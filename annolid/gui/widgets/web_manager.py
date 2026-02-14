from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.widgets.web_viewer import WebViewerWidget

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class WebManager(QtCore.QObject):
    """Manage an embedded web viewer in the shared stacked viewer area."""

    def __init__(
        self, window: "AnnolidWindow", viewer_stack: QtWidgets.QStackedWidget
    ) -> None:
        super().__init__(window)
        self.window = window
        self.viewer_stack = viewer_stack
        self.web_viewer: Optional[WebViewerWidget] = None

    def ensure_web_viewer(self) -> WebViewerWidget:
        if self.web_viewer is None:
            viewer = WebViewerWidget(self.window)
            viewer.status_changed.connect(
                lambda msg: self.window.statusBar().showMessage(msg, 3000)
            )
            self.viewer_stack.addWidget(viewer)
            self.web_viewer = viewer
        return self.web_viewer

    def show_url_in_viewer(self, url: str) -> bool:
        viewer = self.ensure_web_viewer()
        if not viewer.webengine_available:
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Embedded Browser"),
                self.window.tr(
                    "Qt WebEngine is unavailable, cannot open URL in Annolid canvas."
                ),
            )
            return False
        if not viewer.load_url(url):
            return False
        self.window.set_unrelated_docks_visible(False)
        self.window._set_active_view("web")
        try:
            close_action = getattr(getattr(self.window, "actions", None), "close", None)
            if close_action is not None:
                close_action.setEnabled(True)
        except Exception:
            pass
        return True

    def viewer_widget(self) -> Optional[WebViewerWidget]:
        return self.web_viewer

    def get_page_text(self, max_chars: int = 8000) -> dict:
        viewer = self.ensure_web_viewer()
        return viewer.get_page_text(max_chars=max_chars)

    def click_selector(self, selector: str) -> dict:
        viewer = self.ensure_web_viewer()
        return viewer.click_selector(selector)

    def type_selector(self, selector: str, text: str, submit: bool = False) -> dict:
        viewer = self.ensure_web_viewer()
        return viewer.type_selector(selector, text, submit=submit)

    def scroll_by(self, delta_y: int = 800) -> dict:
        viewer = self.ensure_web_viewer()
        return viewer.scroll_by(delta_y=delta_y)

    def find_forms(self) -> dict:
        viewer = self.ensure_web_viewer()
        return viewer.find_forms()

    def get_web_state(self) -> dict:
        viewer = self.web_viewer
        if viewer is None:
            return {
                "ok": True,
                "webengine_available": True,
                "has_page": False,
                "url": "",
                "title": "",
            }
        return viewer.get_state()

    def close_web(self) -> None:
        """Close embedded web view and return to canvas."""
        viewer = self.web_viewer
        if viewer is not None:
            try:
                viewer.clear()
            except Exception:
                pass
        try:
            self.window._set_active_view("canvas")
        except Exception:
            pass
