from __future__ import annotations

from typing import Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.widgets.ai_chat_widget import AIChatWidget


class AIChatManager(QtCore.QObject):
    """Manage the dedicated right-side Annolid Bot dock."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.ai_chat_dock: Optional[QtWidgets.QDockWidget] = None
        self.ai_chat_widget: Optional[AIChatWidget] = None

    def _on_dock_visibility_changed(self, visible: bool) -> None:
        if not visible:
            self.window.set_unrelated_docks_visible(True)

    def _ensure_dock(self) -> tuple[QtWidgets.QDockWidget, AIChatWidget]:
        if self.ai_chat_dock is not None and self.ai_chat_widget is not None:
            return self.ai_chat_dock, self.ai_chat_widget

        dock = QtWidgets.QDockWidget(self.window.tr("Annolid Bot"), self.window)
        dock.setObjectName("aiChatDock")
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        widget = AIChatWidget(dock)
        widget.set_canvas(getattr(self.window, "canvas", None))
        widget.set_host_window(self.window)
        widget.set_default_visual_share_mode(attach_canvas=False, attach_window=False)

        dock.setWidget(widget)
        dock.visibilityChanged.connect(self._on_dock_visibility_changed)

        self.ai_chat_dock = dock
        self.ai_chat_widget = widget
        return dock, widget

    def show_chat_dock(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        hide_other_docks: bool = True,
    ) -> None:
        dock, widget = self._ensure_dock()
        self.window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        widget.set_canvas(getattr(self.window, "canvas", None))
        widget.set_host_window(self.window)
        image_path = getattr(self.window, "filename", None)
        if image_path:
            suffix = str(image_path).lower()
            if suffix.endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
            ):
                widget.set_image_path(image_path)
        if provider:
            widget.set_provider_and_model(provider, model or "")
        elif model:
            widget.set_provider_and_model(widget.selected_provider, model)

        if not dock.isVisible():
            if hide_other_docks:
                self.window.set_unrelated_docks_visible(False, exclude=[dock])
            dock.show()

        dock.raise_()
        widget.prompt_text_edit.setFocus()

    def initialize_annolid_bot(self, *, start_visible: bool = True) -> None:
        """Start bot session resources when the app launches."""
        dock, widget = self._ensure_dock()
        self.window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        widget.set_canvas(getattr(self.window, "canvas", None))
        widget.set_host_window(self.window)
        if start_visible:
            if not dock.isVisible():
                self.show_chat_dock(hide_other_docks=False)
        else:
            # Some Qt setups briefly show dock widgets after addDockWidget.
            # Hide immediately and once again on the next event-loop tick.
            dock.hide()
            QtCore.QTimer.singleShot(0, dock.hide)
