from __future__ import annotations

from qtpy import QtCore, QtWidgets


class LargeImageStatusOverlay(QtWidgets.QFrame):
    """Small translucent status/debug overlay for the tiled large-image viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("largeImageStatusOverlay")
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self._surface_label = QtWidgets.QLabel(self)
        self._surface_label.setObjectName("largeImageSurfaceLabel")
        self._surface_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._surface_label)

        self._details_label = QtWidgets.QLabel(self)
        self._details_label.setObjectName("largeImageDetailsLabel")
        self._details_label.setWordWrap(True)
        self._details_label.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        layout.addWidget(self._details_label)

        self.setStyleSheet(
            """
            QFrame#largeImageStatusOverlay {
                background: rgba(24, 28, 35, 180);
                color: white;
                border: 1px solid rgba(255, 255, 255, 36);
                border-radius: 8px;
            }
            QLabel#largeImageSurfaceLabel {
                color: rgb(240, 244, 255);
            }
            QLabel#largeImageDetailsLabel {
                color: rgb(222, 230, 240);
            }
            """
        )
        self.hide()

    def set_status(
        self,
        *,
        surface_text: str,
        details_text: str,
        visible: bool = True,
    ) -> None:
        self._surface_label.setText(str(surface_text or ""))
        self._details_label.setText(str(details_text or ""))
        self.setVisible(bool(visible and (surface_text or details_text)))
        if self.isVisible():
            self.adjustSize()

    def current_text(self) -> str:
        return "\n".join(
            part
            for part in (
                str(self._surface_label.text() or "").strip(),
                str(self._details_label.text() or "").strip(),
            )
            if part
        )
