from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtCore

if TYPE_CHECKING:  # pragma: no cover
    from annolid.gui.widgets.pdf_viewer_impl import PdfViewerWidget


class _SpeakToken:
    def __init__(self) -> None:
        self.cancelled = False


class _PdfReaderBridge(QtCore.QObject):
    def __init__(self, viewer: "PdfViewerWidget") -> None:
        super().__init__(viewer)
        self._viewer = viewer

    @QtCore.Slot("QVariant")
    def onParagraphClicked(self, payload: object) -> None:
        self._viewer._handle_reader_click(payload)

    @QtCore.Slot(str, result="QVariant")
    def getUserState(self, pdfKey: str) -> object:  # noqa: N802 - Qt slot name
        return self._viewer._get_pdf_user_state(pdfKey)

    @QtCore.Slot("QVariant")
    def saveUserState(self, payload: object) -> None:  # noqa: N802 - Qt slot name
        self._viewer._handle_pdf_user_state_save(payload)

    @QtCore.Slot(str)
    def clearUserState(self, pdfKey: str) -> None:  # noqa: N802 - Qt slot name
        self._viewer._clear_pdf_user_state(pdfKey)

    @QtCore.Slot("QVariant")
    def logEvent(self, payload: object) -> None:  # noqa: N802 - Qt slot name
        self._viewer._handle_pdf_log_event(payload)


__all__ = ["_PdfReaderBridge", "_SpeakToken"]

