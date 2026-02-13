from pathlib import Path
from typing import TYPE_CHECKING

from qtpy import QtWidgets

from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class PdfImportWidget(QtWidgets.QWidget):
    """Open PDF files directly in the integrated viewer."""

    def __init__(self, window: "AnnolidWindow") -> None:
        super().__init__()
        self._window = window

    def open_pdf(self) -> None:
        """Prompt for a PDF and display it in the viewer."""
        window = self._window

        if window.video_loader is not None:
            previous_loader = window.video_loader
            window.closeFile(suppress_tracking_prompt=True)
            if window.video_loader is previous_loader:
                return
        elif not window.mayContinue():
            return

        start_dir = Path(window.lastOpenDir) if window.lastOpenDir else Path.home()
        pdf_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            window,
            window.tr("Select PDF File"),
            str(start_dir),
            window.tr("PDF Files (*.pdf)"),
        )
        if not pdf_file:
            return

        self.open_pdf_path(pdf_file)

    def open_pdf_path(self, pdf_file: str) -> bool:
        """Open a specific PDF path directly without prompting for file selection."""
        window = self._window
        pdf_text = str(pdf_file or "").strip()
        if not pdf_text:
            return False

        try:
            import fitz  # type: ignore[import]
        except ImportError:
            QtWidgets.QMessageBox.critical(
                window,
                window.tr("PyMuPDF Required"),
                window.tr(
                    "PyMuPDF (pymupdf) is required to open PDF files.\n"
                    "Install it with 'pip install pymupdf' and try again."
                ),
            )
            return False

        pdf_path = Path(pdf_text).expanduser()
        if not pdf_path.exists() or not pdf_path.is_file():
            QtWidgets.QMessageBox.warning(
                window,
                window.tr("File Not Found"),
                window.tr("The selected PDF file does not exist:\n%s") % str(pdf_path),
            )
            return False

        try:
            with fitz.open(str(pdf_path)) as pdf_doc:
                if pdf_doc.page_count == 0:
                    QtWidgets.QMessageBox.information(
                        window,
                        window.tr("Empty PDF"),
                        window.tr("The selected PDF does not contain any pages."),
                    )
                    return False
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", pdf_path, exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                window,
                window.tr("Failed to Open PDF"),
                window.tr("Could not open the selected PDF:\n%s") % str(exc),
            )
            return False

        window.show_pdf_in_viewer(str(pdf_path))
        window.lastOpenDir = str(pdf_path.parent)
        return True
