import contextlib
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

from qtpy import QtWidgets

from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class PdfImportWidget(QtWidgets.QWidget):
    """Handle importing PDF pages as images with captions."""

    def __init__(self, window: "AnnolidWindow") -> None:
        super().__init__(window)
        self._window = window

    def open_pdf(self) -> None:
        """Prompt for a PDF, convert pages, and load them into the main window."""
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
            return

        pdf_path = Path(pdf_file)

        try:
            pdf_doc = fitz.open(pdf_file)
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", pdf_file, exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                window,
                window.tr("Failed to Open PDF"),
                window.tr("Could not open the selected PDF:\n%s") % str(exc),
            )
            return

        if pdf_doc.page_count == 0:
            QtWidgets.QMessageBox.information(
                window,
                window.tr("Empty PDF"),
                window.tr("The selected PDF does not contain any pages."),
            )
            pdf_doc.close()
            return

        output_dir = pdf_path.parent / f"{pdf_path.stem}_pdf_pages"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            pdf_doc.close()
            logger.error("Failed to create directory %s: %s", output_dir, exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                window,
                window.tr("Directory Error"),
                window.tr("Could not create a directory for PDF pages:\n%s") % output_dir,
            )
            return

        pattern = f"{pdf_path.stem}_page_*.png"
        for existing_image in output_dir.glob(pattern):
            with contextlib.suppress(Exception):
                existing_image.unlink()
            json_path = existing_image.with_suffix(".json")
            with contextlib.suppress(Exception):
                json_path.unlink()

        matrix = fitz.Matrix(2.0, 2.0)
        generated: List[Path] = []
        errors: List[Tuple[int, Exception]] = []

        try:
            for page_index in range(pdf_doc.page_count):
                try:
                    page = pdf_doc.load_page(page_index)
                    pixmap = page.get_pixmap(matrix=matrix)
                    image_path = output_dir / f"{pdf_path.stem}_page_{page_index + 1:04}.png"
                    pixmap.save(image_path.as_posix())
                    caption_text = (page.get_text("text") or "").strip()
                    self._write_caption_file(
                        image_path=image_path,
                        width=pixmap.width,
                        height=pixmap.height,
                        caption=caption_text,
                    )
                    generated.append(image_path)
                except Exception as exc:
                    errors.append((page_index + 1, exc))
                    logger.error(
                        "Failed to convert page %s of %s: %s",
                        page_index + 1,
                        pdf_path,
                        exc,
                        exc_info=True,
                    )
        finally:
            pdf_doc.close()

        if not generated:
            QtWidgets.QMessageBox.critical(
                window,
                window.tr("Conversion Failed"),
                window.tr("Could not generate images from the selected PDF."),
            )
            return

        window.importDirImages(str(output_dir))
        window.status(window.tr("Loaded PDF pages from %s") % pdf_path.name)

        if errors:
            preview = "\n".join(f"Page {page}: {err}" for page, err in errors[:3])
            QtWidgets.QMessageBox.warning(
                window,
                window.tr("Partial Conversion"),
                window.tr(
                    "Some pages could not be converted. First issues:\n%s"
                ) % preview,
            )

        window.lastOpenDir = str(pdf_path.parent)

    def _write_caption_file(self, image_path: Path, width: int, height: int, caption: str) -> None:
        """Create or update a label file for the generated image with caption text."""
        label_path = image_path.with_suffix(".json")
        label_file = LabelFile()
        try:
            label_file.save(
                filename=str(label_path),
                shapes=[],
                imagePath=image_path.name,
                imageData=None,
                imageHeight=height,
                imageWidth=width,
                otherData=None,
                flags={},
                caption=caption,
            )
        except LabelFileError as exc:
            logger.error(
                "Failed to write caption file %s: %s", label_path, exc, exc_info=True
            )
            raise
