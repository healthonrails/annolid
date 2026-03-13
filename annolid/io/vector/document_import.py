from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .svg_import import ImportedVectorDocument, import_svg_document, import_svg_string


def _looks_like_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _pdf_to_svg_text_with_pymupdf(path: Path) -> str:
    import fitz

    document = fitz.open(path)
    try:
        if document.page_count < 1:
            raise ValueError(f"Vector document has no pages: {path}")
        page = document.load_page(0)
        return str(page.get_svg_image())
    finally:
        document.close()


def _pdf_to_svg_text_with_inkscape(path: Path) -> str:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        raise RuntimeError("Inkscape is not available")
    with tempfile.TemporaryDirectory(prefix="annolid_vector_") as tmpdir:
        output_path = Path(tmpdir) / f"{path.stem}.svg"
        command = [
            inkscape,
            str(path),
            "--export-type=svg",
            f"--export-filename={output_path}",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                stderr or f"Failed to convert vector document with Inkscape: {path}"
            )
        return output_path.read_text(encoding="utf-8")


def _pdf_like_to_svg_text(path: Path) -> str:
    errors: list[str] = []
    for converter in (_pdf_to_svg_text_with_pymupdf, _pdf_to_svg_text_with_inkscape):
        try:
            return converter(path)
        except Exception as exc:
            errors.append(str(exc))
    joined = "; ".join(item for item in errors if item)
    raise RuntimeError(
        "Could not convert PDF-compatible vector document to SVG. "
        "Install PyMuPDF or Inkscape. "
        f"Details: {joined}"
    )


def import_vector_document(path: str | Path) -> ImportedVectorDocument:
    resolved = Path(path).expanduser()
    suffix = resolved.suffix.lower()
    if suffix == ".svg":
        return import_svg_document(resolved)
    if suffix in {".pdf", ".ai"}:
        if suffix == ".ai" and not _looks_like_pdf(resolved):
            raise ValueError(
                "This Illustrator file is not PDF-compatible. Export it as SVG or save it with PDF compatibility enabled."
            )
        svg_text = _pdf_like_to_svg_text(resolved)
        return import_svg_string(svg_text, source_path=str(resolved))
    raise ValueError(f"Unsupported vector import format: {resolved.suffix}")
