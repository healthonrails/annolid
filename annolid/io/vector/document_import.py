from __future__ import annotations

from dataclasses import replace
import shutil
import subprocess
import tempfile
from pathlib import Path

from .svg_import import ImportedVectorDocument, import_svg_document, import_svg_string


_GENERIC_LABEL_PREFIXES = ("path", "shape", "layer")
_PDF_LIKE_CACHE: dict[
    str,
    tuple[
        tuple[int, int],
        str,
        list[dict[str, object]],
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ],
] = {}


def _looks_like_generic_label(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    return any(lowered.startswith(prefix) for prefix in _GENERIC_LABEL_PREFIXES)


def _polygon_contains_point(
    points: list[tuple[float, float]],
    x: float,
    y: float,
) -> bool:
    if len(points) < 3:
        return False
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < ((xj - xi) * (y - yi) / max((yj - yi), 1e-12)) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _shape_bounds(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [float(px) for px, _ in points]
    ys = [float(py) for _, py in points]
    return min(xs), min(ys), max(xs), max(ys)


def _distance_to_bounds(
    bounds: tuple[float, float, float, float],
    x: float,
    y: float,
) -> float:
    min_x, min_y, max_x, max_y = bounds
    dx = 0.0 if min_x <= x <= max_x else min(abs(x - min_x), abs(x - max_x))
    dy = 0.0 if min_y <= y <= max_y else min(abs(y - min_y), abs(y - max_y))
    return float((dx * dx + dy * dy) ** 0.5)


def _extract_pdf_text_labels(path: Path) -> list[dict[str, object]]:
    signature = _path_signature(path)
    cache_key = str(path.expanduser())
    if signature is not None:
        cached = _PDF_LIKE_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return list(cached[2])

    import fitz

    document = fitz.open(path)
    try:
        if document.page_count < 1:
            return []
        page = document.load_page(0)
        items: list[dict[str, object]] = []
        # Prefer line/span extraction because Illustrator/PDF labels often split
        # poorly in "words" mode (single-character fragments).
        text_dict = page.get_text("dict")
        blocks = (
            list(text_dict.get("blocks") or []) if isinstance(text_dict, dict) else []
        )
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if int(block.get("type", 0) or 0) != 0:
                continue
            for line in list(block.get("lines") or []):
                spans = list((line or {}).get("spans") or [])
                if not spans:
                    continue
                parts = []
                x0 = y0 = x1 = y1 = None
                for span in spans:
                    text = str((span or {}).get("text") or "").strip()
                    if not text:
                        continue
                    parts.append(text)
                    bbox = list((span or {}).get("bbox") or [])
                    if len(bbox) >= 4:
                        sx0, sy0, sx1, sy1 = map(float, bbox[:4])
                        x0 = sx0 if x0 is None else min(x0, sx0)
                        y0 = sy0 if y0 is None else min(y0, sy0)
                        x1 = sx1 if x1 is None else max(x1, sx1)
                        y1 = sy1 if y1 is None else max(y1, sy1)
                merged = " ".join(part for part in parts if part).strip()
                if not merged or x0 is None or y0 is None or x1 is None or y1 is None:
                    continue
                items.append(
                    {
                        "text": merged,
                        "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                        "bbox": (x0, y0, x1, y1),
                    }
                )
        if items:
            return items

        items: list[dict[str, object]] = []
        for word in page.get_text("words"):
            if len(word) < 5:
                continue
            x0, y0, x1, y1, text = word[:5]
            text = str(text or "").strip()
            if not text:
                continue
            items.append(
                {
                    "text": text,
                    "center": (
                        (float(x0) + float(x1)) / 2.0,
                        (float(y0) + float(y1)) / 2.0,
                    ),
                    "bbox": (float(x0), float(y0), float(x1), float(y1)),
                }
            )
        return items
    finally:
        document.close()


def _extract_pdf_page_box(path: Path) -> tuple[float, float, float, float] | None:
    signature = _path_signature(path)
    cache_key = str(path.expanduser())
    if signature is not None:
        cached = _PDF_LIKE_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[3]

    import fitz

    try:
        document = fitz.open(path)
    except Exception:
        return None
    try:
        if document.page_count < 1:
            return None
        page = document.load_page(0)
        rect = getattr(page, "rect", None) or page.bound()
        if rect is None:
            return None
        return (
            float(rect.x0),
            float(rect.y0),
            float(rect.x1),
            float(rect.y1),
        )
    finally:
        document.close()


def _extract_pdf_art_box(path: Path) -> tuple[float, float, float, float] | None:
    signature = _path_signature(path)
    cache_key = str(path.expanduser())
    if signature is not None:
        cached = _PDF_LIKE_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[4]

    import fitz

    try:
        document = fitz.open(path)
    except Exception:
        return None
    try:
        if document.page_count < 1:
            return None
        page = document.load_page(0)
        rect = getattr(page, "artbox", None)
        if rect is None:
            return None
        rect = rect or None
        if rect is None:
            return None
        return (
            float(rect.x0),
            float(rect.y0),
            float(rect.x1),
            float(rect.y1),
        )
    finally:
        document.close()


def _path_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return int(stat.st_mtime_ns), int(stat.st_size)


def _extract_pdf_text_labels_from_page(
    page,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    # Prefer line/span extraction because Illustrator/PDF labels often split
    # poorly in "words" mode (single-character fragments).
    text_dict = page.get_text("dict")
    blocks = list(text_dict.get("blocks") or []) if isinstance(text_dict, dict) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if int(block.get("type", 0) or 0) != 0:
            continue
        for line in list(block.get("lines") or []):
            spans = list((line or {}).get("spans") or [])
            if not spans:
                continue
            parts = []
            x0 = y0 = x1 = y1 = None
            for span in spans:
                text = str((span or {}).get("text") or "").strip()
                if not text:
                    continue
                parts.append(text)
                bbox = list((span or {}).get("bbox") or [])
                if len(bbox) >= 4:
                    sx0, sy0, sx1, sy1 = map(float, bbox[:4])
                    x0 = sx0 if x0 is None else min(x0, sx0)
                    y0 = sy0 if y0 is None else min(y0, sy0)
                    x1 = sx1 if x1 is None else max(x1, sx1)
                    y1 = sy1 if y1 is None else max(y1, sy1)
            merged = " ".join(part for part in parts if part).strip()
            if not merged or x0 is None or y0 is None or x1 is None or y1 is None:
                continue
            items.append(
                {
                    "text": merged,
                    "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                    "bbox": (x0, y0, x1, y1),
                }
            )
    if items:
        return items

    for word in page.get_text("words"):
        if len(word) < 5:
            continue
        x0, y0, x1, y1, text = word[:5]
        text = str(text or "").strip()
        if not text:
            continue
        items.append(
            {
                "text": text,
                "center": (
                    (float(x0) + float(x1)) / 2.0,
                    (float(y0) + float(y1)) / 2.0,
                ),
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
            }
        )
    return items


def _pdf_like_to_svg_and_labels_with_pymupdf(
    path: Path,
) -> tuple[
    str,
    list[dict[str, object]],
    tuple[float, float, float, float] | None,
]:
    import fitz

    document = fitz.open(path)
    try:
        if document.page_count < 1:
            raise ValueError(f"Vector document has no pages: {path}")
        page = document.load_page(0)
        svg_text = str(page.get_svg_image())
        text_items = _extract_pdf_text_labels_from_page(page)
        rect = getattr(page, "rect", None) or page.bound()
        page_box = (
            (
                float(rect.x0),
                float(rect.y0),
                float(rect.x1),
                float(rect.y1),
            )
            if rect is not None
            else None
        )
        signature = _path_signature(path)
        if signature is not None:
            _PDF_LIKE_CACHE[str(path.expanduser())] = (
                signature,
                svg_text,
                list(text_items),
                page_box,
                _extract_pdf_art_box(path),
            )
        return svg_text, text_items, page_box
    finally:
        document.close()


def _assign_text_labels_to_shapes(
    vector_document: ImportedVectorDocument,
    text_items: list[dict[str, object]],
) -> ImportedVectorDocument:
    if not text_items or not vector_document.shapes:
        return vector_document
    shapes = list(vector_document.shapes)
    candidate_indices = [
        index
        for index, shape in enumerate(shapes)
        if shape.kind != "text" and shape.points
    ]
    if not candidate_indices:
        return vector_document
    document_area = None
    if vector_document.view_box and len(vector_document.view_box) == 4:
        document_area = max(
            1.0, float(vector_document.view_box[2]) * float(vector_document.view_box[3])
        )
    elif vector_document.width and vector_document.height:
        document_area = max(
            1.0, float(vector_document.width) * float(vector_document.height)
        )

    assignments: dict[int, list[str]] = {}
    for text_item in text_items:
        text = str(text_item.get("text") or "").strip()
        center = tuple(text_item.get("center") or ())
        if not text or len(center) != 2:
            continue
        cx, cy = float(center[0]), float(center[1])
        containing: list[tuple[float, int]] = []
        nearby: list[tuple[float, int]] = []
        for index in candidate_indices:
            shape = shapes[index]
            bounds = _shape_bounds(shape.points)
            if bounds is None:
                continue
            width = max(1.0, bounds[2] - bounds[0])
            height = max(1.0, bounds[3] - bounds[1])
            area = width * height
            if shape.kind == "polygon" and _polygon_contains_point(
                shape.points, cx, cy
            ):
                containing.append((area, index))
                continue
            distance = _distance_to_bounds(bounds, cx, cy)
            threshold = max(12.0, min(width, height) * 0.35)
            if distance <= threshold:
                normalized = distance / max(1.0, min(width, height))
                nearby.append((normalized, index))
        chosen_index = None
        if containing:
            if document_area is not None:
                bounded = [
                    item for item in containing if item[0] <= document_area * 0.30
                ]
                if bounded:
                    containing = bounded
            containing.sort(key=lambda item: item[0])
            chosen_index = containing[0][1]
        elif nearby:
            nearby.sort(key=lambda item: item[0])
            chosen_index = nearby[0][1]
        if chosen_index is not None:
            assignments.setdefault(chosen_index, []).append(text)

    if not assignments:
        return vector_document

    updated_shapes = list(shapes)
    for index, labels in assignments.items():
        shape = updated_shapes[index]
        merged_label = "/".join(
            part for part in [str(item).strip() for item in labels] if part
        )
        if not merged_label:
            continue
        chosen_label = (
            merged_label
            if _looks_like_generic_label(shape.label)
            else str(shape.label or "").strip()
        )
        updated_shapes[index] = replace(shape, label=chosen_label, text=merged_label)
    return replace(vector_document, shapes=updated_shapes)


def _looks_like_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _pdf_to_svg_text_with_pymupdf(path: Path) -> str:
    svg_text, _, _ = _pdf_like_to_svg_and_labels_with_pymupdf(path)
    return svg_text


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
        document = import_svg_string(svg_text, source_path=str(resolved))
        page_box = _extract_pdf_page_box(resolved)
        art_box = _extract_pdf_art_box(resolved)
        document = replace(
            document,
            source_kind="ai" if suffix == ".ai" else "pdf",
            page_box=list(page_box) if page_box is not None else None,
            art_box=list(art_box) if art_box is not None else None,
        )
        try:
            text_items = _extract_pdf_text_labels(resolved)
        except Exception:
            text_items = []
        return _assign_text_labels_to_shapes(document, text_items)
    raise ValueError(f"Unsupported vector import format: {resolved.suffix}")
