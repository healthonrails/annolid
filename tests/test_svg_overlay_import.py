from __future__ import annotations

from pathlib import Path
import sys
import types

from annolid.gui.svg_overlay import flatten_svg_path, import_svg_shapes
from annolid.gui.svg_overlay import import_vector_shapes
from annolid.io.vector import import_vector_document
from annolid.io.vector.svg_import import (
    ImportedPath,
    ImportedVectorDocument,
    import_svg_document,
)
from annolid.io.vector.document_import import _assign_text_labels_to_shapes
from annolid.io.vector.document_import import _extract_pdf_text_labels


def test_flatten_svg_path_samples_curves() -> None:
    points = flatten_svg_path("M 0 0 C 10 0, 10 10, 20 10 Z")
    assert len(points) > 4
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (0.0, 0.0)


def test_flatten_svg_path_simplifies_dense_curve_samples() -> None:
    points = flatten_svg_path("M 0 0 C 10 0, 20 10, 30 10")
    assert len(points) < 25
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (30.0, 10.0)


def test_import_svg_shapes_preserves_source_metadata(tmp_path: Path) -> None:
    svg_path = tmp_path / "atlas.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <g id="brain">
            <path id="region_a" d="M 10 10 L 40 10 L 40 40 L 10 40 Z" />
          </g>
          <circle id="landmark" cx="70" cy="70" r="10" />
        </svg>
        """,
        encoding="utf-8",
    )

    result = import_svg_shapes(svg_path)

    assert len(result.shapes) == 2
    labels = {shape.label for shape in result.shapes}
    assert "region_a" in labels
    assert "landmark" in labels
    for shape in result.shapes:
        assert shape.other_data["overlay_source"] == str(svg_path)
        assert shape.other_data["overlay_id"] == result.metadata["id"]
        assert shape.other_data["overlay_visible"] is True
        assert len(shape.other_data["overlay_transform"]) == 9
    assert result.metadata["view_box"] == [0.0, 0.0, 100.0, 100.0]
    assert result.metadata["shape_count"] == 2
    assert result.metadata["transform"]["opacity"] == 0.5


def test_import_svg_skips_hidden_elements(tmp_path: Path) -> None:
    svg_path = tmp_path / "hidden.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg">
          <path id="visible" d="M 0 0 L 10 0 L 10 10 Z" />
          <path id="hidden_a" style="display:none" d="M 0 0 L 2 0 L 2 2 Z" />
          <g visibility="hidden">
            <path id="hidden_b" d="M 0 0 L 3 0 L 3 3 Z" />
          </g>
        </svg>
        """,
        encoding="utf-8",
    )

    result = import_svg_shapes(svg_path)

    assert [shape.label for shape in result.shapes] == ["visible"]


def test_import_svg_document_preserves_text_and_style(tmp_path: Path) -> None:
    svg_path = tmp_path / "labels.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="layer_a">
            <polygon id="region" points="0,0 10,0 10,10" fill="#ff0000" stroke="#00ff00" />
            <text id="label" x="12" y="14" fill="#123456">Area A</text>
          </g>
        </svg>
        """,
        encoding="utf-8",
    )

    document = import_svg_document(svg_path)

    assert len(document.shapes) == 2
    polygon = next(item for item in document.shapes if item.id == "region")
    text = next(item for item in document.shapes if item.id == "label")
    assert polygon.fill == "#ff0000"
    assert polygon.stroke == "#00ff00"
    assert polygon.layer_name == "layer_a"
    assert text.kind == "text"
    assert text.text == "Area A"


def test_import_vector_document_dispatches_pdf_compatible_ai(
    monkeypatch, tmp_path: Path
) -> None:
    ai_path = tmp_path / "atlas.ai"
    ai_path.write_bytes(b"%PDF-1.5\n%fake\n")

    monkeypatch.setattr(
        "annolid.io.vector.document_import._pdf_like_to_svg_text",
        lambda path: "<svg xmlns='http://www.w3.org/2000/svg'><path id='region' d='M 0 0 L 10 0 L 10 10 Z'/></svg>",
    )

    document = import_vector_document(ai_path)

    assert isinstance(document, ImportedVectorDocument)
    assert document.source_path == str(ai_path)
    assert len(document.shapes) == 1
    assert document.shapes[0].id == "region"


def test_assign_text_labels_to_shapes_prefers_containing_polygon() -> None:
    document = ImportedVectorDocument(
        source_path="atlas.ai",
        width=100.0,
        height=100.0,
        view_box=[0.0, 0.0, 100.0, 100.0],
        shapes=[
            ImportedPath(
                id="path1",
                label="path1",
                kind="polygon",
                points=[(0.0, 0.0), (90.0, 0.0), (90.0, 90.0), (0.0, 90.0)],
            ),
            ImportedPath(
                id="path2",
                label="path2",
                kind="polygon",
                points=[(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)],
            ),
        ],
    )
    updated = _assign_text_labels_to_shapes(
        document,
        [{"text": "AreaA", "center": (20.0, 20.0), "bbox": (18.0, 18.0, 22.0, 22.0)}],
    )
    assert updated.shapes[0].label == "path1"
    assert updated.shapes[0].text is None
    assert updated.shapes[1].label == "AreaA"
    assert updated.shapes[1].text == "AreaA"


def test_import_vector_document_uses_extracted_pdf_text_as_shape_labels(
    monkeypatch, tmp_path: Path
) -> None:
    ai_path = tmp_path / "atlas.ai"
    ai_path.write_bytes(b"%PDF-1.5\n%fake\n")

    monkeypatch.setattr(
        "annolid.io.vector.document_import._pdf_like_to_svg_text",
        lambda path: """
        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>
          <path id='path1' d='M 0 0 L 80 0 L 80 80 L 0 80 Z'/>
          <path id='path2' d='M 10 10 L 30 10 L 30 30 L 10 30 Z'/>
        </svg>
        """,
    )
    monkeypatch.setattr(
        "annolid.io.vector.document_import._extract_pdf_text_labels",
        lambda path: [
            {"text": "Hp", "center": (20.0, 20.0), "bbox": (18.0, 18.0, 22.0, 22.0)}
        ],
    )

    document = import_vector_document(ai_path)

    labels = {shape.id: shape.label for shape in document.shapes}
    texts = {shape.id: shape.text for shape in document.shapes}
    assert labels["path1"] == "path1"
    assert labels["path2"] == "Hp"
    assert texts["path2"] == "Hp"


def test_import_vector_shapes_preserves_assigned_overlay_text(
    monkeypatch, tmp_path: Path
) -> None:
    svg_path = tmp_path / "atlas.svg"
    svg_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' />", encoding="utf-8")
    monkeypatch.setattr(
        "annolid.gui.svg_overlay.import_vector_document",
        lambda path: ImportedVectorDocument(
            source_path=str(path),
            width=100.0,
            height=100.0,
            view_box=[0.0, 0.0, 100.0, 100.0],
            shapes=[
                ImportedPath(
                    id="path2",
                    label="Hp",
                    kind="polygon",
                    points=[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)],
                    text="Hp",
                )
            ],
        ),
    )

    result = import_vector_shapes(svg_path)

    assert result.shapes[0].label == "Hp"
    assert result.shapes[0].other_data["overlay_text"] == "Hp"


def test_import_vector_shapes_disambiguates_duplicate_labels(
    monkeypatch, tmp_path: Path
) -> None:
    svg_path = tmp_path / "atlas.svg"
    svg_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' />", encoding="utf-8")
    monkeypatch.setattr(
        "annolid.gui.svg_overlay.import_vector_document",
        lambda path: ImportedVectorDocument(
            source_path=str(path),
            width=100.0,
            height=100.0,
            view_box=[0.0, 0.0, 100.0, 100.0],
            shapes=[
                ImportedPath(
                    id="shape_0",
                    label="Layer 1",
                    kind="polygon",
                    points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                ),
                ImportedPath(
                    id="shape_1",
                    label="Layer 1",
                    kind="polygon",
                    points=[(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)],
                ),
            ],
        ),
    )

    result = import_vector_shapes(svg_path)

    labels = [shape.label for shape in result.shapes]
    assert labels == ["Layer 1_1", "Layer 1_2"]


def test_import_vector_shapes_marks_bbox_only_overlay_for_ai(
    monkeypatch, tmp_path: Path
) -> None:
    ai_path = tmp_path / "atlas.ai"
    ai_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "annolid.gui.svg_overlay.import_vector_document",
        lambda path: ImportedVectorDocument(
            source_path=str(path),
            width=400.0,
            height=100.0,
            view_box=[0.0, 0.0, 400.0, 100.0],
            source_kind="ai",
            shapes=[
                ImportedPath(
                    id="shape_0",
                    label="Layer 1",
                    kind="polygon",
                    points=[
                        (0.0, 0.0),
                        (100.0, 0.0),
                        (100.0, 40.0),
                        (0.0, 40.0),
                        (0.0, 0.0),
                    ],
                ),
                ImportedPath(
                    id="shape_1",
                    label="Layer 1",
                    kind="polygon",
                    points=[
                        (120.0, 0.0),
                        (200.0, 0.0),
                        (200.0, 40.0),
                        (120.0, 40.0),
                        (120.0, 0.0),
                    ],
                ),
            ],
        ),
    )

    result = import_vector_shapes(ai_path)

    assert result.metadata["bbox_only_overlay"] is True
    assert "rectangular" in str(result.metadata.get("import_warning") or "").lower()


def test_import_svg_skips_defs_and_clip_paths(tmp_path: Path) -> None:
    svg_path = tmp_path / "clip.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="clipA">
              <path id="clip_shape" d="M 0 0 L 10 0 L 10 10 Z" />
            </clipPath>
          </defs>
          <g id="layer_a">
            <path id="visible_shape" d="M 20 20 L 40 20 L 40 40 Z" />
          </g>
        </svg>
        """,
        encoding="utf-8",
    )

    result = import_svg_shapes(svg_path)

    assert [shape.label for shape in result.shapes] == ["visible_shape"]


def test_import_svg_open_path_uses_linestrip_shape_type(tmp_path: Path) -> None:
    svg_path = tmp_path / "open_path.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg">
          <path id="open_curve" d="M 0 0 L 10 0 L 10 10" />
        </svg>
        """,
        encoding="utf-8",
    )

    result = import_svg_shapes(svg_path)

    assert len(result.shapes) == 1
    assert result.shapes[0].shape_type == "linestrip"


def test_extract_pdf_text_labels_prefers_line_spans(
    monkeypatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "atlas.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    class _FakePage:
        def get_text(self, mode):
            if mode == "dict":
                return {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [
                                {
                                    "spans": [
                                        {"text": "Area", "bbox": [10, 20, 40, 30]},
                                        {"text": "A", "bbox": [42, 20, 52, 30]},
                                    ]
                                }
                            ],
                        }
                    ]
                }
            if mode == "words":
                return [(10, 20, 20, 30, "A"), (21, 20, 30, 30, "B")]
            return {}

    class _FakeDoc:
        page_count = 1

        def load_page(self, _idx):
            return _FakePage()

        def close(self):
            return None

    fake_fitz = types.SimpleNamespace(open=lambda _path: _FakeDoc())
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    items = _extract_pdf_text_labels(pdf_path)

    assert len(items) == 1
    assert items[0]["text"] == "Area A"
    assert tuple(items[0]["bbox"]) == (10.0, 20.0, 52.0, 30.0)
