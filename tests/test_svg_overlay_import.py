from __future__ import annotations

from pathlib import Path

from annolid.gui.svg_overlay import flatten_svg_path, import_svg_shapes
from annolid.io.vector import import_vector_document
from annolid.io.vector.svg_import import ImportedVectorDocument, import_svg_document


def test_flatten_svg_path_samples_curves() -> None:
    points = flatten_svg_path("M 0 0 C 10 0, 10 10, 20 10 Z")
    assert len(points) > 4
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (0.0, 0.0)


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
