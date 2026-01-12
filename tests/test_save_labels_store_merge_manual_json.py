import json
from pathlib import Path

from annolid.annotation.keypoints import save_labels
from annolid.gui.shape import Shape
from annolid.utils.annotation_store import AnnotationStore


def test_save_labels_writes_store_even_when_manual_json_exists(tmp_path: Path) -> None:
    # Create a "manually labeled" frame: PNG + JSON exist.
    frame_stem = f"{tmp_path.name}_000000000"
    png_path = tmp_path / f"{frame_stem}.png"
    json_path = tmp_path / f"{frame_stem}.json"

    png_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal header; content not used
    manual_payload = {
        "version": "5.5.0",
        "flags": {},
        "imagePath": str(png_path),
        "imageHeight": 64,
        "imageWidth": 96,
        "shapes": [
            {
                "label": "resident",
                "points": [[5, 5], [45, 5], [45, 45], [5, 45]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "visible": True,
                "description": "manual",
            }
        ],
    }
    json_text = json.dumps(manual_payload, separators=(",", ":"))
    json_path.write_text(json_text, encoding="utf-8")

    # Write a prediction record to the store only (persist_json=False) without overwriting the manual JSON.
    pt = Shape(label="nose", shape_type="point", flags={"score": 0.9})
    pt.group_id = 0
    pt.points = [[20.0, 20.0]]
    save_labels(
        filename=str(json_path),
        imagePath=str(png_path),
        label_list=[pt],
        height=64,
        width=96,
        save_image_to_json=False,
        persist_json=False,
    )

    store = AnnotationStore.for_frame_path(json_path)
    rec = store.get_frame(0)
    assert isinstance(rec, dict)
    shapes = rec.get("shapes") or []
    assert isinstance(shapes, list)
    # Polygon from the manual JSON is preserved, point prediction is added.
    assert any(s.get("shape_type") == "polygon" and s.get("label") == "resident" for s in shapes)
    assert any(s.get("shape_type") == "point" and s.get("label") == "nose" and s.get("group_id") == 0 for s in shapes)

    # Manual JSON should remain unchanged (no overwrite when persist_json=False).
    assert json_path.read_text(encoding="utf-8") == json_text

