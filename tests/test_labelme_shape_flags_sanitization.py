import json
from pathlib import Path

from annolid.annotation.keypoints import save_labels
from annolid.gui.shape import Shape


def test_save_labels_sanitizes_shape_flags_for_labelme(tmp_path: Path) -> None:
    json_path = tmp_path / f"{tmp_path.name}_000000000.json"

    pt = Shape(
        label="nose",
        shape_type="point",
        flags={
            "score": 0.9,
            "instance_id": 3,
            "kp_visibility": 1,
            "kp_visible": False,
            "keep": True,
        },
    )
    pt.group_id = 3
    pt.points = [[20.0, 20.0]]

    save_labels(
        filename=str(json_path),
        imagePath="",
        label_list=[pt],
        height=64,
        width=96,
        flags={"trial_ok": "yes", "confidence": 0.25},
        persist_json=True,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = payload.get("shapes") or []
    assert isinstance(shapes, list) and len(shapes) == 1
    shape0 = shapes[0]
    assert shape0.get("label") == "nose"
    assert shape0.get("flags") == {"keep": True, "kp_visible": False}
    # Non-bool metadata is preserved outside of `flags`.
    assert shape0.get("score") == 0.9
    assert shape0.get("instance_id") == 3
    assert shape0.get("kp_visibility") == 1

    # Image-level flags are boolean-only; unparseable entries are preserved in meta.
    assert payload.get("flags") == {"trial_ok": True}
    assert payload.get("annolid_flags_meta") == {"confidence": 0.25}

