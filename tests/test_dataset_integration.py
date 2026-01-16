from pathlib import Path
import json
from annolid.utils.files import find_manual_labeled_json_files
from annolid.annotation.labelme2yolo import Labelme2YOLO


def create_pair(folder: Path, stem: str, has_json=True, has_png=True):
    if has_json:
        (folder / f"{stem}.json").write_text(
            json.dumps({"imagePath": f"{stem}.png"}), encoding="utf-8"
        )
    if has_png:
        (folder / f"{stem}.png").write_bytes(b"")


def test_find_manual_labeled_json_files_only_returns_valid_pairs(tmp_path):
    # Setup
    folder = tmp_path / "data"
    folder.mkdir()

    # 1. Valid pair
    create_pair(folder, "valid_01")

    # 2. Orphaned JSON (no image)
    create_pair(folder, "orphan_json", has_png=False)

    # 3. Orphaned PNG (no JSON)
    create_pair(folder, "orphan_png", has_json=False)

    # 4. JSON with sidecar image mismatch (filename based)
    # The unified logic should handle files that exist.
    # resolve_image_path prioritizes finding an existing image.

    files = find_manual_labeled_json_files(str(folder))

    assert "valid_01.json" in files
    assert "orphan_json.json" not in files
    assert len(files) == 1


def test_labelme2yolo_uses_unified_resolver(tmp_path):
    # Setup
    folder = tmp_path / "yolo_test"
    folder.mkdir()

    # Create a JSON that points to a non-existent absolute path in 'imagePath'
    # but has a valid sidecar image. Unified logic should find the sidecar.
    p = folder / "test_sidecar.json"
    p.write_text(
        json.dumps({"imagePath": "/non/existent/path.png", "shapes": []}),
        encoding="utf-8",
    )
    (folder / "test_sidecar.png").write_bytes(b"")

    converter = Labelme2YOLO(str(folder), recursive=False)
    items = converter._discover_items()

    assert len(items) == 1
    assert items[0].image_path is not None
    assert items[0].image_path.name == "test_sidecar.png"
