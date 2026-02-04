import json
from pathlib import Path

from annolid.datasets.labelme_collection import (
    index_labelme_dataset,
    index_labelme_pair,
)


def _write_labelme_pair(
    folder: Path,
    *,
    stem: str,
    labeled: bool = True,
) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"{stem}.png"
    json_path = folder / f"{stem}.json"

    image_path.write_bytes(b"not-a-real-png")
    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": (
            [
                {
                    "label": "animal",
                    "points": [[0, 0], [1, 0], [1, 1]],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                }
            ]
            if labeled
            else []
        ),
        "imagePath": image_path.name,
        "imageHeight": 10,
        "imageWidth": 10,
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path, image_path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_index_labelme_pair_appends_absolute_paths(tmp_path):
    src_json, src_img = _write_labelme_pair(
        tmp_path / "session_a", stem="clip_000000000", labeled=True
    )
    index_file = tmp_path / "label_index.jsonl"

    record = index_labelme_pair(
        json_path=src_json,
        index_file=index_file,
        image_path=src_img,
        source="test",
    )
    assert record is not None
    assert index_file.exists()

    records = _read_jsonl(index_file)
    assert len(records) == 1
    assert records[0]["image_path"] == str(src_img.resolve())
    assert records[0]["json_path"] == str(src_json.resolve())
    assert records[0]["shapes_count"] == 1


def test_index_labelme_pair_skips_empty_shapes(tmp_path):
    src_json, src_img = _write_labelme_pair(
        tmp_path / "session_b", stem="clip_000000001", labeled=False
    )
    index_file = tmp_path / "label_index.jsonl"

    record = index_labelme_pair(
        json_path=src_json,
        index_file=index_file,
        image_path=src_img,
        include_empty=False,
        source="test",
    )
    assert record is None
    assert not index_file.exists()


def test_index_labelme_dataset_dedupes_by_image_path(tmp_path):
    _write_labelme_pair(tmp_path / "s1", stem="a_000000000", labeled=True)
    _write_labelme_pair(tmp_path / "s2", stem="b_000000000", labeled=True)
    index_file = tmp_path / "label_index.jsonl"

    summary1 = index_labelme_dataset([tmp_path], index_file=index_file, dedupe=True)
    assert summary1["appended"] == 2

    summary2 = index_labelme_dataset([tmp_path], index_file=index_file, dedupe=True)
    assert summary2["appended"] == 0
    assert summary2["skipped"] >= 2
