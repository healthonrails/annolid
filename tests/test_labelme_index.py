import json
from threading import Event
from pathlib import Path

from annolid.datasets.labelme_collection import (
    DEFAULT_LABEL_INDEX_NAME,
    default_annolid_logs_root,
    index_labelme_dataset,
    index_labelme_pair,
    resolve_label_index_path,
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


def test_index_labelme_dataset_reports_skip_reasons(tmp_path):
    # one valid
    _write_labelme_pair(tmp_path / "s1", stem="ok_a", labeled=True)
    # one empty
    _write_labelme_pair(tmp_path / "s1", stem="empty_a", labeled=False)
    # one missing sidecar image
    missing_json = tmp_path / "s1" / "missing_a.json"
    missing_json.write_text(
        json.dumps(
            {
                "version": "5.5.0",
                "flags": {},
                "shapes": [],
                "imagePath": "missing_a.png",
            }
        ),
        encoding="utf-8",
    )
    # one invalid JSON with sidecar image
    bad_json = tmp_path / "s1" / "bad_a.json"
    bad_img = tmp_path / "s1" / "bad_a.png"
    bad_img.write_bytes(b"not-a-real-png")
    bad_json.write_text("{invalid", encoding="utf-8")
    index_file = tmp_path / "label_index.jsonl"

    summary = index_labelme_dataset(
        [tmp_path],
        index_file=index_file,
        include_empty=False,
        dedupe=False,
        event_sample_limit=10,
    )
    assert summary["appended"] == 1
    assert summary["missing_image"] == 1
    assert summary["skip_reasons"]["empty_labels"] == 1
    assert summary["skip_reasons"]["missing_image"] == 1
    assert summary["skip_reasons"]["invalid_json"] == 1
    assert summary["total_json"] == 4
    assert summary["processed"] == 4
    assert isinstance(summary["events_sample"], list)
    assert len(summary["events_sample"]) == 4


def test_index_labelme_dataset_supports_progress_and_stop(tmp_path):
    for i in range(5):
        _write_labelme_pair(tmp_path / "src", stem=f"item_{i:02d}", labeled=True)
    index_file = tmp_path / "label_index.jsonl"
    stop_event = Event()
    updates: list[dict] = []

    def _on_progress(payload: dict) -> None:
        updates.append(dict(payload))
        if int(payload.get("processed", 0)) >= 2:
            stop_event.set()

    summary = index_labelme_dataset(
        [tmp_path / "src"],
        index_file=index_file,
        dedupe=False,
        progress_callback=_on_progress,
        stop_event=stop_event,
    )
    assert summary["stopped"] is True
    assert int(summary["processed"]) >= 2
    assert int(summary["processed"]) < 5
    assert updates


def test_resolve_label_index_path_anchors_under_logs_layout(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    logs_root = default_annolid_logs_root(dataset_root)
    assert (
        resolve_label_index_path(
            Path("annolid_logs/annolid_dataset.jsonl"), dataset_root
        )
        == (logs_root / "label_index" / "annolid_dataset.jsonl").resolve()
    )
    assert (
        resolve_label_index_path(Path("custom_name.jsonl"), dataset_root)
        == (logs_root / "label_index" / "custom_name.jsonl").resolve()
    )
    assert (
        resolve_label_index_path(Path("nested"), dataset_root)
        == (logs_root / "nested" / DEFAULT_LABEL_INDEX_NAME).resolve()
    )
