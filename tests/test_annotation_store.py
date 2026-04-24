import json

from annolid.utils.annotation_store import AnnotationStore, load_labelme_json


def _append_dummy_record(store: AnnotationStore, frame: int) -> None:
    store.append_frame(
        {
            "frame": frame,
            "shapes": [],
            "flags": {},
            "imagePath": "",
            "imageHeight": 1,
            "imageWidth": 1,
        }
    )


def test_remove_frames_after_prunes_future_frames(tmp_path):
    frame_path = tmp_path / "clip" / "clip_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)

    for frame in range(5):
        _append_dummy_record(store, frame)

    removed = store.remove_frames_after(2)
    assert removed == 2
    assert set(store.iter_frames()) == {0, 1, 2}


def test_remove_frames_after_respects_protected_frames(tmp_path):
    frame_path = tmp_path / "video" / "video_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)

    for frame in range(7):
        _append_dummy_record(store, frame)

    removed = store.remove_frames_after(3, protected_frames={5})
    assert removed == 2  # frames 4 and 6 removed, 5 preserved
    assert set(store.iter_frames()) == {0, 1, 2, 3, 5}


def test_legacy_store_rows_skip_manual_seed_frames(tmp_path):
    results_dir = tmp_path / "mouse"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "mouse_000000000.png").write_bytes(b"")
    (results_dir / "mouse_000000000.json").write_text(
        json.dumps({"version": "annolid", "shapes": []}),
        encoding="utf-8",
    )
    store = AnnotationStore(results_dir / "mouse_annotations.ndjson")
    store.store_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"shapes": [{"label": "box"}], "imageHeight": 1, "imageWidth": 1}
                ),
                json.dumps(
                    {"shapes": [{"label": "box"}], "imageHeight": 1, "imageWidth": 1}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert set(store.iter_frames()) == {1, 2}
    assert store.get_frame(0) is None
    assert store.get_frame(1)["shapes"][0]["label"] == "box"
    assert store.get_frame(2)["shapes"][0]["label"] == "box"
    migrated_rows = [
        json.loads(line)
        for line in store.store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["frame"] for row in migrated_rows] == [1, 2]


def test_update_frame_rewrites_legacy_store_using_inferred_frames(tmp_path):
    results_dir = tmp_path / "mouse"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "mouse_000000000.png").write_bytes(b"")
    (results_dir / "mouse_000000000.json").write_text(
        json.dumps({"version": "annolid", "shapes": []}),
        encoding="utf-8",
    )
    store = AnnotationStore(results_dir / "mouse_annotations.ndjson")
    store.store_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "shapes": [{"label": "teaball"}],
                        "imageHeight": 1,
                        "imageWidth": 1,
                    }
                ),
                json.dumps(
                    {"shapes": [{"label": "mouse"}], "imageHeight": 1, "imageWidth": 1}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store.update_frame(
        2,
        {
            "version": "annolid",
            "flags": {},
            "shapes": [{"label": "box"}],
            "imagePath": None,
            "imageData": None,
            "imageHeight": 1,
            "imageWidth": 1,
            "caption": None,
            "otherData": {},
        },
    )

    rows = [
        json.loads(line)
        for line in store.store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["frame"] for row in rows] == [1, 2]
    assert rows[0]["shapes"][0]["label"] == "teaball"
    assert rows[1]["shapes"][0]["label"] == "box"


def test_update_frames_rewrites_multiple_records_once(tmp_path, monkeypatch):
    frame_path = tmp_path / "video" / "video_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)
    for frame in range(4):
        _append_dummy_record(store, frame)

    writes = {"count": 0}
    original_write_lines = store._write_lines_atomically

    def _counting_write_lines(lines):
        writes["count"] += 1
        original_write_lines(lines)

    monkeypatch.setattr(store, "_write_lines_atomically", _counting_write_lines)

    store.update_frames(
        {
            1: {"version": "annolid", "shapes": [{"label": "a"}]},
            3: {"version": "annolid", "shapes": [{"label": "b"}]},
        }
    )

    assert writes["count"] == 1
    assert store.get_frame(1)["frame"] == 1
    assert store.get_frame(1)["shapes"][0]["label"] == "a"
    assert store.get_frame(3)["frame"] == 3
    assert store.get_frame(3)["shapes"][0]["label"] == "b"


def test_get_frame_uses_cache_without_rerunning_migration(tmp_path, monkeypatch):
    frame_path = tmp_path / "video" / "video_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)
    store.store_path.write_text(
        json.dumps(
            {
                "frame": 0,
                "shapes": [{"label": "mouse"}],
                "imageHeight": 1,
                "imageWidth": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls = {"count": 0}

    def _count_migration():
        calls["count"] += 1

    monkeypatch.setattr(store, "_ensure_explicit_frame_metadata", _count_migration)

    assert store.get_frame(0)["shapes"][0]["label"] == "mouse"
    assert store.get_frame(0)["shapes"][0]["label"] == "mouse"
    assert calls["count"] == 1


def test_legacy_store_migration_falls_back_to_memory_on_permission_error(
    tmp_path, monkeypatch
):
    frame_path = tmp_path / "video" / "video_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)
    store.store_path.write_text(
        json.dumps(
            {
                "shapes": [{"label": "mouse"}],
                "imageHeight": 1,
                "imageWidth": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _deny_replace(_lines):
        raise PermissionError("locked")

    monkeypatch.setattr(store, "_write_lines_atomically", _deny_replace)

    record = store.get_frame(0)
    assert record is not None
    assert record["frame"] == 0
    assert record["shapes"][0]["label"] == "mouse"


def test_load_labelme_json_prefers_fast_single_frame_lookup(tmp_path, monkeypatch):
    frame_path = tmp_path / "video" / "video_000000007.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)
    for frame in range(20):
        _append_dummy_record(store, frame)

    def _fail_full_parse(_frame):
        raise AssertionError("Full-store get_frame should not be required here.")

    monkeypatch.setattr(AnnotationStore, "get_frame", _fail_full_parse)

    payload = load_labelme_json(frame_path)
    assert isinstance(payload, dict)
    assert payload.get("shapes") == []
    assert payload.get("imageHeight") == 1
    assert payload.get("imageWidth") == 1


def test_get_frames_fast_reads_many_records_in_one_scan(tmp_path, monkeypatch):
    frame_path = tmp_path / "video" / "video_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)
    for frame in range(20):
        _append_dummy_record(store, frame)

    def _fail_full_parse(_force_reload=False):
        raise AssertionError("Bulk lookup should not require full-store parsing.")

    monkeypatch.setattr(store, "_load_records", _fail_full_parse)

    records = store.get_frames_fast([3, 7, 12])

    assert sorted(records) == [3, 7, 12]
    assert all(record["imageHeight"] == 1 for record in records.values())


def test_append_frame_handles_fast_scan_only_cache_state(tmp_path):
    frame_path = tmp_path / "video" / "video_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)
    _append_dummy_record(store, 0)

    # Prime fast-scan cache without building full `records` cache.
    assert store.get_frame_fast(0) is not None

    # Appending should not fail even when cache entry lacks a `records` dict.
    _append_dummy_record(store, 1)
    assert store.get_frame_fast(1) is not None


def test_get_frame_handles_fast_scan_only_cache_state(tmp_path):
    frame_path = tmp_path / "video" / "video_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)
    _append_dummy_record(store, 0)

    # Prime fast-scan cache without building full `records` cache.
    assert store.get_frame_fast(0) is not None
    cached = AnnotationStore._CACHE.get(store.store_path) or {}
    assert "records" not in cached

    # Full lookup should gracefully treat this as stale cache and reload records.
    loaded = store.get_frame(0)
    assert loaded is not None
    assert loaded.get("frame") == 0


def test_append_after_fast_scan_keeps_full_get_frame_consistent(tmp_path):
    frame_path = tmp_path / "video" / "video_000000000.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore.for_frame_path(frame_path)
    _append_dummy_record(store, 0)

    # Prime the fast-scan cache shape first.
    assert store.get_frame_fast(0) is not None

    # Append another frame after fast-scan cache has been created.
    _append_dummy_record(store, 1)

    # Full get_frame path must still return both frames, not just the appended one.
    frame0 = store.get_frame(0)
    frame1 = store.get_frame(1)
    assert frame0 is not None and int(frame0.get("frame")) == 0
    assert frame1 is not None and int(frame1.get("frame")) == 1
