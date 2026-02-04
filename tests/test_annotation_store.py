from annolid.utils.annotation_store import AnnotationStore


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
