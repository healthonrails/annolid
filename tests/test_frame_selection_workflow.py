"""Comprehensive tests for embedding-aware frame selection, queue-vs-extract workflows, and GUI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ANNOLID_DISABLE_BOT_AUTOSTART", "1")

from annolid.data.frame_sampling import (
    discover_labeled_frame_indices,
    extract_frames_smart,
    extract_pose_outlier_frames,
    select_farthest_point_indices,
    select_flow_frame_indices,
    select_hybrid_frame_indices,
)
from annolid.engine import cli as engine_cli
from annolid.infrastructure import AnnotationStore
from annolid.services.embedding_search import (
    FrameEmbeddingBackend,
    VideoFrameEmbeddingIndex,
    default_video_index_cache_path,
)
from annolid.services.frame_selection import (
    FrameSelectionConfig,
    build_selection_result_from_indices,
    find_cached_video_embedding_indices,
    list_available_feature_backends,
    load_frame_selection_queue,
    materialize_selected_frames,
    save_frame_selection_queue,
    select_frames_for_labeling,
)


def _write_multi_state_video(
    path: Path,
    *,
    num_frames: int = 45,
    width: int = 160,
    height: int = 120,
    fps: float = 30.0,
    inject_dark_frames: Sequence[int] = (),
) -> Path:
    """Create a deterministic fixed-arena video with distinct moving-animal postures."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open synthetic video writer at {path}"
    dark_set = set(inject_dark_frames)
    try:
        for idx in range(num_frames):
            if idx in dark_set:
                writer.write(np.zeros((height, width, 3), dtype=np.uint8))
                continue
            # Fixed static textured arena background.
            frame = np.full((height, width, 3), (65, 70, 68), dtype=np.uint8)
            cv2.rectangle(frame, (8, 8), (width - 9, height - 9), (120, 120, 120), 2)
            # Three distinct visual/posture states plus one rare extreme state at frames 40..44.
            if idx < 15:
                cx = 35 + (idx % 5) * 2
                cv2.ellipse(frame, (cx, 40), (18, 9), 0, 0, 360, (235, 225, 210), -1)
                cv2.circle(frame, (cx + 16, 40), 5, (40, 40, 220), -1)
            elif idx < 30:
                cx = 115
                cy = 40 + (idx % 5) * 2
                cv2.ellipse(frame, (cx, cy), (9, 20), 0, 0, 360, (220, 240, 180), -1)
                cv2.circle(frame, (cx, cy - 18), 5, (30, 220, 30), -1)
            elif idx < 40:
                cx = 80
                cy = 88
                cv2.ellipse(frame, (cx, cy), (22, 10), 25, 0, 360, (240, 185, 90), -1)
                cv2.circle(frame, (cx - 18, cy - 6), 5, (230, 50, 210), -1)
            else:
                # Rare posture in bottom-left corner with bright rearing silhouette.
                cv2.rectangle(frame, (18, 70), (54, 108), (255, 255, 255), -1)
                cv2.line(frame, (20, 72), (52, 106), (20, 20, 240), 3)
            writer.write(frame)
    finally:
        writer.release()
    return path


class _DeterministicFakeEmbeddingBackend(FrameEmbeddingBackend):
    """Lightweight deterministic multimodal embedding backend for unit tests."""

    supports_text: bool = True
    supports_video_clip: bool = False

    def __init__(self) -> None:
        self.embed_calls = 0

    @property
    def key(self):
        return ("embeddinggemma2_onnx", "fake-model", "q4", "8", "70", "70", "")

    def embed_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        self.embed_calls += 1
        # Encode spatial quadrant mean intensities into an 8-D L2-normalized vector.
        h, w = frame_rgb.shape[:2]
        mh, mw = max(1, h // 2), max(1, w // 2)
        quads = [
            frame_rgb[:mh, :mw],
            frame_rgb[:mh, mw:],
            frame_rgb[mh:, :mw],
            frame_rgb[mh:, mw:],
        ]
        vals = []
        for q in quads:
            vals.append(float(np.mean(q[..., 0])) / 255.0)
            vals.append(float(np.mean(q[..., 2])) / 255.0)
        vec = np.asarray(vals, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else vec

    def embed_text(self, text: str, *, prompt_name=None) -> np.ndarray:
        _ = prompt_name
        # Aim directly at the bottom-left rare rearing quadrant (quad 2 -> indices 4, 5).
        if "rear" in text.lower():
            vec = np.asarray(
                [0.05, 0.05, 0.05, 0.05, 1.0, 1.0, 0.05, 0.05], dtype=np.float32
            )
        else:
            vec = np.asarray([1.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        return vec / float(np.linalg.norm(vec))


def _ensure_qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# ---------------------------------------------------------------------------
# 1. Core Selection Algorithms (Farthest Coreset, Hybrid, Flow, Dual Storage)
# ---------------------------------------------------------------------------


def test_farthest_and_hybrid_respect_seeds_and_min_gap() -> None:
    # Build 4 clear clusters in 4-D unit space (10 points each).
    rng = np.random.default_rng(0)
    bases = np.eye(4, dtype=np.float32)
    feats = np.vstack([bases[i] + rng.normal(0, 0.02, size=(10, 4)) for i in range(4)])
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    frame_indices = list(range(40))

    # If cluster 0 is already labeled (seed_features), farthest-point should pick clusters 1, 2, 3 first!
    picked = select_farthest_point_indices(
        feats,
        3,
        frame_indices=frame_indices,
        seed_features=feats[:5],
        min_frame_gap=5,
    )
    picked_clusters = {r // 10 for r in picked}
    assert picked_clusters == {1, 2, 3}
    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            assert abs(picked[i] - picked[j]) >= 5

    # Hybrid selection covers clusters AND assigns representative + diverse_extreme roles.
    h_rows, roles, labels = select_hybrid_frame_indices(
        feats,
        6,
        frame_indices=frame_indices,
        min_frame_gap=3,
        representative_ratio=0.5,
        include_start_boundary=True,
        seed=42,
    )
    assert len(h_rows) == 6
    assert 0 in h_rows
    assert roles[0] == "boundary"
    assert "representative" in set(roles.values())
    assert "diverse_extreme" in set(roles.values())
    assert labels.shape == (40,)


def test_dual_storage_labeled_discovery_and_exclusion(tmp_path: Path) -> None:
    vid = _write_multi_state_video(tmp_path / "arena.mp4", num_frames=45)
    ann_dir = tmp_path / "arena"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Frame 5 labeled as per-frame disk LabelMe JSON.
    (ann_dir / "arena_000000005.json").write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "shapes": [
                    {"label": "nose", "points": [[10, 10]], "shape_type": "point"}
                ],
                "imagePath": "arena_000000005.png",
            }
        ),
        encoding="utf-8",
    )
    # Frame 18 labeled inside NDJSON AnnotationStore (`*_annotations.ndjson`).
    store = AnnotationStore.for_frame_path(ann_dir / "arena_000000000.json")
    store.append_frame(
        {
            "frame": 18,
            "version": "5.0.1",
            "shapes": [
                {
                    "label": "mouse_1",
                    "points": [[10, 10], [30, 10], [30, 30]],
                    "shape_type": "polygon",
                }
            ],
            "imagePath": "arena_000000018.png",
        }
    )
    # Empty JSON (no shapes) on frame 25 must NOT count as labeled.
    (ann_dir / "arena_000000025.json").write_text(
        json.dumps(
            {"version": "5.0.1", "shapes": [], "imagePath": "arena_000000025.png"}
        ),
        encoding="utf-8",
    )

    discovered = discover_labeled_frame_indices(ann_dir, require_shapes=True)
    assert discovered == {5, 18}

    # Queue-only selection (NO PNG extraction) excludes frames 5 and 18!
    res = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "pose",
            num_frames=8,
            feature_backend="visual_fg",
            extract_images=False,
            annotation_dir=str(ann_dir),
            exclude_labeled=True,
            seed_from_labeled=True,
            min_frame_gap=2,
        ),
    )
    assert res.extract_images is False
    assert res.saved_paths == ()
    assert len(res.selected_items) == 8
    assert 5 not in set(res.selected_indices)
    assert 18 not in set(res.selected_indices)
    assert res.excluded_labeled_count == 2
    assert res.manifest_path is not None and Path(res.manifest_path).is_file()
    # Confirm NO PNGs were written!
    assert list(ann_dir.glob("*.png")) == []


# ---------------------------------------------------------------------------
# 2. Extract vs No-Extract, Materialize, Quality Filtering & Flow
# ---------------------------------------------------------------------------


def test_queue_vs_extract_materialize_blur_filter_and_flow(tmp_path: Path) -> None:
    vid = _write_multi_state_video(
        tmp_path / "session.mp4",
        num_frames=45,
        inject_dark_frames=(2, 3, 12, 13, 22, 23),
    )

    # Segmentation task pins start anchor and filters injected dark frames.
    res_seg = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "segmentation",
            num_frames=6,
            feature_backend="visual_fg",
            extract_images=False,
            filter_blur=True,
            include_boundaries=True,
            min_frame_gap=2,
            persist_queue_manifest=True,
        ),
    )
    assert res_seg.task == "segmentation"
    assert res_seg.extract_images is False
    assert len(res_seg.selected_indices) == 6
    assert res_seg.filtered_blur_count >= 6
    assert not ({2, 3, 12, 13, 22, 23} & set(res_seg.selected_indices))
    assert res_seg.selected_items[0].selection_role == "boundary"

    # Queue JSON round-trip persistence.
    qpath = tmp_path / "custom_queue.json"
    save_frame_selection_queue(res_seg, qpath)
    reloaded = load_frame_selection_queue(qpath)
    assert reloaded.selected_indices == res_seg.selected_indices
    assert reloaded.task == "segmentation"

    # Later materialize the curated queue to disk.
    export_dir = tmp_path / "materialized"
    mat = materialize_selected_frames(
        vid,
        res_seg.selected_items,
        export_dir,
        image_format="png",
        base_result=res_seg,
    )
    assert mat.extract_images is True
    assert len(mat.saved_paths) == 6
    assert all(Path(p).is_file() for p in mat.saved_paths)
    assert mat.manifest_path is not None and Path(mat.manifest_path).is_file()

    # Flow selection works without holding full frames in RAM and returns sorted picks.
    flow_idx, flow_scores = select_flow_frame_indices(vid, 5, min_frame_gap=3)
    assert len(flow_idx) == 5
    assert flow_idx == sorted(flow_idx)
    assert flow_scores

    # Backward-compatible extract_frames_smart supports hybrid & farthest.
    res_hybrid = extract_frames_smart(
        vid, tmp_path / "out_hybrid", algo="hybrid", num_frames=5, seed=7
    )
    assert len(res_hybrid.saved_paths) == 5
    assert len(res_hybrid.selected_indices) == 5


# ---------------------------------------------------------------------------
# 3. Neural Embedding Backend, Persistent .npz Cache Auto-Discovery & Text Focus
# ---------------------------------------------------------------------------


def test_embedding_backend_caching_and_text_guided_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "emb_cache"
    monkeypatch.setenv("ANNOLID_EMBEDDING_CACHE_DIR", str(cache_dir))

    vid = _write_multi_state_video(tmp_path / "mouse_trial.mp4", num_frames=45)
    fake_backend = _DeterministicFakeEmbeddingBackend()

    # First run encodes frames via fake backend and writes .npz cache.
    res1 = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "pose",
            algo="hybrid",
            num_frames=6,
            feature_backend="embeddinggemma2_onnx",
            extract_images=False,
            max_candidates=45,
            min_frame_gap=2,
            use_embedding_disk_cache=True,
        ),
        backend=fake_backend,
        cache_dir=cache_dir,
    )
    assert len(res1.selected_indices) == 6
    assert res1.embedding_cache_hit is False
    assert res1.embedding_cache_path is not None
    assert Path(res1.embedding_cache_path).is_file()
    first_calls = fake_backend.embed_calls
    assert first_calls > 0

    # Auto-discovery finds the cached .npz index for this video!
    discovered = find_cached_video_embedding_indices(vid, cache_dir=cache_dir)
    assert len(discovered) == 1
    assert discovered[0].backend_name == "embeddinggemma2_onnx"
    assert discovered[0].embedding_dim == 8

    # Feature backend listing reports Auto reusing the cached index!
    opts = {
        opt.key: opt
        for opt in list_available_feature_backends(vid, cache_dir=cache_dir)
    }
    assert opts["auto"].cached_index is not None
    assert opts["cached_index"].cached_index is not None

    # Second run with feature_backend="auto" reuses cached .npz with ZERO backend calls!
    res2 = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "pose",
            algo="farthest",
            num_frames=5,
            feature_backend="auto",
            extract_images=False,
            min_frame_gap=2,
            use_embedding_disk_cache=True,
        ),
        cache_dir=cache_dir,
    )
    assert res2.embedding_cache_hit is True
    assert len(res2.selected_indices) == 5
    assert fake_backend.embed_calls == first_calls

    # Text-guided diversity run biases selection toward the rearing frames (40..44)
    # while still reusing the cached index (0 extra frame embeddings!).
    res_text = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "pose",
            algo="farthest",
            num_frames=4,
            feature_backend="embeddinggemma2_onnx",
            query_text="mouse rearing",
            extract_images=False,
            min_frame_gap=1,
            use_embedding_disk_cache=True,
        ),
        backend=fake_backend,
        cache_dir=cache_dir,
    )
    assert res_text.embedding_cache_hit is True
    assert fake_backend.embed_calls == first_calls
    assert any(idx >= 40 for idx in res_text.selected_indices)
    assert all(item.relevance_score is not None for item in res_text.selected_items)


def test_precomputed_npz_index_direct_reuse(tmp_path: Path) -> None:
    vid = _write_multi_state_video(tmp_path / "cached_demo.mp4", num_frames=30)
    idx_path = default_video_index_cache_path(
        video_path=vid,
        backend_key=("dinov3", "mock"),
        stride=2,
        max_frames=None,
        clip_frames=1,
        cache_dir=tmp_path,
    )
    frame_indices = np.arange(0, 30, 2, dtype=np.int64)
    rng = np.random.default_rng(123)
    emb = rng.normal(size=(len(frame_indices), 16)).astype(np.float32)
    index = VideoFrameEmbeddingIndex(
        frame_indices=frame_indices,
        embeddings=emb,
        metadata={
            "video_path": str(vid.resolve()),
            "backend_key": ["dinov3", "mock"],
            "stride": 2,
            "completed": True,
        },
    )
    index.save_npz(idx_path)

    res = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig(
            task="segmentation",
            algo="hybrid",
            num_frames=5,
            feature_backend="cached_index",
            embedding_index_path=str(idx_path),
            extract_images=False,
            min_frame_gap=2,
        ),
    )
    assert res.embedding_cache_hit is True
    assert res.feature_backend == "dinov3"
    assert len(res.selected_indices) == 5
    assert set(res.selected_indices).issubset(set(frame_indices.tolist()))


# ---------------------------------------------------------------------------
# 4. Pose Outlier Queue Without Extraction & CLI --no-extract
# ---------------------------------------------------------------------------


def test_pose_outliers_queue_without_extraction_and_cli_no_extract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from annolid.annotation.keypoint_visibility import KeypointVisibility
    from annolid.annotation.pose_interop.model import (
        PoseDataset,
        PoseFrame,
        PoseInstance,
        PosePoint,
    )
    from annolid.annotation.pose_schema import get_pose_schema_preset

    vid = _write_multi_state_video(tmp_path / "cli_vid.mp4", num_frames=20)
    schema = get_pose_schema_preset("Mouse (5 keypoints)")
    frames = []
    for fidx in range(12):
        score = 0.15 if fidx == 4 else 0.95
        frames.append(
            PoseFrame(
                frame_index=fidx,
                instances=[
                    PoseInstance(
                        instance_name="mouse_1",
                        keypoints={
                            "nose": PosePoint(
                                40.0 + fidx * 2.0,
                                40.0,
                                score=score,
                                visibility=int(KeypointVisibility.VISIBLE),
                            ),
                            "centroid": PosePoint(
                                30.0 + fidx * 2.0,
                                40.0,
                                score=0.95,
                                visibility=int(KeypointVisibility.VISIBLE),
                            ),
                        },
                    )
                ],
            )
        )
    ds = PoseDataset(
        frames=frames, keypoints=list(schema.keypoints), edges=list(schema.edges)
    )

    out_res = extract_pose_outlier_frames(
        ds,
        vid,
        output_dir=None,
        pose_schema=schema,
        num_frames=2,
        min_frame_gap=2,
        extract_images=False,
    )
    assert out_res.frames_extracted == 0
    assert out_res.saved_images == ()
    assert any(item.frame_index == 4 for item in out_res.selected_frames)

    # CLI `annolid-run extract-frames --no-extract --task pose --algo hybrid --json`
    rc = engine_cli.main(
        [
            "extract-frames",
            str(vid),
            "--task",
            "pose",
            "--algo",
            "hybrid",
            "--features",
            "visual_fg",
            "--num-frames",
            "5",
            "--no-extract",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["extract_images"] is False
    assert payload["num_extracted"] == 0
    assert payload["num_selected"] == 5
    assert Path(payload["manifest_path"]).is_file()


# ---------------------------------------------------------------------------
# 5. GUI Dialog & Interactive Label Queue Dock Tests
# ---------------------------------------------------------------------------


def test_gui_extract_dialog_modes_presets_and_config(tmp_path: Path) -> None:
    _ensure_qapp()
    from annolid.gui.widgets.extract_frame_dialog import ExtractFrameDialog

    vid = _write_multi_state_video(tmp_path / "gui_demo.mp4", num_frames=30)
    dlg = ExtractFrameDialog(
        str(vid),
        total_frames=30,
        fps=30.0,
        slider_selection=(6, 24),
    )
    try:
        # With an open video, defaults to Queue in Video (No Extraction).
        assert dlg.extract_images is False
        assert dlg.slider.value() == 100

        # Switching task to segmentation pins boundary anchor automatically.
        dlg.task_seg_radio.setChecked(True)
        assert dlg.task == "segmentation"
        assert dlg.include_boundaries_cb.isChecked() is True

        # Using timeline selection fills start & end seconds accurately.
        dlg._fill_from_slider_selection()
        assert dlg.start_seconds == pytest.approx(0.2, abs=0.01)
        assert dlg.end_seconds == pytest.approx(0.8, abs=0.01)
        assert dlg.start_sconds == dlg.start_seconds

        # Selecting EmbeddingGemma 2 ONNX exposes ONNX preset + focus prompt.
        idx_onnx = dlg.feature_backend_combo.findData("embeddinggemma2_onnx")
        assert idx_onnx >= 0
        dlg.feature_backend_combo.setCurrentIndex(idx_onnx)
        dlg.query_text_edit.setText("grooming")

        # Count sync without recursion.
        dlg.framesLineEdit.setText("25")
        assert dlg.num_frames == 25
        assert dlg.slider.value() == 25

        cfg = dlg.build_config()
        assert cfg.task == "segmentation"
        assert cfg.extract_images is False
        assert cfg.num_frames == 25
        assert cfg.feature_backend == "embeddinggemma2_onnx"
        assert cfg.query_text == "grooming"
        assert cfg.include_boundaries is True

        # Switching to Extract mode builds extract_images=True config.
        dlg.mode_extract_radio.setChecked(True)
        cfg_ext = dlg.build_config()
        assert cfg_ext.extract_images is True
    finally:
        dlg.close()


def test_gui_frame_selection_queue_widget_workflow(tmp_path: Path) -> None:
    _ensure_qapp()
    from annolid.gui.widgets.frame_selection_queue_widget import (
        FrameSelectionQueueWidget,
    )

    vid = _write_multi_state_video(tmp_path / "queue_demo.mp4", num_frames=30)
    ann_dir = tmp_path / "queue_demo"
    ann_dir.mkdir(parents=True, exist_ok=True)

    queue = FrameSelectionQueueWidget()
    jumped: list[int] = []
    marks_updates: list[list[int]] = []
    queue.jumpToFrameRequested.connect(jumped.append)
    queue.queueMarksChanged.connect(marks_updates.append)

    try:
        res = build_selection_result_from_indices(
            vid,
            [2, 10, 20, 28],
            task="pose",
            algo="hybrid",
            feature_backend="visual_fg",
            annotation_dir=ann_dir,
            reasons_by_frame={10: "Cluster #2 medoid"},
            persist_queue_manifest=True,
        )
        queue.set_selection_result(res, persist=False, load_thumbnails=False)
        assert queue.total_count() == 4
        assert queue.labeled_count() == 0
        assert marks_updates[-1] == [2, 10, 20, 28]
        assert "Cluster #2 medoid" in queue.tooltip_for_frame(10)

        # Jump through unlabeled frames.
        assert queue.jump_to_next_unlabeled() == 2
        assert jumped[-1] == 2

        # Marking frame 2 labeled updates progress and triggers auto-advance flag when checked.
        queue.auto_advance_cb.setChecked(True)
        should_advance = queue.mark_frame_labeled(2, labeled=True)
        assert should_advance is True
        assert queue.labeled_count() == 1
        assert queue.jump_to_next_unlabeled() == 10
        assert jumped[-1] == 10

        # Filtering to unlabeled hides frame 2.
        unlabeled_idx = queue.filter_combo.findData("unlabeled")
        queue.filter_combo.setCurrentIndex(unlabeled_idx)
        assert queue.frame_list.count() == 3

        # Add manual frame and verify persistence reloads on a fresh widget!
        assert queue.add_frame(15, reason="Manual edge case") is True
        assert 15 in queue.queued_frame_indices()

        queue2 = FrameSelectionQueueWidget()
        try:
            restored = queue2.set_video_context(
                vid, annotation_dir=ann_dir, fps=30.0, autoload_queue=True
            )
            assert restored is True
            assert 15 in queue2.queued_frame_indices()
        finally:
            queue2.shutdown_workers()
            queue2.close()
    finally:
        queue.shutdown_workers()
        queue.close()


def test_gui_embedding_search_to_queue_and_background_worker(tmp_path: Path) -> None:
    app = _ensure_qapp()
    from annolid.gui.widgets.embedding_search_widget import EmbeddingSearchWidget
    from annolid.gui.widgets.extract_frame_dialog import FrameSelectionWorker

    # 1. EmbeddingSearchWidget "To Queue" emits all matched frames when none selected.
    search_w = EmbeddingSearchWidget()
    queued_batches: list[list[int]] = []
    search_w.queueFramesRequested.connect(queued_batches.append)
    try:
        search_w._populate_results(
            [
                {"frame_index": 12, "similarity": 0.91},
                {"frame_index": 4, "similarity": 0.88},
                {"frame_index": 25, "similarity": 0.85},
            ]
        )
        search_w._queue_selected_frames()
        assert queued_batches == [[4, 12, 25]]
    finally:
        search_w.close()

    # 2. FrameSelectionWorker runs off-thread and emits FrameSelectionResult cleanly.
    vid = _write_multi_state_video(tmp_path / "worker_demo.mp4", num_frames=30)
    results = []
    errors = []
    worker = FrameSelectionWorker(
        vid,
        FrameSelectionConfig.for_task(
            "pose",
            num_frames=5,
            feature_backend="visual_fg",
            extract_images=False,
            min_frame_gap=2,
        ),
    )
    worker.finishedResult.connect(results.append)
    worker.failed.connect(errors.append)
    worker.start()
    assert worker.wait(10_000) is True
    app.processEvents()
    assert errors == []
    assert len(results) == 1
    assert results[0].extract_images is False
    assert len(results[0].selected_indices) == 5


# ---------------------------------------------------------------------------
# 6. Residual-Risk Hardening Tests (Offline Cache, Real ONNX, Dual Export, Batch)
# ---------------------------------------------------------------------------


def test_offline_cache_hit_never_loads_model_and_sparse_cache_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from annolid.services import embedding_search as emb_mod

    cache_dir = tmp_path / "offline_cache"
    monkeypatch.setenv("ANNOLID_EMBEDDING_CACHE_DIR", str(cache_dir))

    vid = _write_multi_state_video(tmp_path / "offline_vid.mp4", num_frames=45)
    bparams = {"quantization": "q4", "matryoshka_dim": 128, "image_max_soft_tokens": 70}
    bkey = emb_mod.backend_cache_key("embeddinggemma2_onnx", bparams)
    cache_path = default_video_index_cache_path(
        video_path=vid,
        backend_key=bkey,
        stride=1,
        max_frames=None,
        clip_frames=1,
        cache_dir=cache_dir,
    )
    rng = np.random.default_rng(7)
    full_indices = np.arange(0, 45, dtype=np.int64)
    stat = vid.stat()
    VideoFrameEmbeddingIndex(
        frame_indices=full_indices,
        embeddings=rng.normal(size=(45, 128)).astype(np.float32),
        metadata={
            "video_path": str(vid.resolve()),
            "video_mtime_ns": int(stat.st_mtime_ns),
            "video_size": int(stat.st_size),
            "stride": 1,
            "clip_frames": 1,
            "overlay_shapes": False,
            "backend_key": list(bkey),
            "completed": True,
        },
    ).save_npz(cache_path)

    # Hard guard: build_embedding_backend MUST NOT be called when disk cache satisfies request!
    def _fail_if_loaded(*_args, **_kwargs):
        raise AssertionError(
            "build_embedding_backend must not run on offline cache hit"
        )

    monkeypatch.setattr(emb_mod, "build_embedding_backend", _fail_if_loaded)

    res = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "pose",
            algo="hybrid",
            num_frames=8,
            feature_backend="embeddinggemma2_onnx",
            backend_params=bparams,
            extract_images=False,
            min_frame_gap=2,
            use_embedding_disk_cache=True,
        ),
        cache_dir=cache_dir,
    )
    assert res.embedding_cache_hit is True
    assert len(res.selected_indices) == 8

    # Sparse cache guard: cache with only 4 frames in [10, 30] must NOT truncate a k=10 auto request.
    sparse_cache_dir = tmp_path / "sparse_cache"
    sparse_path = default_video_index_cache_path(
        video_path=vid,
        backend_key=bkey,
        stride=10,
        max_frames=None,
        clip_frames=1,
        cache_dir=sparse_cache_dir,
    )
    VideoFrameEmbeddingIndex(
        frame_indices=np.asarray([0, 10, 20, 30, 40], dtype=np.int64),
        embeddings=rng.normal(size=(5, 128)).astype(np.float32),
        metadata={
            "video_path": str(vid.resolve()),
            "video_mtime_ns": int(stat.st_mtime_ns),
            "video_size": int(stat.st_size),
            "stride": 10,
            "clip_frames": 1,
            "overlay_shapes": False,
            "backend_key": list(bkey),
            "completed": True,
        },
    ).save_npz(sparse_path)

    res_sub = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "pose",
            algo="hybrid",
            num_frames=10,
            feature_backend="auto",
            extract_images=False,
            start_frame=10,
            end_frame=30,
            min_frame_gap=1,
            use_embedding_disk_cache=True,
        ),
        cache_dir=sparse_cache_dir,
    )
    assert res_sub.embedding_cache_hit is False
    assert res_sub.feature_backend == "visual_fg"
    assert len(res_sub.selected_indices) == 10


def test_real_onnx_embeddinggemma2_frame_selection_and_matryoshka_text_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    pytest.importorskip("tokenizers")

    from tests.test_embeddinggemma2_onnx_behavior_search import (
        _create_mini_embeddinggemma2_onnx_dir,
    )

    cache_dir = tmp_path / "real_onnx_cache"
    monkeypatch.setenv("ANNOLID_EMBEDDING_CACHE_DIR", str(cache_dir))

    model_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "mini_gemma2")
    # Build a synthetic video with distinct R/G/B segments matching the mini ONNX model
    # ('grooming'->R, 'rearing'->G, 'sniffing'->B).
    vid = tmp_path / "rgb_behaviors.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid), fourcc, 10.0, (96, 96))
    assert writer.isOpened()
    try:
        for i in range(18):
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            if i < 6:
                frame[:, :] = (20, 20, 235)  # BGR red -> R channel high (grooming)
            elif i < 12:
                frame[:, :] = (20, 235, 20)  # BGR green -> G channel high (rearing)
            else:
                frame[:, :] = (235, 20, 20)  # BGR blue -> B channel high (sniffing)
            writer.write(frame)
    finally:
        writer.release()

    # Run real ONNX Runtime EmbeddingGemma 2 frame selection (no fake backend!).
    res = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "pose",
            algo="hybrid",
            num_frames=3,
            feature_backend="embeddinggemma2_onnx",
            backend_params={
                "model_id": str(model_dir),
                "quantization": "q4",
                "matryoshka_dim": 128,
                "local_files_only": True,
            },
            extract_images=False,
            max_candidates=18,
            min_frame_gap=2,
            use_embedding_disk_cache=True,
        ),
        cache_dir=cache_dir,
    )
    assert res.feature_backend == "embeddinggemma2_onnx"
    assert len(res.selected_indices) == 3
    assert res.embedding_cache_path is not None
    assert Path(res.embedding_cache_path).is_file()
    # Should cover all 3 distinct behavior segments [0..5], [6..11], [12..17].
    assert {idx // 6 for idx in res.selected_indices} == {0, 1, 2}

    # Re-run with a real text focus prompt ("rearing" -> green frames [6..11])
    # reusing the cached 128-d .npz index via real ONNX text encoder!
    res_q = select_frames_for_labeling(
        vid,
        config=FrameSelectionConfig.for_task(
            "pose",
            algo="farthest",
            num_frames=3,
            feature_backend="embeddinggemma2_onnx",
            backend_params={
                "model_id": str(model_dir),
                "quantization": "q4",
                "matryoshka_dim": 128,
                "local_files_only": True,
            },
            query_text="rearing",
            extract_images=False,
            max_candidates=18,
            min_frame_gap=1,
            use_embedding_disk_cache=True,
        ),
        cache_dir=cache_dir,
    )
    assert res_q.embedding_cache_hit is True
    assert all(item.relevance_score is not None for item in res_q.selected_items)
    assert any(6 <= idx < 12 for idx in res_q.selected_indices)


def test_materialize_dual_storage_labels_relocated_queue_and_cli_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ensure_qapp()
    from annolid.gui.widgets.frame_selection_queue_widget import (
        FrameSelectionQueueWidget,
    )

    vid = _write_multi_state_video(tmp_path / "dual_export.mp4", num_frames=20)
    ann_dir = tmp_path / "dual_export"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Frame 3 in disk JSON with stale imagePath; Frame 9 only in AnnotationStore NDJSON.
    (ann_dir / "dual_export_000000003.json").write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "shapes": [
                    {"label": "nose", "points": [[12.0, 14.0]], "shape_type": "point"}
                ],
                "imagePath": "old_name_000000003.png",
            }
        ),
        encoding="utf-8",
    )
    store = AnnotationStore.for_frame_path(ann_dir / "dual_export_000000000.json")
    store.append_frame(
        {
            "frame": 9,
            "version": "5.0.1",
            "shapes": [
                {
                    "label": "mouse_1",
                    "points": [[10, 10], [40, 10], [40, 40]],
                    "shape_type": "polygon",
                }
            ],
            "imagePath": "dual_export_000000009.png",
            "imageHeight": 120,
            "imageWidth": 160,
        }
    )

    export_dir = tmp_path / "exported_bundle"
    mat = materialize_selected_frames(
        vid,
        [3, 9, 15],
        export_dir,
        image_format="png",
        copy_existing_labels=True,
        annotation_dir=ann_dir,
    )
    assert len(mat.saved_paths) == 3
    j3 = json.loads(
        (export_dir / "dual_export_000000003.json").read_text(encoding="utf-8")
    )
    j9 = json.loads(
        (export_dir / "dual_export_000000009.json").read_text(encoding="utf-8")
    )
    assert j3["imagePath"] == "dual_export_000000003.png"
    assert len(j3["shapes"]) == 1
    assert j9["imagePath"] == "dual_export_000000009.png"
    assert j9["shapes"][0]["label"] == "mouse_1"
    assert not (export_dir / "dual_export_000000015.json").exists()

    # Relocated queue manifest pointing to a nonexistent old path keeps live video path.
    qpath = ann_dir / "frame_selection_queue.json"
    stale_res = build_selection_result_from_indices(
        vid, [3, 9], annotation_dir=ann_dir, persist_queue_manifest=False
    )
    stale_payload = stale_res.to_dict()
    stale_payload["video_path"] = "/nonexistent/moved/old_video.mp4"
    qpath.write_text(json.dumps(stale_payload), encoding="utf-8")

    queue = FrameSelectionQueueWidget()
    try:
        assert (
            queue.set_video_context(
                vid, annotation_dir=ann_dir, fps=30.0, autoload_queue=True
            )
            is True
        )
        assert queue._video_path == vid.resolve()
        # Unlabeling frame 3 and re-labeling last frame stops auto-advance cleanly.
        queue.auto_advance_cb.setChecked(True)
        assert queue.mark_frame_labeled(3, labeled=False) is False
        assert queue.labeled_count() == 1
        assert queue.mark_frame_labeled(3, labeled=True) is False  # last unlabeled!
        assert queue.labeled_count() == 2
    finally:
        queue.shutdown_workers()
        queue.close()

    # CLI --no-extract honors custom --output JSON path.
    custom_q = tmp_path / "my_custom_queue.json"
    rc = engine_cli.main(
        [
            "extract-frames",
            str(vid),
            "--task",
            "segmentation",
            "--algo",
            "hybrid",
            "--features",
            "visual_fg",
            "--num-frames",
            "4",
            "--no-extract",
            "--output",
            str(custom_q),
            "--json",
        ]
    )
    assert rc == 0
    out_data = json.loads(capsys.readouterr().out)
    assert Path(out_data["manifest_path"]).resolve() == custom_q.resolve()
    assert custom_q.is_file()


def test_batch_frame_extractor_worker_isolates_per_video_manifests(
    tmp_path: Path,
) -> None:
    app = _ensure_qapp()
    from annolid.gui.workers import FrameExtractorWorker

    vid_a = _write_multi_state_video(tmp_path / "trial_a.mp4", num_frames=20)
    vid_b = _write_multi_state_video(tmp_path / "trial_b.mp4", num_frames=20)
    batch_out = tmp_path / "batch_out"

    done_paths: list[str] = []
    errors: list[str] = []
    worker = FrameExtractorWorker(
        [str(vid_a), str(vid_b)],
        str(batch_out),
        num_frames=4,
        algo="hybrid",
        task="pose",
        feature_backend="visual_fg",
        extract_images=True,
    )
    worker.finished.connect(done_paths.append)
    worker.error.connect(errors.append)
    worker.start()
    assert worker.wait(15_000) is True
    app.processEvents()
    assert errors == []
    assert done_paths == [str(batch_out)]
    assert (batch_out / "trial_a_hybrid" / "frame_selection_manifest.json").is_file()
    assert (batch_out / "trial_b_hybrid" / "frame_selection_manifest.json").is_file()
    assert len(list((batch_out / "trial_a_hybrid").glob("*.png"))) == 4
    assert len(list((batch_out / "trial_b_hybrid").glob("*.png"))) == 4
