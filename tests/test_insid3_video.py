from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from annolid.segmentation.crf_refinement import CrfMaskRefiner, CrfRefinementConfig
from annolid.segmentation.insid3_video import (
    Insid3VideoProcessor,
    Insid3VideoConfig,
    Insid3VideoSegmenter,
    _ReferenceObject,
    _neighbor_connected_component_labels,
)
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.domain import InstanceRegistry
from annolid.utils.annotation_store import AnnotationStore


class _FakeExtractor:
    patch_size = 1

    def __init__(self) -> None:
        self._queue: list[torch.Tensor] = []

    def queue(self, feats: torch.Tensor) -> None:
        self._queue.append(feats)

    def extract(self, _image, return_layer="all", normalize=True):
        _ = return_layer, normalize
        if not self._queue:
            raise AssertionError("No queued features")
        return self._queue.pop(0)


class _SharedFakeExtractor(_FakeExtractor):
    queue_values: list[torch.Tensor] = []

    def __init__(self, _cfg) -> None:
        super().__init__()
        self._queue = self.queue_values


class _FakeVideo:
    def __init__(self, _path: str) -> None:
        self.frames = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
        ]

    def total_frames(self) -> int:
        return len(self.frames)

    def get_height(self) -> int:
        return 4

    def get_width(self) -> int:
        return 4

    def load_frame(self, frame_number: int) -> np.ndarray:
        return self.frames[int(frame_number)]

    def release(self) -> None:
        return None


def test_insid3_segmenter_tracks_reference_mask_by_cluster_similarity(monkeypatch):
    monkeypatch.setattr(
        "annolid.segmentation.insid3_video._sklearn_agglomerative_labels",
        lambda _features, _tau: None,
    )
    extractor = _FakeExtractor()
    segmenter = Insid3VideoSegmenter(
        config=Insid3VideoConfig(tau=0.95, merge_threshold=0.2),
        extractor=extractor,
    )

    reference_features = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    target_features = reference_features.clone()
    extractor.queue(reference_features)
    extractor.queue(target_features)

    mask = np.array([[1, 0], [1, 0]], dtype=bool)
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    references = segmenter.build_references(image, {"animal": mask})
    predictions = segmenter.segment(image, references, output_size=(2, 2))

    assert set(predictions) == {"animal"}
    np.testing.assert_array_equal(predictions["animal"], mask)


def test_neighbor_connected_component_labels_groups_adjacent_similar_patches():
    features = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    features = features / torch.sqrt((features * features).sum(dim=0, keepdim=True))

    labels = _neighbor_connected_component_labels(features, tau=0.95)

    assert labels[0, 0].item() == labels[0, 1].item()
    assert labels[0, 0].item() == labels[1, 0].item()
    assert labels[0, 2].item() == labels[1, 1].item()
    assert labels[0, 0].item() != labels[0, 2].item()


def test_candidate_localization_requires_backward_match_to_reference_mask():
    ref_features = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    target_features = torch.tensor(
        [
            [[1.0, 0.2, 0.2]],
            [[0.0, 0.98, 0.98]],
        ],
        dtype=torch.float32,
    )
    ref_features = ref_features / torch.sqrt(
        (ref_features * ref_features).sum(dim=0, keepdim=True)
    )
    target_features = target_features / torch.sqrt(
        (target_features * target_features).sum(dim=0, keepdim=True)
    )
    candidate = Insid3VideoSegmenter._locate_candidate_mask(
        debiased=target_features,
        reference=_ReferenceObject(
            label="animal",
            mask=torch.tensor([[True, False, False]]),
            prototype=torch.tensor([1.0, 0.0]),
            features=ref_features,
        ),
        prototype_similarity=torch.einsum(
            "chw,c->hw",
            target_features,
            torch.tensor([1.0, 0.0]),
        ),
    )

    np.testing.assert_array_equal(candidate.numpy(), np.array([[True, False, False]]))


def test_crf_refiner_disabled_returns_mask_copy():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    refiner = CrfMaskRefiner(CrfRefinementConfig(enabled=False))

    refined = refiner.refine(image, mask)

    np.testing.assert_array_equal(refined, mask)
    assert refined is not mask


def test_opencv_crf_fallback_removes_isolated_boundary_noise():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    mask[0, 0] = True
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    refiner = CrfMaskRefiner(
        CrfRefinementConfig(enabled=True, backend="opencv", band_px=1)
    )

    refined = refiner.refine(image, mask)

    assert refined[4, 4]
    assert not refined[0, 0]


def test_insid3_processor_parses_crf_runtime_options(tmp_path, monkeypatch):
    monkeypatch.setattr("annolid.segmentation.insid3_video.CV2Video", _FakeVideo)
    monkeypatch.setattr(
        "annolid.segmentation.insid3_video.Dinov3FeatureExtractor",
        _SharedFakeExtractor,
    )
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"")

    processor = Insid3VideoProcessor(
        video_path=str(video_path),
        results_folder=str(tmp_path / "demo"),
        insid3_crf_refine="true",
        insid3_crf_backend="opencv",
        insid3_crf_band_px=3,
        insid3_crf_p_core=0.9,
        insid3_crf_iterations=5,
    )

    assert processor.config.crf_refine is True
    assert processor.config.crf_backend == "opencv"
    assert processor.config.crf_band_px == 3
    assert processor.config.crf_p_core == 0.9
    assert processor.config.crf_iterations == 5


def test_insid3_video_processor_writes_labelme_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "annolid.segmentation.insid3_video._sklearn_agglomerative_labels",
        lambda _features, _tau: None,
    )
    monkeypatch.setattr("annolid.segmentation.insid3_video.CV2Video", _FakeVideo)
    monkeypatch.setattr(
        "annolid.segmentation.insid3_video.Dinov3FeatureExtractor",
        _SharedFakeExtractor,
    )
    features = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    _SharedFakeExtractor.queue_values = [features, features, features]

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"")
    output_dir = tmp_path / "demo"
    output_dir.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
        output_dir / "demo_000000000.png"
    )

    adapter = AnnotationAdapter(image_height=4, image_width=4, description="seed")
    registry = InstanceRegistry()
    registry.ensure_instance("animal").set_mask(
        bitmap=None,
        polygon=[(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)],
    )
    adapter.write_annotation(frame_number=0, registry=registry, output_dir=output_dir)

    processor = Insid3VideoProcessor(
        video_path=str(video_path),
        results_folder=str(output_dir),
        insid3_short_side=2,
    )
    message = processor.process_video_frames(start_frame=0, end_frame=1)

    assert message == "INSID3 video segmentation completed for 2 frames."
    assert not (output_dir / "demo_000000001.json").exists()
    store = AnnotationStore(output_dir / "demo_annotations.ndjson")
    frame_one = store.get_frame(1)
    assert frame_one is not None
    assert frame_one["shapes"][0]["label"] == "animal"
    assert frame_one["shapes"][0]["description"] == "INSID3"
