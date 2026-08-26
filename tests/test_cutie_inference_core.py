import numpy as np
import pytest
import torch

from annolid.segmentation.cutie_vos.inference import inference_core
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.segmentation.cutie_vos.inference.memory_manager import MemoryManager
from annolid.segmentation.cutie_vos.inference.object_manager import ObjectManager
from annolid.segmentation.cutie_vos.interactive_utils import (
    image_to_torch,
    resize_frame_for_inference,
    torch_prob_to_numpy_mask,
)


class _FeatureStore:
    def __init__(self) -> None:
        self.deleted = []

    def get_features(self, _index, _image):
        return (), torch.zeros((1, 1, 1, 1))

    def get_key(self, _index, _image):
        value = torch.zeros((1, 1, 1, 1))
        return value, value, value

    def delete(self, index) -> None:
        self.deleted.append(index)


def _core_for_mask_merge(*, existing_objects=()) -> InferenceCore:
    core = InferenceCore.__new__(InferenceCore)
    core.max_internal_size = -1
    core.curr_ti = -1
    core.last_mem_ti = 0
    core.mem_every = 5
    core.stagger_ti = set()
    core.flip_aug = False
    core.object_manager = ObjectManager()
    core.object_manager.add_new_objects(list(existing_objects))
    core.image_feature_store = _FeatureStore()
    core._add_memory = lambda *_args, **_kwargs: None
    return core


def test_resize_index_mask_uses_nearest_without_align_corners() -> None:
    mask = torch.tensor(
        [
            [1, 2],
            [3, 0],
        ],
        dtype=torch.int64,
    )

    resized = InferenceCore._resize_index_mask(mask, (4, 4))

    assert resized.dtype == torch.int64
    assert set(torch.unique(resized).tolist()) == {0, 1, 2, 3}
    assert resized[0, 0].item() == 1
    assert resized[0, -1].item() == 2
    assert resized[-1, 0].item() == 3
    assert resized[-1, -1].item() == 0


def test_resize_input_index_mask_preserves_object_ids() -> None:
    mask = torch.tensor(
        [
            [0, 5],
            [9, 0],
        ],
        dtype=torch.int64,
    )

    resized = InferenceCore._resize_input_mask(
        mask,
        (8, 8),
        idx_mask=True,
    )

    assert resized.shape == (8, 8)
    assert resized.dtype == torch.int64
    assert set(torch.unique(resized).tolist()) == {0, 5, 9}


def test_finalize_index_output_resizes_only_one_label_channel(monkeypatch) -> None:
    probabilities = torch.tensor(
        [
            [[0.9, 0.1], [0.1, 0.1]],
            [[0.1, 0.8], [0.2, 0.1]],
            [[0.0, 0.1], [0.7, 0.8]],
        ],
        dtype=torch.float32,
    )
    interpolate_shapes = []
    original_interpolate = inference_core.F.interpolate

    def _capture_interpolate(input_tensor, *args, **kwargs):
        interpolate_shapes.append(tuple(input_tensor.shape))
        return original_interpolate(input_tensor, *args, **kwargs)

    monkeypatch.setattr(inference_core.F, "interpolate", _capture_interpolate)

    output = InferenceCore._finalize_output(
        probabilities,
        output_size=(8, 8),
        return_index_mask=True,
    )

    assert output.shape == (8, 8)
    assert output.dtype == torch.int32
    assert set(torch.unique(output).tolist()) == {0, 1, 2}
    assert interpolate_shapes == [(1, 1, 2, 2)]


def test_finalize_probability_output_preserves_default_behavior(monkeypatch) -> None:
    probabilities = torch.rand((4, 2, 3), dtype=torch.float32)
    interpolate_shapes = []
    original_interpolate = inference_core.F.interpolate

    def _capture_interpolate(input_tensor, *args, **kwargs):
        interpolate_shapes.append(tuple(input_tensor.shape))
        return original_interpolate(input_tensor, *args, **kwargs)

    monkeypatch.setattr(inference_core.F, "interpolate", _capture_interpolate)

    output = InferenceCore._finalize_output(
        probabilities,
        output_size=(8, 12),
    )

    assert output.shape == (4, 8, 12)
    assert output.dtype == probabilities.dtype
    assert interpolate_shapes == [(1, 4, 2, 3)]


def test_resize_frame_for_inference_caps_short_side_before_tensor_conversion() -> None:
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    resized = resize_frame_for_inference(frame, 480)

    assert resized.shape == (480, 853, 3)
    assert resized.dtype == np.uint8


def test_resize_frame_for_inference_reuses_small_frame() -> None:
    frame = np.zeros((230, 280, 3), dtype=np.uint8)

    resized = resize_frame_for_inference(frame, 480)

    assert resized is frame


def test_image_to_torch_returns_contiguous_float_chw_tensor() -> None:
    frame = np.full((3, 5, 3), 255, dtype=np.uint8)

    tensor = image_to_torch(frame, device="cpu")

    assert tensor.shape == (3, 3, 5)
    assert tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    assert torch.all(tensor == 1.0)


def test_finalize_index_output_skips_same_size_interpolation(monkeypatch) -> None:
    probabilities = torch.rand((3, 4, 5), dtype=torch.float32)

    def _unexpected_interpolate(*_args, **_kwargs):
        raise AssertionError("same-size output should not be interpolated")

    monkeypatch.setattr(inference_core.F, "interpolate", _unexpected_interpolate)

    output = InferenceCore._finalize_output(
        probabilities,
        output_size=(4, 5),
        return_index_mask=True,
    )

    assert output.shape == (4, 5)
    assert output.dtype == torch.int32


def test_partial_per_object_correction_preserves_unspecified_objects() -> None:
    core = _core_for_mask_merge(existing_objects=(10, 20))

    def _segment(*_args, **_kwargs):
        probabilities = torch.zeros((3, 16, 16))
        probabilities[0] = 0.1
        probabilities[1] = 0.9
        return probabilities

    core._segment = _segment
    correction = torch.zeros((1, 2, 2))
    correction[0, 0, 1] = 1

    output = core.step(
        torch.zeros((3, 2, 2)),
        correction,
        objects=[20],
        idx_mask=False,
        return_index_mask=True,
    )

    assert output.tolist() == [[1, 2], [1, 1]]
    assert core.image_feature_store.deleted == [0]


def test_per_object_correction_respects_caller_channel_order() -> None:
    core = _core_for_mask_merge(existing_objects=(10, 20))
    core._segment = lambda *_args, **_kwargs: pytest.fail(
        "complete correction should not run segmentation"
    )
    correction = torch.zeros((2, 2, 2))
    correction[0, 0, 0] = 1  # object 20 -> temporary channel 2
    correction[1, 0, 1] = 1  # object 10 -> temporary channel 1

    output = core.step(
        torch.zeros((3, 2, 2)),
        correction,
        objects=[20, 10],
        idx_mask=False,
        return_index_mask=True,
    )

    assert output.tolist() == [[2, 1], [0, 0]]


def test_new_object_correction_keeps_existing_prediction() -> None:
    core = _core_for_mask_merge(existing_objects=(10,))

    def _segment(*_args, **_kwargs):
        probabilities = torch.zeros((2, 16, 16))
        probabilities[0] = 0.1
        probabilities[1] = 0.9
        return probabilities

    core._segment = _segment
    new_object_mask = torch.zeros((1, 2, 2))
    new_object_mask[0, 1, 1] = 1

    output = core.step(
        torch.zeros((3, 2, 2)),
        new_object_mask,
        objects=[30],
        idx_mask=False,
        return_index_mask=True,
    )

    assert core.object_manager.all_obj_ids == [10, 30]
    assert output.tolist() == [[1, 1], [1, 2]]


def test_step_releases_cached_features_when_segmentation_fails() -> None:
    core = _core_for_mask_merge(existing_objects=(1,))

    def _fail(*_args, **_kwargs):
        raise RuntimeError("segmentation failed")

    core._segment = _fail

    with pytest.raises(RuntimeError, match="segmentation failed"):
        core.step(torch.zeros((3, 2, 2)))

    assert core.image_feature_store.deleted == [0]


@pytest.mark.parametrize(
    ("mask", "objects", "idx_mask", "message"),
    [
        (torch.zeros((2, 2)), [1, 1], True, "must not contain duplicates"),
        (torch.zeros((1, 2, 2)), [1, 2], False, "channels must match"),
        (torch.zeros((2, 2)), [], True, "at least one foreground"),
        (
            torch.tensor([[0, 2], [0, 0]]),
            [1],
            True,
            "missing from the objects list",
        ),
        (
            torch.tensor([[0.0, 1.5], [0.0, 0.0]]),
            None,
            True,
            "must contain integer object ids",
        ),
    ],
)
def test_step_rejects_invalid_mask_metadata_before_inference(
    mask, objects, idx_mask, message
) -> None:
    core = _core_for_mask_merge()

    with pytest.raises(ValueError, match=message):
        core.step(
            torch.zeros((3, 2, 2)),
            mask,
            objects=objects,
            idx_mask=idx_mask,
        )

    assert core.curr_ti == -1
    assert core.image_feature_store.deleted == []


def test_numpy_mask_conversion_preserves_ids_above_uint8_range() -> None:
    mask = torch.tensor([[0, 256], [511, 1]], dtype=torch.int64)

    converted = torch_prob_to_numpy_mask(mask)

    assert converted.dtype == np.int32
    assert converted.tolist() == [[0, 256], [511, 1]]


def test_memory_purge_drops_deleted_object_query_values() -> None:
    class _WorkMemory:
        def __init__(self):
            self.kept = None

        def purge_except(self, kept):
            self.kept = list(kept)

        def engaged(self):
            return True

    manager = MemoryManager.__new__(MemoryManager)
    manager.work_mem = _WorkMemory()
    manager.use_long_term = False
    manager.sensory = {1: object(), 2: object()}
    manager.obj_v = {1: object(), 2: object()}
    manager.engaged = True

    manager.purge_except([2])

    assert manager.work_mem.kept == [2]
    assert set(manager.sensory) == {2}
    assert set(manager.obj_v) == {2}


def test_delete_objects_keeps_last_mask_aligned_with_reindexed_objects() -> None:
    class _Memory:
        def __init__(self):
            self.kept = None

        def purge_except(self, kept):
            self.kept = list(kept)

    core = InferenceCore.__new__(InferenceCore)
    core.object_manager = ObjectManager()
    core.object_manager.add_new_objects([10, 20, 30])
    core.last_mask = torch.tensor([[[[0.1]], [[0.2]], [[0.3]]]])
    core.memory = _Memory()

    core.delete_objects([20])

    assert core.object_manager.all_obj_ids == [10, 30]
    assert core.last_mask.flatten().tolist() == pytest.approx([0.1, 0.3])
    assert core.memory.kept == [10, 30]


def test_output_prob_to_mask_restores_sparse_stable_object_ids() -> None:
    core = InferenceCore.__new__(InferenceCore)
    core.object_manager = ObjectManager()
    core.object_manager.add_new_objects([7, 511])
    temporary_mask = torch.tensor([[0, 1], [2, 1]], dtype=torch.int64)

    stable_mask = core.output_prob_to_mask(temporary_mask)

    assert stable_mask.tolist() == [[0, 7], [511, 7]]


def test_object_manager_exports_stable_to_temporary_id_mapping() -> None:
    manager = ObjectManager()
    manager.add_new_objects([7, 511])

    assert manager.get_tmp_to_obj_mapping() == {7: 1, 511: 2}
