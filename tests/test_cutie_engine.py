from types import SimpleNamespace

import numpy as np
import torch

from annolid.segmentation.cutie_vos.engine import CutieEngine


class _Capture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.position = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return 0

    def set(self, _prop, value):
        self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()


class _Core:
    def __init__(self):
        self.calls = []

    def step(self, frame, mask=None, **kwargs):
        self.calls.append((frame, mask, kwargs))
        output_size = kwargs["output_size"]
        return torch.ones(output_size, dtype=torch.int64)

    def output_prob_to_mask(self, output):
        return torch.where(output == 1, 257, 0)


def test_process_frames_uses_index_masks_and_pretransfer_resize() -> None:
    engine = CutieEngine.__new__(CutieEngine)
    engine.device = "cpu"
    engine.cfg = SimpleNamespace(amp=False, max_internal_size=2)
    engine.cutie_model = object()
    engine.inference_core = _Core()

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    initial_mask = np.zeros((4, 6), dtype=np.int32)
    initial_mask[:, :3] = 7

    results = list(
        engine.process_frames(
            _Capture([frame]),
            start_frame_index=0,
            initial_mask_np=initial_mask,
            num_objects_in_mask=1,
            frames_to_propagate=1,
            reset_core=False,
        )
    )

    assert len(results) == 1
    _, _, output_mask = results[0]
    assert output_mask.dtype == np.int32
    assert int(output_mask.max()) == 257

    frame_tensor, mask_tensor, kwargs = engine.inference_core.calls[0]
    assert tuple(frame_tensor.shape) == (3, 2, 3)
    assert tuple(mask_tensor.shape) == (4, 6)
    assert mask_tensor.device.type == "cpu"
    assert kwargs["objects"] == [7]
    assert kwargs["idx_mask"] is True
    assert kwargs["return_index_mask"] is True
    assert kwargs["output_size"] == (4, 6)


def test_cleanup_accepts_string_cuda_device(monkeypatch) -> None:
    engine = CutieEngine.__new__(CutieEngine)
    engine.device = "cuda:0"
    engine.inference_core = object()
    engine.cutie_model = object()
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))

    engine.cleanup()

    assert engine.inference_core is None
    assert engine.cutie_model is None
    assert emptied == [True]
