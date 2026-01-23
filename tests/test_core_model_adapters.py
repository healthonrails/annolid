from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from annolid.core.models.base import ModelRequest
from annolid.core.models.pipeline import run_caption
from annolid.core.models.adapters.llm_chat import LLMChatAdapter, _OpenAICompatConfig
from annolid.core.models.adapters.maskrcnn_torchvision import (
    TorchvisionMaskRCNNAdapter,
)


class _FakeOpenAIChatCompletions:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **kwargs):  # noqa: ANN003
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))]
        )


class _FakeOpenAIClient:
    def __init__(self, text: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeOpenAIChatCompletions(text))


def test_llm_adapter_caption_swappable_without_openai(tmp_path: Path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "img.png"

    try:
        import cv2  # type: ignore
    except ImportError:
        pytest.skip("cv2 is required for this test.")

    assert cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def factory(cfg: _OpenAICompatConfig):
        assert cfg.model
        return _FakeOpenAIClient("fake caption")

    adapter = LLMChatAdapter(
        provider="ollama",
        model="qwen3-vl",
        persist=False,
        client_factory=factory,
    )

    resp = run_caption(adapter, img_path, prompt="Describe")
    assert resp.text == "fake caption"
    assert resp.output["text"] == "fake caption"


def test_cv_adapter_caption_swappable_with_fake_model(tmp_path: Path):
    torch = pytest.importorskip("torch")

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "img.png"

    try:
        import cv2  # type: ignore
    except ImportError:
        pytest.skip("cv2 is required for this test.")

    assert cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    class FakeModel:
        def to(self, device):  # noqa: ANN001
            return self

        def eval(self):
            return self

        def __call__(self, batch):  # noqa: ANN001
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                }
            ]

    adapter = TorchvisionMaskRCNNAdapter(
        pretrained=False,
        score_threshold=0.1,
        model_factory=FakeModel,
        label_names=["bg", "mouse"],
    )

    resp = run_caption(adapter, img_path, prompt="Describe")
    assert "Detected:" in (resp.text or "")
    assert resp.output["detections"][0]["label_id"] == 1


def test_pipeline_interface_stays_constant():
    req = ModelRequest(task="caption", image_path="x.png", text="Describe")
    assert req.task == "caption"
