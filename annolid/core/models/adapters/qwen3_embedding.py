from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from annolid.core.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RuntimeModel,
)


@dataclass(frozen=True)
class Qwen3EmbeddingConfig:
    instruction: str = "Represent the user's input."
    max_length: int = 8192
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    fps: float = 1.0
    max_frames: int = 128
    total_pixels: int = 1280 * 28 * 28


class Qwen3EmbeddingAdapter(RuntimeModel):
    """Multimodal embedding adapter for Qwen3-VL."""

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-VL-Embedding-8B",
        config: Optional[Qwen3EmbeddingConfig] = None,
        torch_dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._config = config or Qwen3EmbeddingConfig()
        self._torch_dtype = torch_dtype
        self._attn_implementation = attn_implementation
        self._model = None
        self._processor = None
        self._process_vision_info = None
        self._torch = None

    @property
    def model_id(self) -> str:
        return f"qwen3-vl-embedding:{self._model_id}"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tasks=("embed",),
            input_modalities=("text", "image", "video"),
            output_modalities=("embedding",),
        )

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import Qwen3VLModel, Qwen3VLProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3EmbeddingAdapter requires transformers>=4.57.0 and qwen-vl-utils>=0.0.14."
            ) from exc

        processor = Qwen3VLProcessor.from_pretrained(
            self._model_id, min_pixels=self._config.min_pixels
        )
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self._torch_dtype:
            model_kwargs["torch_dtype"] = getattr(torch, self._torch_dtype)
        if self._attn_implementation:
            model_kwargs["attn_implementation"] = self._attn_implementation
        model = Qwen3VLModel.from_pretrained(self._model_id, **model_kwargs)
        model.eval()

        self._processor = processor
        self._model = model
        self._process_vision_info = process_vision_info
        self._torch = torch

    def predict(self, request: ModelRequest) -> ModelResponse:
        self.load()
        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen3EmbeddingAdapter not loaded.")

        text = request.text
        if text is None:
            messages = request.messages
            if messages:
                text = str(messages[-1].get("content", ""))
        text = str(text or "").strip()

        image = request.image or request.image_path
        params = request.params or {}
        video = params.get("video") or params.get("video_path")

        if not text and image is None and video is None:
            raise ValueError("Embedding request requires text, image, or video.")

        instruction = str(params.get("instruction") or self._config.instruction).strip()
        if instruction and instruction[-1] not in {".", "?", "!"}:
            instruction = f"{instruction}."

        conversation = self._format_conversation(
            text=text, image=image, video=video, instruction=instruction
        )
        embeddings = self._embed_conversations([conversation])
        return ModelResponse(
            task=request.task,
            output={"embedding": embeddings[0]},
            meta={"model_id": self._model_id},
        )

    def close(self) -> None:
        self._model = None
        self._processor = None

    def _format_conversation(
        self,
        *,
        text: str,
        image: Any,
        video: Any,
        instruction: str,
    ) -> Sequence[Dict[str, Any]]:
        content: list[Dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if video is not None:
            content.append(
                {"type": "video", "video": video, "max_pixels": self._config.max_pixels}
            )
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            content.append({"type": "text", "text": "NULL"})
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    def _embed_conversations(
        self, conversations: Sequence[Sequence[Mapping[str, Any]]]
    ) -> list[list[float]]:
        processor = self._processor
        model = self._model
        torch = self._torch
        if (
            processor is None
            or model is None
            or torch is None
            or self._process_vision_info is None
        ):
            raise RuntimeError("Qwen3EmbeddingAdapter not loaded.")

        texts = [
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_kwargs is None:
            video_kwargs = {}
        video_metadata = video_kwargs.pop("video_metadata", None)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {key: val.to(model.device) for key, val in inputs.items()}
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        embeddings = self._pool_last_token(outputs.last_hidden_state, attention_mask)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    @staticmethod
    def _pool_last_token(last_hidden: Any, attention_mask: Any) -> Any:
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        return last_hidden[
            torch.arange(batch_size, device=last_hidden.device), sequence_lengths
        ]
