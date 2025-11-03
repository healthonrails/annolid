"""
Florence-2 integration utilities.

This module exposes a lightweight predictor class that can be reused from both
CLI workflows and the Qt GUI. The predictor takes care of lazily loading the
model/processor pair, running caption and segmentation tasks, and providing
results in a dictionary-friendly structure. Helper functions are provided for
converting the Florence masks into `Shape` objects so the annotations can be
saved in LabelMe format or injected directly into the canvas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.generation.utils import GenerationMixin

try:  # Optional import; Florence-2 is still in rapid development upstream.
    from transformers.models.florence2.modeling_florence2 import (
        Florence2ForConditionalGeneration,
    )
except Exception:  # pragma: no cover - if transformers version changes/remove symbol
    Florence2ForConditionalGeneration = None  # type: ignore

from annolid.annotation.keypoints import save_labels
from annolid.gui.shape import Shape
from annolid.utils.logger import logger
from annolid.data.videos import video_loader
import numpy as np
import numpy as np

try:  # Qt is available in the GUI environment, but guard for CLI use.
    from qtpy import QtCore
except Exception:  # pragma: no cover - headless environments
    QtCore = None  # type: ignore

FlorencePolygons = List[List[List[Tuple[float, float]]]]


@dataclass
class Florence2Result:
    """Container for Florence-2 outputs."""

    mask_dict: Dict[str, Any]
    caption: Optional[str]
    raw_outputs: Dict[str, Any]

    def has_polygons(self) -> bool:
        polygons = self.mask_dict.get("polygons")
        return bool(polygons)


def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    """Resolve the preferred torch device and dtype."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32
    return device, torch_dtype


def _pairwise(iterable: Iterable[float]) -> Iterable[Tuple[float, float]]:
    """Yield coordinate pairs from a flat list."""
    iterator = iter(iterable)
    while True:
        try:
            x = next(iterator)
            y = next(iterator)
            yield float(x), float(y)
        except StopIteration:
            break


def convert_to_mask_dict(
    results: Dict[str, Any],
    text_prompt: str = "<REFERRING_EXPRESSION_SEGMENTATION>",
    text_input: Optional[str] = None,
) -> Dict[str, FlorencePolygons]:
    """
    Converts the model results into a valid mask dictionary that can be used
    for saving annotations or rendering in the GUI.
    """
    task_result = results.get(text_prompt) or {}
    polygons = task_result.get("polygons") or []
    labels = task_result.get("labels") or []

    mask_dict: Dict[str, FlorencePolygons] = {
        "polygons": [],
        "labels": [],
    }

    for idx, polygon_set in enumerate(polygons):
        label = labels[idx] if idx < len(labels) and labels[idx] else text_input
        formatted_polygons: List[List[Tuple[float, float]]] = []
        if polygon_set is None:
            continue

        for polygon in polygon_set:
            if polygon is None:
                continue
            if isinstance(polygon, Iterable) and not isinstance(polygon, (str, bytes)):
                if all(isinstance(coord, (int, float)) for coord in polygon):
                    coords = list(polygon)
                    if len(coords) % 2 != 0:
                        logger.warning(
                            "Skipping Florence polygon with odd number of coords: %s",
                            coords,
                        )
                        continue
                    formatted_polygons.append(list(_pairwise(coords)))
                else:
                    try:
                        formatted_polygons.append(
                            [(float(x), float(y)) for x, y in polygon]
                        )
                    except Exception:  # pragma: no cover
                        logger.warning("Skipping malformed polygon: %s", polygon)
                        continue

        if formatted_polygons:
            mask_dict["polygons"].append(formatted_polygons)
            mask_dict["labels"].append(label or "florence")

    return mask_dict


def load_model_and_processor(
    model_name: str, device: str, torch_dtype: torch.dtype
) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load the Florence-2 model and processor with safe defaults."""
    from_pretrained_kwargs = {
        "dtype": torch_dtype,
        "trust_remote_code": True,
    }
    try:
        from_pretrained_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **from_pretrained_kwargs,
        )
    except TypeError:
        # Older transformer versions may not accept attn_implementation
        from_pretrained_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **from_pretrained_kwargs,
        )
    model = model.to(device).eval()
    # Some Florence checkpoints do not define `_supports_sdpa`, which recent
    # Transformers versions expect when deciding whether to use scaled-dot
    # product attention. Default it to False when missing.
    _ensure_sdpa_flag(model)

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True)
    return model, processor


_SDPA_PATCHED = False


def _patch_generation_defaults() -> None:
    """Ensure global transformer helpers expose SDPA hints we rely on."""
    global _SDPA_PATCHED
    if _SDPA_PATCHED:
        return

    try:
        if not hasattr(GenerationMixin, "_supports_sdpa"):
            setattr(GenerationMixin, "_supports_sdpa", False)
    except Exception:
        pass

    if Florence2ForConditionalGeneration is not None and not hasattr(
        Florence2ForConditionalGeneration, "_supports_sdpa"
    ):
        try:
            setattr(Florence2ForConditionalGeneration, "_supports_sdpa", False)
        except Exception:
            pass

    if Florence2ForConditionalGeneration is not None:
        cfg_cls = getattr(Florence2ForConditionalGeneration, "config_class", None)
        if cfg_cls is not None:
            for attr in ("_attn_implementation", "attn_implementation"):
                if not hasattr(cfg_cls, attr):
                    try:
                        setattr(cfg_cls, attr, "eager")
                    except Exception:
                        pass

    _SDPA_PATCHED = True


def _normalize_image_input(image: Any) -> Tuple[Image.Image, np.ndarray]:
    """Coerce different image inputs into a contiguous RGB representation."""
    if image is None:
        raise ValueError("Florence-2 received an empty image.")

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
        np_image = np.array(pil_image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        np_image = image
        if np_image.ndim == 2:
            np_image = np.stack([np_image] * 3, axis=-1)
        if np_image.ndim != 3:
            raise ValueError(f"Unsupported numpy image shape: {np_image.shape}")
        if np_image.shape[2] == 1:
            np_image = np.repeat(np_image, 3, axis=2)
        elif np_image.shape[2] >= 4:
            np_image = np_image[:, :, :3]
        np_image = np.ascontiguousarray(np_image.astype(np.uint8))
        pil_image = Image.fromarray(np_image)
    elif torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        np_image = tensor.numpy()
        return _normalize_image_input(np_image)
    else:
        raise TypeError(f"Unsupported image type for Florence-2: {type(image)}")

    if np_image.ndim != 3 or np_image.shape[2] != 3:
        raise ValueError(f"Florence-2 expects 3-channel RGB images, got {np_image.shape}")

    return pil_image, np_image


def _ensure_sdpa_flag(model: AutoModelForCausalLM) -> None:
    """
    Make sure the HF model object and all its submodules expose `_supports_sdpa`
    so newer generation helpers do not raise attribute errors.
    """
    _patch_generation_defaults()

    def _set_flag(module: Any) -> None:
        try:
            if not hasattr(module, "_supports_sdpa"):
                setattr(module, "_supports_sdpa", False)
        except Exception:
            pass
        module_cls = getattr(module, "__class__", None)
        if module_cls is not None:
            try:
                if not hasattr(module_cls, "_supports_sdpa"):
                    setattr(module_cls, "_supports_sdpa", False)
            except Exception:
                pass

    _set_flag(model)
    try:
        for _, submodule in model.named_modules():
            _set_flag(submodule)
    except Exception:
        pass

    try:
        for base in type(model).mro():
            if base in (object, GenerationMixin):
                continue
            if not hasattr(base, "_supports_sdpa"):
                try:
                    setattr(base, "_supports_sdpa", False)
                except Exception:
                    continue
    except Exception:
        pass

    try:
        if hasattr(model, "config"):
            cfg = model.config
            for attr in ("_attn_implementation", "attn_implementation"):
                if not getattr(cfg, attr, None):
                    setattr(cfg, attr, "eager")
    except Exception:
        pass

def florence2(
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    task_prompt: str,
    image: Image.Image,
    text_input: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Florence-2 for a single task prompt and image.
    """
    _ensure_sdpa_flag(model)
    prompt = f"{task_prompt}{text_input}" if text_input else task_prompt
    pil_image, np_image = _normalize_image_input(image)
    inputs = processor(text=[prompt], images=[np_image], return_tensors="pt")

    # Move tensors to the model device if possible
    device = model.device
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            inputs[key] = value.to(device)

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise ValueError(
            "Florence-2 processor returned no pixel values."
            " Ensure the image is valid and supported by the processor."
        )

    try:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    except AttributeError as exc:
        logger.warning(
            "Florence-2 generation hit AttributeError (%s); retrying with use_cache=False.",
            exc,
        )
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
            use_cache=False,
        )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=pil_image.size,
    )
    return parsed_answer


def _create_shape_points(
    polygon: Sequence[Tuple[float, float]]
) -> List[Any]:
    """Convert polygon coordinates into Qt points when available."""
    if QtCore is None:
        return [[float(x), float(y)] for x, y in polygon]
    return [QtCore.QPointF(float(x), float(y)) for x, y in polygon]  # type: ignore[attr-defined]


def create_shapes_from_mask_dict(
    mask_dict: Dict[str, FlorencePolygons],
    description: str = "florence",
) -> List[Shape]:
    """
    Create Shape objects from a Florence mask dictionary.

    Returns:
        list[Shape]: The shapes corresponding to the polygons.
    """
    label_list: List[Shape] = []
    polygons = mask_dict.get("polygons") or []
    labels = mask_dict.get("labels") or []

    for label, polygon_group in zip(labels, polygons):
        if not polygon_group:
            continue

        for polygon in polygon_group:
            if not polygon or len(polygon) < 3:
                logger.warning(
                    "Skipping invalid polygon with %d points: %s",
                    0 if polygon is None else len(polygon),
                    polygon,
                )
                continue

            points = _create_shape_points(polygon)
            current_shape = Shape(
                label=label or "florence",
                description=description,
                flags={},
            )
            current_shape.points = points  # type: ignore[assignment]
            if hasattr(current_shape, "close"):
                current_shape.close()
            label_list.append(current_shape)

    return label_list


def save_annotations(
    filename: str,
    mask_dict: Dict[str, FlorencePolygons],
    frame_shape: Tuple[int, int, int],
    caption: Optional[str] = None,
    description: str = "florence",
) -> List[Shape]:
    """
    Persist Florence-2 annotations in LabelMe format.
    """
    if len(frame_shape) >= 2:
        height, width = frame_shape[:2]
    else:
        raise ValueError("Invalid frame shape provided to save_annotations")
    image_path = os.path.splitext(filename)[0] + ".png"

    label_list = create_shapes_from_mask_dict(
        mask_dict, description=description)
    save_labels(
        filename=filename,
        imagePath=image_path,
        label_list=label_list,
        height=height,
        width=width,
        save_image_to_json=False,
        caption=caption,
    )
    return label_list


class Florence2Predictor:
    """
    High-level predictor wrapper that caches the Florence-2 model and processor.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name = model_name
        self._requested_device = device
        self._requested_dtype = torch_dtype
        self._model: Optional[AutoModelForCausalLM] = None
        self._processor: Optional[AutoProcessor] = None
        self._device: Optional[str] = None
        self._dtype: Optional[torch.dtype] = None

    def ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            _ensure_sdpa_flag(self._model)
            return
        device, torch_dtype = get_device_and_dtype()
        if self._requested_device:
            device = self._requested_device
        if self._requested_dtype:
            torch_dtype = self._requested_dtype

        logger.info("Loading Florence-2 model '%s' on %s", self.model_name, device)
        self._model, self._processor = load_model_and_processor(
            self.model_name, device, torch_dtype
        )
        _ensure_sdpa_flag(self._model)
        self._device = device
        self._dtype = torch_dtype

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self._dtype

    def predict(
        self,
        image: Any,
        *,
        text_input: Optional[str],
        segmentation_task: Optional[str] = "<REFERRING_EXPRESSION_SEGMENTATION>",
        include_caption: bool = True,
        caption_task: str = "<MORE_DETAILED_CAPTION>",
    ) -> Florence2Result:
        """
        Run Florence-2 on a PIL image and return segmentation/caption results.
        """
        self.ensure_loaded()
        assert self._model is not None and self._processor is not None

        raw_outputs: Dict[str, Any] = {}
        caption: Optional[str] = None
        mask_dict: Dict[str, FlorencePolygons] = {
            "polygons": [],
            "labels": [],
        }

        if include_caption and caption_task:
            caption_answer = florence2(
                self._processor,
                self._model,
                caption_task,
                image,
            )
            raw_outputs["caption"] = caption_answer
            caption = caption_answer.get(caption_task)

        if segmentation_task:
            # Ensure SDPA guard remains in effect prior to generation
            _ensure_sdpa_flag(self._model)
            segmentation_answer = florence2(
                self._processor,
                self._model,
                segmentation_task,
                image,
                text_input=text_input,
            )
            raw_outputs["segmentation"] = segmentation_answer
            mask_dict = convert_to_mask_dict(
                segmentation_answer,
                text_prompt=segmentation_task,
                text_input=text_input,
            )

        return Florence2Result(mask_dict=mask_dict, caption=caption, raw_outputs=raw_outputs)


def process_nth_frame_from_video(
    video_path: str,
    n: int,
    predictor: Florence2Predictor,
    *,
    segmentation_task: Optional[str],
    text_input: Optional[str],
    caption_task: Optional[str] = "<MORE_DETAILED_CAPTION>",
    description: str = "florence",
) -> None:
    """
    Stream a video and process every n-th frame, saving Florence annotations.

    This leverages Annolid's ``CV2Video`` loader to ensure consistent frame
    decoding and metadata reuse.
    """
    if n <= 0:
        raise ValueError("Parameter 'n' must be a positive integer.")

    video_reader = video_loader(video_path)
    if video_reader is None:
        raise ValueError(f"Cannot open video file: {video_path}")

    video_dir = os.path.splitext(video_path)[0]
    os.makedirs(video_dir, exist_ok=True)

    processed_frames = 0
    skipped_frames = 0

    total_frames = video_reader.total_frames()
    try:
        for frame_idx in range(0, total_frames, n):
            try:
                frame_rgb = video_reader.load_frame(frame_idx)
            except KeyError as exc:
                skipped_frames += 1
                logger.warning("Florence-2 skipped frame %s: %s", frame_idx, exc)
                continue

            if frame_rgb is None or not hasattr(frame_rgb, "shape"):
                skipped_frames += 1
                logger.warning(
                    "Florence-2 skipped frame %s: loader returned empty frame.",
                    frame_idx,
                )
                continue

            frame_shape = frame_rgb.shape
            if len(frame_shape) < 2:
                skipped_frames += 1
                logger.warning(
                    "Florence-2 skipped frame %s: unexpected frame shape %s",
                    frame_idx,
                    frame_shape,
                )
                continue

            try:
                result = predictor.predict(
                    frame_rgb,
                    text_input=text_input,
                    segmentation_task=segmentation_task,
                    include_caption=bool(caption_task),
                    caption_task=caption_task or "<MORE_DETAILED_CAPTION>",
                )

                mask_dict = result.mask_dict or {"polygons": [], "labels": []}
                filename = os.path.join(video_dir, f"{frame_idx:09}.json")
                save_annotations(
                    filename,
                    mask_dict,
                    frame_shape,
                    caption=result.caption,
                    description=description,
                )
                processed_frames += 1
            except Exception as exc:
                skipped_frames += 1
                logger.error(
                    "Florence-2 failed to process frame %s (%s): %s",
                    frame_idx,
                    video_path,
                    exc,
                    exc_info=True,
                )
                continue
    finally:
        try:
            video_reader.cap.release()
        except Exception:
            pass

    logger.info(
        "Florence-2 video processing finished: processed=%s skipped=%s path=%s",
        processed_frames,
        skipped_frames,
        video_path,
    )


def run_prediction(
    model_name: str,
    video_path: str,
    n: int,
    prompt: str,
    text_input: Optional[str],
    task: Optional[str] = "<REFERRING_EXPRESSION_SEGMENTATION>",
) -> None:
    """
    Entry point for the CLI usage of Florence-2 segmentation on videos.
    """
    predictor = Florence2Predictor(model_name=model_name)
    process_nth_frame_from_video(
        video_path,
        n,
        predictor,
        segmentation_task=task or prompt,
        text_input=text_input,
        caption_task="<MORE_DETAILED_CAPTION>",
        description="florence",
    )


if __name__ == "__main__":
    model_name = "microsoft/Florence-2-large"
    video_path = os.path.expanduser("mouse.mp4")
    n = 10
    text_input = "a black mouse"
    task = "<REFERRING_EXPRESSION_SEGMENTATION>"
    prompt = task

    run_prediction(model_name, video_path, n, prompt, text_input, task)
