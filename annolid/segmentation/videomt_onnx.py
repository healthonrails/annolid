"""VideoMT ONNX runtime integration for Annolid.

Source repository:
https://github.com/tue-mps/videomt

Citation:
@inproceedings{norouzi2026videomt,
  author     = {Norouzi, Narges and Zulfikar, Idil and Cavagnero, Niccol\`{o}
                and Kerssies, Tommie and Leibe, Bastian and Dubbelman, Gijs
                and {de Geus}, Daan},
  title      = {{VidEoMT: Your ViT is Secretly Also a Video Segmentation Model}},
  booktitle  = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year       = {2026},
}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from annolid.annotation.keypoints import save_labels
from annolid.annotation.masks import mask_to_polygons
from annolid.core.media.video import CV2Video
from annolid.gui.shape import Shape
from annolid.utils.annotation_store import load_labelme_json
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger


def _choose_seed_frame(requested_frame: int, available_frames: list[int]) -> int | None:
    if not available_frames:
        return None
    unique_sorted = sorted(set(int(v) for v in available_frames))
    req = int(requested_frame)
    if req in unique_sorted:
        return req
    previous = [idx for idx in unique_sorted if idx <= req]
    if previous:
        return previous[-1]
    return unique_sorted[0]


def _causal_window_indices(
    target_frame: int, window_size: int
) -> tuple[list[int], int]:
    """Return a fixed-size causal window ending at target_frame.

    Indices may be < 0 and should be clamped by caller to valid frame bounds.
    """
    w = max(1, int(window_size))
    end = int(target_frame)
    start = end - w + 1
    idx = list(range(start, end + 1))
    return idx, w - 1


def _preflight_validate_onnx_file(model_path: Path) -> None:
    """Fail fast for obviously invalid model files before ONNX Runtime init."""
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    if not path.is_file():
        raise RuntimeError(f"Model path is not a file: {path}")
    if path.suffix.lower() != ".onnx":
        raise RuntimeError(f"Expected an .onnx file, got: {path.name}")

    header = b""
    try:
        with path.open("rb") as handle:
            header = handle.read(4096)
    except Exception as exc:
        raise RuntimeError(f"Cannot read ONNX file: {path} ({exc})") from exc

    header_lc = header.lower()
    if (
        b"<!doctype html" in header_lc
        or b"<html" in header_lc
        or b"<?xml" in header_lc
        or b"<body" in header_lc
    ):
        raise RuntimeError(
            f"File is not an ONNX model (looks like HTML/XML): {path}. "
            "Please re-download the actual .onnx artifact."
        )

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = 0
    if size < 1024:
        raise RuntimeError(
            f"ONNX file appears truncated ({size} bytes): {path}. "
            "Please re-download the model."
        )

    # Optional deep validation when `onnx` is available. This catches malformed
    # protobuf content before entering ONNX Runtime.
    try:
        import onnx  # type: ignore
    except Exception:
        return
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid ONNX model file: {path}. Please re-download it. ({exc})"
        ) from exc


def _require_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional dependency 'onnxruntime' is required for VideoMT ONNX tracking. "
            "Install it with: pip install onnxruntime"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'onnxruntime' for VideoMT ONNX tracking. "
            f"Original error: {exc}"
        ) from exc
    return ort


@dataclass(slots=True)
class _SeedMask:
    label: str
    mask: np.ndarray  # bool [H, W] at model input resolution


def _infer_videomt_input_mode(input_shape: list[Any]) -> str:
    """Infer how to feed ONNX input tensors from shape metadata."""
    rank = len(input_shape)
    if rank == 5:
        return "video_5d"
    if rank != 4:
        raise RuntimeError(
            f"Unsupported VideoMT ONNX input rank {rank}; expected 4D or 5D."
        )

    dim0, dim1 = input_shape[0], input_shape[1]
    dim3 = input_shape[3] if len(input_shape) > 3 else None
    # Common clip layout: [T, 3, H, W] or [N, 3, H, W], often with fixed T/N=5.
    if dim1 == 3:
        if isinstance(dim0, int) and dim0 == 1:
            return "image_4d"
        return "clip_4d_nchw"
    # Common TensorFlow/CoreML layout: [T, H, W, 3] or [1, H, W, 3].
    if dim3 == 3:
        if isinstance(dim0, int) and dim0 == 1:
            return "image_4d_nhwc"
        return "clip_4d_nhwc"
    # Common image layout: [1, 3, H, W].
    if dim0 == 1:
        return "image_4d"
    # Fallback to clip mode to avoid underfeeding temporal models.
    return "clip_4d_nchw"


def _has_dynamic_temporal_axis(input_shape: list[Any], mode: str) -> bool:
    if mode == "video_5d":
        t_dim = input_shape[2] if len(input_shape) > 2 else None
    elif mode in {"clip_4d_nchw", "clip_4d_nhwc"}:
        t_dim = input_shape[0] if len(input_shape) > 0 else None
    else:
        return False
    return not (isinstance(t_dim, int) and t_dim > 0)


def _select_onnx_providers(
    available_providers: list[str], *, input_shape: list[Any], input_mode: str
) -> list[str]:
    available = set(available_providers or [])
    providers: list[str] = []

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    # CoreML is reliable for simple image-only models but has recurrent failures
    # for temporal clip/video models (sequence resize/runtime shape issues).
    # Keep it disabled for clip/video modes.
    dynamic_temporal = _has_dynamic_temporal_axis(input_shape, input_mode)
    allow_coreml = (input_mode in {"image_4d", "image_4d_nhwc"}) and (
        not dynamic_temporal
    )
    if allow_coreml and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")

    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


class VideoMTOnnxVideoProcessor:
    """Video instance segmentation backend for VideoMT ONNX models."""

    def __init__(self, video_path: str, *args: Any, **kwargs: Any):
        _ = args
        self.video_path = str(video_path)
        self.video_loader = CV2Video(self.video_path)
        first_frame = self.video_loader.get_first_frame()
        self.video_height, self.video_width = (
            int(first_frame.shape[0]),
            int(first_frame.shape[1]),
        )
        self.num_frames = int(self.video_loader.total_frames())
        self.results_folder = Path(
            kwargs.get("results_folder") or Path(self.video_path).with_suffix("")
        )
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.pred_worker = None

        self.mask_threshold = float(kwargs.get("videomt_mask_threshold", 0.5))
        self.logit_threshold = float(kwargs.get("videomt_logit_threshold", -2.0))
        self.seed_iou_threshold = float(kwargs.get("videomt_seed_iou_threshold", 0.01))
        self.description = str(kwargs.get("videomt_description") or "VideoMT")

        self.model_path = self._resolve_model_path(kwargs)
        _preflight_validate_onnx_file(self.model_path)
        self._coreml_retry_done = False
        self._ort_available_providers = []
        self._providers: list[str] = []
        self._input_name = ""
        self._input_shape: list[Any] = []
        self._input_rank = 0
        self.input_mode = ""
        self.input_height = 0
        self.input_width = 0
        self.temporal_window = 1

        self._session = self._build_session(self.model_path)
        self._initialize_input_layout(kwargs)

        logger.info(
            "Initialized VideoMT ONNX processor: model=%s mode=%s size=%sx%s window=%s providers=%s",
            self.model_path,
            self.input_mode,
            self.input_width,
            self.input_height,
            self.temporal_window,
            self._providers,
        )

    def set_pred_worker(self, pred_worker: Any) -> None:
        self.pred_worker = pred_worker

    def get_total_frames(self) -> int:
        return int(self.num_frames)

    def cleanup(self) -> None:
        try:
            self.video_loader.release()
        except Exception:
            pass
        self._session = None

    def _resolve_model_path(self, kwargs: dict[str, Any]) -> Path:
        candidates: list[str] = []
        for key in (
            "model_weight",
            "model_path",
            "onnx_path",
            "weight_file",
            "model_name",
        ):
            value = str(kwargs.get(key) or "").strip()
            if value and value.lower().endswith(".onnx"):
                candidates.append(value)

        candidates.extend(
            [
                "downloads/videomt_yt_2019_vit_small_52.8.onnx",
                str(
                    Path.home()
                    / ".annolid/workspace/downloads/videomt_yt_2019_vit_small_52.8.onnx"
                ),
            ]
        )

        for candidate in candidates:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if path.exists():
                return path
        raise FileNotFoundError(
            "VideoMT ONNX model not found. Expected e.g. "
            "'downloads/videomt_yt_2019_vit_small_52.8.onnx'."
        )

    def _build_session(self, model_path: Path):
        ort = _require_onnxruntime()
        self._ort_available_providers = list(ort.get_available_providers() or [])
        providers = list(self._providers or [])
        if not providers:
            available = set(self._ort_available_providers)
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            if not providers:
                providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            return ort.InferenceSession(
                str(model_path), providers=providers, sess_options=opts
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize VideoMT ONNX session. "
                f"Model path: {model_path}. Error: {exc}"
            ) from exc

    def _initialize_input_layout(self, kwargs: dict[str, Any]) -> None:
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = list(self._session.get_inputs()[0].shape)
        self._input_rank = len(self._input_shape)
        self.input_mode = _infer_videomt_input_mode(self._input_shape)

        selected = _select_onnx_providers(
            self._ort_available_providers,
            input_shape=self._input_shape,
            input_mode=self.input_mode,
        )
        if selected != self._providers:
            self._providers = selected
            self._session = self._build_session(self.model_path)
            self._input_name = self._session.get_inputs()[0].name
            self._input_shape = list(self._session.get_inputs()[0].shape)
            self._input_rank = len(self._input_shape)
            self.input_mode = _infer_videomt_input_mode(self._input_shape)

        self.input_height, self.input_width = self._resolve_input_hw(kwargs)
        self.temporal_window = self._resolve_temporal_window(kwargs)

    def _should_retry_without_coreml(self, exc: Exception) -> bool:
        if self._coreml_retry_done:
            return False
        if "CoreMLExecutionProvider" not in self._providers:
            return False
        text = str(exc)
        if "CoreMLExecutionProvider" in text:
            return True
        if "dynamically resizing for sequence length" in text:
            return True
        return False

    def _rebuild_cpu_only_session(self) -> None:
        self._providers = ["CPUExecutionProvider"]
        self._session = self._build_session(self.model_path)
        self._input_name = self._session.get_inputs()[0].name

    def _resolve_input_hw(self, kwargs: dict[str, Any]) -> tuple[int, int]:
        shape_h = None
        shape_w = None
        if self.input_mode == "video_5d":
            shape_h = self._input_shape[3] if len(self._input_shape) > 3 else None
            shape_w = self._input_shape[4] if len(self._input_shape) > 4 else None
        elif self.input_mode in {"clip_4d_nchw", "image_4d"}:
            shape_h = self._input_shape[2] if len(self._input_shape) > 2 else None
            shape_w = self._input_shape[3] if len(self._input_shape) > 3 else None
        elif self.input_mode in {"clip_4d_nhwc", "image_4d_nhwc"}:
            shape_h = self._input_shape[1] if len(self._input_shape) > 1 else None
            shape_w = self._input_shape[2] if len(self._input_shape) > 2 else None

        fallback_h = int(kwargs.get("videomt_input_height") or self.video_height)
        fallback_w = int(kwargs.get("videomt_input_width") or self.video_width)

        height = (
            int(shape_h) if isinstance(shape_h, int) and shape_h > 0 else fallback_h
        )
        width = int(shape_w) if isinstance(shape_w, int) and shape_w > 0 else fallback_w
        return height, width

    def _resolve_temporal_window(self, kwargs: dict[str, Any]) -> int:
        if self.input_mode in {"image_4d", "image_4d_nhwc"}:
            return 1
        if self.input_mode == "video_5d":
            shape_t = self._input_shape[2] if len(self._input_shape) > 2 else None
            if isinstance(shape_t, int) and shape_t > 0:
                return int(shape_t)
        elif self.input_mode in {"clip_4d_nchw", "clip_4d_nhwc"}:
            shape_t = self._input_shape[0] if len(self._input_shape) > 0 else None
            if isinstance(shape_t, int) and shape_t > 0:
                return int(shape_t)
        return max(1, int(kwargs.get("videomt_window") or 8))

    def _should_stop(self) -> bool:
        if self.pred_worker is None:
            return False
        return bool(getattr(self.pred_worker, "is_stopped", lambda: False)())

    def _frame_json_path(self, frame_idx: int) -> Path:
        return (
            self.results_folder
            / f"{self.results_folder.name}_{int(frame_idx):09d}.json"
        )

    def _resolve_seed_json(self, frame_idx: int) -> Path:
        primary = self._frame_json_path(frame_idx)
        legacy = self.results_folder / f"{int(frame_idx):09d}.json"
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy

        candidates_by_frame: dict[int, Path] = {}
        for path in self.results_folder.glob("*.json"):
            try:
                idx = int(get_frame_number_from_json(path.name))
            except Exception:
                continue
            candidates_by_frame.setdefault(idx, path)

        for filename in find_manual_labeled_json_files(str(self.results_folder)):
            try:
                idx = int(get_frame_number_from_json(filename))
                candidate = self.results_folder / filename
                if candidate.exists():
                    candidates_by_frame.setdefault(idx, candidate)
            except Exception:
                continue

        chosen = _choose_seed_frame(int(frame_idx), list(candidates_by_frame.keys()))
        if chosen is not None:
            chosen_path = candidates_by_frame.get(chosen)
            if chosen_path is not None and chosen_path.exists():
                if int(chosen) != int(frame_idx):
                    logger.info(
                        "VideoMT seed fallback: requested frame %s, using seed frame %s (%s).",
                        int(frame_idx),
                        int(chosen),
                        chosen_path.name,
                    )
                return chosen_path

        available = sorted(candidates_by_frame.keys())
        raise FileNotFoundError(
            "No seed annotation JSON found for requested frame "
            f"{frame_idx} in {self.results_folder}. Available seed frames: {available}"
        )

    def _frame_index_from_json_path(self, path: Path) -> int:
        try:
            return int(get_frame_number_from_json(path.name))
        except Exception:
            return 0

    def _load_seed_masks(self, seed_json_path: Path) -> list[_SeedMask]:
        payload = load_labelme_json(seed_json_path)
        shapes = payload.get("shapes", []) or []
        sx = float(self.input_width) / float(max(self.video_width, 1))
        sy = float(self.input_height) / float(max(self.video_height, 1))
        by_label: dict[str, np.ndarray] = {}
        for shape in shapes:
            if str(shape.get("shape_type") or "").lower() != "polygon":
                continue
            label = str(shape.get("label") or "").strip()
            points = shape.get("points") or []
            if not label or len(points) < 3:
                continue
            pts = np.asarray(points, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 2:
                continue
            pts[:, 0] = np.clip(pts[:, 0] * sx, 0, max(self.input_width - 1, 0))
            pts[:, 1] = np.clip(pts[:, 1] * sy, 0, max(self.input_height - 1, 0))
            polygon = np.round(pts[:, :2]).astype(np.int32)
            mask = np.zeros((self.input_height, self.input_width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            if label in by_label:
                by_label[label] = np.logical_or(by_label[label], mask > 0)
            else:
                by_label[label] = mask > 0
        return [
            _SeedMask(label=label, mask=mask.astype(bool))
            for label, mask in by_label.items()
        ]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _prepare_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        return self._prepare_frame_to(frame_rgb, self.input_height, self.input_width)

    def _prepare_frame_to(
        self, frame_rgb: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        if frame_rgb.shape[0] != int(target_h) or frame_rgb.shape[1] != int(target_w):
            frame_rgb = cv2.resize(
                frame_rgb,
                (int(target_w), int(target_h)),
                interpolation=cv2.INTER_LINEAR,
            )
        arr = frame_rgb.astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))

    def _prepare_frame_to_nhwc(
        self, frame_rgb: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        if frame_rgb.shape[0] != int(target_h) or frame_rgb.shape[1] != int(target_w):
            frame_rgb = cv2.resize(
                frame_rgb,
                (int(target_w), int(target_h)),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame_rgb.astype(np.float32) / 255.0

    def _parse_outputs(
        self, outputs: list[np.ndarray], output_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        by_name = {name: value for name, value in zip(output_names, outputs)}
        logits = by_name.get("pred_logits")
        masks = by_name.get("pred_masks")
        if logits is None:
            logits = next((x for x in outputs if getattr(x, "ndim", 0) == 3), None)
        if masks is None:
            masks = next((x for x in outputs if getattr(x, "ndim", 0) == 5), None)
        if logits is None or masks is None:
            raise RuntimeError(
                "VideoMT ONNX outputs must include 'pred_logits' [1,Q,C] and "
                "'pred_masks' [1,Q,T,H,W]."
            )
        if logits.shape[0] != 1 or masks.shape[0] != 1:
            raise RuntimeError("VideoMT ONNX outputs must have batch size 1.")
        return logits, masks

    def _run_inference(self, input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            outputs = self._session.run(None, {self._input_name: input_tensor})
        except Exception as exc:
            if self._should_retry_without_coreml(exc):
                logger.warning(
                    "VideoMT ONNX CoreML runtime failure detected; retrying on CPU. Error: %s",
                    exc,
                )
                self._coreml_retry_done = True
                self._rebuild_cpu_only_session()
                outputs = self._session.run(None, {self._input_name: input_tensor})
            else:
                raise
        output_names = [out.name for out in self._session.get_outputs()]
        return self._parse_outputs(outputs, output_names)

    def _foreground_scores(self, logits: np.ndarray) -> np.ndarray:
        qxc = np.asarray(logits[0], dtype=np.float32)
        if qxc.shape[1] <= 1:
            return qxc[:, 0]
        bg = qxc[:, -1]
        fg = np.max(qxc[:, :-1], axis=1)
        return fg - bg

    def _binarize_masks(self, masks: np.ndarray) -> np.ndarray:
        qthw = np.asarray(masks[0], dtype=np.float32)
        if np.min(qthw) < 0.0 or np.max(qthw) > 1.0:
            qthw = self._sigmoid(qthw)
        return qthw >= self.mask_threshold

    @staticmethod
    def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        mask_a = np.squeeze(np.asarray(mask_a).astype(bool))
        mask_b = np.squeeze(np.asarray(mask_b).astype(bool))
        if mask_a.ndim != 2 or mask_b.ndim != 2:
            return 0.0
        if mask_a.shape != mask_b.shape:
            mask_a = cv2.resize(
                mask_a.astype(np.uint8),
                (int(mask_b.shape[1]), int(mask_b.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        inter = float(np.logical_and(mask_a, mask_b).sum())
        if inter <= 0:
            return 0.0
        union = float(np.logical_or(mask_a, mask_b).sum())
        return inter / max(union, 1.0)

    def _build_query_label_map(
        self,
        binary_masks: np.ndarray,
        scores: np.ndarray,
        seed_masks: list[_SeedMask],
        *,
        seed_timestep: int = 0,
    ) -> dict[int, str]:
        query_map: dict[int, str] = {}
        if not seed_masks or binary_masks.shape[1] < 1:
            return query_map
        used_labels: set[str] = set()
        used_queries: set[int] = set()
        t = int(np.clip(int(seed_timestep), 0, max(binary_masks.shape[1] - 1, 0)))
        frame0 = binary_masks[:, t, :, :]

        # First pass: confidence-aware matching.
        for query_idx in np.argsort(scores)[::-1]:
            if float(scores[query_idx]) < self.logit_threshold:
                continue
            best_label = None
            best_iou = 0.0
            query_mask = frame0[int(query_idx)]
            if not np.any(query_mask):
                continue
            for seed in seed_masks:
                if seed.label in used_labels:
                    continue
                iou = self._iou(query_mask, seed.mask)
                if iou > best_iou:
                    best_iou = iou
                    best_label = seed.label
            if best_label is None or best_iou < self.seed_iou_threshold:
                continue
            query_map[int(query_idx)] = best_label
            used_labels.add(best_label)
            used_queries.add(int(query_idx))

        # Second pass: recover still-unmatched labels by best IoU, even when
        # logits are weak, so seeded instances are less likely to disappear.
        remaining = [seed for seed in seed_masks if seed.label not in used_labels]
        if remaining:
            candidate_queries = list(np.argsort(scores)[::-1])
            for seed in remaining:
                best_query = None
                best_iou = 0.0
                for query_idx in candidate_queries:
                    q_idx = int(query_idx)
                    if q_idx in used_queries:
                        continue
                    query_mask = frame0[q_idx]
                    if not np.any(query_mask):
                        continue
                    iou = self._iou(query_mask, seed.mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_query = q_idx
                if best_query is None or best_iou <= 0.0:
                    continue
                query_map[best_query] = seed.label
                used_queries.add(best_query)
                used_labels.add(seed.label)
        return query_map

    @staticmethod
    def _should_skip_saving_frame(
        frame_idx: int, requested_start_frame: int, seed_frame_idx: int
    ) -> bool:
        # Never overwrite the seed frame annotation.
        if int(frame_idx) == int(seed_frame_idx):
            return True
        # Guard against padded/causal frames before requested range.
        if int(frame_idx) < int(requested_start_frame):
            return True
        return False

    def _mask_to_video_size(self, mask_hw: np.ndarray) -> np.ndarray:
        if (
            mask_hw.shape[0] == self.video_height
            and mask_hw.shape[1] == self.video_width
        ):
            return mask_hw.astype(bool)
        resized = cv2.resize(
            mask_hw.astype(np.uint8),
            (self.video_width, self.video_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized > 0

    def _save_frame_masks(
        self, frame_idx: int, masks_by_label: dict[str, np.ndarray]
    ) -> bool:
        if not masks_by_label:
            return False
        label_list: list[Shape] = []
        for label, mask in masks_by_label.items():
            if not np.any(mask):
                continue
            polygons, _ = mask_to_polygons(mask.astype(np.uint8))
            for polygon in polygons:
                coords = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if coords.shape[0] < 3:
                    continue
                shape = Shape(
                    label=label,
                    shape_type="polygon",
                    flags={},
                    description=self.description,
                )
                shape.points = [[float(x), float(y)] for x, y in coords]
                label_list.append(shape)
        if not label_list:
            return False

        save_labels(
            filename=self._frame_json_path(frame_idx),
            imagePath="",
            label_list=label_list,
            height=self.video_height,
            width=self.video_width,
            save_image_to_json=False,
            persist_json=False,
        )
        return True

    def _run_5d_clip(
        self, frames_rgb: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        clip = np.stack([self._prepare_frame(frame) for frame in frames_rgb], axis=0)
        # [T,C,H,W] -> [1,C,T,H,W]
        input_tensor = np.transpose(clip, (1, 0, 2, 3))[None].astype(np.float32)
        return self._run_inference(input_tensor)

    def _run_4d_clip_nchw(
        self, frames_rgb: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        strategies: list[tuple[str, int, int]] = []
        seen: set[tuple[str, int, int]] = set()

        def add(layout: str, h: int, w: int) -> None:
            key = (layout, int(h), int(w))
            if key in seen:
                return
            seen.add(key)
            strategies.append(key)

        add("nchw", self.input_height, self.input_width)
        if self.input_height != self.input_width:
            add("nchw", self.input_width, self.input_height)

        last_exc: Exception | None = None
        attempted_error_dims = False
        idx = 0
        while idx < len(strategies):
            layout, h, w = strategies[idx]
            idx += 1
            try:
                if layout == "nchw":
                    tensor = np.stack(
                        [
                            self._prepare_frame_to(frame, target_h=h, target_w=w)
                            for frame in frames_rgb
                        ],
                        axis=0,
                    ).astype(np.float32)
                else:
                    tensor = np.stack(
                        [
                            self._prepare_frame_to_nhwc(frame, target_h=h, target_w=w)
                            for frame in frames_rgb
                        ],
                        axis=0,
                    ).astype(np.float32)
                logits, pred_masks = self._run_inference(tensor)
                self.input_height, self.input_width = int(h), int(w)
                self.input_mode = "clip_4d_nhwc" if layout == "nhwc" else "clip_4d_nchw"
                return logits, pred_masks
            except Exception as exc:
                last_exc = exc
                if self._is_shape_related_error(exc) and not attempted_error_dims:
                    dims = self._extract_broadcast_dims(exc)
                    if dims is not None:
                        a, b = dims
                        actual_tokens = int(min(a, b))
                        expected_tokens = int(max(a, b))
                        inferred_square = self._infer_square_side_from_token_mismatch(
                            actual_tokens=actual_tokens,
                            expected_tokens=expected_tokens,
                            current_h=int(h),
                            current_w=int(w),
                        )
                        logger.warning(
                            "VideoMT ONNX broadcast mismatch (%s). Retrying using error-derived sizes (%sx%s)%s.",
                            exc,
                            a,
                            b,
                            (
                                f" and inferred square {inferred_square}x{inferred_square}"
                                if inferred_square is not None
                                else ""
                            ),
                        )
                        add("nchw", a, b)
                        add("nchw", b, a)
                        if inferred_square is not None:
                            add("nchw", inferred_square, inferred_square)
                        attempted_error_dims = True
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "VideoMT 4D clip inference failed with no strategies attempted."
        )

    def _run_4d_clip_nhwc(
        self, frames_rgb: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        clip = np.stack(
            [
                self._prepare_frame_to_nhwc(frame, self.input_height, self.input_width)
                for frame in frames_rgb
            ],
            axis=0,
        )
        input_tensor = clip.astype(np.float32)
        return self._run_inference(input_tensor)

    @staticmethod
    def _is_shape_related_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "invalid dimensions" in text
            or "broadcast" in text
            or "attempting to broadcast" in text
            or "shape" in text
        )

    @staticmethod
    def _extract_broadcast_dims(exc: Exception) -> tuple[int, int] | None:
        text = str(exc)
        # Example: "... Attempting to broadcast ... 540 by 1600"
        match = re.search(r"(\d+)\s+by\s+(\d+)", text)
        if match is None:
            return None
        try:
            a = int(match.group(1))
            b = int(match.group(2))
        except Exception:
            return None
        if a <= 0 or b <= 0:
            return None
        return a, b

    @staticmethod
    def _infer_square_side_from_token_mismatch(
        *, actual_tokens: int, expected_tokens: int, current_h: int, current_w: int
    ) -> int | None:
        """Infer a likely square image side from token-count mismatch.

        For ViT-like patchification:
          tokens ~= (H / p) * (W / p)
        If `actual_tokens` came from current (H, W) and model expects
        `expected_tokens`, infer patch size `p` then back-calculate square side.
        """
        if (
            actual_tokens <= 0
            or expected_tokens <= 0
            or current_h <= 0
            or current_w <= 0
        ):
            return None
        patch = np.sqrt((float(current_h) * float(current_w)) / float(actual_tokens))
        if not np.isfinite(patch) or patch <= 0:
            return None
        grid = np.sqrt(float(expected_tokens))
        if not np.isfinite(grid) or grid <= 0:
            return None
        side = int(round(grid * patch))
        patch_int = max(1, int(round(patch)))
        side = max(patch_int, int(round(float(side) / float(patch_int))) * patch_int)
        if side <= 0 or side > 4096:
            return None
        return int(side)

    def _run_4d_frame(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.input_mode == "image_4d_nhwc":
            input_tensor = self._prepare_frame_to_nhwc(
                frame_rgb, self.input_height, self.input_width
            )[None].astype(np.float32)
        else:
            input_tensor = self._prepare_frame(frame_rgb)[None].astype(np.float32)
        return self._run_inference(input_tensor)

    def process_video_frames(
        self,
        start_frame: int = 0,
        end_frame: int = -1,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        if self.num_frames <= 0:
            return "Video has no frames."

        start = max(0, int(start_frame))
        end = (
            self.num_frames - 1
            if int(end_frame) < 0
            else min(int(end_frame), self.num_frames - 1)
        )
        if start > end:
            return f"Invalid frame range: start={start}, end={end}."

        seed_json = self._resolve_seed_json(start)
        seed_frame_idx = self._frame_index_from_json_path(seed_json)
        seed_masks = self._load_seed_masks(seed_json)
        if not seed_masks:
            return (
                f"No valid seed polygons found in {seed_json.name}. "
                "Please label at least one polygon on the seed frame."
            )

        query_map: dict[int, str] = {}
        saved = 0

        if self.input_mode == "video_5d":
            for chunk_start in range(start, end + 1, self.temporal_window):
                if self._should_stop():
                    return f"VideoMT ONNX stopped by user. Saved {saved} frames."
                chunk_end = min(end, chunk_start + self.temporal_window - 1)
                frame_indices = list(range(chunk_start, chunk_end + 1))
                frames = [self.video_loader.load_frame(idx) for idx in frame_indices]
                if len(frames) < self.temporal_window:
                    pad = frames[-1]
                    frames.extend([pad] * (self.temporal_window - len(frames)))

                logits, pred_masks = self._run_5d_clip(frames)
                scores = self._foreground_scores(logits)
                binary_masks = self._binarize_masks(pred_masks)
                if not query_map:
                    seed_timestep = 0
                    if seed_frame_idx in frame_indices:
                        seed_timestep = frame_indices.index(seed_frame_idx)
                    query_map = self._build_query_label_map(
                        binary_masks, scores, seed_masks, seed_timestep=seed_timestep
                    )
                    if not query_map:
                        return (
                            "VideoMT ONNX could not match model queries to seed instances. "
                            "Try a clearer seed frame or adjust threshold settings."
                        )

                for local_idx, global_idx in enumerate(frame_indices):
                    if self._should_skip_saving_frame(
                        global_idx, start, seed_frame_idx
                    ):
                        continue
                    masks_by_label: dict[str, np.ndarray] = {}
                    for query_idx, label in query_map.items():
                        if (
                            query_idx >= binary_masks.shape[0]
                            or local_idx >= binary_masks.shape[1]
                        ):
                            continue
                        mask = binary_masks[query_idx, local_idx]
                        if not np.any(mask):
                            continue
                        resized = self._mask_to_video_size(mask)
                        if label in masks_by_label:
                            masks_by_label[label] = np.logical_or(
                                masks_by_label[label], resized
                            )
                        else:
                            masks_by_label[label] = resized
                    if self._save_frame_masks(global_idx, masks_by_label):
                        saved += 1
        elif self.input_mode in {"clip_4d_nchw", "clip_4d_nhwc"}:
            for global_idx in range(start, end + 1):
                if self._should_stop():
                    return f"VideoMT ONNX stopped by user. Saved {saved} frames."
                raw_window, local_idx = _causal_window_indices(
                    target_frame=global_idx, window_size=self.temporal_window
                )
                frame_indices = [
                    int(np.clip(idx, 0, max(self.num_frames - 1, 0)))
                    for idx in raw_window
                ]
                frames = [self.video_loader.load_frame(idx) for idx in frame_indices]
                if self.input_mode == "clip_4d_nhwc":
                    logits, pred_masks = self._run_4d_clip_nhwc(frames)
                else:
                    logits, pred_masks = self._run_4d_clip_nchw(frames)
                scores = self._foreground_scores(logits)
                binary_masks = self._binarize_masks(pred_masks)
                if not query_map:
                    seed_timestep = 0
                    if seed_frame_idx in frame_indices:
                        seed_timestep = frame_indices.index(seed_frame_idx)
                    query_map = self._build_query_label_map(
                        binary_masks, scores, seed_masks, seed_timestep=seed_timestep
                    )
                    if not query_map:
                        return (
                            "VideoMT ONNX could not match model queries to seed instances. "
                            "Try a clearer seed frame or adjust threshold settings."
                        )
                if self._should_skip_saving_frame(global_idx, start, seed_frame_idx):
                    continue
                masks_by_label: dict[str, np.ndarray] = {}
                for query_idx, label in query_map.items():
                    if (
                        query_idx >= binary_masks.shape[0]
                        or local_idx >= binary_masks.shape[1]
                    ):
                        continue
                    mask = binary_masks[query_idx, local_idx]
                    if not np.any(mask):
                        continue
                    resized = self._mask_to_video_size(mask)
                    if label in masks_by_label:
                        masks_by_label[label] = np.logical_or(
                            masks_by_label[label], resized
                        )
                    else:
                        masks_by_label[label] = resized
                if self._save_frame_masks(global_idx, masks_by_label):
                    saved += 1
        else:
            for frame_idx in range(start, end + 1):
                if self._should_stop():
                    return f"VideoMT ONNX stopped by user. Saved {saved} frames."
                frame = self.video_loader.load_frame(frame_idx)
                logits, pred_masks = self._run_4d_frame(frame)
                scores = self._foreground_scores(logits)
                binary_masks = self._binarize_masks(pred_masks)
                if binary_masks.shape[1] < 1:
                    continue
                if not query_map:
                    query_map = self._build_query_label_map(
                        binary_masks, scores, seed_masks
                    )
                    if not query_map:
                        return (
                            "VideoMT ONNX could not match model queries to seed instances. "
                            "Try a clearer seed frame or adjust threshold settings."
                        )
                if self._should_skip_saving_frame(frame_idx, start, seed_frame_idx):
                    continue
                masks_by_label: dict[str, np.ndarray] = {}
                for query_idx, label in query_map.items():
                    if query_idx >= binary_masks.shape[0]:
                        continue
                    mask = binary_masks[query_idx, 0]
                    if not np.any(mask):
                        continue
                    resized = self._mask_to_video_size(mask)
                    if label in masks_by_label:
                        masks_by_label[label] = np.logical_or(
                            masks_by_label[label], resized
                        )
                    else:
                        masks_by_label[label] = resized
                if self._save_frame_masks(frame_idx, masks_by_label):
                    saved += 1

        return f"VideoMT ONNX completed. Saved predictions for {saved} frame(s)."
