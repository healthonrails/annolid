"""AI prompt, SAM/GroundingDINO/CountGD orchestration, and async worker mixin for Canvas."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import inspect
import os
import sys
import threading
import time
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from qtpy import QtCore, QtWidgets

from annolid.annotation.polygon_constraints import (
    make_instance_masks_exclusive as _default_make_instance_masks_exclusive,
    resolve_polygon_shape_conflicts as _default_resolve_polygon_shape_conflicts,
)
from annolid.gui.cursor_utils import set_widget_busy_cursor
from annolid.gui.qt_threads import (
    DEFAULT_WORKER_THREAD_STACK_SIZE,
    configure_worker_thread_stack,
)
from annolid.gui.shape import MaskShape, Shape
from annolid.gui.workers import FlexibleWorker
from annolid.utils.annotation_compat import (
    AI_MODELS as _DEFAULT_AI_MODELS,
    utils as _default_utils,
)
from annolid.utils.cancellation import InferenceCancelledError, ObservableStopEvent
from annolid.utils.devices import clear_device_cache
from annolid.utils.logger import logger
from annolid.utils.mask_geometry import tight_mask_bbox
from annolid.utils.prompts import (
    DetectionBoxCandidate,
    assign_instance_labels,
    is_sliver_mask_remnant,
    match_predicted_phrase_to_label,
    parse_every_prompt,
    parse_text_prompt,
    sanitize_and_nms_box_candidates,
)
from annolid.utils.qt2cv import convert_qt_image_to_rgb_cv_image


def _canvas_attr(name: str, default):
    """Resolve monkeypatchable symbols from annolid.gui.widgets.canvas when loaded."""
    mod = sys.modules.get("annolid.gui.widgets.canvas")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default


_PROMPT_WORKER_STACK_SIZE = DEFAULT_WORKER_THREAD_STACK_SIZE
_ANCHORED_QT_WORKERS_LOCK = threading.Lock()
_ANCHORED_QT_WORKERS_SEQ = 0
_ANCHORED_QT_WORKERS: dict[int, tuple[QtCore.QThread, FlexibleWorker]] = {}


def _anchor_qt_worker(thread: QtCore.QThread, worker: FlexibleWorker) -> int:
    global _ANCHORED_QT_WORKERS_SEQ
    with _ANCHORED_QT_WORKERS_LOCK:
        _ANCHORED_QT_WORKERS_SEQ += 1
        anchor_id = _ANCHORED_QT_WORKERS_SEQ
        _ANCHORED_QT_WORKERS[anchor_id] = (thread, worker)
    return anchor_id


def _release_anchored_qt_worker(anchor_id: int) -> None:
    with _ANCHORED_QT_WORKERS_LOCK:
        entry = _ANCHORED_QT_WORKERS.pop(anchor_id, None)
    if entry is not None:
        thread, worker = entry
        try:
            worker.deleteLater()
        except RuntimeError:
            pass
        try:
            thread.deleteLater()
        except RuntimeError:
            pass


@dataclass(frozen=True)
class TextPromptComputationResult:
    """Thread-safe output produced by background or synchronous text-prompt inference."""

    shapes: list[Shape]
    priorities: list[float]
    conflicts_resolved: bool = False
    warning_title: str | None = None
    warning_message: str | None = None
    cancelled: bool = False


class CanvasAiPromptMixin:
    """AI model lifecycle, text-prompt segmentation, and async worker orchestration."""

    @staticmethod
    def _release_device_cache() -> None:
        """Best-effort cache release after unloading GPU-backed models."""
        gc.collect()
        clear_device_cache()

    def _release_model(self, model, *, context: str) -> None:
        """Best-effort close hook for optional AI model instances."""
        if model is None:
            return
        try:
            close_fn = getattr(model, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            logger.debug("Failed to close %s model cleanly.", context, exc_info=True)
        self._release_device_cache()

    def _release_all_prompt_models(self) -> None:
        """Close and clear all prompt/segmentation models held by Canvas (caller holds lock)."""
        self._release_model(self._ai_model, context="canvas AI")
        self._release_model(self.sam_hq_model, context="SAM HQ")
        self._release_model(self._ai_model_rect, context="rectangle detector")
        self._release_model(self._countgd_counter, context="CountGD")
        self._ai_model = None
        self._ai_model_rgb_digest = None
        self.sam_hq_model = None
        self._ai_model_rect = None
        self._countgd_counter = None
        self._pending_close_model_release = False

    def initializeAiModel(self, name, _custom_ai_models=None):
        init_start = time.perf_counter()
        model_class = self._resolve_ai_model_class(
            name=name,
            custom_model_names=_custom_ai_models,
        )
        if model_class is None:
            elapsed_ms = (time.perf_counter() - init_start) * 1000.0
            logger.info("AI model init skipped for '%s' (%.1fms).", name, elapsed_ms)
            return

        reused_model = (
            self._ai_model is not None and self._ai_model.name == model_class.name
        )
        if reused_model:
            logger.debug("AI model is already initialized: %r" % model_class.name)
        else:
            logger.debug("Initializing AI model: %r" % model_class.name)
            with self._prompt_inference_lock:
                self._release_model(self._ai_model, context="previous AI")
                self._ai_model = None
                self._ai_model_pixmap_key = None
                self._ai_model_rgb_digest = None
                self._ai_model = self._instantiate_ai_model(
                    model_class,
                    requested_name=name,
                )
            if self._ai_model is None:
                elapsed_ms = (time.perf_counter() - init_start) * 1000.0
                logger.warning(
                    "AI model init failed for '%s' in %.1fms.",
                    getattr(model_class, "name", name),
                    elapsed_ms,
                )
                return

        # Check if self.pixmap is None before calling isNull()
        if self.pixmap is None or self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return

        # Avoid rebuilding embeddings for the same frame/model pair.
        self._sync_ai_model_image(force=not reused_model)
        elapsed_ms = (time.perf_counter() - init_start) * 1000.0
        if reused_model:
            logger.info(
                "AI model '%s' reused and synced in %.1fms.",
                getattr(model_class, "name", name),
                elapsed_ms,
            )
        else:
            logger.info(
                "AI model '%s' initialized and synced in %.1fms.",
                getattr(model_class, "name", name),
                elapsed_ms,
            )

    def _resolve_ai_model_class(self, name: str, custom_model_names=None):
        model_class = None
        for candidate in _canvas_attr("AI_MODELS", _DEFAULT_AI_MODELS):
            if candidate.name == name:
                model_class = candidate
                break

        if model_class is None:
            if custom_model_names and name in custom_model_names:
                logger.debug(
                    "Custom model selected: %s, skipping SAM initialization", name
                )
                return None
            for candidate in _canvas_attr("AI_MODELS", _DEFAULT_AI_MODELS):
                if getattr(candidate, "name", None) == "EfficientSam (speed)":
                    model_class = candidate
                    break
            ai_models = _canvas_attr("AI_MODELS", _DEFAULT_AI_MODELS)
            if model_class is None and ai_models:
                model_class = ai_models[0]
            if model_class is None:
                logger.warning(
                    "No AI segmentation models are available; skipping initialization."
                )
                return None
        return model_class

    def _instantiate_ai_model(self, model_class, *, requested_name: str):
        try:
            return model_class()
        except Exception as exc:
            self._ai_model = None
            QtWidgets.QMessageBox.warning(
                self,
                "AI Model Unavailable",
                f"Failed to initialize AI model '{getattr(model_class, 'name', requested_name)}'.\n\n{exc}",
            )
            return None

    def _sync_ai_model_image(self, *, force: bool = False) -> bool:
        if self._ai_model is None:
            return False
        if self.pixmap is None or self.pixmap.isNull():
            return False
        source_signature = self._ai_model_image_signature_value()
        image_dirty = bool(getattr(self, "_ai_model_image_dirty", True))
        if not force and not image_dirty:
            # The semantic image/frame identity is the real cache key.
            # `QPixmap.cacheKey()` is only a fallback when signature generation is unavailable.
            if (
                source_signature is not None
                and self._ai_model_image_signature == source_signature
            ):
                return True
            if source_signature is None:
                try:
                    current_key = int(self.pixmap.cacheKey())
                except Exception:
                    current_key = None
                if (
                    current_key is not None
                    and self._ai_model_pixmap_key is not None
                    and int(self._ai_model_pixmap_key) == int(current_key)
                ):
                    return True
        try:
            rgb_arr = _canvas_attr("utils", _default_utils).img_qt_to_arr(
                self.pixmap.toImage()
            )
            with self._prompt_inference_lock:
                self._ai_model.set_image(image=rgb_arr)
                self._ai_model_pixmap_key = int(self.pixmap.cacheKey())
                self._ai_model_image_signature = source_signature
                self._ai_model_image_dirty = False
                self._ai_model_rgb_digest = hashlib.blake2b(
                    np.ascontiguousarray(rgb_arr).view(np.uint8), digest_size=16
                ).hexdigest()
            return True
        except Exception:
            logger.debug("Failed to sync AI model image.", exc_info=True)
            return False

    def _invalidate_ai_model_image_signature_cache(self) -> None:
        self._ai_model_signature_cache_token = None
        self._ai_model_signature_cache_value = None

    def _ai_model_image_signature_value(self):
        if self.pixmap is None or self.pixmap.isNull():
            return None
        source_hint = self._ai_model_source_hint()
        cache_key = None
        try:
            cache_key = int(self.pixmap.cacheKey())
        except Exception:
            cache_key = None
        signature_cache_token = (cache_key, source_hint)
        if (
            self._ai_model_signature_cache_token == signature_cache_token
            and self._ai_model_signature_cache_value is not None
        ):
            return self._ai_model_signature_cache_value
        try:
            qimage = self.pixmap.toImage()
            if qimage.isNull():
                return None
            image_data = _canvas_attr("utils", _default_utils).img_qt_to_arr(qimage)
            digest = hashlib.sha1(np.ascontiguousarray(image_data)).hexdigest()
            signature = (
                "image",
                digest,
                int(self.pixmap.width()),
                int(self.pixmap.height()),
                source_hint,
            )
            self._ai_model_signature_cache_token = signature_cache_token
            self._ai_model_signature_cache_value = signature
            return signature
        except Exception:
            signature = source_hint
            if signature is None:
                try:
                    signature = ("pixmap", int(self.pixmap.cacheKey()))
                except Exception:
                    signature = None
            self._ai_model_signature_cache_token = signature_cache_token
            self._ai_model_signature_cache_value = signature
            return signature

    def _ai_model_source_hint(self):
        candidates = []
        try:
            candidates.append(self.window())
        except Exception:
            pass
        candidates.append(self)
        for owner in candidates:
            if owner is None:
                continue
            source_path = str(
                getattr(owner, "filename", "") or getattr(owner, "imagePath", "") or ""
            ).strip()
            frame_number = getattr(owner, "frame_number", None)
            if source_path:
                if frame_number is not None:
                    try:
                        return (source_path, int(frame_number))
                    except Exception:
                        return (source_path, None)
                return (source_path, None)
        return None

    def _ensure_ai_model_initialized(self, *, force_sync: bool = False) -> bool:
        if self._ai_model is not None:
            return self._sync_ai_model_image(force=force_sync)
        default_name = None
        try:
            ai_models = _canvas_attr("AI_MODELS", _DEFAULT_AI_MODELS)
            for m in ai_models:
                if getattr(m, "name", None) == "EfficientSam (speed)":
                    default_name = m.name
                    break
            if default_name is None and ai_models:
                default_name = ai_models[0].name
        except Exception:
            default_name = None
        if default_name:
            self.initializeAiModel(default_name)
        return self._ai_model is not None and self._sync_ai_model_image(
            force=force_sync
        )

    def _prompt_polygon_epsilon(self) -> float:
        try:
            win = self.window()
            raw = getattr(win, "epsilon_for_polygon", None)
            if raw is not None:
                val = float(raw)
                if np.isfinite(val) and 0.0 <= val <= 25.0:
                    return val
        except Exception:
            pass
        return 2.0

    def _remove_existing_prompt_shapes(self, description: str = "grounding_sam") -> int:
        if not self.shapes:
            return 0
        removed = [
            shape
            for shape in self.shapes
            if getattr(shape, "description", None) == description
        ]
        if not removed:
            return 0
        removed_ids = {id(shape) for shape in removed}
        self.shapes = [shape for shape in self.shapes if id(shape) not in removed_ids]
        self.selectedShapes = [
            shape for shape in self.selectedShapes if id(shape) not in removed_ids
        ]
        for shape_id in removed_ids:
            self.visible.pop(shape_id, None)
        if self.hShape is not None and id(self.hShape) in removed_ids:
            self.hShape = None
            self.hVertex = None
            self.hEdge = None
        try:
            win = self.window()
            rem_labels = getattr(win, "remLabels", None)
            if callable(rem_labels):
                rem_labels(removed)
        except Exception:
            logger.debug(
                "Failed to sync label list after removing prompt shapes.", exc_info=True
            )
        return len(removed)

    @staticmethod
    def _is_stop_requested(stop_event: threading.Event | None) -> bool:
        return bool(stop_event is not None and stop_event.is_set())

    @staticmethod
    def _call_with_optional_stop(
        fn, *args, stop_event: threading.Event | None = None, **kwargs
    ):
        if stop_event is not None:
            try:
                params = inspect.signature(fn).parameters
                accepts_stop = "stop_event" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                accepts_stop = False
            if accepts_stop:
                return fn(*args, stop_event=stop_event, **kwargs)
        return fn(*args, **kwargs)

    @staticmethod
    def _extract_exemplar_boxes(rectangle_shapes) -> list[list[int]]:
        exemplar_boxes: list[list[int]] = []
        for rectangle_shape in rectangle_shapes or []:
            x1 = int(min(rectangle_shape.points[0].x(), rectangle_shape.points[1].x()))
            y1 = int(min(rectangle_shape.points[0].y(), rectangle_shape.points[1].y()))
            x2 = int(max(rectangle_shape.points[0].x(), rectangle_shape.points[1].x()))
            y2 = int(max(rectangle_shape.points[0].y(), rectangle_shape.points[1].y()))
            exemplar_boxes.append([x1, y1, x2, y2])
        return exemplar_boxes

    def _ensure_lazy_prompt_model(
        self,
        attr_name: str,
        factory: Callable[..., object],
        *,
        context: str,
        stop_event: threading.Event | None = None,
        reset_rgb_digest: bool = False,
    ) -> object:
        """Instantiate a lazy prompt model outside ``_prompt_inference_lock`` and publish it safely."""
        with self._prompt_inference_lock:
            if self._is_stop_requested(stop_event) or self._canvas_closing:
                raise InferenceCancelledError(f"{context} initialization cancelled.")
            existing = getattr(self, attr_name, None)
            if existing is not None:
                return existing

        instance = self._call_with_optional_stop(factory, stop_event=stop_event)
        ensure_fn = getattr(instance, "_ensure_model", None)
        if callable(ensure_fn):
            try:
                self._call_with_optional_stop(ensure_fn, stop_event=stop_event)
            except BaseException:
                self._release_model(instance, context=context)
                raise

        with self._prompt_inference_lock:
            if self._is_stop_requested(stop_event) or self._canvas_closing:
                self._release_model(instance, context=context)
                raise InferenceCancelledError(f"{context} initialization cancelled.")
            existing = getattr(self, attr_name, None)
            if existing is not None:
                self._release_model(instance, context=f"duplicate {context}")
                return existing
            setattr(self, attr_name, instance)
            if reset_rgb_digest:
                self._ai_model_rgb_digest = None
            return instance

    def _ensure_prompt_fallback_ai_model(
        self,
        image_data: np.ndarray,
        *,
        stop_event: threading.Event | None = None,
    ) -> tuple[object | None, str | None]:
        """Initialize and sync the built-in ONNX SAM model without touching QPixmap/QMessageBox."""
        if self._is_stop_requested(stop_event) or self._canvas_closing:
            raise InferenceCancelledError("Prompt fallback SAM cancelled.")

        if self._ai_model is None:
            model_class = self._resolve_ai_model_class("EfficientSam (speed)")
            if model_class is None:
                return (
                    None,
                    "Neither 'segment_anything' (SAM-HQ) nor a built-in ONNX SAM model is available.",
                )
            try:
                self._ensure_lazy_prompt_model(
                    "_ai_model",
                    model_class,
                    context="canvas AI",
                    stop_event=stop_event,
                    reset_rgb_digest=True,
                )
            except InferenceCancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Failed to initialize built-in AI fallback model '%s': %s",
                    getattr(model_class, "name", "SAM"),
                    exc,
                )
                return (
                    None,
                    f"Neither 'segment_anything' (SAM-HQ) nor built-in AI model "
                    f"'{getattr(model_class, 'name', 'SAM')}' could be initialized ({exc}).",
                )

        with self._prompt_inference_lock:
            if self._is_stop_requested(stop_event) or self._canvas_closing:
                raise InferenceCancelledError("Prompt fallback SAM cancelled.")
            if self._ai_model is None:
                raise InferenceCancelledError("Prompt fallback SAM released.")

            rgb_digest = hashlib.blake2b(
                np.ascontiguousarray(image_data).view(np.uint8), digest_size=16
            ).hexdigest()
            if self._ai_model_rgb_digest != rgb_digest:
                try:
                    self._call_with_optional_stop(
                        self._ai_model.set_image,
                        image=image_data,
                        stop_event=stop_event,
                    )
                except InferenceCancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Failed to sync built-in AI fallback model image: %s", exc
                    )
                    return (
                        None,
                        f"Failed to prepare built-in AI model image embedding ({exc}).",
                    )
                if self._is_stop_requested(stop_event) or self._canvas_closing:
                    raise InferenceCancelledError("Prompt fallback SAM cancelled.")
                self._ai_model_rgb_digest = rgb_digest
                self._ai_model_image_dirty = True
            return self._ai_model, None

    def _segment_boxes_for_prompt_internal(
        self,
        image_data: np.ndarray,
        bboxes: list[list[float]],
        *,
        stop_event: threading.Event | None = None,
    ) -> tuple[object, object, object, str | None]:
        """Thread-safe box-to-mask segmentation via SAM-HQ or built-in ONNX SAM."""
        if not bboxes or self._is_stop_requested(stop_event) or self._canvas_closing:
            return [], [], [], None

        if self.sam_hq_model is None:
            try:
                from annolid.segmentation.SAM.sam_hq import SamHQSegmenter

                self._ensure_lazy_prompt_model(
                    "sam_hq_model",
                    SamHQSegmenter,
                    context="SAM HQ",
                    stop_event=stop_event,
                )
            except InferenceCancelledError:
                return [], [], [], None
            except Exception as exc:
                logger.info(
                    "SAM-HQ unavailable (%s); falling back to built-in AI model.",
                    exc,
                )

        if self._is_stop_requested(stop_event) or self._canvas_closing:
            return [], [], [], None

        if self.sam_hq_model is not None:
            with self._prompt_inference_lock:
                if (
                    self._is_stop_requested(stop_event)
                    or self._canvas_closing
                    or self.sam_hq_model is None
                ):
                    return [], [], [], None
                try:
                    masks, scores, boxes = self._call_with_optional_stop(
                        self.sam_hq_model.segment_objects,
                        image_data,
                        bboxes,
                        stop_event=stop_event,
                    )
                except InferenceCancelledError:
                    return [], [], [], None
                return masks, scores, boxes, None

        try:
            ai_model, warning_msg = self._ensure_prompt_fallback_ai_model(
                image_data,
                stop_event=stop_event,
            )
        except InferenceCancelledError:
            return [], [], [], None
        if ai_model is None:
            return None, None, None, warning_msg

        with self._prompt_inference_lock:
            fallback_masks: list[np.ndarray] = []
            fallback_scores: list[float] = []
            valid_boxes: list[list[float]] = []
            height, width = image_data.shape[:2]
            for box in bboxes:
                if self._is_stop_requested(stop_event) or self._canvas_closing:
                    return [], [], [], None
                x1, y1, x2, y2 = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )
                try:
                    mask = self._call_with_optional_stop(
                        ai_model.predict_mask_from_points,
                        points=[[x1, y1], [x2, y2]],
                        point_labels=[2, 3],
                        stop_event=stop_event,
                    )
                    mask_bool = np.asarray(mask, dtype=bool)
                    if mask_bool.shape != (height, width):
                        mask_u8 = cv2.resize(
                            mask_bool.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        mask_bool = mask_u8 > 0
                except InferenceCancelledError:
                    return [], [], [], None
                except Exception as exc:
                    logger.warning(
                        "Built-in SAM box fallback failed for %s: %s", box, exc
                    )
                    continue
                fallback_masks.append(mask_bool)
                fallback_scores.append(1.0)
                valid_boxes.append([x1, y1, x2, y2])
            return fallback_masks, fallback_scores, valid_boxes, None

    def _segment_boxes_for_prompt(
        self,
        image_data: np.ndarray,
        bboxes: list[list[float]],
        *,
        stop_event: threading.Event | None = None,
    ):
        """Segment candidate boxes via SAM-HQ, falling back to built-in ONNX SAM."""
        masks, scores, boxes, warning_msg = self._segment_boxes_for_prompt_internal(
            image_data,
            bboxes,
            stop_event=stop_event,
        )
        if masks is None and warning_msg:
            QtWidgets.QMessageBox.about(
                self,
                "Missing Segment Anything dependency",
                warning_msg,
            )
        return masks, scores, boxes

    def _build_exclusive_prompt_shapes(
        self,
        *,
        raw_masks: list[np.ndarray],
        raw_scores: list[float],
        labels: list[str],
        extra_metadata: list[dict[str, str]] | None = None,
        is_polygon_output: bool = True,
        use_group_id: bool = False,
        polygon_epsilon: float = 2.0,
        resolve_conflicts: bool = True,
        stop_event: threading.Event | None = None,
    ) -> TextPromptComputationResult:
        """Convert candidate masks into exclusive non-sliver polygon/mask shapes."""
        make_exclusive = _canvas_attr(
            "make_instance_masks_exclusive",
            _default_make_instance_masks_exclusive,
        )
        exclusive_masks = make_exclusive(raw_masks, scores=raw_scores)

        surviving: list[tuple[str, np.ndarray, float, dict[str, str]]] = []
        for idx, (raw_mask, excl_mask, score, lbl) in enumerate(
            zip(raw_masks, exclusive_masks, raw_scores, labels)
        ):
            if is_sliver_mask_remnant(excl_mask, raw_mask):
                continue
            meta = (
                dict(extra_metadata[idx])
                if extra_metadata is not None and idx < len(extra_metadata)
                else {}
            )
            surviving.append((lbl, excl_mask, float(score), meta))

        label_assignments = assign_instance_labels(
            [item[0] for item in surviving],
            use_group_id=use_group_id,
        )
        generated_shapes: list[Shape] = []
        generated_priorities: list[float] = []

        for (_lbl, excl_mask, score, meta), (inst_label, inst_group_id) in zip(
            surviving, label_assignments
        ):
            if self._is_stop_requested(stop_event):
                return TextPromptComputationResult(
                    shapes=[], priorities=[], cancelled=True
                )
            bbox = tight_mask_bbox(excl_mask)
            if bbox is None:
                continue
            if is_polygon_output:
                mask_shape = MaskShape(
                    label=inst_label,
                    group_id=inst_group_id,
                    flags={},
                    description="grounding_sam",
                )
                mask_shape.mask = excl_mask
                polygons = mask_shape.toPolygons(
                    epsilon=polygon_epsilon,
                    fill_holes=False,
                    bbox=bbox,
                )
                if not polygons:
                    logger.warning(
                        "Skipping empty text-prompt mask for '%s'.", inst_label
                    )
                    continue
                built_shape = polygons[0]
            else:
                x1, y1, x2, y2 = bbox
                built_shape = Shape(
                    label=inst_label,
                    group_id=inst_group_id,
                    flags={},
                    description="grounding_sam",
                )
                built_shape.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=excl_mask[int(y1) : int(y2), int(x1) : int(x2)],
                )
            built_shape.other_data["score"] = str(score)
            for k, v in meta.items():
                built_shape.other_data[k] = v
            generated_shapes.append(built_shape)
            generated_priorities.append(score)

        if resolve_conflicts and generated_shapes:
            resolve_conflicts_fn = _canvas_attr(
                "resolve_polygon_shape_conflicts",
                _default_resolve_polygon_shape_conflicts,
            )
            resolution = resolve_conflicts_fn(
                generated_shapes,
                priorities=generated_priorities,
            )
            if resolution.adjusted_shape_indices:
                logger.info(
                    "Resolved geometric conflicts for text-prompt polygon(s) %s.",
                    resolution.adjusted_shape_indices,
                )
            if resolution.dropped_shape_indices:
                logger.warning(
                    "Dropped fully occluded text-prompt polygon(s) %s.",
                    resolution.dropped_shape_indices,
                )
            return TextPromptComputationResult(
                shapes=list(resolution.shapes),
                priorities=[],
                conflicts_resolved=True,
            )

        return TextPromptComputationResult(
            shapes=generated_shapes,
            priorities=generated_priorities,
            conflicts_resolved=not generated_shapes,
        )

    def _compute_auto_masks_result(
        self,
        image_data: np.ndarray,
        label: str,
        *,
        points_per_side: int = 32,
        is_polygon_output: bool = True,
        use_group_id: bool = False,
        polygon_epsilon: float = 2.0,
        resolve_conflicts: bool = True,
        stop_event: threading.Event | None = None,
    ) -> TextPromptComputationResult:
        if self._is_stop_requested(stop_event) or self._canvas_closing:
            return TextPromptComputationResult(shapes=[], priorities=[], cancelled=True)

        if self.sam_hq_model is None:
            try:
                from annolid.segmentation.SAM.sam_hq import SamHQSegmenter

                self._ensure_lazy_prompt_model(
                    "sam_hq_model",
                    SamHQSegmenter,
                    context="SAM HQ",
                    stop_event=stop_event,
                )
            except InferenceCancelledError:
                return TextPromptComputationResult(
                    shapes=[], priorities=[], cancelled=True
                )
            except Exception as exc:
                return TextPromptComputationResult(
                    shapes=[],
                    priorities=[],
                    warning_title="Missing Segment Anything dependency",
                    warning_message=str(exc),
                )

        with self._prompt_inference_lock:
            if (
                self._is_stop_requested(stop_event)
                or self._canvas_closing
                or self.sam_hq_model is None
            ):
                return TextPromptComputationResult(
                    shapes=[], priorities=[], cancelled=True
                )

            try:
                anns = self._call_with_optional_stop(
                    self.sam_hq_model.segment_everything,
                    image_data,
                    points_per_side=points_per_side,
                    stop_event=stop_event,
                )
            except InferenceCancelledError:
                return TextPromptComputationResult(
                    shapes=[], priorities=[], cancelled=True
                )

        if self._is_stop_requested(stop_event):
            return TextPromptComputationResult(shapes=[], priorities=[], cancelled=True)
        if not anns:
            return TextPromptComputationResult(
                shapes=[], priorities=[], conflicts_resolved=True
            )

        raw_masks = [ann["segmentation"] for ann in anns]
        raw_scores = [float(ann.get("stability_score", 0.0)) for ann in anns]
        return self._build_exclusive_prompt_shapes(
            raw_masks=raw_masks,
            raw_scores=raw_scores,
            labels=[label or "object"] * len(raw_masks),
            is_polygon_output=bool(is_polygon_output),
            use_group_id=bool(use_group_id),
            polygon_epsilon=float(polygon_epsilon),
            resolve_conflicts=bool(resolve_conflicts),
            stop_event=stop_event,
        )

    def auto_mask_generator(
        self,
        image_data,
        label,
        points_per_side=32,
        is_polygon_output=True,
        *,
        replace_existing=False,
        use_group_id=False,
    ) -> int:
        """
        Generate masks or polygons from segmentation annotations.
        """
        result = self._compute_auto_masks_result(
            image_data,
            label,
            points_per_side=points_per_side,
            is_polygon_output=is_polygon_output,
            use_group_id=use_group_id,
            polygon_epsilon=self._prompt_polygon_epsilon(),
            resolve_conflicts=True,
        )
        return self._apply_text_prompt_result(
            result,
            replace_existing=replace_existing,
        )

    def _notify_window_prompt_mutation(self) -> None:
        try:
            win = self.window()
            if win is None:
                return
            actions = getattr(win, "actions", None)
            undo_action = (
                getattr(actions, "undo", None) if actions is not None else None
            )
            if undo_action is not None and hasattr(undo_action, "setEnabled"):
                undo_action.setEnabled(self.isShapeRestorable)
            set_dirty = getattr(win, "setDirty", None)
            if callable(set_dirty):
                set_dirty()
        except Exception:
            logger.debug(
                "Failed to notify window after prompt mutation.", exc_info=True
            )

    def _finalise_generated_prompt_shapes(
        self,
        shapes,
        priorities=None,
        *,
        replace_existing=False,
        already_resolved=False,
    ) -> int:
        """Finalize one prompt batch atomically after enforcing exclusive geometry."""
        if not shapes:
            self.current = None
            if replace_existing:
                if not self.shapesBackups:
                    self.storeShapes()
                if self._remove_existing_prompt_shapes() > 0:
                    self._shared_finalize_topology_edit()
                    self.storeShapes()
                    self._notify_window_prompt_mutation()
                    self.update()
            return 0

        if already_resolved:
            accepted = list(shapes)
        else:
            resolution = _canvas_attr(
                "resolve_polygon_shape_conflicts",
                _default_resolve_polygon_shape_conflicts,
            )(
                shapes,
                priorities=priorities,
            )
            if resolution.adjusted_shape_indices:
                logger.info(
                    "Resolved geometric conflicts for text-prompt polygon(s) %s.",
                    resolution.adjusted_shape_indices,
                )
            if resolution.dropped_shape_indices:
                logger.warning(
                    "Dropped fully occluded text-prompt polygon(s) %s.",
                    resolution.dropped_shape_indices,
                )
            accepted = list(resolution.shapes)

        if not accepted:
            self.current = None
            if replace_existing:
                if not self.shapesBackups:
                    self.storeShapes()
                if self._remove_existing_prompt_shapes() > 0:
                    self._shared_finalize_topology_edit()
                    self.storeShapes()
                    self._notify_window_prompt_mutation()
                    self.update()
            return 0

        if not self.shapesBackups:
            self.storeShapes()

        removed_count = self._remove_existing_prompt_shapes() if replace_existing else 0
        shapes_before = len(self.shapes)
        self._createMode = "grounding_sam"
        self._batch_finalising_prompt_shapes = True
        try:
            for shape in accepted:
                self.current = shape
                self.finalise()
        finally:
            self._batch_finalising_prompt_shapes = False
            self.current = None

        if len(self.shapes) != shapes_before or removed_count > 0:
            self._shared_finalize_topology_edit()
            self.storeShapes()
            self._notify_window_prompt_mutation()
            self.update()

        return len(accepted)

    def _apply_text_prompt_result(
        self,
        result: TextPromptComputationResult,
        *,
        replace_existing: bool = False,
    ) -> int:
        """Apply a computed text-prompt result onto the canvas on the Qt UI thread."""
        if result.cancelled:
            return 0
        if result.warning_message:
            QtWidgets.QMessageBox.about(
                self,
                result.warning_title or "Segmentation Unavailable",
                result.warning_message,
            )
            return 0
        return self._finalise_generated_prompt_shapes(
            result.shapes,
            priorities=result.priorities,
            replace_existing=replace_existing,
            already_resolved=result.conflicts_resolved,
        )

    def compute_text_prompt_shapes(
        self,
        image_data: np.ndarray,
        prompt: str,
        *,
        exemplar_boxes: list[list[int]] | None = None,
        is_polygon_output: bool = True,
        use_countgd: bool = False,
        box_threshold: float | None = None,
        iou_threshold: float = 0.50,
        output_mode: str | None = None,
        use_group_id: bool = False,
        polygon_epsilon: float = 2.0,
        resolve_conflicts: bool = True,
        stop_event: threading.Event | None = None,
    ) -> TextPromptComputationResult:
        """Thread-safe text-prompt detection + segmentation without touching Qt GUI state."""
        if self._is_stop_requested(stop_event):
            return TextPromptComputationResult(shapes=[], priorities=[], cancelled=True)

        resolved_mode = str(output_mode or "").strip().lower()
        if resolved_mode in {"box", "boxes", "bbox"}:
            resolved_mode = "rectangle"
        elif resolved_mode not in {"polygon", "mask", "rectangle"}:
            resolved_mode = "polygon" if is_polygon_output else "mask"

        every_spec = parse_every_prompt(prompt)
        if every_spec is not None and exemplar_boxes is None:
            points_per_side, every_label = every_spec
            logger.info(
                "%d points to be sampled along one side of the image for '%s'",
                points_per_side,
                every_label,
            )
            return self._compute_auto_masks_result(
                image_data,
                every_label,
                points_per_side=points_per_side,
                is_polygon_output=(resolved_mode != "mask"),
                use_group_id=use_group_id,
                polygon_epsilon=polygon_epsilon,
                resolve_conflicts=resolve_conflicts,
                stop_event=stop_event,
            )

        parsed_prompt = parse_text_prompt(prompt)
        default_label = (
            parsed_prompt.phrases[0].label
            if not parsed_prompt.is_empty
            else (str(prompt or "").strip() or "object")
        )

        raw_candidates: list[DetectionBoxCandidate] = []
        countgd_conf = float(box_threshold) if box_threshold is not None else 0.23

        if self._is_stop_requested(stop_event) or self._canvas_closing:
            return TextPromptComputationResult(shapes=[], priorities=[], cancelled=True)

        if exemplar_boxes is not None:
            query_text = (
                parsed_prompt.phrases[0].query
                if not parsed_prompt.is_empty
                else str(prompt or "")
            )
            cgd_boxes = self._predict_similar_rectangles_from_boxes(
                exemplar_boxes=exemplar_boxes,
                prompt=query_text,
                confidence_threshold=countgd_conf,
                image_data=image_data,
                stop_event=stop_event,
            )
            for box in cgd_boxes or []:
                raw_candidates.append(
                    DetectionBoxCandidate(
                        bbox=(
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ),
                        label=default_label,
                        score=max(countgd_conf, 0.5),
                        source="countgd_exemplar",
                    )
                )
        else:
            if use_countgd:
                phrases_for_cgd = (
                    parsed_prompt.phrases
                    if not parsed_prompt.is_empty
                    else (parse_text_prompt(default_label).phrases or ())
                )
                for phrase in phrases_for_cgd:
                    if self._is_stop_requested(stop_event) or self._canvas_closing:
                        return TextPromptComputationResult(
                            shapes=[], priorities=[], cancelled=True
                        )
                    cgd_boxes = self._predict_similar_rectangles_from_boxes(
                        exemplar_boxes=[],
                        prompt=phrase.query,
                        confidence_threshold=countgd_conf,
                        image_data=image_data,
                        stop_event=stop_event,
                    )
                    for box in cgd_boxes or []:
                        raw_candidates.append(
                            DetectionBoxCandidate(
                                bbox=(
                                    float(box[0]),
                                    float(box[1]),
                                    float(box[2]),
                                    float(box[3]),
                                ),
                                label=phrase.label,
                                score=max(countgd_conf, 0.45),
                                source="countgd",
                            )
                        )

            if self._is_stop_requested(stop_event) or self._canvas_closing:
                return TextPromptComputationResult(
                    shapes=[], priorities=[], cancelled=True
                )

            if self._ai_model_rect is None:
                try:
                    from annolid.detector.grounding_dino import GroundingDINO

                    self._ensure_lazy_prompt_model(
                        "_ai_model_rect",
                        GroundingDINO,
                        context="rectangle detector",
                        stop_event=stop_event,
                    )
                except InferenceCancelledError:
                    return TextPromptComputationResult(
                        shapes=[], priorities=[], cancelled=True
                    )
                except ModuleNotFoundError as exc:
                    if not raw_candidates:
                        return TextPromptComputationResult(
                            shapes=[],
                            priorities=[],
                            warning_title="Missing GroundingDINO dependency",
                            warning_message=str(exc),
                        )
                    logger.warning("GroundingDINO unavailable: %s", exc)

            if self._is_stop_requested(stop_event) or self._canvas_closing:
                return TextPromptComputationResult(
                    shapes=[], priorities=[], cancelled=True
                )

            with self._prompt_inference_lock:
                if self._is_stop_requested(stop_event) or self._canvas_closing:
                    return TextPromptComputationResult(
                        shapes=[], priorities=[], cancelled=True
                    )
                if self._ai_model_rect is not None:
                    detect_fn = getattr(self._ai_model_rect, "predict_detections", None)
                    try:
                        if callable(detect_fn):
                            gd_candidates = self._call_with_optional_stop(
                                detect_fn,
                                image_data,
                                prompt,
                                box_threshold=box_threshold,
                                nms_iou_threshold=iou_threshold,
                                stop_event=stop_event,
                            )
                            raw_candidates.extend(gd_candidates)
                        else:
                            bboxes = self._call_with_optional_stop(
                                self._ai_model_rect.predict_bboxes,
                                image_data,
                                prompt,
                                stop_event=stop_event,
                            )
                            for box, meta in bboxes or []:
                                if isinstance(
                                    meta, (int, float, np.floating, np.integer)
                                ) and not isinstance(meta, bool):
                                    cand_label = default_label
                                    cand_score = float(meta)
                                else:
                                    cand_label = match_predicted_phrase_to_label(
                                        str(meta or ""),
                                        parsed_prompt,
                                        fallback_label=default_label,
                                    )
                                    cand_score = 1.0
                                raw_candidates.append(
                                    DetectionBoxCandidate(
                                        bbox=(
                                            float(box[0]),
                                            float(box[1]),
                                            float(box[2]),
                                            float(box[3]),
                                        ),
                                        label=cand_label,
                                        score=cand_score,
                                        source="grounding_dino",
                                    )
                                )
                    except InferenceCancelledError:
                        return TextPromptComputationResult(
                            shapes=[], priorities=[], cancelled=True
                        )

        if self._is_stop_requested(stop_event):
            return TextPromptComputationResult(shapes=[], priorities=[], cancelled=True)

        candidates = sanitize_and_nms_box_candidates(
            raw_candidates,
            image_shape=image_data.shape[:2],
            min_score=float(box_threshold) if box_threshold is not None else None,
            same_label_iou_threshold=float(iou_threshold),
        )

        if not candidates:
            logger.info("No candidate boxes found for prompt '%s'.", prompt)
            return TextPromptComputationResult(
                shapes=[], priorities=[], conflicts_resolved=True
            )

        if resolved_mode == "rectangle":
            rect_labels = assign_instance_labels(
                [cand.label for cand in candidates],
                use_group_id=use_group_id,
            )
            rect_shapes: list[Shape] = []
            rect_priorities: list[float] = []
            for cand, (inst_label, inst_group_id) in zip(candidates, rect_labels):
                x1, y1, x2, y2 = cand.bbox
                rect_shape = Shape(
                    label=inst_label,
                    group_id=inst_group_id,
                    shape_type="rectangle",
                    flags={},
                    description="grounding_sam",
                )
                rect_shape.addPoint(QtCore.QPointF(x1, y1))
                rect_shape.addPoint(QtCore.QPointF(x2, y2))
                rect_shape.other_data["score"] = str(cand.score)
                rect_shape.other_data["detector_score"] = str(cand.score)
                rect_shape.other_data["source"] = cand.source
                rect_shapes.append(rect_shape)
                rect_priorities.append(cand.score)
            return TextPromptComputationResult(
                shapes=rect_shapes,
                priorities=rect_priorities,
                conflicts_resolved=True,
            )

        candidate_boxes = [list(cand.bbox) for cand in candidates]
        raw_masks, scores, out_boxes, seg_warning = (
            self._segment_boxes_for_prompt_internal(
                image_data,
                candidate_boxes,
                stop_event=stop_event,
            )
        )
        if self._is_stop_requested(stop_event):
            return TextPromptComputationResult(shapes=[], priorities=[], cancelled=True)
        if raw_masks is None:
            return TextPromptComputationResult(
                shapes=[],
                priorities=[],
                warning_title="Missing Segment Anything dependency",
                warning_message=seg_warning
                or "Neither 'segment_anything' (SAM-HQ) nor a built-in ONNX SAM model is available.",
            )
        if len(raw_masks) == 0:
            logger.info("No masks decoded for prompt '%s'.", prompt)
            return TextPromptComputationResult(
                shapes=[], priorities=[], conflicts_resolved=True
            )

        score_list = [float(s) for s in scores]
        matched_cands = [
            candidates[idx] if idx < len(candidates) else candidates[-1]
            for idx in range(len(out_boxes))
        ]
        matched_scores = [
            score_list[idx] if idx < len(score_list) else matched_cands[idx].score
            for idx in range(len(out_boxes))
        ]
        extra_meta = [
            {"detector_score": str(cand.score), "source": cand.source}
            for cand in matched_cands
        ]
        return self._build_exclusive_prompt_shapes(
            raw_masks=list(raw_masks[: len(out_boxes)]),
            raw_scores=matched_scores,
            labels=[cand.label for cand in matched_cands],
            extra_metadata=extra_meta,
            is_polygon_output=(resolved_mode == "polygon"),
            use_group_id=bool(use_group_id),
            polygon_epsilon=float(polygon_epsilon),
            resolve_conflicts=bool(resolve_conflicts),
            stop_event=stop_event,
        )

    compute_text_prompt_shapes.allow_force_thread_terminate = False

    def _capture_prompt_frame_token(self) -> tuple | None:
        if self.pixmap is None or self.pixmap.isNull():
            return None
        return (self._ai_model_image_signature_value(), self._ai_model_source_hint())

    def is_text_prompt_running(self) -> bool:
        return self._active_prompt_request_id is not None

    def cancel_async_text_prompt(
        self,
        *,
        wait_ms: int = 0,
        reason: str = "cancelled",
        notify_callback: bool = True,
    ) -> bool:
        """Cooperatively cancel any running background text-prompt job."""
        had_active = self._active_prompt_request_id is not None or bool(
            self._inflight_prompt_workers
        )
        active_id = self._active_prompt_request_id
        self._active_prompt_request_id = None
        set_widget_busy_cursor(self, False)

        for _rid, (_thread, worker, cancel_event, _anchor_id) in list(
            self._inflight_prompt_workers.items()
        ):
            cancel_event.set()
            try:
                worker.request_stop()
            except Exception:
                pass

        pending_cb = (
            self._prompt_callbacks.pop(active_id, None)
            if active_id is not None
            else None
        )
        self._prompt_callbacks.clear()
        if notify_callback and pending_cb is not None:
            _frame_token, _replace_existing, started_ts, cb = pending_cb
            if callable(cb):
                elapsed = max(0.0, time.perf_counter() - started_ts)
                try:
                    cb("cancelled", 0, elapsed, reason)
                except Exception:
                    logger.debug(
                        "Prompt cancellation callback raised an error.", exc_info=True
                    )

        if wait_ms > 0 and self._inflight_prompt_workers:
            deadline = time.perf_counter() + (float(wait_ms) / 1000.0)
            for rid, (thread, _worker, _ev, anchor_id) in list(
                self._inflight_prompt_workers.items()
            ):
                remaining_ms = max(
                    1, int(round((deadline - time.perf_counter()) * 1000.0))
                )
                try:
                    if thread.isRunning():
                        thread.quit()
                        thread.wait(remaining_ms)
                    if not thread.isRunning():
                        self._inflight_prompt_workers.pop(rid, None)
                        _release_anchored_qt_worker(anchor_id)
                except RuntimeError:
                    self._inflight_prompt_workers.pop(rid, None)
                    _release_anchored_qt_worker(anchor_id)
        return had_active

    def _has_other_running_prompt_workers(self, exclude_req_id: int) -> bool:
        for rid, (thread, _worker, _ev, _anchor_id) in list(
            self._inflight_prompt_workers.items()
        ):
            if rid == exclude_req_id:
                continue
            try:
                if thread.isRunning():
                    return True
            except RuntimeError:
                continue
        return False

    def _cleanup_finished_prompt_threads(self) -> None:
        for rid, (thread, _worker, _ev, anchor_id) in list(
            self._inflight_prompt_workers.items()
        ):
            try:
                running = thread.isRunning()
            except RuntimeError:
                self._inflight_prompt_workers.pop(rid, None)
                _release_anchored_qt_worker(anchor_id)
                continue
            if not running:
                self._inflight_prompt_workers.pop(rid, None)
                _release_anchored_qt_worker(anchor_id)

    def _on_prompt_worker_finished(self, payload: object) -> None:
        if not (isinstance(payload, tuple) and len(payload) == 2):
            return
        req_id, outcome = payload

        worker_entry = self._inflight_prompt_workers.get(req_id)
        cancel_event = worker_entry[2] if worker_entry is not None else None
        if worker_entry is not None:
            thread = worker_entry[0]
            try:
                thread.quit()
            except RuntimeError:
                pass

        cb_entry = self._prompt_callbacks.pop(req_id, None)
        if self._active_prompt_request_id == req_id:
            self._active_prompt_request_id = None
            set_widget_busy_cursor(self, False)

        if cb_entry is None:
            return

        request_frame_token, replace_existing, started_ts, on_finished = cb_entry
        elapsed = max(0.0, time.perf_counter() - started_ts)

        if isinstance(outcome, BaseException):
            logger.error(
                "Background text-prompt segmentation failed: %s",
                outcome,
                exc_info=True,
            )
            if callable(on_finished):
                on_finished("error", 0, elapsed, str(outcome))
            return

        if not isinstance(outcome, TextPromptComputationResult):
            if callable(on_finished):
                on_finished("error", 0, elapsed, "Unexpected worker result payload.")
            return

        if outcome.cancelled or self._is_stop_requested(cancel_event):
            if callable(on_finished):
                on_finished("cancelled", 0, elapsed, "cancelled")
            return

        current_token = self._capture_prompt_frame_token()
        if current_token is None or current_token != request_frame_token:
            logger.info(
                "Discarding stale text-prompt segmentation result because the active frame changed."
            )
            if callable(on_finished):
                on_finished("stale", 0, elapsed, "frame_changed")
            return

        created = self._apply_text_prompt_result(
            outcome,
            replace_existing=replace_existing,
        )
        if callable(on_finished):
            on_finished("completed", created, elapsed, outcome.warning_message)

    def start_async_text_prompt(
        self,
        prompt: str,
        rectangle_shapes=None,
        is_polygon_output: bool = True,
        use_countgd: bool = False,
        *,
        box_threshold: float | None = None,
        iou_threshold: float = 0.50,
        output_mode: str | None = None,
        replace_existing: bool = False,
        use_group_id: bool = False,
        on_finished: Callable[[str, int, float, str | None], None] | None = None,
    ) -> bool:
        """Launch text-prompt segmentation on a background QThread with stale-frame protection."""
        if self.pixmap is None or self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return False

        if self.is_text_prompt_running():
            self.cancel_async_text_prompt(
                wait_ms=0,
                reason="superseded",
                notify_callback=False,
            )

        qt_image = self.pixmap.toImage()
        image_data = np.array(
            convert_qt_image_to_rgb_cv_image(qt_image),
            dtype=np.uint8,
            copy=True,
            order="C",
        )
        del qt_image
        exemplar_boxes = (
            self._extract_exemplar_boxes(rectangle_shapes)
            if rectangle_shapes is not None
            else None
        )
        polygon_epsilon = self._prompt_polygon_epsilon()
        frame_token = self._capture_prompt_frame_token()

        self._prompt_request_seq += 1
        req_id = self._prompt_request_seq
        cancel_event = ObservableStopEvent()
        started_ts = time.perf_counter()

        def _task_runner() -> tuple[int, object]:
            try:
                res = self.compute_text_prompt_shapes(
                    image_data,
                    prompt,
                    exemplar_boxes=exemplar_boxes,
                    is_polygon_output=is_polygon_output,
                    use_countgd=use_countgd,
                    box_threshold=box_threshold,
                    iou_threshold=iou_threshold,
                    output_mode=output_mode,
                    use_group_id=use_group_id,
                    polygon_epsilon=polygon_epsilon,
                    resolve_conflicts=True,
                    stop_event=cancel_event,
                )
                return (req_id, res)
            except InferenceCancelledError:
                return (
                    req_id,
                    TextPromptComputationResult(
                        shapes=[], priorities=[], cancelled=True
                    ),
                )
            except Exception as exc:
                return (req_id, exc)
            finally:
                if self._canvas_closing or self._pending_close_model_release:
                    with self._prompt_inference_lock:
                        if (
                            self._canvas_closing or self._pending_close_model_release
                        ) and not self._has_other_running_prompt_workers(req_id):
                            self._release_all_prompt_models()

        _task_runner.allow_force_thread_terminate = False

        thread = configure_worker_thread_stack(
            QtCore.QThread(), _PROMPT_WORKER_STACK_SIZE
        )
        worker = FlexibleWorker(_task_runner)
        worker.moveToThread(thread)
        anchor_id = _anchor_qt_worker(thread, worker)

        self._active_prompt_request_id = req_id
        self._inflight_prompt_workers[req_id] = (
            thread,
            worker,
            cancel_event,
            anchor_id,
        )
        self._prompt_callbacks[req_id] = (
            frame_token,
            bool(replace_existing),
            started_ts,
            on_finished,
        )

        thread.started.connect(worker.run)
        worker.finished_signal.connect(thread.quit)
        worker.finished_signal.connect(
            self._on_prompt_worker_finished, QtCore.Qt.QueuedConnection
        )
        thread.finished.connect(
            lambda aid=anchor_id: _release_anchored_qt_worker(aid),
            QtCore.Qt.QueuedConnection,
        )
        thread.finished.connect(
            self._cleanup_finished_prompt_threads, QtCore.Qt.QueuedConnection
        )

        set_widget_busy_cursor(self, True)
        thread.start()
        return True

    def predictAiRectangle(
        self,
        prompt,
        rectangle_shapes=None,
        is_polygon_output=True,
        use_countgd=False,
        *,
        box_threshold=None,
        iou_threshold=0.50,
        output_mode=None,
        replace_existing=False,
        use_group_id=False,
    ) -> int:
        """
        Predict bounding boxes and then polygons, masks, or boxes from a text prompt.

        Returns the number of shapes finalized onto the canvas synchronously.
        """
        if self.pixmap is None or self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return 0

        if self._inflight_prompt_workers:
            self.cancel_async_text_prompt(wait_ms=2000, reason="sync_override")

        qt_image = self.pixmap.toImage()
        image_data = convert_qt_image_to_rgb_cv_image(qt_image)
        exemplar_boxes = (
            self._extract_exemplar_boxes(rectangle_shapes)
            if rectangle_shapes is not None
            else None
        )
        polygon_epsilon = self._prompt_polygon_epsilon()

        result = self.compute_text_prompt_shapes(
            image_data,
            prompt,
            exemplar_boxes=exemplar_boxes,
            is_polygon_output=is_polygon_output,
            use_countgd=use_countgd,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            output_mode=output_mode,
            use_group_id=use_group_id,
            polygon_epsilon=polygon_epsilon,
            resolve_conflicts=True,
        )
        return self._apply_text_prompt_result(
            result,
            replace_existing=replace_existing,
        )

    def loadSamPredictor(
        self,
    ):
        """
                The code requires Python version 3.8 or higher, along with PyTorch version 1.7 or higher,
                as well as TorchVision version 0.8 or higher. To install these dependencies,
                kindly refer to the instructions provided here. It's strongly recommended to install both PyTorch and TorchVision with CUDA support.
        To install Segment Anything, run the following command in your terminal:
        pip install git+https://github.com/facebookresearch/segment-anything.git

        """
        self._sam_last_load_error = None
        if self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            self._sam_last_load_error = "no_pixmap"
            return
        if not self.sam_predictor:
            try:
                import torch
                from pathlib import Path
                from segment_anything import sam_model_registry, SamPredictor
                from annolid.utils.devices import has_gpu
            except ImportError:
                self._sam_last_load_error = "missing_dependency"
                return
            if not has_gpu() and not torch.cuda.is_available():
                QtWidgets.QMessageBox.about(
                    self,
                    "GPU not available",
                    """For optimal performance, it is recommended to \
                                                use a GPU for running the Segment Anything model. \
                                                    Running the model on a CPU will result \
                                                        in significantly longer processing times.""",
                )
            here = (
                Path(os.path.abspath(__file__)).parent.parent.parent
                / "segmentation"
                / "SAM"
            )
            cachedir = str(here)
            os.makedirs(cachedir, exist_ok=True)
            weight_file = os.path.join(cachedir, self.sam_config["weights"] + ".pth")
            weight_urls = {
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            }
            if not os.path.isfile(weight_file):
                torch.hub.download_url_to_file(
                    weight_urls[self.sam_config["weights"]], weight_file
                )
            sam = sam_model_registry[self.sam_config["weights"]](checkpoint=weight_file)
            if self.sam_config["device"] == "cuda" and torch.cuda.is_available():
                sam.to(device="cuda")
            self.sam_predictor = SamPredictor(sam)
            self._sam_predictor_missing_logged = False
        self.samEmbedding()

    def samEmbedding(
        self,
    ):
        image = convert_qt_image_to_rgb_cv_image(self.pixmap.toImage())
        h, w, _ = image.shape
        self.sam_image_scale = self.sam_config["maxside"] / max(h, w)
        self.sam_image_scale = min(1, self.sam_image_scale)
        image = cv2.resize(
            image,
            None,
            fx=self.sam_image_scale,
            fy=self.sam_image_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        self.sam_predictor.set_image(image)
        self.update()

    def samPrompt(self, points, labels):
        if self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return
        if not self.sam_predictor:
            if not self._sam_predictor_missing_logged:
                logger.warning(
                    "SAM predictor is not initialized; run 'Segment Anything' to load the model."
                )
                self._sam_predictor_missing_logged = True
            return
        if not hasattr(self, "sam_image_scale"):
            try:
                self.samEmbedding()
            except Exception:
                logger.exception("Failed to set SAM image embedding.")
                return
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points * self.sam_image_scale,
            point_labels=labels,
            mask_input=self.sam_mask.logits[None, :, :]
            if self.sam_mask.logits is not None
            else None,
            multimask_output=False,
        )
        self.sam_mask.logits = logits[np.argmax(scores), :, :]
        mask = masks[np.argmax(scores), :, :]
        self.sam_mask.setScaleMask(self.sam_image_scale, mask)

    def _predict_similar_rectangles_from_boxes(
        self,
        exemplar_boxes: list[list[int]] | None = None,
        prompt=None,
        confidence_threshold: float = 0.23,
        *,
        image_data: np.ndarray,
        stop_event: threading.Event | None = None,
    ):
        """Thread-safe CountGD prediction accepting pre-extracted exemplar boxes and RGB array."""
        detected_boxes = []
        if (
            image_data is None
            or self._is_stop_requested(stop_event)
            or self._canvas_closing
        ):
            return detected_boxes
        try:
            from annolid.detector.countgd.predict import ObjectCounter

            pil_img = Image.fromarray(image_data)
            if self._countgd_counter is None:
                self._ensure_lazy_prompt_model(
                    "_countgd_counter",
                    ObjectCounter,
                    context="CountGD",
                    stop_event=stop_event,
                )
            with self._prompt_inference_lock:
                if (
                    self._is_stop_requested(stop_event)
                    or self._canvas_closing
                    or self._countgd_counter is None
                ):
                    return []
                boxes = list(exemplar_boxes or [])
                detected_boxes = self._call_with_optional_stop(
                    self._countgd_counter.count_objects,
                    pil_img,
                    text_prompt=prompt or "object",
                    exemplar_image=pil_img if boxes else None,
                    exemplar_boxes=boxes,
                    confidence_threshold=confidence_threshold,
                    stop_event=stop_event,
                )
        except InferenceCancelledError:
            return []
        except ImportError:
            logger.warning(
                "CountGD is not available. Install optional dependency and "
                "its requirements to enable CountGD-assisted prompting."
            )
        except Exception as e:
            logger.error(f"Error in CountGD: {e}")
        return detected_boxes

    def _predict_similar_rectangles(
        self,
        rectangle_shapes=None,
        prompt=None,
        confidence_threshold=0.23,
        *,
        image_data=None,
    ):
        """Predict more rectangle shapes without modifying self.shapes directly here."""
        if image_data is None:
            if not (self.pixmap and not self.pixmap.isNull()):
                return []
            qt_image = self.pixmap.toImage()
            image_data = convert_qt_image_to_rgb_cv_image(qt_image)
        exemplar_boxes = self._extract_exemplar_boxes(rectangle_shapes)
        return self._predict_similar_rectangles_from_boxes(
            exemplar_boxes=exemplar_boxes,
            prompt=prompt,
            confidence_threshold=confidence_threshold,
            image_data=image_data,
        )
