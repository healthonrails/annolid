from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from annolid.segmentation.SAM.sam_v2 import BaseSAMVideoProcessor
from annolid.utils.logger import logger
from .aliases import ensure_sam3_aliases
from .sam3.utils import set_default_device
from .video_window_inference import run_video_sliding_window

SAM3_IMPORT_ERROR: Optional[Exception] = None
_SAM3_REQUIRED_MODULES = ("iopath", "ftfy")

for _mod in _SAM3_REQUIRED_MODULES:
    try:
        importlib.import_module(_mod)
    except Exception as exc:  # pragma: no cover - import guard
        SAM3_IMPORT_ERROR = ImportError(
            f"SAM3 requires '{_mod}'. Install via `pip install .[sam3]` or "
            f"`pip install iopath ftfy`. Original error: {exc}"
        )
        break

if SAM3_IMPORT_ERROR is None:
    try:
        ensure_sam3_aliases()
        importlib.import_module("annolid.segmentation.SAM.sam3.sam3")
        from annolid.segmentation.SAM.sam3.sam3.model.sam3_video_predictor import (
            Sam3VideoPredictor,
        )
    except Exception as exc:  # pragma: no cover - import guard
        SAM3_IMPORT_ERROR = exc


def _default_bpe_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"


def _is_mps_oom(exc: BaseException) -> bool:
    msg = str(exc)
    return "MPS backend out of memory" in msg or "mps backend out of memory" in msg


def _clear_mps_cache() -> None:
    """Best-effort cache clearing after MPS OOM to allow CPU fallback."""
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        pass


@dataclass
class Sam3SessionConfig:
    """Container for session-level settings."""

    checkpoint_path: Optional[str] = None
    text_prompt: Optional[str] = None
    epsilon_for_polygon: float = 2.0
    ndjson_filename: Optional[str] = None
    max_frame_num_to_track: Optional[int] = None
    propagation_direction: str = "both"
    device: Optional[str] = None
    score_threshold_detection: Optional[float] = None
    new_det_thresh: Optional[float] = None
    # Performance knobs (safe defaults).
    compile_model: bool = False
    offload_video_to_cpu: bool = True
    async_loading_frames: bool = False  # keep memory usage low; no preloading
    sliding_window_size: int = 5  # frames per window for text-only runs
    sliding_window_stride: Optional[int] = None
    use_sliding_window_for_text_prompt: bool = True


def _resolve_session_config(
    config: Optional[Sam3SessionConfig],
    *,
    checkpoint_path: Optional[str],
    text_prompt: Optional[str],
    epsilon_for_polygon: float,
    ndjson_filename: Optional[str],
    max_frame_num_to_track: Optional[int],
    propagation_direction: str,
    device: Optional[str],
    score_threshold_detection: Optional[float],
    new_det_thresh: Optional[float],
    compile_model: bool,
    offload_video_to_cpu: bool,
    async_loading_frames: bool,
    sliding_window_size: int,
    sliding_window_stride: Optional[int],
    use_sliding_window_for_text_prompt: bool,
) -> Sam3SessionConfig:
    """Allow legacy kwargs or a config object to drive initialization."""
    if config is not None:
        return config
    return Sam3SessionConfig(
        checkpoint_path=checkpoint_path,
        text_prompt=text_prompt,
        epsilon_for_polygon=epsilon_for_polygon,
        ndjson_filename=ndjson_filename,
        max_frame_num_to_track=max_frame_num_to_track,
        propagation_direction=propagation_direction,
        device=device,
        score_threshold_detection=score_threshold_detection,
        new_det_thresh=new_det_thresh,
        compile_model=compile_model,
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=async_loading_frames,
        sliding_window_size=sliding_window_size,
        sliding_window_stride=sliding_window_stride,
        use_sliding_window_for_text_prompt=use_sliding_window_for_text_prompt,
    )


class Sam3SessionManager(BaseSAMVideoProcessor):
    """
    Thin session wrapper around Sam3VideoPredictor that handles prompt
    injection, propagation, and saving per-frame results (JSON + NDJSON).
    """

    def __init__(
        self,
        video_dir,
        id_to_labels,
        *,
        checkpoint_path: Optional[str] = None,
        text_prompt: Optional[str] = None,
        epsilon_for_polygon: float = 2.0,
        ndjson_filename: Optional[str] = None,
        max_frame_num_to_track: Optional[int] = None,
        propagation_direction: str = "both",
        device: Optional[str] = None,
        score_threshold_detection: Optional[float] = None,
        new_det_thresh: Optional[float] = None,
        compile_model: bool = False,
        offload_video_to_cpu: bool = True,
        async_loading_frames: bool = False,
        sliding_window_size: int = 5,
        sliding_window_stride: Optional[int] = None,
        use_sliding_window_for_text_prompt: bool = True,
        config: Optional[Sam3SessionConfig] = None,
    ):
        if SAM3_IMPORT_ERROR:
            raise RuntimeError(
                f"SAM3 dependencies are missing; install the required packages "
                f"(pip install .[sam3]) and retry. Root cause: {SAM3_IMPORT_ERROR}"
            ) from SAM3_IMPORT_ERROR

        cfg = _resolve_session_config(
            config,
            checkpoint_path=checkpoint_path,
            text_prompt=text_prompt,
            epsilon_for_polygon=epsilon_for_polygon,
            ndjson_filename=ndjson_filename,
            max_frame_num_to_track=max_frame_num_to_track,
            propagation_direction=propagation_direction,
            device=device,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            compile_model=compile_model,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            sliding_window_size=sliding_window_size,
            sliding_window_stride=sliding_window_stride,
            use_sliding_window_for_text_prompt=use_sliding_window_for_text_prompt,
        )

        self.text_prompt = cfg.text_prompt
        self.checkpoint_path = self._sanitize_checkpoint_path(
            cfg.checkpoint_path)
        self.bpe_path = _default_bpe_path()
        if not self.bpe_path.exists():
            raise FileNotFoundError(
                f"SAM3 BPE vocab not found: {self.bpe_path}")

        super().__init__(video_dir, id_to_labels, cfg.epsilon_for_polygon)
        ndjson_name = (
            cfg.ndjson_filename or f"{Path(self.video_dir).name}_annotations.ndjson"
        )
        self._init_ndjson_writer(ndjson_name)

        self._predictor: Optional[Sam3VideoPredictor] = None
        self._session_id: Optional[str] = None
        self._predictor_device: Optional[torch.device] = None
        # Track label hints per SAM3 object id so predictions keep the expected label
        # instead of defaulting to background or raw ids.
        self.obj_id_to_label: Dict[str | int, str] = {}
        # Track which frames produced masks during the main propagation run.
        self._frames_processed: set[int] = set()
        self._frames_with_masks: set[int] = set()
        # Store per-frame masks from the main propagation so we can re-use them
        # as visual prompts when re-acquiring lost tracks.
        self._frame_masks: Dict[int, Dict[str, np.ndarray]] = {}
        self.max_frame_num_to_track = cfg.max_frame_num_to_track
        # Default propagation settings (can be overridden per-call).
        self.propagation_direction = cfg.propagation_direction or "both"
        self.default_device = cfg.device
        self.score_threshold_detection = cfg.score_threshold_detection
        self.new_det_thresh = cfg.new_det_thresh
        self.compile_model = bool(cfg.compile_model)
        self.offload_video_to_cpu = bool(cfg.offload_video_to_cpu)
        self.async_loading_frames = cfg.async_loading_frames
        self.sliding_window_size = cfg.sliding_window_size
        self.sliding_window_stride = cfg.sliding_window_stride
        self.use_sliding_window_for_text_prompt = cfg.use_sliding_window_for_text_prompt
        # Sliding-window text-prompt mode: maintain a global track id mapping so
        # ids remain consistent across windows.
        self._sliding_window_mode_active: bool = False
        self._global_track_next_id: int = 1
        self._global_track_last_box: Dict[int, np.ndarray] = {}

    @staticmethod
    def _sanitize_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
        """Validate checkpoint path; return None to trigger auto-download."""
        if not checkpoint_path:
            return None
        candidate = Path(checkpoint_path).expanduser()
        if candidate.is_dir():
            logger.warning(
                "SAM3 checkpoint path points to a directory (%s); will auto-download instead.",
                candidate,
            )
            return None
        if not candidate.exists():
            logger.warning(
                "SAM3 checkpoint not found at %s; will auto-download.", candidate
            )
            return None
        return str(candidate)

    def _initialize_predictor(self, device: torch.device):
        return Sam3VideoPredictor(
            checkpoint_path=self.checkpoint_path,
            bpe_path=str(self.bpe_path),
            apply_temporal_disambiguation=True,
            device=device,
            compile_model=self.compile_model,
            offload_video_to_cpu=self.offload_video_to_cpu,
            async_loading_frames=self.async_loading_frames,
            score_threshold_detection=self.score_threshold_detection,
            new_det_thresh=self.new_det_thresh,
        )

    def start_session(self, target_device: Optional[torch.device | str] = None) -> str:
        resolved_device = set_default_device(
            target_device or self.default_device)
        if resolved_device.type == "cpu":
            torch.set_default_dtype(torch.float32)

        if self._predictor is None or self._predictor_device != resolved_device:
            self._predictor = self._initialize_predictor(resolved_device)
            self._predictor_device = resolved_device
        session = self._predictor.start_session(
            resource_path=str(self.video_path))
        self._session_id = session["session_id"]
        logger.info(
            "SAM3 session %s started on device=%s (checkpoint=%s)",
            self._session_id,
            resolved_device,
            self.checkpoint_path or "auto-download",
        )
        return self._session_id

    def close_session(self):
        if self._predictor and self._session_id:
            try:
                self._predictor.close_session(self._session_id)
            except Exception:
                pass
        self._session_id = None

    @contextmanager
    def _session_scope(
        self, target_device: Optional[torch.device | str] = None, *, auto_close: bool = True
    ):
        """
        Context manager to ensure sessions are opened and closed safely even on errors.
        """
        self.start_session(target_device)
        try:
            yield self._session_id
        finally:
            if auto_close:
                try:
                    self.close_session()
                except Exception:
                    pass

    def _reset_action_history_if_supported(self):
        """
        Best-effort clearing of cached action history on the underlying predictor
        so the next propagation performs a full pass.
        """
        try:
            state_entry = self._predictor._ALL_INFERENCE_STATES.get(
                self._session_id)  # type: ignore[attr-defined]
            if state_entry and isinstance(state_entry, dict):
                inference_state = state_entry.get("state")
                if isinstance(inference_state, dict) and "action_history" in inference_state:
                    inference_state["action_history"] = []
        except Exception as exc:
            logger.debug("Unable to reset SAM3 action history: %s", exc)

    def add_prompt(
        self,
        frame_idx: int,
        *,
        text: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[int]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
        record_outputs: bool = False,
        label_hints: Optional[List[str]] = None,
    ):
        if not self._predictor or not self._session_id:
            raise RuntimeError("SAM3 session has not been started.")
        logger.debug(
            "SAM3 add_prompt(frame=%s, text=%s, boxes=%d, points=%d)",
            frame_idx,
            bool(text),
            len(boxes or []),
            len(points or []),
        )
        result = self._predictor.add_prompt(
            session_id=self._session_id,
            frame_idx=frame_idx,
            text=text,
            points=points if points else None,
            point_labels=point_labels if point_labels else None,
            bounding_boxes=boxes if boxes else None,
            bounding_box_labels=box_labels if box_labels else None,
            obj_id=obj_id,
        )
        if record_outputs:
            outputs = result.get("outputs", {}) if isinstance(
                result, dict) else {}
            # Save prompt-frame outputs immediately to avoid losing masks if propagation fails.
            self._handle_frame_outputs(
                frame_idx=frame_idx,
                outputs=outputs or {},
                total_frames=max(len(self.frame_names) or 0,
                                 self.max_frame_num_to_track or 0) or None,
                yielded_frames=1,
                label_hints=label_hints,
            )
        return result

    @staticmethod
    def _normalize_points(
        points_abs: List[List[float]], width: float, height: float
    ) -> List[List[float]]:
        return [[x / width, y / height] for x, y in points_abs]

    @staticmethod
    def _normalize_boxes(
        boxes_abs: List[List[float]], width: float, height: float
    ) -> List[List[float]]:
        return [
            [x / width, y / height, w / width, h / height] for x, y, w, h in boxes_abs
        ]

    @staticmethod
    def _box_iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two [x, y, w, h] boxes in pixel space."""
        ax, ay, aw, ah = [float(v) for v in box_a]
        bx, by, bw, bh = [float(v) for v in box_b]
        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0
        union = aw * ah + bw * bh - inter_area
        if union <= 0.0:
            return 0.0
        return float(inter_area / union)

    def _reset_global_tracks(self) -> None:
        """Reset global track id mapping for sliding-window runs."""
        self._global_track_next_id = 1
        self._global_track_last_box.clear()

    def _assign_global_track_id(
        self,
        box_xywh: np.ndarray,
        used_ids: Optional[set[int]] = None,
        iou_threshold: float = 0.3,
    ) -> int:
        """
        Assign a stable global track id for the given box based on IoU with
        previously seen tracks. Ensures each global id is used at most once
        per frame via the optional `used_ids` set.
        """
        if used_ids is None:
            used_ids = set()

        best_gid: Optional[int] = None
        best_iou = 0.0
        best_center_shift = float("inf")
        for gid, prev_box in self._global_track_last_box.items():
            if gid in used_ids:
                continue
            iou = self._box_iou_xywh(prev_box, box_xywh)
            cx_prev = float(prev_box[0] + prev_box[2] * 0.5)
            cy_prev = float(prev_box[1] + prev_box[3] * 0.5)
            cx_curr = float(box_xywh[0] + box_xywh[2] * 0.5)
            cy_curr = float(box_xywh[1] + box_xywh[3] * 0.5)
            center_shift = float(
                np.hypot(cx_prev - cx_curr, cy_prev - cy_curr))

            if iou > best_iou or (np.isclose(iou, best_iou) and center_shift < best_center_shift):
                best_iou = iou
                best_gid = gid
                best_center_shift = center_shift

        if best_gid is not None and best_iou >= iou_threshold:
            self._global_track_last_box[best_gid] = box_xywh
            return best_gid

        # Fallback: if IoU is low but the box stayed near the previous center,
        # keep the same id to avoid unnecessary id churn on small object counts.
        if best_gid is not None:
            prev_box = np.asarray(
                self._global_track_last_box[best_gid], dtype=float)
            prev_diag = float(np.hypot(prev_box[2], prev_box[3]))
            curr_diag = float(np.hypot(box_xywh[2], box_xywh[3]))
            max_diag = max(prev_diag, curr_diag, 1e-6)
            relative_shift = best_center_shift / max_diag
            if best_iou > 0.05 or relative_shift <= 0.75:
                self._global_track_last_box[best_gid] = box_xywh
                return best_gid

        gid = self._global_track_next_id
        self._global_track_next_id += 1
        self._global_track_last_box[gid] = box_xywh
        return gid

    @staticmethod
    def _label_hints_from_ids(labels: List[int], id_to_labels: Dict[int, str]) -> List[str]:
        hints: List[str] = []
        for lid in labels:
            try:
                hints.append(id_to_labels.get(int(lid), str(lid)))
            except Exception:
                hints.append(str(lid))
        return hints

    def _reset_session_state(self) -> None:
        """Best-effort reset of current predictor state without reloading video."""
        try:
            if self._predictor and self._session_id:
                self._predictor.reset_session(self._session_id)
        except Exception as exc:
            logger.debug("Unable to reset SAM3 session state: %s", exc)

    def _derive_boxes_from_previous_masks(
        self, frame_idx: int
    ) -> List[List[float]]:
        """
        Derive a single bounding box from the most recent prior frame with masks.
        Returns an empty list if no usable box is found.
        """
        prev_frames = sorted(
            f for f in self._frame_masks.keys() if f < frame_idx)
        if not prev_frames:
            return []
        prev_frame = prev_frames[-1]
        prev_masks = self._frame_masks.get(prev_frame) or {}
        if not prev_masks:
            return []

        boxes_abs: List[List[float]] = []
        for mask in prev_masks.values():
            arr = np.asarray(mask, dtype=np.uint8)
            ys, xs = np.nonzero(arr)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            boxes_abs.append([x1, y1, w, h])

        if not boxes_abs:
            return []
        # The downstream SAM3 visual prompt path only supports a single box.
        boxes_abs.sort(key=lambda b: b[2] * b[3], reverse=True)
        return [boxes_abs[0]]

    def add_prompt_points_abs(
        self,
        frame_idx: int,
        points_abs: List[List[float]],
        point_labels: List[int],
        *,
        text: Optional[str] = None,
        obj_id: Optional[int] = None,
    ):
        """Convenience: accept pixel coordinates, normalize, then add prompt."""
        if self.frame_shape is None:
            self.frame_shape = self.get_frame_shape()
        h, w = self.frame_shape[:2]
        points = self._normalize_points(points_abs, w, h)
        return self.add_prompt(
            frame_idx,
            text=text,
            points=points,
            point_labels=point_labels,
            obj_id=obj_id,
        )

    def add_prompt_boxes_abs(
        self,
        frame_idx: int,
        boxes_abs: List[List[float]],
        box_labels: List[int],
        *,
        text: Optional[str] = None,
    ):
        """Convenience: accept pixel-space boxes [x,y,w,h], normalize, then add prompt."""
        if self.frame_shape is None:
            self.frame_shape = self.get_frame_shape()
        h, w = self.frame_shape[:2]
        boxes = self._normalize_boxes(boxes_abs, w, h)
        return self.add_prompt(
            frame_idx,
            text=text,
            boxes=boxes,
            box_labels=box_labels,
        )

    def _handle_frame_outputs(
        self,
        *,
        frame_idx: int,
        outputs: Dict[str, object],
        total_frames: Optional[int] = None,
        yielded_frames: int = 0,
        label_hints: Optional[List[str]] = None,
    ) -> Tuple[int, int]:
        obj_ids = outputs.get("out_obj_ids", [])
        masks = outputs.get("out_binary_masks", [])
        probs = outputs.get("out_probs", [])
        boxes = outputs.get("out_boxes_xywh", [])
        mask_dict = {}
        obj_meta: Dict[str, Dict[str, object]] = {}

        frame_stats = outputs.get("frame_stats")
        frame_meta: Optional[dict] = None
        if frame_stats is not None:
            try:
                fs = frame_stats.tolist() if hasattr(frame_stats, "tolist") else frame_stats
                frame_meta = {"sam3_frame_stats": fs}
            except Exception:
                frame_meta = None

        # In sliding-window text-prompt mode, remap per-window SAM3 object ids to
        # stable global track ids so ids stay consistent across windows.
        if (
            self._sliding_window_mode_active
            and self.text_prompt
            and boxes is not None
            and len(boxes) == len(obj_ids)
        ):
            try:
                used_global: set[int] = set()
                new_ids: List[int] = []
                boxes_arr = np.asarray(boxes, dtype=float)
                for idx, _local_id in enumerate(obj_ids):
                    box_xywh = np.asarray(boxes_arr[idx], dtype=float)
                    gid = self._assign_global_track_id(
                        box_xywh=box_xywh,
                        used_ids=used_global,
                    )
                    used_global.add(gid)
                    new_ids.append(int(gid))
                obj_ids = new_ids
                outputs["out_obj_ids"] = np.asarray(new_ids, dtype=np.int64)
            except Exception:
                # Best-effort remapping; fall back to raw ids on failure.
                pass

        for idx, (obj_id, mask) in enumerate(zip(obj_ids, masks)):
            key = str(obj_id)
            meta: Dict[str, object] = {"track_id": int(obj_id)}
            # Propagate label hints to keep stable labels across frames.
            hint = None
            # Prefer an existing mapping for this SAM3 object id when available.
            existing = self.obj_id_to_label.get(key)
            if existing:
                hint = existing
            if label_hints and idx < len(label_hints):
                hint = label_hints[idx]
            if hint is None and self.text_prompt:
                # Use the text prompt plus global object id to keep labels
                # interpretable and stable across windows, e.g. "mouse_1".
                hint = f"{self.text_prompt}_{int(obj_id)}"
            if hint is not None:
                self.obj_id_to_label[key] = hint
                try:
                    self.id_to_labels[int(obj_id)] = hint
                except Exception:
                    self.id_to_labels[key] = hint
            sam3_score: Optional[float] = None
            if idx < len(probs):
                try:
                    sam3_score = float(probs[idx])
                    meta["sam3_score"] = sam3_score
                except Exception:
                    pass
            score_thresh = self.score_threshold_detection
            if score_thresh is not None and sam3_score is not None:
                if sam3_score < float(score_thresh):
                    # Drop low-confidence outputs instead of writing them to disk.
                    continue
            if idx < len(boxes):
                try:
                    meta["sam3_box_xywh"] = [
                        float(v) for v in boxes[idx].tolist()]
                except Exception:
                    pass
            mask_dict[key] = np.asarray(mask, dtype=np.uint8)
            obj_meta[key] = meta

        if self.frame_shape is None:
            self.frame_shape = self.get_frame_shape()

        filename = str(Path(self.video_dir) / f"{frame_idx:09}.json")
        self._save_annotations(
            filename,
            mask_dict,
            self.frame_shape,
            frame_idx=frame_idx,
            obj_meta=obj_meta,
            frame_meta=frame_meta,
        )
        # Track frames for possible later per-frame reacquisition.
        self._frames_processed.add(int(frame_idx))
        if mask_dict:
            self._frames_with_masks.add(int(frame_idx))
            # Persist masks for this frame so they can be used as visual prompts
            # when re-acquiring lost tracks on later frames.
            self._frame_masks[int(frame_idx)] = {
                k: np.asarray(v, dtype=np.uint8) for k, v in mask_dict.items()
            }

        if not total_frames or total_frames <= 0:
            total_frames = self.total_frames_estimate()
        if total_frames:
            progress = int((yielded_frames / total_frames) * 100)
            if progress > 100:
                progress = 100
            if progress % 10 == 0:
                logger.info(
                    f"SAM3 progress: {yielded_frames}/{total_frames} frames "
                    f"({progress}%), frame={frame_idx}, masks={len(mask_dict)}"
                )
        return len(mask_dict), len(mask_dict)

    def propagate(
        self,
        *,
        start_frame_idx: int,
        propagation_direction: str = "both",
        max_frame_num_to_track: Optional[int] = None,
    ) -> Tuple[int, int]:
        @torch.inference_mode()
        def _propagate():
            if not self._predictor or not self._session_id:
                raise RuntimeError("SAM3 session has not been started.")

            total_frames = self.total_frames_estimate()
            yielded_frames = 0
            total_masks = 0
            for result in self._predictor.propagate_in_video(
                session_id=self._session_id,
                propagation_direction=propagation_direction or self.propagation_direction,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track
                or self.max_frame_num_to_track,
            ):
                frame_idx = result["frame_index"]
                outputs = result.get("outputs", {}) or {}
                yielded_frames += 1
                masks_in_frame, _ = self._handle_frame_outputs(
                    frame_idx=frame_idx,
                    outputs=outputs,
                    total_frames=total_frames,
                    yielded_frames=yielded_frames,
                )
                total_masks += masks_in_frame

            if yielded_frames == 0:
                raise RuntimeError("SAM3 propagate_in_video yielded no frames")
            return yielded_frames, total_masks

        return _propagate()

    def _propagate_text_prompt_with_sliding_window(
        self,
        *,
        text_prompt: str,
        target_device: Optional[torch.device | str],
    ) -> Tuple[int, int]:
        """
        Low-RAM propagation path for text-only runs: keep only a small window
        of frames in memory while still writing masks/metadata through the
        usual annotation pipeline.
        """
        resolved_device = set_default_device(
            target_device or self.default_device)
        device_str = str(
            resolved_device) if resolved_device is not None else None
        total_frames = self.total_frames_estimate()
        yielded_frames = 0
        total_masks = 0

        window_size = int(self.sliding_window_size or 1)
        stride = self.sliding_window_stride
        if stride is not None:
            try:
                stride = int(stride)
            except Exception:
                stride = None

        # Heuristic: MPS is prone to OOM on larger windows. Clamp to a safer
        # default while still allowing CUDA users to run larger windows.
        if resolved_device.type == "mps" and window_size > 5:
            logger.warning(
                "SAM3: clamping sliding-window size from %d to %d for MPS to reduce OOM risk.",
                window_size,
                5,
            )
            window_size = 5
            if stride is None or stride > window_size:
                stride = window_size

        # Enable global-id remapping for the sliding-window run.
        self._reset_global_tracks()
        self._sliding_window_mode_active = True
        try:
            try:
                for frame_idx, outputs in run_video_sliding_window(
                    video_path=str(self.video_path),
                    mode="sam3",
                    text_prompt=text_prompt,
                    window_size=window_size,
                    stride=stride,
                    device=device_str,
                ):
                    yielded_frames += 1
                    masks_in_frame, _ = self._handle_frame_outputs(
                        frame_idx=frame_idx,
                        outputs=outputs,
                        total_frames=total_frames,
                        yielded_frames=yielded_frames,
                    )
                    total_masks += masks_in_frame
            except RuntimeError as exc:
                # CPU fallback for Apple MPS OOM to avoid crashing the GUI.
                if resolved_device.type == "mps" and _is_mps_oom(exc) and yielded_frames == 0:
                    logger.warning(
                        "SAM3 sliding-window propagation hit MPS OOM; retrying on CPU. "
                        "Tip: reduce 'Sliding window size' for MPS to lower memory usage. Error: %s",
                        str(exc),
                    )
                    _clear_mps_cache()
                    resolved_device = set_default_device("cpu")
                    device_str = "cpu"
                    for frame_idx, outputs in run_video_sliding_window(
                        video_path=str(self.video_path),
                        mode="sam3",
                        text_prompt=text_prompt,
                        window_size=window_size,
                        stride=stride,
                        device=device_str,
                    ):
                        yielded_frames += 1
                        masks_in_frame, _ = self._handle_frame_outputs(
                            frame_idx=frame_idx,
                            outputs=outputs,
                            total_frames=total_frames,
                            yielded_frames=yielded_frames,
                        )
                        total_masks += masks_in_frame
                else:
                    raise
        finally:
            self._sliding_window_mode_active = False

        if yielded_frames == 0:
            raise RuntimeError(
                "SAM3 sliding-window propagation yielded no frames"
            )
        return yielded_frames, total_masks

    def _prepare_prompts(
        self, annotations: Iterable[dict], text_prompt: Optional[str]
    ) -> Tuple[Optional[int], List[List[float]], List[int], List[int]]:
        """
        Build bounding-box prompts from cached annotations. Returns
        (frame_idx, boxes, labels, obj_ids).
        """
        if self.frame_shape is None:
            self.frame_shape = self.get_frame_shape()

        if not annotations and text_prompt:
            logger.info(
                "SAM3 using text-only prompt; no per-frame annotations found under %s",
                self.video_dir,
            )
            prompt_frame_idx = self._first_frame_index()
            return prompt_frame_idx, [], [], []

        if not annotations:
            raise FileNotFoundError(
                f"No per-frame JSON annotations found under {self.video_dir}"
            )

        height, width = self.frame_shape[:2]
        annotations_by_frame: Dict[int, List[dict]] = {}
        for ann in annotations:
            try:
                frame_idx = int(ann.get("ann_frame_idx", 0))
            except (TypeError, ValueError):
                frame_idx = 0
            annotations_by_frame.setdefault(frame_idx, []).append(ann)

        for frame_idx in sorted(annotations_by_frame):
            boxes: List[List[float]] = []
            labels: List[int] = []
            obj_ids: List[int] = []
            for ann in annotations_by_frame[frame_idx]:
                label_val = int(ann["labels"][0]) if ann.get("labels") else 1
                if ann["type"] == "box":
                    x1, y1, x2, y2 = ann["box"]
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                elif ann["type"] == "mask":
                    mask = np.asarray(ann["mask"], dtype=np.uint8)
                    ys, xs = np.nonzero(mask)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    x1, x2 = float(xs.min()), float(xs.max())
                    y1, y2 = float(ys.min()), float(ys.max())
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                else:
                    continue

                if w <= 0 or h <= 0:
                    continue

                boxes.append([x1 / width, y1 / height, w / width, h / height])
                labels.append(label_val)
                obj_ids.append(int(ann.get("obj_id", label_val)))

            if boxes:
                return frame_idx, boxes, labels, obj_ids

        return None, [], [], []

    def _first_frame_index(self) -> int:
        """
        Infer the first frame index.
        Prefer existing extracted frame names; fall back to 0 when frames
        are not yet materialized on disk (e.g. text-only prompt on raw video).
        """
        if self.frame_names:
            first_name = Path(self.frame_names[0]).stem
            try:
                return int(first_name)
            except ValueError:
                return 0

        # No extracted frames found; default to frame 0 so that SAM3 can
        # operate directly on the video file via Sam3VideoPredictor.
        logger.info(
            "SAM3: no extracted frames found under %s; defaulting prompt frame index to 0.",
            self.video_dir,
        )
        return 0

    def run_offline(
        self,
        annotations: Iterable[dict],
        target_device: Optional[torch.device | str] = None,
        *,
        propagation_direction: Optional[str] = None,
        max_frame_num_to_track: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Run a one-shot SAM3 propagation using stored annotations/text prompt.
        """
        @torch.inference_mode()
        def _run():
            prompt_frame_idx, boxes, labels, _ = self._prepare_prompts(
                annotations, self.text_prompt
            )
            if prompt_frame_idx is None:
                raise FileNotFoundError(
                    f"No usable prompts found under {self.video_dir}"
                )
            # Build label hints for seed boxes.
            label_hints = self._label_hints_from_ids(labels, self.id_to_labels)

            # Reset tracking sets for this run.
            self._frames_processed.clear()
            self._frames_with_masks.clear()

            if (
                self.use_sliding_window_for_text_prompt
                and self.text_prompt
                and not boxes
            ):
                resolved_device = set_default_device(
                    target_device or self.default_device
                )
                window_size = int(self.sliding_window_size or 1)
                stride = self.sliding_window_stride or window_size
                if resolved_device.type == "mps" and window_size > 5:
                    window_size = 5
                    stride = min(int(stride or window_size), window_size)
                logger.info(
                    "SAM3: running sliding-window text propagation "
                    "(window_size=%d, stride=%s) to keep memory low.",
                    window_size,
                    stride,
                )
                return self._propagate_text_prompt_with_sliding_window(
                    text_prompt=self.text_prompt,
                    target_device=target_device,
                )

            with self._session_scope(target_device) as _:
                # Clear SAM3 action history so the first propagation performs a full
                # video pass instead of a partial/fetch-only update.
                self._reset_action_history_if_supported()

                self.add_prompt(
                    frame_idx=prompt_frame_idx,
                    text=self.text_prompt,
                    boxes=boxes or None,
                    box_labels=labels or None,
                    record_outputs=True,
                    label_hints=label_hints,
                )
                frames, masks = self.propagate(
                    start_frame_idx=prompt_frame_idx,
                    propagation_direction=propagation_direction or self.propagation_direction,
                    max_frame_num_to_track=max_frame_num_to_track or self.max_frame_num_to_track,
                )
                # If any frames were processed without masks, optionally re-run per-frame
                # segmentation using both the text prompt and the last available masks
                # as visual prompts to re-acquire tracking on those frames.
                missing_frames = sorted(
                    self._frames_processed - self._frames_with_masks)
                if missing_frames and self.text_prompt:
                    logger.info(
                        "SAM3: %d frame(s) with no masks; running per-frame visual+text "
                        "reacquisition on these frames.",
                        len(missing_frames),
                    )
                    self._reacquire_frames_with_visual_and_text(
                        missing_frames, target_device
                    )
            return frames, masks

        return _run()

    def _reacquire_frame_with_visual_and_text(
        self,
        frame_idx: int,
        target_device: Optional[torch.device | str] = None,
    ) -> None:
        """
        Run a lightweight SAM3 pass on a single frame using the text prompt
        and the last available masks as visual prompts to recover tracking
        when the main tracker path produced none.
        """
        with torch.inference_mode():
            if not self.text_prompt or not self._frame_masks:
                return

            boxes_abs = self._derive_boxes_from_previous_masks(frame_idx)
            if not boxes_abs:
                return

            with self._session_scope(target_device, auto_close=True):
                if self.frame_shape is None:
                    self.frame_shape = self.get_frame_shape()
                h, w = self.frame_shape[:2]
                boxes = self._normalize_boxes(boxes_abs, w, h)
                self._reset_session_state()
                self._reset_action_history_if_supported()
                self.add_prompt(
                    frame_idx=frame_idx,
                    text=self.text_prompt,
                    boxes=boxes,
                    box_labels=[1] * len(boxes),
                    record_outputs=True,
                    label_hints=None,
                )
                self.propagate(
                    start_frame_idx=frame_idx,
                    propagation_direction="forward",
                    max_frame_num_to_track=1,
                )

    def _reacquire_frames_with_visual_and_text(
        self,
        frame_indices: List[int],
        target_device: Optional[torch.device | str] = None,
    ) -> None:
        """
        Re-acquire multiple frames using a single temporary session to avoid
        reloading video data repeatedly, improving speed on CPU fallback.
        """
        if not frame_indices or not self.text_prompt or not self._frame_masks:
            return

        with torch.inference_mode():
            with self._session_scope(target_device, auto_close=True):
                if self.frame_shape is None:
                    self.frame_shape = self.get_frame_shape()
                h, w = self.frame_shape[:2]

                for frame_idx in frame_indices:
                    boxes_abs = self._derive_boxes_from_previous_masks(
                        frame_idx)
                    if not boxes_abs:
                        continue
                    boxes = self._normalize_boxes(boxes_abs, w, h)
                    self._reset_session_state()
                    self._reset_action_history_if_supported()
                    try:
                        self.add_prompt(
                            frame_idx=frame_idx,
                            text=self.text_prompt,
                            boxes=boxes,
                            box_labels=[1] * len(boxes),
                            record_outputs=True,
                            label_hints=None,
                        )
                        self.propagate(
                            start_frame_idx=frame_idx,
                            propagation_direction="forward",
                            max_frame_num_to_track=1,
                        )
                    except Exception as exc:
                        logger.warning(
                            "SAM3 per-frame reacquisition failed for frame %s: %s",
                            frame_idx,
                            exc,
                        )
