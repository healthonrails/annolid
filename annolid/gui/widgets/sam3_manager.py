from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from qtpy import QtWidgets

import yaml
from annolid.segmentation.SAM.sam3.prompt_builder import normalize_text_prompt
from annolid.utils.annotation_compat import shape_to_mask
from annolid.utils.logger import logger


@dataclass(frozen=True)
class Sam3RuntimeOptions:
    propagation_direction: str
    max_frame_num_to_track: Optional[int]
    device: Optional[str]
    score_threshold_detection: Optional[float]
    new_det_thresh: Optional[float]
    sliding_window_size: int
    sliding_window_stride: Optional[int]
    compile_model: bool
    offload_video_to_cpu: bool
    use_explicit_window_reseed: bool
    boundary_mask_match_iou_threshold: float
    allow_private_state_mutation: bool
    max_num_objects: int
    multiplex_count: int
    agent_det_thresh: Optional[float]
    agent_window_size: int
    agent_stride: Optional[int]
    agent_output_dir: Optional[str]


class Sam3Manager:
    """Encapsulates SAM3 runtime state, prompts, and video processor setup."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        self.window = window
        self.sam3_session = None
        self._last_prompt_frame: Optional[int] = None

        # Runtime overrides (populated via Advanced Parameters dialog).
        self.score_threshold_detection: Optional[float] = None
        self.new_det_thresh: Optional[float] = None
        self.propagation_direction: Optional[str] = None
        self.max_frame_num_to_track: Optional[int] = None
        self.device_override: Optional[str] = None
        self.sliding_window_size: Optional[int] = None
        self.sliding_window_stride: Optional[int] = None
        self.compile_model: Optional[bool] = None
        self.offload_video_to_cpu: Optional[bool] = None
        self.use_explicit_window_reseed: Optional[bool] = None
        self.boundary_mask_match_iou_threshold: Optional[float] = None
        self.allow_private_state_mutation: Optional[bool] = None
        self.max_num_objects: Optional[int] = None
        self.multiplex_count: Optional[int] = None
        self.agent_det_thresh: Optional[float] = None
        self.agent_window_size: Optional[int] = None
        self.agent_stride: Optional[int] = None
        self.agent_output_dir: Optional[str] = None

    @staticmethod
    def is_sam3_model(identifier: str, weight: str) -> bool:
        key = f"{identifier or ''} {weight or ''}".lower()
        return "sam3" in key

    @staticmethod
    def resolve_checkpoint_path(weight: str) -> Optional[str]:
        """
        Resolve a SAM3 checkpoint if the provided weight path exists.
        Otherwise return None to allow the SAM3 package to auto-download
        (requires HF auth and GPU).
        """
        if not weight:
            return None
        candidate = Path(weight).expanduser()
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            pt_files = sorted(candidate.glob("*.pt"))
            if pt_files:
                return str(pt_files[0])
            return None
        return None

    @staticmethod
    def _parse_optional_int(value, *, default: Optional[int] = None) -> Optional[int]:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_optional_float(
        value,
        *,
        default: Optional[float] = None,
    ) -> Optional[float]:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_bool(value, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _pick(override, configured):
        return override if override is not None else configured

    @staticmethod
    def _normalized_rect_from_points(p1, p2) -> tuple[float, float, float, float]:
        x1, y1 = float(p1.x()), float(p1.y())
        x2, y2 = float(p2.x()), float(p2.y())
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return left, top, right, bottom

    def _runtime_options(self, base_config: Dict[str, Any]) -> Sam3RuntimeOptions:
        sam3_cfg = dict((base_config or {}).get("sam3", {}) or {})
        agent_cfg = dict(sam3_cfg.get("agent", {}) or {})

        score_threshold_detection = self._parse_optional_float(
            self._pick(
                getattr(self, "score_threshold_detection", None),
                sam3_cfg.get("score_threshold_detection"),
            )
        )
        new_det_thresh = self._parse_optional_float(
            self._pick(
                getattr(self, "new_det_thresh", None),
                sam3_cfg.get("new_det_thresh"),
            )
        )
        max_frame_num_to_track = self._parse_optional_int(
            self._pick(
                getattr(self, "max_frame_num_to_track", None),
                sam3_cfg.get("max_frame_num_to_track"),
            )
        )
        sliding_window_size = self._parse_optional_int(
            self._pick(
                getattr(self, "sliding_window_size", None),
                sam3_cfg.get("sliding_window_size"),
            ),
            default=5,
        )
        sliding_window_size = max(1, int(sliding_window_size or 5))
        sliding_window_stride = self._parse_optional_int(
            self._pick(
                getattr(self, "sliding_window_stride", None),
                sam3_cfg.get("sliding_window_stride"),
            )
        )
        max_num_objects = self._parse_optional_int(
            self._pick(
                getattr(self, "max_num_objects", None),
                sam3_cfg.get("max_num_objects"),
            ),
            default=16,
        )
        multiplex_count = self._parse_optional_int(
            self._pick(
                getattr(self, "multiplex_count", None),
                sam3_cfg.get("multiplex_count"),
            ),
            default=16,
        )
        boundary_iou = self._parse_optional_float(
            self._pick(
                getattr(self, "boundary_mask_match_iou_threshold", None),
                sam3_cfg.get("boundary_mask_match_iou_threshold"),
            ),
            default=0.2,
        )
        propagation_direction = str(
            self._pick(
                getattr(self, "propagation_direction", None),
                sam3_cfg.get("propagation_direction", "both"),
            )
            or "both"
        )
        device = self._pick(
            getattr(self, "device_override", None),
            sam3_cfg.get("device"),
        )
        device = str(device).strip() if device else None

        compile_model = self._parse_bool(
            self._pick(
                getattr(self, "compile_model", None),
                sam3_cfg.get("compile_model"),
            ),
            False,
        )
        offload_video_to_cpu = self._parse_bool(
            self._pick(
                getattr(self, "offload_video_to_cpu", None),
                sam3_cfg.get("offload_video_to_cpu"),
            ),
            True,
        )
        use_explicit_window_reseed = self._parse_bool(
            self._pick(
                getattr(self, "use_explicit_window_reseed", None),
                sam3_cfg.get("use_explicit_window_reseed"),
            ),
            True,
        )
        allow_private_state_mutation = self._parse_bool(
            self._pick(
                getattr(self, "allow_private_state_mutation", None),
                sam3_cfg.get("allow_private_state_mutation"),
            ),
            False,
        )

        agent_det_thresh = self._parse_optional_float(
            self._pick(
                getattr(self, "agent_det_thresh", None),
                agent_cfg.get("det_thresh"),
            ),
            default=score_threshold_detection,
        )
        agent_window_size = self._parse_optional_int(
            self._pick(
                getattr(self, "agent_window_size", None),
                agent_cfg.get("window_size"),
            ),
            default=sliding_window_size,
        )
        agent_window_size = max(1, int(agent_window_size or sliding_window_size))
        agent_stride = self._parse_optional_int(
            self._pick(getattr(self, "agent_stride", None), agent_cfg.get("stride")),
            default=sliding_window_stride,
        )
        agent_output_dir = self._pick(
            getattr(self, "agent_output_dir", None),
            agent_cfg.get("output_dir"),
        )
        if isinstance(agent_output_dir, str):
            agent_output_dir = agent_output_dir.strip() or None

        return Sam3RuntimeOptions(
            propagation_direction=propagation_direction,
            max_frame_num_to_track=max_frame_num_to_track,
            device=device,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            sliding_window_size=sliding_window_size,
            sliding_window_stride=sliding_window_stride,
            compile_model=compile_model,
            offload_video_to_cpu=offload_video_to_cpu,
            use_explicit_window_reseed=use_explicit_window_reseed,
            boundary_mask_match_iou_threshold=float(boundary_iou or 0.2),
            allow_private_state_mutation=allow_private_state_mutation,
            max_num_objects=max(
                1,
                int(16 if max_num_objects is None else max_num_objects),
            ),
            multiplex_count=max(
                1,
                int(16 if multiplex_count is None else multiplex_count),
            ),
            agent_det_thresh=agent_det_thresh,
            agent_window_size=agent_window_size,
            agent_stride=agent_stride,
            agent_output_dir=agent_output_dir,
        )

    def close_session(self) -> None:
        """Best-effort close of any active SAM3 session."""
        try:
            if self.sam3_session:
                self.sam3_session.close_session()
        except Exception as exc:  # pragma: no cover - shutdown best effort
            logger.warning("Error closing SAM3 session on exit: %s", exc)
        self.sam3_session = None

    def reset_active_session(self) -> bool:
        """
        Best-effort reset for an active SAM3 session state.
        """
        session = self.sam3_session
        if session is None:
            return False
        try:
            session.reset_session_state()
            return True
        except Exception as exc:
            logger.warning("Failed to reset active SAM3 session: %s", exc)
            return False

    def remove_object_from_active_session(
        self, obj_id: int, frame_idx: Optional[int] = None
    ) -> bool:
        """
        Remove one tracked object from active SAM3 session.
        """
        session = self.sam3_session
        if session is None:
            return False
        try:
            if frame_idx is None:
                frame_idx = max(getattr(self.window, "frame_number", 0), 0)
            session.remove_object(obj_id=int(obj_id), frame_idx=int(frame_idx))
            return True
        except Exception as exc:
            logger.warning(
                "Failed to remove SAM3 object id=%s at frame=%s: %s",
                obj_id,
                frame_idx,
                exc,
            )
            return False

    def dialog_defaults(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return the SAM3 runtime defaults for the Advanced Parameters dialog."""
        sam3_cfg = dict((base_config or {}).get("sam3", {}) or {})
        agent_cfg = sam3_cfg.get("agent", {}) or {}

        def _pick(override, cfg_value):
            return override if override is not None else cfg_value

        score_threshold_detection = _pick(
            self.score_threshold_detection,
            sam3_cfg.get("score_threshold_detection"),
        )
        new_det_thresh = _pick(
            self.new_det_thresh,
            sam3_cfg.get("new_det_thresh"),
        )
        propagation_direction = _pick(
            self.propagation_direction,
            sam3_cfg.get("propagation_direction"),
        )
        max_frame_num_to_track = _pick(
            self.max_frame_num_to_track,
            sam3_cfg.get("max_frame_num_to_track"),
        )
        device_override = _pick(
            self.device_override,
            sam3_cfg.get("device"),
        )
        sliding_window_size = _pick(
            self.sliding_window_size,
            sam3_cfg.get("sliding_window_size"),
        )
        sliding_window_stride = _pick(
            self.sliding_window_stride,
            sam3_cfg.get("sliding_window_stride"),
        )
        compile_model = _pick(
            self.compile_model,
            sam3_cfg.get("compile_model"),
        )
        offload_video_to_cpu = _pick(
            self.offload_video_to_cpu,
            sam3_cfg.get("offload_video_to_cpu"),
        )
        use_explicit_window_reseed = _pick(
            self.use_explicit_window_reseed,
            sam3_cfg.get("use_explicit_window_reseed"),
        )
        boundary_mask_match_iou_threshold = _pick(
            self.boundary_mask_match_iou_threshold,
            sam3_cfg.get("boundary_mask_match_iou_threshold"),
        )
        if boundary_mask_match_iou_threshold is None:
            boundary_mask_match_iou_threshold = 0.2
        allow_private_state_mutation = _pick(
            self.allow_private_state_mutation,
            sam3_cfg.get("allow_private_state_mutation"),
        )
        max_num_objects = _pick(
            self.max_num_objects,
            sam3_cfg.get("max_num_objects"),
        )
        multiplex_count = _pick(
            self.multiplex_count,
            sam3_cfg.get("multiplex_count"),
        )
        agent_det_thresh = _pick(
            self.agent_det_thresh,
            agent_cfg.get("det_thresh"),
        )
        agent_window_size = _pick(
            self.agent_window_size,
            agent_cfg.get("window_size"),
        )
        agent_stride = _pick(
            self.agent_stride,
            agent_cfg.get("stride"),
        )
        agent_output_dir = _pick(
            self.agent_output_dir,
            agent_cfg.get("output_dir"),
        )

        return {
            "score_threshold_detection": score_threshold_detection,
            "new_det_thresh": new_det_thresh,
            "propagation_direction": propagation_direction,
            "max_frame_num_to_track": max_frame_num_to_track,
            "device": device_override,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "compile_model": compile_model,
            "offload_video_to_cpu": offload_video_to_cpu,
            "use_explicit_window_reseed": use_explicit_window_reseed,
            "boundary_mask_match_iou_threshold": boundary_mask_match_iou_threshold,
            "allow_private_state_mutation": allow_private_state_mutation,
            "max_num_objects": max_num_objects,
            "multiplex_count": multiplex_count,
            "agent_det_thresh": agent_det_thresh,
            "agent_window_size": agent_window_size
            if agent_window_size is not None
            else sliding_window_size,
            "agent_stride": agent_stride,
            "agent_output_dir": agent_output_dir,
        }

    def apply_dialog_results(
        self, dialog: QtWidgets.QDialog, base_config: Dict[str, Any]
    ) -> None:
        """Persist SAM3 runtime overrides after the Advanced Parameters dialog."""
        sam3_thresholds = dialog.get_sam3_thresholds()
        self.score_threshold_detection = sam3_thresholds.get(
            "score_threshold_detection"
        )
        self.new_det_thresh = sam3_thresholds.get("new_det_thresh")
        sam3_runtime = dialog.get_sam3_runtime_settings()
        self.propagation_direction = sam3_runtime.get("propagation_direction")
        self.max_frame_num_to_track = sam3_runtime.get("max_frame_num_to_track")
        self.device_override = sam3_runtime.get("device")
        self.sliding_window_size = sam3_runtime.get("sliding_window_size")
        self.sliding_window_stride = sam3_runtime.get("sliding_window_stride")
        self.compile_model = sam3_runtime.get("compile_model")
        self.offload_video_to_cpu = sam3_runtime.get("offload_video_to_cpu")
        self.use_explicit_window_reseed = sam3_runtime.get("use_explicit_window_reseed")
        boundary_mask_match_iou_threshold = sam3_runtime.get(
            "boundary_mask_match_iou_threshold", self.boundary_mask_match_iou_threshold
        )
        try:
            self.boundary_mask_match_iou_threshold = (
                None
                if boundary_mask_match_iou_threshold is None
                else float(boundary_mask_match_iou_threshold)
            )
        except Exception:
            self.boundary_mask_match_iou_threshold = 0.2
        self.allow_private_state_mutation = sam3_runtime.get(
            "allow_private_state_mutation"
        )
        self.max_num_objects = sam3_runtime.get("max_num_objects", self.max_num_objects)
        self.multiplex_count = sam3_runtime.get("multiplex_count", self.multiplex_count)
        self.agent_det_thresh = sam3_runtime.get("agent_det_thresh")
        self.agent_window_size = sam3_runtime.get("agent_window_size")
        self.agent_stride = sam3_runtime.get("agent_stride")
        self.agent_output_dir = sam3_runtime.get("agent_output_dir")
        logger.info(
            "SAM3 thresholds updated: score=%.4f, new_det=%.4f",
            float(self.score_threshold_detection or 0.0),
            float(self.new_det_thresh or 0.0),
        )
        sam3_updates = {
            "score_threshold_detection": self.score_threshold_detection,
            "new_det_thresh": self.new_det_thresh,
            "propagation_direction": self.propagation_direction,
            "max_frame_num_to_track": self.max_frame_num_to_track,
            "device": self.device_override,
            "sliding_window_size": self.sliding_window_size,
            "sliding_window_stride": self.sliding_window_stride,
            "compile_model": self.compile_model,
            "offload_video_to_cpu": self.offload_video_to_cpu,
            "use_explicit_window_reseed": self.use_explicit_window_reseed,
            "boundary_mask_match_iou_threshold": self.boundary_mask_match_iou_threshold,
            "allow_private_state_mutation": self.allow_private_state_mutation,
            "max_num_objects": self.max_num_objects,
            "multiplex_count": self.multiplex_count,
        }
        agent_updates = {
            "det_thresh": self.agent_det_thresh,
            "window_size": self.agent_window_size,
            "stride": self.agent_stride,
            "output_dir": self.agent_output_dir,
        }
        self._persist_config_defaults(base_config, sam3_updates, agent_updates)

    def _persist_config_defaults(
        self, base_config: Dict[str, Any], sam3_updates: dict, agent_updates: dict
    ) -> None:
        """
        Persist SAM3/agent runtime overrides to both the in-memory config and
        the user config file (~/.labelmerc) so future sessions pick them up.
        """
        try:
            if isinstance(base_config, dict):
                sam3_block = base_config.setdefault("sam3", {})
                sam3_block.update(sam3_updates)
                agent_block = sam3_block.get("agent", {}) or {}
                agent_block.update(agent_updates)
                sam3_block["agent"] = agent_block
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("Failed to update in-memory SAM3 config: %s", exc)

        config_path = Path.home() / ".labelmerc"
        try:
            disk_cfg: dict = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                    if isinstance(loaded, dict):
                        disk_cfg = loaded

            disk_cfg.setdefault("sam3", {})
            disk_cfg["sam3"].update(sam3_updates)
            agent_block = disk_cfg["sam3"].get("agent", {}) or {}
            agent_block.update(agent_updates)
            disk_cfg["sam3"]["agent"] = agent_block

            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(disk_cfg, fh, sort_keys=False)
            logger.info("SAM3 defaults saved to %s", config_path)
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning(
                "Failed to persist SAM3 defaults to %s: %s", config_path, exc
            )

    def extract_prompts_from_canvas(self) -> dict:
        """
        Extract SAM3-friendly prompts from current canvas shapes.

        Returns a dict with:
          - frame_idx: current frame
          - boxes_abs: list of rectangle prompts as [x, y, w, h] in pixel space
          - box_labels: list of int labels (defaults to 1)
          - polygons_abs: list of polygon prompts in pixel space
          - polygon_labels: list of int labels for polygons
          - points_abs: list of [x, y]
          - point_labels: list of int labels (defaults to 1 for positive clicks)
        """
        frame_idx = max(getattr(self.window, "frame_number", 0), 0)
        boxes_abs = []
        box_labels = []
        polygons_abs = []
        polygon_labels = []
        points_abs = []
        point_labels = []
        canvas = getattr(self.window, "canvas", None)
        if canvas is None or not getattr(canvas, "shapes", None):
            return {
                "frame_idx": frame_idx,
                "boxes_abs": boxes_abs,
                "box_labels": box_labels,
                "polygons_abs": polygons_abs,
                "polygon_labels": polygon_labels,
                "points_abs": points_abs,
                "point_labels": point_labels,
            }

        for shape in canvas.shapes:
            if shape.shape_type == "rectangle" and len(shape.points) == 2:
                p1, p2 = shape.points
                x1, y1, x2, y2 = self._normalized_rect_from_points(p1, p2)
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                boxes_abs.append([x1, y1, w, h])
                box_labels.append(1)
            elif shape.shape_type == "polygon" and shape.points:
                polygon = [[pt.x(), pt.y()] for pt in shape.points]
                polygons_abs.append(polygon)
                polygon_labels.append(1)
            elif shape.shape_type in ["points", "point"]:
                labels = (
                    shape.point_labels
                    if shape.point_labels
                    else [1] * len(shape.points)
                )
                for pt, lbl in zip(shape.points, labels):
                    points_abs.append([pt.x(), pt.y()])
                    point_labels.append(lbl if lbl is not None else 1)
        return {
            "frame_idx": frame_idx,
            "boxes_abs": boxes_abs,
            "box_labels": box_labels,
            "polygons_abs": polygons_abs,
            "polygon_labels": polygon_labels,
            "points_abs": points_abs,
            "point_labels": point_labels,
        }

    @staticmethod
    def _stable_obj_id_for_shape(
        shape,
        *,
        reverse_label_map: Dict[str, int],
        assigned_by_group: Dict[int, int],
        next_id_ref: Dict[str, int],
    ) -> int:
        group_id = getattr(shape, "group_id", None)
        if group_id is not None:
            try:
                gid = int(group_id)
                if gid > 0:
                    if gid not in assigned_by_group:
                        assigned_by_group[gid] = gid
                        next_id_ref["value"] = max(next_id_ref["value"], gid + 1)
                    return assigned_by_group[gid]
            except Exception:
                pass
        label = str(getattr(shape, "label", "") or "").strip()
        if label and label in reverse_label_map:
            try:
                return max(1, int(reverse_label_map[label]))
            except Exception:
                pass
        obj_id = int(next_id_ref["value"])
        next_id_ref["value"] = obj_id + 1
        if label:
            reverse_label_map.setdefault(label, obj_id)
        return obj_id

    def _canvas_prompts_to_annotations(
        self,
        *,
        frame_idx: int,
        id_to_labels: Dict[int, str],
    ) -> list:
        """
        Convert current canvas prompts into annotation records consumed by SAM3.
        """
        canvas = getattr(self.window, "canvas", None)
        if canvas is None or not getattr(canvas, "shapes", None):
            return []

        reverse_label_map: Dict[str, int] = {}
        for key, value in (id_to_labels or {}).items():
            try:
                reverse_label_map[str(value)] = int(key)
            except Exception:
                continue
        assigned_by_group: Dict[int, int] = {}
        next_start = 1
        if reverse_label_map:
            next_start = max(reverse_label_map.values()) + 1
        next_id_ref: Dict[str, int] = {"value": max(1, int(next_start))}

        image_h: Optional[int] = None
        image_w: Optional[int] = None
        frame_image = getattr(self.window, "image", None)
        if isinstance(frame_image, np.ndarray) and frame_image.ndim >= 2:
            image_h, image_w = int(frame_image.shape[0]), int(frame_image.shape[1])

        ann_records: list = []
        for shape in canvas.shapes:
            label = str(getattr(shape, "label", "") or "").strip() or "object"
            obj_id = self._stable_obj_id_for_shape(
                shape,
                reverse_label_map=reverse_label_map,
                assigned_by_group=assigned_by_group,
                next_id_ref=next_id_ref,
            )
            id_to_labels.setdefault(int(obj_id), label)

            if shape.shape_type == "rectangle" and len(shape.points) == 2:
                p1, p2 = shape.points
                x1, y1, x2, y2 = self._normalized_rect_from_points(p1, p2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if image_h and image_w:
                    try:
                        mask = shape_to_mask(
                            img_shape=(int(image_h), int(image_w)),
                            points=[[x1, y1], [x2, y2]],
                            shape_type="rectangle",
                        )
                    except Exception:
                        mask = None
                    if mask is not None and np.any(mask):
                        ann_records.append(
                            {
                                "type": "mask",
                                "ann_frame_idx": int(frame_idx),
                                "mask": np.asarray(mask, dtype=np.uint8),
                                "labels": [int(obj_id)],
                                "obj_id": int(obj_id),
                            }
                        )
                        continue
                ann_records.append(
                    {
                        "type": "box",
                        "ann_frame_idx": int(frame_idx),
                        "box": [x1, y1, x2, y2],
                        "labels": [int(obj_id)],
                        "obj_id": int(obj_id),
                    }
                )
            elif shape.shape_type == "polygon" and shape.points:
                poly = [[float(pt.x()), float(pt.y())] for pt in shape.points]
                if image_h and image_w:
                    try:
                        mask = shape_to_mask(
                            img_shape=(int(image_h), int(image_w)),
                            points=poly,
                            shape_type="polygon",
                        )
                    except Exception:
                        mask = None
                    if mask is not None and np.any(mask):
                        ann_records.append(
                            {
                                "type": "mask",
                                "ann_frame_idx": int(frame_idx),
                                "mask": np.asarray(mask, dtype=np.uint8),
                                "labels": [int(obj_id)],
                                "obj_id": int(obj_id),
                            }
                        )
                        continue
                ann_records.append(
                    {
                        "type": "polygon",
                        "ann_frame_idx": int(frame_idx),
                        "polygon": poly,
                        "labels": [int(obj_id)],
                        "obj_id": int(obj_id),
                    }
                )
            elif shape.shape_type in {"point", "points"} and shape.points:
                labels = list(shape.point_labels or [1] * len(shape.points))
                points = [[float(pt.x()), float(pt.y())] for pt in shape.points]
                prompt_labels: list[int] = []
                for raw_label in labels:
                    try:
                        prompt_labels.append(1 if int(raw_label) > 0 else 0)
                    except Exception:
                        prompt_labels.append(1)
                ann_records.append(
                    {
                        "type": "points",
                        "ann_frame_idx": int(frame_idx),
                        "points": points,
                        "labels": prompt_labels,
                        "obj_id": int(obj_id),
                    }
                )
        return ann_records

    @staticmethod
    def _merge_canvas_annotations(
        annotations: list,
        canvas_annotations: list,
    ) -> list:
        """
        Merge live canvas prompts into loaded annotations.

        Canvas prompts win for the same frame/object pair so that a user who
        edits frame 42 in the GUI does not get overridden by stale saved data.
        """
        if not canvas_annotations:
            return list(annotations or [])

        def _annotation_key(ann: dict) -> tuple[int, int]:
            return (
                int(ann.get("ann_frame_idx", -1)),
                int(ann.get("obj_id", 0)),
            )

        canvas_keys = set()
        for ann in canvas_annotations:
            if not isinstance(ann, dict):
                continue
            try:
                canvas_keys.add(_annotation_key(ann))
            except Exception:
                continue

        merged = []
        for ann in annotations or []:
            if not isinstance(ann, dict):
                continue
            try:
                key = _annotation_key(ann)
            except Exception:
                key = None
            if key is not None and key in canvas_keys:
                continue
            merged.append(dict(ann))

        merged.extend(dict(ann) for ann in canvas_annotations if isinstance(ann, dict))
        merged.sort(
            key=lambda ann: (
                int(ann.get("ann_frame_idx", 0)),
                int(ann.get("obj_id", 0)),
                str(ann.get("type", "")),
            )
        )
        return merged

    def build_video_processor(
        self, model_name: str, model_weight: str, text_prompt: Optional[str]
    ):
        """
        Construct the SAM3 video processor or agent runner callable.
        Returns a callable or None if setup fails before worker start.
        """
        try:
            from annolid.segmentation.SAM.sam3 import adapter as sam3_adapter
            from annolid.segmentation.SAM.sam_v2 import (
                load_manual_seed_annotations_from_video,
            )
        except Exception as exc:  # pragma: no cover - import guard
            QtWidgets.QMessageBox.warning(
                self.window,
                "SAM3 import error",
                f"Failed to load SAM3 packages.\n{exc}",
            )
            return None

        sam3_checkpoint = self.resolve_checkpoint_path(model_weight)
        if sam3_checkpoint:
            # Ensure the SAM3 Agent image path uses the same checkpoint
            os.environ["SAM3_CKPT_PATH"] = sam3_checkpoint
        text_prompt = normalize_text_prompt(
            text_prompt or self.window._current_text_prompt()
        )

        logger.info(
            "Using SAM3 with checkpoint '%s'",
            sam3_checkpoint if sam3_checkpoint else "auto-download",
        )
        try:
            annotations, id_to_labels = load_manual_seed_annotations_from_video(
                self.window.video_file
            )
        except FileNotFoundError:
            annotations, id_to_labels = [], {}
            logger.info(
                "SAM3: no per-frame JSON annotations found under %s; "
                "continuing with text-only prompt if provided.",
                self.window.video_file,
            )

        prompt_frame_idx = max(getattr(self.window, "frame_number", 0), 0)
        canvas_ann = self._canvas_prompts_to_annotations(
            frame_idx=prompt_frame_idx,
            id_to_labels=id_to_labels,
        )
        if canvas_ann:
            annotations = self._merge_canvas_annotations(annotations, canvas_ann)

        if not annotations and not text_prompt:
            QtWidgets.QMessageBox.warning(
                self.window,
                "SAM3 prompts missing",
                "No per-frame annotations were found and no text prompt was "
                "provided. Please add a prompt and try again.",
            )
            return None

        options = self._runtime_options(getattr(self.window, "_config", {}) or {})
        propagation_direction = options.propagation_direction
        max_frame_num_to_track = options.max_frame_num_to_track
        device_override = options.device
        score_threshold_detection = options.score_threshold_detection
        new_det_thresh = options.new_det_thresh
        sliding_window_size = options.sliding_window_size
        sliding_window_stride = options.sliding_window_stride
        compile_model = options.compile_model
        offload_video_to_cpu = options.offload_video_to_cpu
        use_explicit_window_reseed = options.use_explicit_window_reseed
        boundary_mask_match_iou_threshold = options.boundary_mask_match_iou_threshold
        allow_private_state_mutation = options.allow_private_state_mutation
        max_num_objects = options.max_num_objects
        multiplex_count = options.multiplex_count
        agent_det_thresh = options.agent_det_thresh
        agent_window_size = options.agent_window_size
        agent_stride = options.agent_stride
        agent_output_dir = options.agent_output_dir

        if not annotations:
            annotations = canvas_ann

        def _build_standard_sam3_runner():
            try:
                self.sam3_session = sam3_adapter.SAM3VideoProcessor(
                    video_dir=self.window.video_file,
                    id_to_labels=id_to_labels,
                    annotations=annotations,
                    checkpoint_path=sam3_checkpoint,
                    text_prompt=text_prompt,
                    epsilon_for_polygon=self.window.epsilon_for_polygon,
                    propagation_direction=propagation_direction,
                    max_frame_num_to_track=max_frame_num_to_track,
                    device=device_override,
                    score_threshold_detection=score_threshold_detection,
                    new_det_thresh=new_det_thresh,
                    max_num_objects=max_num_objects,
                    multiplex_count=multiplex_count,
                    compile_model=compile_model,
                    offload_video_to_cpu=offload_video_to_cpu,
                    use_explicit_window_reseed=use_explicit_window_reseed,
                    boundary_mask_match_iou_threshold=boundary_mask_match_iou_threshold,
                    allow_private_state_mutation=allow_private_state_mutation,
                    sliding_window_size=sliding_window_size,
                    sliding_window_stride=sliding_window_stride,
                )
                self._last_prompt_frame = None
            except Exception as exc:
                logger.error(
                    "Failed to initialise SAM3 session for '%s': %s",
                    self.window.video_file,
                    exc,
                    exc_info=True,
                )
                return RuntimeError(f"Failed to initialise SAM3 session.\n{exc}")

            def _run_sam3_with_canvas_prompts(pred_worker=None, stop_event=None):
                return self.sam3_session.run(stop_event=stop_event)

            _run_sam3_with_canvas_prompts.request_stop = self.sam3_session.request_stop

            return _run_sam3_with_canvas_prompts

        def _run_sam3_agent_first():
            """
            Prefer agent-seeded SAM3; fall back to standard SAM3 if agent fails.
            """
            if text_prompt:
                try:
                    frames, masks = sam3_adapter.process_video_with_agent(
                        video_path=self.window.video_file,
                        agent_prompt=text_prompt,
                        agent_det_thresh=agent_det_thresh,
                        window_size=agent_window_size,
                        stride=agent_stride,
                        output_dir=agent_output_dir,
                        checkpoint_path=sam3_checkpoint,
                        propagation_direction=propagation_direction,
                        device=device_override,
                        score_threshold_detection=score_threshold_detection,
                        new_det_thresh=new_det_thresh,
                        max_num_objects=max_num_objects,
                        multiplex_count=multiplex_count,
                        compile_model=compile_model,
                        offload_video_to_cpu=offload_video_to_cpu,
                        use_explicit_window_reseed=use_explicit_window_reseed,
                        boundary_mask_match_iou_threshold=boundary_mask_match_iou_threshold,
                        allow_private_state_mutation=allow_private_state_mutation,
                    )
                    if masks <= 0:
                        raise RuntimeError(
                            "Agent mode returned zero masks; falling back to standard SAM3."
                        )
                    return f"SAM3 agent-seeded done#{masks}"
                except Exception as exc:
                    logger.warning(
                        "SAM3 agent mode failed; falling back to standard tracking: %s",
                        exc,
                        exc_info=False,
                    )

            standard_runner = _build_standard_sam3_runner()
            if isinstance(standard_runner, Exception):
                return standard_runner
            if standard_runner is None:
                return RuntimeError("SAM3 init failed.")
            return standard_runner()

        return _run_sam3_agent_first
