from __future__ import annotations

import importlib
import json
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch

from annolid.segmentation.SAM.sam_v2 import BaseSAMVideoProcessor
from annolid.utils.annotation_compat import shape_to_mask
from annolid.utils.logger import logger
from .sam3.utils import set_default_device
from .window_refresh import compute_mid_window_refresh_index, run_mid_window_refresh

SAM3_IMPORT_ERROR: Optional[Exception] = None
_SAM3_REQUIRED_MODULES = ("iopath", "ftfy")
_BUILD_SAM3_PREDICTOR: Optional[Callable[..., Any]] = None


class Sam3StopRequested(RuntimeError):
    """Raised when a SAM3 prediction is cancelled cooperatively."""

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
        external_builder = importlib.import_module("sam3.model_builder")
        candidate = getattr(external_builder, "build_sam3_predictor", None)
        if not callable(candidate):
            raise RuntimeError(
                "Bundled SAM3 runtime is missing `build_sam3_predictor`."
            )
        _BUILD_SAM3_PREDICTOR = candidate
    except Exception as exc:  # pragma: no cover - import guard
        SAM3_IMPORT_ERROR = RuntimeError(
            "Bundled SAM3.1 runtime is unavailable or outdated. Ensure "
            "`annolid/segmentation/SAM/sam3/sam3` contains a build that exposes "
            "`sam3.model_builder.build_sam3_predictor`."
        )


class _PredictorAPIAdapter:
    """
    Normalize predictor APIs across:
    - legacy direct-method predictors (start_session/add_prompt/propagate_in_video)
    - request-based predictors (handle_request/handle_stream_request)
    """

    def __init__(self, predictor: Any):
        self._predictor = predictor

    def start_session(self, *, resource_path: str, offload_video_to_cpu: bool) -> Dict[str, Any]:
        if hasattr(self._predictor, "start_session"):
            try:
                return self._predictor.start_session(
                    resource_path=resource_path,
                    offload_video_to_cpu=offload_video_to_cpu,
                )
            except TypeError:
                return self._predictor.start_session(resource_path=resource_path)
        return self._predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": resource_path,
                "offload_video_to_cpu": bool(offload_video_to_cpu),
            }
        )

    def add_prompt(
        self,
        *,
        session_id: str,
        frame_idx: int,
        text: Optional[str],
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]],
        bounding_boxes: Optional[List[List[float]]],
        bounding_box_labels: Optional[List[int]],
        mask_inputs: Optional[List[object]] = None,
        mask_labels: Optional[List[int]] = None,
        obj_id: Optional[int],
    ) -> Dict[str, Any]:
        if not bounding_boxes:
            bounding_boxes = None
            bounding_box_labels = None
        elif not bounding_box_labels:
            bounding_box_labels = [1] * len(bounding_boxes)

        if hasattr(self._predictor, "add_prompt"):
            return self._predictor.add_prompt(
                session_id=session_id,
                frame_idx=frame_idx,
                text=text,
                points=points,
                point_labels=point_labels,
                bounding_boxes=bounding_boxes,
                bounding_box_labels=bounding_box_labels,
                mask_inputs=mask_inputs,
                mask_labels=mask_labels,
                obj_id=obj_id,
            )
        return self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
                "text": text,
                "points": points,
                "point_labels": point_labels,
                "bounding_boxes": bounding_boxes,
                "bounding_box_labels": bounding_box_labels,
                "mask_inputs": mask_inputs,
                "mask_labels": mask_labels,
                "obj_id": obj_id,
            }
        )

    def propagate_in_video(
        self,
        *,
        session_id: str,
        propagation_direction: str,
        start_frame_idx: Optional[int],
        max_frame_num_to_track: Optional[int],
    ) -> Iterator[Dict[str, Any]]:
        if hasattr(self._predictor, "propagate_in_video"):
            return self._predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction=propagation_direction,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
            )
        return self._predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": propagation_direction,
                "start_frame_index": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
            }
        )

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        if hasattr(self._predictor, "reset_session"):
            return self._predictor.reset_session(session_id)
        return self._predictor.handle_request({"type": "reset_session", "session_id": session_id})

    def remove_object(
        self,
        *,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int,
    ) -> Dict[str, Any]:
        if hasattr(self._predictor, "remove_object"):
            return self._predictor.remove_object(
                session_id=session_id,
                frame_idx=frame_idx,
                obj_id=obj_id,
            )
        return self._predictor.handle_request(
            {
                "type": "remove_object",
                "session_id": session_id,
                "frame_index": frame_idx,
                "obj_id": obj_id,
            }
        )

    def close_session(self, session_id: str) -> Dict[str, Any]:
        if hasattr(self._predictor, "close_session"):
            return self._predictor.close_session(session_id)
        return self._predictor.handle_request({"type": "close_session", "session_id": session_id})

    def cancel_propagation(self, session_id: str) -> Dict[str, Any]:
        if hasattr(self._predictor, "cancel_propagation"):
            return self._predictor.cancel_propagation(session_id=session_id)
        return self._predictor.handle_request(
            {"type": "cancel_propagation", "session_id": session_id}
        )

    @property
    def raw(self) -> Any:
        return self._predictor


def _default_bpe_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"


def _is_mps_oom(exc: BaseException) -> bool:
    msg = str(exc)
    return "MPS backend out of memory" in msg or "mps backend out of memory" in msg


def _clear_mps_cache() -> None:
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
    # Multiplex tuning knobs (SAM3.1).
    max_num_objects: int = 16
    multiplex_count: int = 16
    # Performance knobs.
    compile_model: bool = False
    offload_video_to_cpu: bool = True
    async_loading_frames: bool = False  # keep memory usage low; no preloading
    sliding_window_size: int = 5
    sliding_window_stride: Optional[int] = None
    use_sliding_window_for_text_prompt: bool = False


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
    max_num_objects: int,
    multiplex_count: int,
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
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
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
        max_num_objects: int = 16,
        multiplex_count: int = 16,
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
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
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

        self._predictor: Optional[_PredictorAPIAdapter] = None
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
        # Track ids observed per frame and their most recent seen frame so we can
        # detect partial track loss and recover only the missing instances.
        self._frame_track_ids: Dict[int, set[int]] = {}
        self._track_last_seen_frame: Dict[int, int] = {}
        self.max_frame_num_to_track = cfg.max_frame_num_to_track
        # Default propagation settings (can be overridden per-call).
        self.propagation_direction = cfg.propagation_direction or "both"
        self.default_device = cfg.device
        self.score_threshold_detection = cfg.score_threshold_detection
        self.new_det_thresh = cfg.new_det_thresh
        self.max_num_objects = int(cfg.max_num_objects or 16)
        self.multiplex_count = int(cfg.multiplex_count or 16)
        self.compile_model = bool(cfg.compile_model)
        self.offload_video_to_cpu = bool(cfg.offload_video_to_cpu)
        self.async_loading_frames = cfg.async_loading_frames
        self.sliding_window_size = cfg.sliding_window_size
        self.sliding_window_stride = cfg.sliding_window_stride
        self.use_sliding_window_for_text_prompt = cfg.use_sliding_window_for_text_prompt
        self._stop_event = None
        self._stop_requested = False
        # Cross-window tracking id state used by SAM3.1 windowed mode.
        self._global_track_next_id: int = 1
        self._global_track_last_box: Dict[int, np.ndarray] = {}
        self._global_track_last_seen_frame: Dict[int, int] = {}
        self._global_track_history: Dict[int, deque[np.ndarray]] = {}

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
        if _BUILD_SAM3_PREDICTOR is None:
            raise RuntimeError(
                "SAM3.1 predictor backend is unavailable from the bundled SAM3 runtime."
            )
        try:
            predictor = _BUILD_SAM3_PREDICTOR(
                checkpoint_path=self.checkpoint_path,
                bpe_path=str(self.bpe_path),
                version="sam3.1",
                compile=bool(self.compile_model),
                async_loading_frames=bool(self.async_loading_frames),
                max_num_objects=self.max_num_objects,
                multiplex_count=self.multiplex_count,
                device=str(device),
            )
        except TypeError as exc:
            raise RuntimeError(
                "Installed `sam3` package is too old for SAM3.1-only mode. "
                "Please upgrade `sam3`."
            ) from exc
        return _PredictorAPIAdapter(predictor)

    def _resolve_runtime_device(
        self, target_device: Optional[torch.device | str]
    ) -> torch.device:
        """
        Resolve the runtime device for SAM3.1 inference.

        Note: current SAM3.1 multiplex stack is unstable on MPS for some kernels/
        dtype paths. Prefer CPU fallback over hard abort on macOS.
        """
        resolved = set_default_device(target_device or self.default_device)
        if resolved.type == "mps":
            logger.warning(
                "SAM3.1 multiplex runtime on MPS is unstable in this environment; "
                "falling back to CPU."
            )
            resolved = torch.device("cpu")
            try:
                torch.set_default_device(resolved)
            except Exception:
                pass
        if resolved.type == "cpu":
            torch.set_default_dtype(torch.float32)
        return resolved

    def start_session(self, target_device: Optional[torch.device | str] = None) -> str:
        resolved_device = self._resolve_runtime_device(target_device)

        if self._predictor is None or self._predictor_device != resolved_device:
            self._predictor = self._initialize_predictor(resolved_device)
            self._predictor_device = resolved_device
        session = self._predictor.start_session(
            resource_path=str(self.video_path),
            offload_video_to_cpu=self.offload_video_to_cpu,
        )
        self._session_id = session["session_id"]
        logger.info(
            "SAM3.1 session %s started on device=%s (checkpoint=%s)",
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

    def bind_stop_event(self, stop_event) -> None:
        self._stop_event = stop_event

    def request_stop(self) -> None:
        self._stop_requested = True
        stop_event = getattr(self, "_stop_event", None)
        try:
            if stop_event is not None:
                stop_event.set()
        except Exception:
            pass
        session_id = getattr(self, "_session_id", None)
        predictor = getattr(self, "_predictor", None)
        if session_id and predictor is not None:
            try:
                predictor.cancel_propagation(session_id=session_id)
            except Exception:
                logger.debug("SAM3 cancel_propagation request failed.", exc_info=True)

    def _check_stop_requested(self) -> None:
        stop_event = getattr(self, "_stop_event", None)
        if self._stop_requested or (stop_event is not None and stop_event.is_set()):
            self.request_stop()
            raise Sam3StopRequested("Stopped by user.")

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

    def _reset_action_history_if_supported(self, session_id: Optional[str] = None):
        """
        Best-effort clearing of cached action history on the underlying predictor
        so the next propagation performs a full pass.
        """
        try:
            raw_predictor = self._predictor.raw if self._predictor is not None else None
            states = getattr(raw_predictor, "_ALL_INFERENCE_STATES", None)
            if states is None:
                states = getattr(raw_predictor, "_all_inference_states", None)
            target_session_id = str(session_id or self._session_id or "")
            state_entry = (
                states.get(target_session_id)
                if isinstance(states, dict) and target_session_id
                else None
            )
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
        mask_inputs: Optional[List[object]] = None,
        mask_labels: Optional[List[int]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
        session_id: Optional[str] = None,
        record_outputs: bool = False,
        merge_existing_on_record: bool = False,
        label_hints: Optional[List[str]] = None,
    ):
        target_session_id = str(session_id or self._session_id or "")
        if not self._predictor or not target_session_id:
            raise RuntimeError("SAM3 session has not been started.")
        logger.debug(
            "SAM3 add_prompt(frame=%s, text=%s, boxes=%d, masks=%d, points=%d)",
            frame_idx,
            bool(text),
            len(boxes or []),
            len(mask_inputs or []),
            len(points or []),
        )
        result = self._execute_prompt_transaction(
            session_id=target_session_id,
            frame_idx=frame_idx,
            text=text,
            boxes=boxes,
            box_labels=box_labels,
            mask_inputs=mask_inputs,
            mask_labels=mask_labels,
            points=points,
            point_labels=point_labels,
            obj_id=obj_id,
        )
        if record_outputs:
            step_results = []
            if isinstance(result, dict):
                step_results = list(result.get("transaction_steps", []) or [])
            if not step_results:
                step_results = [result if isinstance(result, dict) else {}]
            for idx, step_result in enumerate(step_results):
                outputs = step_result.get("outputs", {}) if isinstance(step_result, dict) else {}
                # Save prompt-frame outputs immediately to avoid losing masks if propagation fails.
                self._handle_frame_outputs(
                    frame_idx=frame_idx,
                    outputs=outputs or {},
                    total_frames=max(len(self.frame_names) or 0,
                                     self.max_frame_num_to_track or 0) or None,
                    yielded_frames=1,
                    label_hints=label_hints,
                    apply_score_threshold=False,
                    merge_existing=bool(merge_existing_on_record) or idx > 0,
                )
        return result

    @staticmethod
    def _has_text_prompt(text: Optional[str]) -> bool:
        return bool(isinstance(text, str) and text.strip())

    @staticmethod
    def _has_prompt_items(items: Optional[List[List[float]]]) -> bool:
        return bool(items and len(items) > 0)

    @staticmethod
    def _polygon_seed_points(
        polygon: List[List[float]], max_points: int = 8
    ) -> List[List[float]]:
        """
        Derive a small, deterministic set of positive seed points from a polygon.

        We keep this conservative: one centroid-like anchor plus a handful of
        evenly sampled boundary vertices. That preserves the polygon geometry
        better than a bbox-only seed while staying within the existing point
        prompt contract.
        """
        if not polygon:
            return []
        arr = np.asarray(polygon, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return []
        if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
            arr = arr[:-1]
        if len(arr) == 0:
            return []

        points: List[List[float]] = []
        centroid = np.mean(arr, axis=0)
        points.append([float(centroid[0]), float(centroid[1])])

        if len(arr) == 1 or max_points <= 1:
            return points

        sample_count = min(len(arr), max(1, max_points - 1))
        sample_indices = np.linspace(0, len(arr) - 1, num=sample_count, dtype=int)
        seen = {tuple(points[0])}
        for idx in sample_indices:
            pt = [float(arr[idx, 0]), float(arr[idx, 1])]
            key = (round(pt[0], 4), round(pt[1], 4))
            if key in seen:
                continue
            points.append(pt)
            seen.add(key)
        return points

    @staticmethod
    def _shape_points_to_mask(
        points: List[List[float]],
        frame_shape: Tuple[int, int, int] | Tuple[int, int],
        *,
        shape_type: str = "polygon",
    ) -> Optional[np.ndarray]:
        if not points:
            return None
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
            arr = arr[:-1]
        if len(arr) < 3:
            return None
        try:
            mask = shape_to_mask(
                img_shape=(int(frame_shape[0]), int(frame_shape[1])),
                points=arr.tolist(),
                shape_type=shape_type or "polygon",
            )
        except Exception:
            return None
        if mask is None:
            return None
        return np.asarray(mask, dtype=np.uint8)

    def _build_prompt_transaction_steps(
        self,
        *,
        text: Optional[str],
        boxes: Optional[List[List[float]]],
        box_labels: Optional[List[int]],
        mask_inputs: Optional[List[object]],
        mask_labels: Optional[List[int]],
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]],
        obj_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Build a prompt transaction that keeps semantic prompts together.
        Sequencing rules: text + boxes + masks in one semantic request, then points.
        """
        normalized_boxes = boxes if self._has_prompt_items(boxes) else None
        normalized_masks = mask_inputs if self._has_prompt_items(mask_inputs) else None
        normalized_points = points if self._has_prompt_items(points) else None
        safe_box_labels = self._sanitize_prompt_labels(
            box_labels,
            expected_len=len(normalized_boxes) if normalized_boxes else 0,
            default_value=1,
        )
        safe_mask_labels = self._sanitize_prompt_labels(
            mask_labels,
            expected_len=len(normalized_masks) if normalized_masks else 0,
            default_value=1,
        )
        safe_point_labels = self._sanitize_prompt_labels(
            point_labels,
            expected_len=len(normalized_points) if normalized_points else 0,
            default_value=1,
        )

        steps: List[Dict[str, Any]] = []
        if self._has_text_prompt(text) or normalized_boxes is not None or normalized_masks is not None:
            semantic_step: Dict[str, Any] = {"kind": "semantic"}
            if self._has_text_prompt(text):
                semantic_step["text"] = str(text).strip()
            if normalized_boxes is not None:
                semantic_step["bounding_boxes"] = normalized_boxes
                semantic_step["bounding_box_labels"] = safe_box_labels
            if normalized_masks is not None:
                semantic_step["mask_inputs"] = normalized_masks
                semantic_step["mask_labels"] = safe_mask_labels
            steps.append(semantic_step)
        if normalized_points is not None:
            point_obj_id = obj_id if obj_id is not None else 1
            steps.append(
                {
                    "kind": "points",
                    "points": normalized_points,
                    "point_labels": safe_point_labels,
                    "obj_id": int(point_obj_id),
                }
            )
        if not steps:
            raise ValueError(
                "No prompt payload provided. Supply one of: text, boxes, masks, points."
            )
        return steps

    @classmethod
    def _prompt_result_mask_count(cls, result: Any) -> int:
        if not isinstance(result, dict):
            return 0
        step_results = list(result.get("transaction_steps", []) or [])
        if not step_results:
            step_results = [result]
        total = 0
        for step in step_results:
            outputs = step.get("outputs", {}) if isinstance(step, dict) else {}
            total += cls._output_candidate_mask_count(outputs or {})
        return int(total)

    @staticmethod
    def _as_sequence(value: object) -> List[object]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @classmethod
    def _merge_prompt_outputs(cls, outputs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge prompt-step outputs so transaction callers see the combined result.

        Later prompt steps override earlier entries for the same obj_id, which
        matches the expected "text -> boxes -> points" refinement sequence.
        """
        merged: Dict[str, Any] = {}
        ordered_obj_ids: List[int] = []
        entries: Dict[int, Dict[str, Any]] = {}

        for outputs in outputs_list:
            if not isinstance(outputs, dict):
                continue
            obj_ids = cls._as_sequence(outputs.get("out_obj_ids"))
            probs = cls._as_sequence(outputs.get("out_probs"))
            boxes = cls._as_sequence(outputs.get("out_boxes_xywh"))
            masks = cls._as_sequence(outputs.get("out_binary_masks"))
            if not obj_ids:
                continue
            for idx, raw_obj_id in enumerate(obj_ids):
                try:
                    obj_id = int(raw_obj_id)
                except Exception:
                    continue
                record = entries.get(obj_id)
                if record is None:
                    record = {
                        "out_obj_ids": obj_id,
                        "out_probs": None,
                        "out_boxes_xywh": None,
                        "out_binary_masks": None,
                    }
                    entries[obj_id] = record
                    ordered_obj_ids.append(obj_id)
                if idx < len(probs):
                    record["out_probs"] = probs[idx]
                if idx < len(boxes):
                    record["out_boxes_xywh"] = boxes[idx]
                if idx < len(masks):
                    record["out_binary_masks"] = masks[idx]

            if "frame_stats" in outputs and outputs.get("frame_stats") is not None:
                merged["frame_stats"] = outputs.get("frame_stats")

        if ordered_obj_ids:
            merged["out_obj_ids"] = np.asarray(ordered_obj_ids, dtype=np.int64)
            merged["out_probs"] = np.asarray(
                [entries[obj_id]["out_probs"] if entries[obj_id]["out_probs"] is not None else 0.0 for obj_id in ordered_obj_ids],
                dtype=np.float32,
            )
            merged["out_boxes_xywh"] = np.asarray(
                [
                    entries[obj_id]["out_boxes_xywh"]
                    if entries[obj_id]["out_boxes_xywh"] is not None
                    else [0.0, 0.0, 0.0, 0.0]
                    for obj_id in ordered_obj_ids
                ],
                dtype=np.float32,
            )
            merged["out_binary_masks"] = np.asarray(
                [
                    entries[obj_id]["out_binary_masks"]
                    if entries[obj_id]["out_binary_masks"] is not None
                    else np.zeros(0, dtype=np.uint8)
                    for obj_id in ordered_obj_ids
                ],
                dtype=object,
            )
        return merged

    def _execute_prompt_transaction(
        self,
        *,
        session_id: str,
        frame_idx: int,
        text: Optional[str],
        boxes: Optional[List[List[float]]],
        box_labels: Optional[List[int]],
        mask_inputs: Optional[List[object]] = None,
        mask_labels: Optional[List[int]] = None,
        points: Optional[List[List[float]]],
        point_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self._predictor:
            raise RuntimeError("SAM3 predictor is not initialized.")
        steps = self._build_prompt_transaction_steps(
            text=text,
            boxes=boxes,
            box_labels=box_labels,
            mask_inputs=mask_inputs,
            mask_labels=mask_labels,
            points=points,
            point_labels=point_labels,
            obj_id=obj_id,
        )
        if len(steps) > 1:
            logger.info(
                "SAM3 prompt transaction split into %d single-type requests (order: %s).",
                len(steps),
                " -> ".join(str(s.get("kind", "?")) for s in steps),
            )

        step_results: List[Dict[str, Any]] = []
        last_result: Dict[str, Any] = {"frame_index": frame_idx, "outputs": {}}
        step_outputs: List[Dict[str, Any]] = []
        for step in steps:
            kind = step.get("kind")
            result = self._predictor.add_prompt(
                session_id=session_id,
                frame_idx=frame_idx,
                text=step.get("text") if kind == "semantic" else None,
                points=step.get("points") if kind == "points" else None,
                point_labels=step.get("point_labels") if kind == "points" else None,
                bounding_boxes=step.get("bounding_boxes") if kind == "semantic" else None,
                bounding_box_labels=step.get("bounding_box_labels")
                if kind == "semantic"
                else None,
                mask_inputs=step.get("mask_inputs") if kind == "semantic" else None,
                mask_labels=step.get("mask_labels") if kind == "semantic" else None,
                obj_id=step.get("obj_id") if kind == "points" else None,
            )
            if isinstance(result, dict):
                result = dict(result)
                result["prompt_kind"] = kind
                step_results.append(result)
                last_result = result
                step_outputs.append(result.get("outputs", {}) or {})
            else:
                step_results.append({"frame_index": frame_idx, "outputs": {}, "prompt_kind": kind})

        merged_result = dict(last_result)
        merged_result["outputs"] = self._merge_prompt_outputs(step_outputs)
        merged_result["transaction_steps"] = step_results
        merged_result["transaction_step_kinds"] = [s.get("prompt_kind") for s in step_results]
        return merged_result

    @staticmethod
    def _sanitize_prompt_labels(
        labels: Optional[List[int]],
        *,
        expected_len: int,
        default_value: int = 1,
    ) -> Optional[List[int]]:
        if expected_len <= 0:
            return None
        src = list(labels) if labels is not None else []
        out: List[int] = []
        for idx in range(expected_len):
            raw = src[idx] if idx < len(src) else default_value
            try:
                out.append(1 if int(raw) > 0 else 0)
            except Exception:
                out.append(1 if int(default_value) > 0 else 0)
        return out

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
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        union = aw * ah + bw * bh - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    @staticmethod
    def _box_center_xywh(box: np.ndarray) -> Tuple[float, float]:
        x, y, w, h = [float(v) for v in np.asarray(box, dtype=float).tolist()]
        return x + (w / 2.0), y + (h / 2.0)

    @staticmethod
    def _box_area_xywh(box: np.ndarray) -> float:
        _, _, w, h = [float(v) for v in np.asarray(box, dtype=float).tolist()]
        return max(0.0, w) * max(0.0, h)

    def _track_match_max_gap(self) -> int:
        """
        Maximum allowed gap for cross-window matching.

        Use the larger of window size and stride so the matcher survives the
        actual handoff between windows instead of treating the next window as
        stale by default.
        """
        return max(
            2,
            int(self.sliding_window_size or 0),
            int(self.sliding_window_stride or 0),
        )

    def _box_track_match_score(
        self,
        prev_box: np.ndarray,
        candidate_box: np.ndarray,
        *,
        age_frames: int = 0,
        max_gap: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Score how likely a detection continues a previously seen track.

        The score blends overlap, center proximity, and scale similarity. A
        modest age penalty keeps stale tracks from stealing identities after a
        long gap.
        """
        prev = np.asarray(prev_box, dtype=float)
        cand = np.asarray(candidate_box, dtype=float)
        iou = self._box_iou_xywh(prev, cand)
        prev_cx, prev_cy = self._box_center_xywh(prev)
        cand_cx, cand_cy = self._box_center_xywh(cand)
        center_dx = cand_cx - prev_cx
        center_dy = cand_cy - prev_cy
        prev_w, prev_h = float(prev[2]), float(prev[3])
        cand_w, cand_h = float(cand[2]), float(cand[3])
        prev_diag = float(np.hypot(max(prev_w, cand_w), max(prev_h, cand_h)))
        if prev_diag <= 0.0:
            prev_diag = 1.0
        center_dist = float(np.hypot(center_dx, center_dy) / prev_diag)
        center_score = float(max(0.0, 1.0 - min(center_dist, 2.0)))

        prev_area = self._box_area_xywh(prev)
        cand_area = self._box_area_xywh(cand)
        if prev_area <= 0.0 or cand_area <= 0.0:
            size_score = 0.0
        else:
            size_score = float(min(prev_area, cand_area) / max(prev_area, cand_area))

        if max_gap is None:
            max_gap = self._track_match_max_gap()
        gap_norm = max(1.0, float(max_gap))
        age_penalty = min(0.3, max(0.0, float(age_frames)) / gap_norm * 0.15)

        score = (0.6 * iou) + (0.25 * center_score) + (0.15 * size_score) - age_penalty
        return float(score), float(iou), float(center_score), float(size_score)

    def _track_match_candidates(
        self,
        box_xywh: np.ndarray,
        *,
        frame_idx: Optional[int] = None,
        used_ids: Optional[set[int]] = None,
        min_score: float = 0.35,
    ) -> List[Tuple[float, int]]:
        if used_ids is None:
            used_ids = set()

        max_gap = self._track_match_max_gap()
        candidates: List[Tuple[float, int]] = []
        for gid, prev_box in self._global_track_last_box.items():
            if gid in used_ids:
                continue
            last_seen = self._global_track_last_seen_frame.get(gid)
            age = 0
            if frame_idx is not None and last_seen is not None:
                age = max(0, int(frame_idx) - int(last_seen))
                if age > max_gap:
                    continue
            reference_box = self._predict_global_track_box(
                gid,
                frame_idx=frame_idx,
            )
            score, iou, center_score, _ = self._box_track_match_score(
                reference_box,
                box_xywh,
                age_frames=age,
                max_gap=max_gap,
            )
            if score < min_score:
                continue
            if iou < 0.05 and center_score < 0.45:
                continue
            candidates.append((float(score), int(gid)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _reset_global_tracks(self) -> None:
        self._global_track_next_id = 1
        self._global_track_last_box.clear()
        self._global_track_last_seen_frame.clear()
        self._global_track_history.clear()

    def _record_global_track_observation(
        self,
        gid: int,
        box_xywh: np.ndarray,
        *,
        frame_idx: Optional[int] = None,
    ) -> None:
        box_arr = np.asarray(box_xywh, dtype=float)
        self._global_track_last_box[int(gid)] = box_arr
        if frame_idx is not None:
            self._global_track_last_seen_frame[int(gid)] = int(frame_idx)
        history = self._global_track_history.setdefault(int(gid), deque(maxlen=4))
        history.append(box_arr)

    def _predict_global_track_box(
        self,
        gid: int,
        *,
        frame_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict the reference box for a track using its recent motion history.

        The last observed box remains the fallback. When we have at least two
        observations and a positive frame gap, extrapolate a simple linear
        motion estimate. This improves matching continuity across windows when
        objects move steadily between refresh points.
        """
        prev_box = self._global_track_last_box.get(int(gid))
        if prev_box is None:
            return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)

        history = self._global_track_history.get(int(gid))
        last_seen = self._global_track_last_seen_frame.get(int(gid))
        if (
            frame_idx is None
            or last_seen is None
            or history is None
            or len(history) < 2
        ):
            return np.asarray(prev_box, dtype=float)

        age = max(0, int(frame_idx) - int(last_seen))
        if age <= 0:
            return np.asarray(prev_box, dtype=float)

        # Use short-range motion only. Longer gaps should rely on the last
        # observed box rather than a linear extrapolation that can overshoot.
        motion_horizon = max(1, min(self._track_match_max_gap(), 3))
        if age > motion_horizon:
            return np.asarray(prev_box, dtype=float)

        last_box = np.asarray(history[-1], dtype=float)
        prev_hist_box = np.asarray(history[-2], dtype=float)
        delta = last_box - prev_hist_box
        predicted = last_box + (delta * float(age))
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != (4,):
            return np.asarray(prev_box, dtype=float)
        predicted[2] = max(1.0, float(predicted[2]))
        predicted[3] = max(1.0, float(predicted[3]))
        if not np.all(np.isfinite(predicted)):
            return np.asarray(prev_box, dtype=float)
        return predicted

    def _assign_global_track_id(
        self,
        box_xywh: np.ndarray,
        used_ids: Optional[set[int]] = None,
        *,
        frame_idx: Optional[int] = None,
        iou_threshold: float = 0.3,
    ) -> int:
        if used_ids is None:
            used_ids = set()

        candidates = self._track_match_candidates(
            box_xywh,
            frame_idx=frame_idx,
            used_ids=used_ids,
            min_score=max(0.2, float(iou_threshold) * 0.9),
        )
        if candidates:
            best_score, best_gid = candidates[0]
            if best_score >= max(0.2, float(iou_threshold) * 0.9):
                self._record_global_track_observation(
                    best_gid,
                    box_xywh,
                    frame_idx=frame_idx,
                )
                return best_gid

        gid = self._global_track_next_id
        self._global_track_next_id += 1
        self._record_global_track_observation(
            gid,
            box_xywh,
            frame_idx=frame_idx,
        )
        return gid

    @staticmethod
    def _empty_window_dir(window_dir: Path) -> None:
        for child in window_dir.glob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)

    @staticmethod
    def _write_window_frames(
        window_dir: Path,
        frames: List[np.ndarray],
        *,
        previous_count: int = 0,
        shift: int = 0,
    ) -> int:
        """
        Materialize a window using stable local filenames.

        For heavily overlapping windows, reuse prior files by shifting existing
        frame files left and writing only the new tail frames.
        """
        frame_count = len(frames)
        prev_count = max(0, int(previous_count))
        shift_count = max(0, int(shift))

        can_shift_reuse = (
            prev_count > 0
            and frame_count > 0
            and shift_count > 0
            and shift_count < prev_count
            and frame_count == prev_count
        )
        if can_shift_reuse:
            # Shift files left in-place (0..n-1) => (0..n-1-shift), then write
            # only the last `shift` positions for the new tail frames.
            for src_idx in range(shift_count, prev_count):
                src_path = window_dir / f"{src_idx:06d}.jpg"
                dst_path = window_dir / f"{src_idx - shift_count:06d}.jpg"
                if src_path.exists():
                    src_path.replace(dst_path)
            start_write = max(0, frame_count - shift_count)
        else:
            start_write = 0

        for local_idx in range(start_write, frame_count):
            frame = frames[local_idx]
            out_path = window_dir / f"{local_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)

        for stale_idx in range(frame_count, max(frame_count, prev_count)):
            stale_path = window_dir / f"{stale_idx:06d}.jpg"
            stale_path.unlink(missing_ok=True)
        return frame_count

    def _iter_video_windows(
        self,
        *,
        window_size: int,
        stride: int,
    ) -> Iterator[Tuple[int, int, List[np.ndarray]]]:
        # For file-backed videos, always read windows from the actual video.
        # Sidecar images in the results folder may include stale prediction
        # artifacts and must not redefine the source timeline.
        if (not Path(self.video_path).is_file()) and self.frame_names:
            total = len(self.frame_names)
            for start in range(0, total, stride):
                end = min(start + window_size, total)
                if end <= start:
                    break
                frames: List[np.ndarray] = []
                for idx in range(start, end):
                    frame_path = Path(self.video_dir) / self.frame_names[idx]
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        continue
                    frames.append(frame)
                if frames:
                    yield start, start + len(frames), frames
            return

        if not Path(self.video_path).is_file():
            return
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        try:
            frames_buf: deque[np.ndarray] = deque()

            while len(frames_buf) < window_size:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_buf.append(frame)

            start = 0
            while frames_buf:
                frames = list(frames_buf)
                end = start + len(frames)
                yield start, end, frames
                if total_frames and end >= total_frames:
                    break

                advance = min(stride, len(frames_buf))
                for _ in range(advance):
                    frames_buf.popleft()
                    start += 1

                while len(frames_buf) < window_size:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames_buf.append(frame)

                if not frames_buf or (total_frames and start >= total_frames):
                    break
        finally:
            cap.release()

    def _map_outputs_to_global_ids(self, outputs: Dict[str, object]) -> Dict[str, object]:
        return self._map_outputs_to_global_ids_at_frame(outputs, frame_idx=None)

    def _map_outputs_to_global_ids_at_frame(
        self,
        outputs: Dict[str, object],
        *,
        frame_idx: Optional[int],
    ) -> Dict[str, object]:
        obj_ids = outputs.get("out_obj_ids", [])
        boxes = outputs.get("out_boxes_xywh", [])
        if boxes is None:
            return outputs
        try:
            boxes_arr = np.asarray(boxes, dtype=float)
        except Exception:
            return outputs
        if len(boxes_arr) != len(obj_ids):
            return outputs

        used: set[int] = set()
        pairs: List[Tuple[float, int, int]] = []
        for idx, _ in enumerate(obj_ids):
            candidates = self._track_match_candidates(
                np.asarray(boxes_arr[idx], dtype=float),
                frame_idx=frame_idx,
                used_ids=used,
            )
            for score, gid in candidates:
                pairs.append((float(score), int(idx), int(gid)))

        pairs.sort(key=lambda item: item[0], reverse=True)
        mapped: List[Optional[int]] = [None] * len(obj_ids)
        assigned_gids: set[int] = set()
        assigned_dets: set[int] = set()
        for score, det_idx, gid in pairs:
            if det_idx in assigned_dets or gid in assigned_gids:
                continue
            if score < 0.35:
                continue
            mapped[det_idx] = gid
            assigned_dets.add(det_idx)
            assigned_gids.add(gid)
            used.add(gid)
            self._record_global_track_observation(
                gid,
                np.asarray(boxes_arr[det_idx], dtype=float),
                frame_idx=frame_idx,
            )

        for idx, mapped_gid in enumerate(mapped):
            if mapped_gid is not None:
                continue
            gid = self._assign_global_track_id(
                box_xywh=np.asarray(boxes_arr[idx], dtype=float),
                used_ids=used,
                frame_idx=frame_idx,
            )
            used.add(gid)
            mapped[idx] = int(gid)
        outputs = dict(outputs)
        outputs["out_obj_ids"] = np.asarray([int(v) for v in mapped], dtype=np.int64)
        return outputs

    @staticmethod
    def _output_candidate_mask_count(outputs: Dict[str, object]) -> int:
        """Estimate how many masks an output payload can materialize."""
        if not isinstance(outputs, dict):
            return 0
        obj_ids = outputs.get("out_obj_ids", [])
        masks = outputs.get("out_binary_masks", [])
        if obj_ids is None:
            obj_ids = []
        if masks is None:
            masks = []
        try:
            return max(0, min(len(obj_ids), len(masks)))
        except Exception:
            return 0

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

    def reset_session_state(self) -> None:
        """Public wrapper for resetting active SAM3 session state."""
        self._reset_session_state()

    def _derive_boxes_from_neighbor_masks(
        self, frame_idx: int, *, max_boxes: int = 4
    ) -> List[List[float]]:
        """
        Derive visual prompt boxes from the nearest frame that has masks.

        This is boundary-safe for windowed inference: if the immediate previous
        frame has no masks (common near window cuts), we can still seed from the
        nearest valid frame on either side.
        """
        if not self._frame_masks:
            return []
        candidate_frames = [
            f for f, masks in self._frame_masks.items() if masks and int(f) != int(frame_idx)
        ]
        if not candidate_frames:
            return []
        # Prefer closest frame; break ties toward the previous frame.
        nearest_frame = min(
            candidate_frames,
            key=lambda f: (abs(int(f) - int(frame_idx)), 0 if int(f) < int(frame_idx) else 1),
        )
        masks_for_frame = self._frame_masks.get(int(nearest_frame)) or {}

        boxes_abs: List[List[float]] = []
        for mask in masks_for_frame.values():
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
        boxes_abs.sort(key=lambda b: b[2] * b[3], reverse=True)
        limit = max(1, int(max_boxes))
        return boxes_abs[:limit]

    def _derive_boxes_for_track_ids(
        self,
        frame_idx: int,
        track_ids: Iterable[int],
        *,
        max_boxes: Optional[int] = None,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Derive carry boxes for specific global track ids.

        Prefer the nearest stored mask for that track. Fall back to the last
        known global track box when a per-frame mask is unavailable.
        """
        requested = []
        for track_id in track_ids:
            try:
                requested.append(int(track_id))
            except Exception:
                continue
        if not requested:
            return [], []

        if max_boxes is None:
            max_boxes = len(requested)
        limit = max(1, int(max_boxes))

        boxes_abs: List[List[float]] = []
        box_track_ids: List[int] = []
        for track_id in requested[:limit]:
            track_key = str(int(track_id))
            nearest_box: Optional[List[float]] = None
            candidate_frames = [
                int(f)
                for f, masks in self._frame_masks.items()
                if masks and track_key in masks and int(f) != int(frame_idx)
            ]
            if candidate_frames:
                nearest_frame = min(
                    candidate_frames,
                    key=lambda f: (abs(int(f) - int(frame_idx)), 0 if int(f) < int(frame_idx) else 1),
                )
                arr = np.asarray(
                    self._frame_masks.get(int(nearest_frame), {}).get(track_key),
                    dtype=np.uint8,
                )
                ys, xs = np.nonzero(arr)
                if len(xs) > 0 and len(ys) > 0:
                    x1, x2 = float(xs.min()), float(xs.max())
                    y1, y2 = float(ys.min()), float(ys.max())
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        nearest_box = [x1, y1, w, h]
            if nearest_box is None:
                prev_box = self._global_track_last_box.get(int(track_id))
                if prev_box is not None:
                    try:
                        nearest_box = [float(v) for v in np.asarray(prev_box, dtype=float).tolist()]
                    except Exception:
                        nearest_box = None
            if nearest_box is None:
                continue
            boxes_abs.append(nearest_box)
            box_track_ids.append(int(track_id))
        return boxes_abs, box_track_ids

    def _expected_track_ids_for_frame(
        self,
        frame_idx: int,
        *,
        max_gap: Optional[int] = None,
    ) -> set[int]:
        """Return recently active tracks that should still be considered live."""
        if max_gap is None:
            max_gap = max(3, min(int(self.sliding_window_size or 5), 10))
        expected: set[int] = set()
        start_frame = max(0, int(frame_idx) - int(max_gap))
        for prev_frame in range(start_frame, int(frame_idx)):
            expected.update(int(v) for v in self._frame_track_ids.get(int(prev_frame), set()))
        return expected

    def _missing_track_ids_for_frame(self, frame_idx: int) -> List[int]:
        current = {int(v) for v in self._frame_track_ids.get(int(frame_idx), set())}
        expected = self._expected_track_ids_for_frame(int(frame_idx))
        missing = sorted(expected - current)
        return missing

    @staticmethod
    def _normalize_window_schedule(
        *,
        window_size: int,
        stride: Optional[int],
    ) -> Tuple[int, int]:
        window_size = max(1, int(window_size or 1))
        try:
            stride_val = int(stride) if stride is not None else max(1, window_size - 1)
        except Exception:
            stride_val = max(1, window_size - 1)
        if stride_val <= 0:
            stride_val = max(1, window_size - 1)
        if window_size > 1 and stride_val >= window_size:
            stride_val = window_size - 1
        return window_size, stride_val

    def _resolve_window_schedule(
        self,
        *,
        resolved_device: torch.device,
        total_frames: Optional[int],
    ) -> Tuple[int, int]:
        """
        Choose a device-aware window schedule for long-video text prompting.

        Explicit user settings win. Otherwise prefer larger windows on CUDA and
        moderate overlap on long CPU runs to reduce session churn.
        """
        user_window_size = self.sliding_window_size
        user_stride = self.sliding_window_stride
        if user_window_size is not None or user_stride is not None:
            window_size, stride = self._normalize_window_schedule(
                window_size=user_window_size or 5,
                stride=user_stride,
            )
            return window_size, stride

        total = max(0, int(total_frames or 0))
        if resolved_device.type == "cuda":
            window_size = 48 if total >= 200 else 24
            overlap = max(4, window_size // 6)
        elif resolved_device.type == "cpu":
            window_size = 32 if total >= 2000 else 24 if total >= 400 else 12
            overlap = max(2, window_size // 5)
        else:
            window_size = 8
            overlap = 1

        if total > 0:
            window_size = min(window_size, total)
        stride = max(1, window_size - overlap)
        return self._normalize_window_schedule(window_size=window_size, stride=stride)

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

    def remove_object(
        self,
        obj_id: int,
        *,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Remove an object from an active session and persist resulting outputs.
        """
        if not self._predictor or not self._session_id:
            raise RuntimeError("SAM3 session has not been started.")
        if frame_idx is None:
            frame_idx = 0
        result = self._predictor.remove_object(
            session_id=self._session_id,
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
        )
        outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
        self._handle_frame_outputs(
            frame_idx=int(frame_idx),
            outputs=outputs or {},
            total_frames=self.total_frames_estimate(),
            yielded_frames=max(len(self._frames_processed), 1),
            apply_score_threshold=False,
            merge_existing=False,
        )
        return result

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
        apply_score_threshold: bool = True,
        merge_existing: bool = False,
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
            if (
                apply_score_threshold
                and score_thresh is not None
                and sam3_score is not None
            ):
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

        current_track_ids = {int(k) for k in mask_dict.keys()}
        if merge_existing:
            existing_masks = self._frame_masks.get(int(frame_idx)) or {}
            if existing_masks:
                merged_masks = {
                    str(k): np.asarray(v, dtype=np.uint8) for k, v in existing_masks.items()
                }
                merged_masks.update(mask_dict)
                mask_dict = merged_masks
                current_track_ids = {int(k) for k in mask_dict.keys()}

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
            persist_empty_frame=True,
        )
        # Track frames for possible later per-frame reacquisition.
        self._frames_processed.add(int(frame_idx))
        self._frame_track_ids[int(frame_idx)] = set(current_track_ids)
        if mask_dict:
            self._frames_with_masks.add(int(frame_idx))
            # Persist masks for this frame so they can be used as visual prompts
            # when re-acquiring lost tracks on later frames.
            self._frame_masks[int(frame_idx)] = {
                k: np.asarray(v, dtype=np.uint8) for k, v in mask_dict.items()
            }
            for track_id in current_track_ids:
                self._track_last_seen_frame[int(track_id)] = int(frame_idx)

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

    def _ensure_prediction_json_coverage(
        self,
        *,
        expected_frames: Iterable[int],
    ) -> Tuple[int, int]:
        """Ensure expected frames have valid per-frame JSON prediction files.

        Repairs missing/corrupt files by materializing empty prediction JSON.
        """
        if self.frame_shape is None:
            self.frame_shape = self.get_frame_shape()

        repaired = 0
        invalid = 0
        for frame_idx in sorted({int(f) for f in expected_frames if f is not None}):
            if frame_idx < 0:
                continue
            frame_path = Path(self.video_dir) / f"{int(frame_idx):09d}.json"
            needs_repair = False
            if not frame_path.exists():
                needs_repair = True
            else:
                try:
                    json.loads(frame_path.read_text(encoding="utf-8"))
                except Exception:
                    needs_repair = True
                    invalid += 1
            if not needs_repair:
                continue
            self._save_annotations(
                str(frame_path),
                {},
                self.frame_shape,
                frame_idx=int(frame_idx),
                persist_empty_frame=True,
            )
            repaired += 1
        return int(repaired), int(invalid)

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
                self._check_stop_requested()
                frame_idx = result["frame_index"]
                outputs = result.get("outputs", {}) or {}
                yielded_frames += 1
                masks_in_frame, _ = self._handle_frame_outputs(
                    frame_idx=frame_idx,
                    outputs=outputs,
                    total_frames=total_frames,
                    yielded_frames=yielded_frames,
                    apply_score_threshold=False,
                )
                total_masks += masks_in_frame

            if yielded_frames == 0:
                raise RuntimeError("SAM3 propagate_in_video yielded no frames")
            return yielded_frames, total_masks

        return _propagate()

    @staticmethod
    def _build_window_telemetry_entry(
        *,
        window_index: int,
        window_start_idx: int,
        window_end_idx: int,
        local_mask_counts: Dict[int, int],
        boundary_empty_skips: int = 0,
        latency_ms: float = 0.0,
        reacquired_frames: int = 0,
    ) -> Dict[str, object]:
        covered_frames = sorted(local_mask_counts.keys())
        frames_in_window = len(covered_frames)
        zero_mask_frames = sum(
            1 for f in covered_frames if int(local_mask_counts.get(int(f), 0)) <= 0
        )
        nonzero_frames = max(0, frames_in_window - zero_mask_frames)
        dropped_rate = (
            float(zero_mask_frames) / float(frames_in_window)
            if frames_in_window > 0
            else 0.0
        )
        return {
            "window_index": int(window_index),
            "start": int(window_start_idx),
            "end": int(window_end_idx),
            "frames": int(frames_in_window),
            "nonzero_frames": int(nonzero_frames),
            "zero_mask_frames": int(zero_mask_frames),
            "dropped_mask_rate": float(dropped_rate),
            "boundary_empty_skips": int(boundary_empty_skips),
            "latency_ms": float(latency_ms),
            "reacquired_frames": int(reacquired_frames),
        }

    def _propagate_text_prompt_windowed(
        self,
        *,
        text_prompt: str,
        target_device: Optional[torch.device | str],
    ) -> Tuple[int, int]:
        """
        SAM3.1 windowed inference for long videos:
        - small frame windows to bound memory
        - carry visual boxes from previous masks to stabilize continuity
        - remap local object ids to stable global ids across windows
        """
        resolved_device = self._resolve_runtime_device(target_device)
        self._reset_global_tracks()
        total_frames = self.total_frames_estimate()
        window_size, stride = self._resolve_window_schedule(
            resolved_device=resolved_device,
            total_frames=total_frames,
        )
        logger.info(
            "SAM3.1 windowed mode: resolved schedule window_size=%s stride=%s device=%s total_frames=%s.",
            window_size,
            stride,
            resolved_device,
            total_frames if total_frames else "unknown",
        )
        frame_to_masks: Dict[int, int] = {}
        window_frame_to_index: Dict[int, int] = {}
        window_telemetry: List[Dict[str, object]] = []

        if self._predictor is None or self._predictor_device != resolved_device:
            self._predictor = self._initialize_predictor(resolved_device)
            self._predictor_device = resolved_device

        with tempfile.TemporaryDirectory(prefix="annolid_sam3p1_windows_") as tmp_root:
            window_dir = Path(tmp_root) / "frames"
            window_dir.mkdir(parents=True, exist_ok=True)
            previous_window_frame_count = 0
            previous_window_start_idx: Optional[int] = None

            for window_idx, (start_idx, end_idx, frames) in enumerate(
                self._iter_video_windows(
                    window_size=window_size,
                    stride=stride,
                )
            ):
                self._check_stop_requested()
                window_t0 = time.perf_counter()
                local_mask_counts: Dict[int, int] = {}
                boundary_empty_skips = 0
                window_start_idx = int(start_idx)
                window_end_idx = int(end_idx)
                refresh_local_idx = compute_mid_window_refresh_index(
                    len(frames),
                    "forward",
                )
                # When windows overlap and slide forward by stride, reuse temp
                # files by shifting existing files and writing only the new tail.
                shift = 0
                if (
                    previous_window_start_idx is not None
                    and window_start_idx > int(previous_window_start_idx)
                    and len(frames) == int(previous_window_frame_count)
                ):
                    shift = min(
                        int(window_start_idx - int(previous_window_start_idx)),
                        max(0, len(frames) - 1),
                    )
                previous_window_frame_count = self._write_window_frames(
                    window_dir,
                    frames,
                    previous_count=previous_window_frame_count,
                    shift=shift,
                )
                previous_window_start_idx = window_start_idx

                session_resp = self._predictor.start_session(
                    resource_path=str(window_dir),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                )
                session_id = str(session_resp["session_id"])
                try:
                    propagation_direction = "forward"

                    def seed_first_frame() -> None:
                        prompt_result = self._execute_prompt_transaction(
                            session_id=session_id,
                            frame_idx=0,
                            text=text_prompt,
                            boxes=None,
                            box_labels=None,
                            points=None,
                            point_labels=None,
                            obj_id=None,
                        )
                        prompt_outputs = (
                            prompt_result.get("outputs", {})
                            if isinstance(prompt_result, dict)
                            else {}
                        ) or {}
                        prompt_outputs = self._map_outputs_to_global_ids_at_frame(
                            prompt_outputs,
                            frame_idx=int(start_idx),
                        )
                        prompt_global_frame = int(start_idx)
                        prompt_masks_in_frame, _ = self._handle_frame_outputs(
                            frame_idx=prompt_global_frame,
                            outputs=prompt_outputs,
                            total_frames=total_frames,
                            yielded_frames=len(frame_to_masks) + 1,
                            apply_score_threshold=False,
                        )
                        frame_to_masks[prompt_global_frame] = max(
                            int(frame_to_masks.get(prompt_global_frame, 0)),
                            int(prompt_masks_in_frame),
                        )
                        window_frame_to_index[int(prompt_global_frame)] = int(window_idx)
                        local_mask_counts[int(prompt_global_frame)] = max(
                            int(local_mask_counts.get(prompt_global_frame, 0)),
                            int(prompt_masks_in_frame),
                        )

                    def propagate_segment(
                        segment_start_local_idx: int,
                        segment_len: int,
                    ) -> Tuple[int, int]:
                        nonlocal boundary_empty_skips
                        frames_processed = 0
                        masks_written = 0
                        for result in self._predictor.propagate_in_video(
                            session_id=session_id,
                            propagation_direction=propagation_direction,
                            start_frame_idx=int(segment_start_local_idx),
                            max_frame_num_to_track=int(segment_len),
                        ):
                            self._check_stop_requested()
                            local_frame = int(result.get("frame_index", 0))
                            global_frame = start_idx + local_frame
                            outputs = result.get("outputs", {}) or {}
                            if (
                                int(frame_to_masks.get(global_frame, 0)) > 0
                                and self._output_candidate_mask_count(outputs) == 0
                            ):
                                boundary_empty_skips += 1
                                continue
                            outputs = self._map_outputs_to_global_ids_at_frame(
                                outputs,
                                frame_idx=global_frame,
                            )
                            masks_in_frame, _ = self._handle_frame_outputs(
                                frame_idx=global_frame,
                                outputs=outputs,
                                total_frames=total_frames,
                                yielded_frames=len(frame_to_masks) + 1,
                                apply_score_threshold=False,
                            )
                            frames_processed += 1
                            masks_written += int(masks_in_frame)
                            frame_to_masks[global_frame] = max(
                                int(frame_to_masks.get(global_frame, 0)),
                                int(masks_in_frame),
                            )
                            window_frame_to_index[int(global_frame)] = int(window_idx)
                            local_mask_counts[int(global_frame)] = max(
                                int(local_mask_counts.get(global_frame, 0)),
                                int(masks_in_frame),
                            )
                        return int(frames_processed), int(masks_written)

                    def refresh_mid_frame(refresh_local_idx: int) -> Tuple[int, int]:
                        refresh_global_frame = int(start_idx + int(refresh_local_idx))
                        refresh_result = self._execute_prompt_transaction(
                            session_id=session_id,
                            frame_idx=int(refresh_local_idx),
                            text=text_prompt,
                            boxes=None,
                            box_labels=None,
                            points=None,
                            point_labels=None,
                            obj_id=None,
                        )
                        refresh_outputs = (
                            refresh_result.get("outputs", {})
                            if isinstance(refresh_result, dict)
                            else {}
                        ) or {}
                        refresh_outputs = self._map_outputs_to_global_ids_at_frame(
                            refresh_outputs,
                            frame_idx=refresh_global_frame,
                        )
                        refresh_masks_in_frame, _ = self._handle_frame_outputs(
                            frame_idx=refresh_global_frame,
                            outputs=refresh_outputs,
                            total_frames=total_frames,
                            yielded_frames=len(frame_to_masks) + 1,
                            apply_score_threshold=False,
                        )
                        frame_to_masks[refresh_global_frame] = max(
                            int(frame_to_masks.get(refresh_global_frame, 0)),
                            int(refresh_masks_in_frame),
                        )
                        window_frame_to_index[int(refresh_global_frame)] = int(window_idx)
                        local_mask_counts[int(refresh_global_frame)] = max(
                            int(local_mask_counts.get(refresh_global_frame, 0)),
                            int(refresh_masks_in_frame),
                        )
                        return 1, int(refresh_masks_in_frame)

                    _frames_processed, _masks_written, _ = run_mid_window_refresh(
                        len(frames),
                        propagation_direction,
                        seed_first_frame=seed_first_frame,
                        propagate_segment=propagate_segment,
                        refresh_mid_frame=refresh_mid_frame,
                    )
                finally:
                    try:
                        self._predictor.close_session(session_id)
                    except Exception:
                        pass
                latency_ms = float((time.perf_counter() - window_t0) * 1000.0)
                telemetry = self._build_window_telemetry_entry(
                    window_index=int(window_idx),
                    window_start_idx=int(window_start_idx),
                    window_end_idx=int(window_end_idx),
                    local_mask_counts=local_mask_counts,
                    boundary_empty_skips=int(boundary_empty_skips),
                    latency_ms=float(latency_ms),
                )
                window_telemetry.append(telemetry)
                logger.info(
                    "SAM3.1 window telemetry #%d [%d,%d): frames=%d nonzero=%d zero=%d "
                    "drop_rate=%.3f boundary_skips=%d latency_ms=%.1f",
                    int(window_idx),
                    int(window_start_idx),
                    int(window_end_idx),
                    int(telemetry["frames"]),
                    int(telemetry["nonzero_frames"]),
                    int(telemetry["zero_mask_frames"]),
                    float(telemetry["dropped_mask_rate"]),
                    int(boundary_empty_skips),
                    float(latency_ms),
                )

        if not frame_to_masks:
            raise RuntimeError("SAM3.1 windowed propagation yielded no frames")
        if total_frames and self.text_prompt:
            missing_frames = sorted(
                set(range(int(total_frames))) - set(self._frames_with_masks)
            )
            if missing_frames:
                logger.info(
                    "SAM3.1: %d frame(s) with no masks after windowed propagation; "
                    "running per-frame visual+text reacquisition.",
                    len(missing_frames),
                )
                before_masks = {
                    int(f): int(frame_to_masks.get(int(f), 0)) for f in missing_frames
                }
                self._reacquire_frames_with_visual_and_text(
                    missing_frames, target_device
                )
                reacquired = 0
                for frame_idx in missing_frames:
                    after_mask_count = int(
                        len(self._frame_masks.get(int(frame_idx), {}) or {})
                    )
                    if after_mask_count > int(before_masks.get(int(frame_idx), 0)):
                        reacquired += 1
                        owning_window = window_frame_to_index.get(int(frame_idx))
                        if owning_window is not None and 0 <= owning_window < len(window_telemetry):
                            window_telemetry[int(owning_window)]["reacquired_frames"] = int(
                                window_telemetry[int(owning_window)].get("reacquired_frames", 0)
                            ) + 1
                        frame_to_masks[int(frame_idx)] = max(
                            int(frame_to_masks.get(int(frame_idx), 0)),
                            int(after_mask_count),
                        )
                logger.info(
                    "SAM3.1 windowed reacquisition: recovered %d/%d missing frame(s).",
                    int(reacquired),
                    int(len(missing_frames)),
                )
                if window_telemetry:
                    for t in window_telemetry:
                        logger.info(
                            "SAM3.1 window telemetry post-reacquire #%d: reacquired_frames=%d",
                            int(t.get("window_index", -1)),
                            int(t.get("reacquired_frames", 0)),
                        )
        expected_frames: Iterable[int]
        if total_frames and total_frames > 0:
            expected_frames = range(int(total_frames))
        else:
            expected_frames = frame_to_masks.keys()
        repaired, invalid = self._ensure_prediction_json_coverage(
            expected_frames=expected_frames
        )
        if repaired > 0:
            logger.info(
                "SAM3.1: repaired %d frame JSON file(s)%s during coverage finalization.",
                int(repaired),
                f" (invalid={int(invalid)})" if invalid > 0 else "",
            )
        total_masks = int(sum(frame_to_masks.values()))
        return len(frame_to_masks), total_masks

    @staticmethod
    def _shift_annotations_to_window(
        annotations: Iterable[dict],
        start_idx: int,
        end_idx: int,
    ) -> Dict[int, List[dict]]:
        """
        Group annotations by local frame index within [start_idx, end_idx).
        Returned annotations have ann_frame_idx rewritten to local window coords.
        """
        grouped: Dict[int, List[dict]] = {}
        for ann in annotations:
            try:
                frame_idx = int(ann.get("ann_frame_idx", 0))
            except (TypeError, ValueError):
                continue
            if frame_idx < start_idx or frame_idx >= end_idx:
                continue
            local_idx = frame_idx - start_idx
            shifted = dict(ann)
            shifted["ann_frame_idx"] = local_idx
            grouped.setdefault(local_idx, []).append(shifted)
        return grouped

    @staticmethod
    def _build_window_seed_segments(
        seed_frame_indices: Iterable[int],
        window_length: int,
        *,
        has_text_prompt: bool,
    ) -> List[Tuple[int, int]]:
        """
        Normalize seeded frames into ordered local segments for one window.

        This mirrors CUTIE's segment-oriented handling of multiple seed frames:
        each seed starts a contiguous pass until the next seed, and text-only
        fallback anchors the window at frame 0 when no manual labels exist.
        """
        normalized = sorted(
            {
                int(idx)
                for idx in seed_frame_indices or []
                if idx is not None and 0 <= int(idx) < int(window_length)
            }
        )
        if not normalized and has_text_prompt:
            normalized = [0]
        elif has_text_prompt and 0 not in normalized:
            normalized = [0, *normalized]
            normalized = sorted(set(normalized))

        segments: List[Tuple[int, int]] = []
        for idx, start_local in enumerate(normalized):
            next_local = (
                normalized[idx + 1] if idx + 1 < len(normalized) else int(window_length)
            )
            if int(next_local) <= int(start_local):
                continue
            segments.append((int(start_local), int(next_local)))
        return segments

    def _propagate_annotations_windowed(
        self,
        annotations: Iterable[dict],
        *,
        target_device: Optional[torch.device | str],
        propagation_direction: Optional[str],
        max_frame_num_to_track: Optional[int],
    ) -> Tuple[int, int]:
        """
        Windowed inference for long videos with labeled seed frames.

        Each window is decoded independently, prompts are applied on the
        labeled frame(s) inside that window, and results are flushed before
        moving to the next window. This avoids loading the full video tensor.
        """
        resolved_device = self._resolve_runtime_device(target_device)
        self._reset_global_tracks()
        total_frames = self.total_frames_estimate()
        window_size, stride = self._resolve_window_schedule(
            resolved_device=resolved_device,
            total_frames=total_frames,
        )
        logger.info(
            "SAM3.1 annotated windowed mode: window_size=%s stride=%s device=%s total_frames=%s.",
            window_size,
            stride,
            resolved_device,
            total_frames if total_frames else "unknown",
        )
        frame_to_masks: Dict[int, int] = {}
        window_frame_to_index: Dict[int, int] = {}
        window_telemetry: List[Dict[str, object]] = []
        annotations_by_window = list(annotations)

        if self._predictor is None or self._predictor_device != resolved_device:
            self._predictor = self._initialize_predictor(resolved_device)
            self._predictor_device = resolved_device

        with tempfile.TemporaryDirectory(prefix="annolid_sam3p1_windows_") as tmp_root:
            window_dir = Path(tmp_root) / "frames"
            window_dir.mkdir(parents=True, exist_ok=True)
            previous_window_frame_count = 0
            previous_window_start_idx: Optional[int] = None

            for window_idx, (start_idx, end_idx, frames) in enumerate(
                self._iter_video_windows(
                    window_size=window_size,
                    stride=stride,
                )
            ):
                self._check_stop_requested()
                if not frames:
                    continue
                window_t0 = time.perf_counter()
                window_start_idx = int(start_idx)
                window_end_idx = int(end_idx)
                local_ann_groups = self._shift_annotations_to_window(
                    annotations_by_window,
                    window_start_idx,
                    window_end_idx,
                )
                local_prompt_frames = sorted(local_ann_groups.keys())
                local_mask_counts: Dict[int, int] = {}
                boundary_empty_skips = 0

                shift = 0
                if (
                    previous_window_start_idx is not None
                    and window_start_idx > int(previous_window_start_idx)
                    and len(frames) == int(previous_window_frame_count)
                ):
                    shift = min(
                        int(window_start_idx - int(previous_window_start_idx)),
                        max(0, len(frames) - 1),
                    )
                previous_window_frame_count = self._write_window_frames(
                    window_dir,
                    frames,
                    previous_count=previous_window_frame_count,
                    shift=shift,
                )
                previous_window_start_idx = window_start_idx

                session_resp = self._predictor.start_session(
                    resource_path=str(window_dir),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                )
                session_id = str(session_resp["session_id"])
                try:
                    propagation_direction_local = (propagation_direction or self.propagation_direction or "forward").lower()
                    if propagation_direction_local not in {"forward", "both"}:
                        propagation_direction_local = "forward"

                    if not local_prompt_frames and self.text_prompt:
                        local_prompt_frames = [0]

                    def _seed_frame(local_frame_idx: int) -> bool:
                        abs_frame_idx = window_start_idx + int(local_frame_idx)
                        frame_annotations = local_ann_groups.get(int(local_frame_idx), [])
                        seed_mask_count = 0
                        if frame_annotations:
                            (
                                prompt_frame_idx,
                                boxes,
                                labels,
                                mask_inputs,
                                mask_labels,
                                points,
                                point_labels,
                                obj_ids,
                                point_obj_ids,
                            ) = self._prepare_prompts(frame_annotations, self.text_prompt)
                            if prompt_frame_idx is not None:
                                label_hints = self._label_hints_from_ids(obj_ids, self.id_to_labels)
                                seed_mask_count = self._apply_seed_prompts(
                                    frame_idx=int(prompt_frame_idx),
                                    session_id=session_id,
                                    boxes=boxes,
                                    labels=labels,
                                    mask_inputs=mask_inputs,
                                    mask_labels=mask_labels,
                                    points=points,
                                    point_labels=point_labels,
                                    point_obj_ids=point_obj_ids,
                                    label_hints=label_hints,
                                )
                                if seed_mask_count <= 0:
                                    logger.info(
                                        "SAM3.1 annotated window #%d frame=%d produced no seed masks; skipping propagation from this prompt frame.",
                                        int(window_idx),
                                        int(abs_frame_idx),
                                    )
                                    return False
                                return True
                            if not self.text_prompt:
                                return False
                            logger.info(
                                "SAM3.1 annotated window #%d frame=%d had no usable boxes, masks, or points; falling back to text prompt.",
                                int(window_idx),
                                int(abs_frame_idx),
                            )
                            seed_mask_count = self._apply_seed_prompts(
                                frame_idx=int(local_frame_idx),
                                session_id=session_id,
                                boxes=[],
                                labels=[],
                                mask_inputs=[],
                                mask_labels=[],
                                points=[],
                                point_labels=[],
                                point_obj_ids=[],
                                label_hints=[],
                            )
                        elif self.text_prompt:
                            logger.info(
                                "SAM3.1 annotated window #%d frame=%d had no local annotations; falling back to text prompt.",
                                int(window_idx),
                                int(abs_frame_idx),
                            )
                            seed_mask_count = self._apply_seed_prompts(
                                frame_idx=int(local_frame_idx),
                                session_id=session_id,
                                boxes=[],
                                labels=[],
                                mask_inputs=[],
                                mask_labels=[],
                                points=[],
                                point_labels=[],
                                point_obj_ids=[],
                                label_hints=[],
                            )
                        else:
                            return False
                        if seed_mask_count <= 0:
                            logger.info(
                                "SAM3.1 annotated window #%d frame=%d text prompt produced no seed masks; skipping propagation from this prompt frame.",
                                int(window_idx),
                                int(abs_frame_idx),
                                )
                            return False
                        return True

                    def _propagate_segment(
                        segment_start_local_idx: int,
                        segment_end_local_exclusive: int,
                    ) -> Tuple[int, int]:
                        if segment_end_local_exclusive <= segment_start_local_idx:
                            return 0, 0
                        frames_processed = 0
                        masks_written = 0
                        max_track = max(
                            1,
                            int(segment_end_local_exclusive) - int(segment_start_local_idx) - 1,
                        )
                        for result in self._predictor.propagate_in_video(
                            session_id=session_id,
                            propagation_direction=propagation_direction_local,
                            start_frame_idx=int(segment_start_local_idx),
                            max_frame_num_to_track=max_track,
                        ):
                            self._check_stop_requested()
                            local_frame = int(result.get("frame_index", 0))
                            global_frame = window_start_idx + local_frame
                            outputs = result.get("outputs", {}) or {}
                            outputs = self._map_outputs_to_global_ids_at_frame(
                                outputs,
                                frame_idx=global_frame,
                            )
                            masks_in_frame, _ = self._handle_frame_outputs(
                                frame_idx=global_frame,
                                outputs=outputs,
                                total_frames=total_frames,
                                yielded_frames=len(frame_to_masks) + 1,
                                apply_score_threshold=False,
                            )
                            frames_processed += 1
                            masks_written += int(masks_in_frame)
                            frame_to_masks[global_frame] = max(
                                int(frame_to_masks.get(global_frame, 0)),
                                int(masks_in_frame),
                            )
                            window_frame_to_index[int(global_frame)] = int(window_idx)
                            local_mask_counts[int(global_frame)] = max(
                                int(local_mask_counts.get(global_frame, 0)),
                                int(masks_in_frame),
                            )
                        return int(frames_processed), int(masks_written)

                    seed_segments = self._build_window_seed_segments(
                        local_prompt_frames,
                        len(frames),
                        has_text_prompt=bool(self.text_prompt),
                    )
                    for start_local, next_local in seed_segments:
                        if int(start_local) not in local_ann_groups and not (
                            int(start_local) == 0 and self.text_prompt
                        ):
                            continue
                        if not _seed_frame(int(start_local)):
                            continue
                        _propagate_segment(int(start_local), int(next_local))
                finally:
                    try:
                        self._predictor.close_session(session_id)
                    except Exception:
                        pass

                latency_ms = float((time.perf_counter() - window_t0) * 1000.0)
                telemetry = self._build_window_telemetry_entry(
                    window_index=int(window_idx),
                    window_start_idx=int(window_start_idx),
                    window_end_idx=int(window_end_idx),
                    local_mask_counts=local_mask_counts,
                    boundary_empty_skips=int(boundary_empty_skips),
                    latency_ms=float(latency_ms),
                )
                window_telemetry.append(telemetry)
                logger.info(
                    "SAM3.1 annotated telemetry #%d [%d,%d): frames=%d nonzero=%d zero=%d drop_rate=%.3f latency_ms=%.1f",
                    int(window_idx),
                    int(window_start_idx),
                    int(window_end_idx),
                    int(telemetry["frames"]),
                    int(telemetry["nonzero_frames"]),
                    int(telemetry["zero_mask_frames"]),
                    float(telemetry["dropped_mask_rate"]),
                    float(latency_ms),
                )

        if not frame_to_masks:
            raise RuntimeError("SAM3.1 annotated windowed propagation yielded no frames")
        expected_frames: Iterable[int]
        if total_frames and total_frames > 0:
            expected_frames = range(int(total_frames))
        else:
            expected_frames = frame_to_masks.keys()
        repaired, invalid = self._ensure_prediction_json_coverage(
            expected_frames=expected_frames
        )
        if repaired > 0:
            logger.info(
                "SAM3.1: repaired %d frame JSON file(s)%s during coverage finalization.",
                int(repaired),
                f" (invalid={int(invalid)})" if invalid > 0 else "",
            )
        total_masks = int(sum(frame_to_masks.values()))
        return len(frame_to_masks), total_masks

    def _prepare_prompts(
        self, annotations: Iterable[dict], text_prompt: Optional[str]
    ) -> Tuple[
        Optional[int],
        List[List[float]],
        List[int],
        List[np.ndarray],
        List[int],
        List[int],
        List[int],
        List[int],
        List[int],
    ]:
        """
        Build SAM3 prompts from cached annotations.

        Returns:
            (frame_idx, boxes, box_labels, mask_inputs, mask_labels, points, point_labels, obj_ids, point_obj_ids)
        """
        if self.frame_shape is None:
            self.frame_shape = self.get_frame_shape()

        if not annotations:
            if text_prompt:
                logger.info(
                    "SAM3 using text-only prompt; no per-frame annotations found under %s",
                    self.video_dir,
                )
                prompt_frame_idx = self._first_frame_index()
                return prompt_frame_idx, [], [], [], [], [], [], [], []
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
            box_labels: List[int] = []
            mask_inputs: List[np.ndarray] = []
            mask_labels: List[int] = []
            points: List[List[float]] = []
            point_labels: List[int] = []
            obj_ids: List[int] = []
            point_obj_ids: List[int] = []
            for ann in annotations_by_frame[frame_idx]:
                label_val = int(ann["labels"][0]) if ann.get("labels") else 1
                if ann["type"] == "box":
                    x1, y1, x2, y2 = ann["box"]
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                elif ann["type"] == "mask":
                    mask = np.asarray(ann["mask"], dtype=np.uint8)
                    if mask.ndim != 2:
                        continue
                    if mask.shape[:2] != (height, width):
                        try:
                            mask = cv2.resize(
                                mask,
                                (width, height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        except Exception:
                            continue
                    if not np.any(mask):
                        continue
                    mask_inputs.append(mask.astype(np.uint8))
                    mask_labels.append(1)
                    try:
                        obj_ids.append(int(ann.get("obj_id", label_val)))
                    except Exception:
                        obj_ids.append(int(label_val))
                    continue
                elif ann["type"] in {"polygon", "polyline"}:
                    poly_pts = ann.get("polygon") or ann.get("polyline") or []
                    if not poly_pts:
                        continue
                    try:
                        mask = self._shape_points_to_mask(
                            poly_pts,
                            self.frame_shape,
                            shape_type="polygon",
                        )
                        if mask is None or not np.any(mask):
                            continue
                    except Exception:
                        continue
                    mask_inputs.append(mask)
                    mask_labels.append(1)
                    try:
                        obj_ids.append(int(ann.get("obj_id", label_val)))
                    except Exception:
                        obj_ids.append(int(label_val))
                    continue
                elif ann["type"] in {"points", "point"}:
                    ann_points = ann.get("points") or []
                    if not ann_points and ann.get("point"):
                        ann_points = [ann.get("point")]
                    if not ann_points:
                        continue
                    ann_point_labels = ann.get("labels") or []
                    added_any = False
                    for idx, pt in enumerate(ann_points):
                        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                            continue
                        try:
                            x = float(pt[0])
                            y = float(pt[1])
                        except Exception:
                            continue
                        if x < 0.0 or y < 0.0:
                            continue
                        points.append([x / width, y / height])
                        raw_label = ann_point_labels[idx] if idx < len(ann_point_labels) else 1
                        try:
                            point_labels.append(1 if int(raw_label) > 0 else 0)
                        except Exception:
                            point_labels.append(1)
                        try:
                            point_obj_ids.append(int(ann.get("obj_id", label_val)))
                        except Exception:
                            point_obj_ids.append(int(label_val))
                        added_any = True
                    if added_any:
                        try:
                            obj_ids.append(int(ann.get("obj_id", label_val)))
                        except Exception:
                            obj_ids.append(int(label_val))
                    continue
                else:
                    continue

                if w <= 0 or h <= 0:
                    continue

                boxes.append([x1 / width, y1 / height, w / width, h / height])
                box_labels.append(label_val)
                obj_ids.append(int(ann.get("obj_id", label_val)))

            if boxes or mask_inputs or points:
                return (
                    frame_idx,
                    boxes,
                    box_labels,
                    mask_inputs,
                    mask_labels,
                    points,
                    point_labels,
                    obj_ids,
                    point_obj_ids,
                )

        if text_prompt:
            logger.info(
                "SAM3 using text-only prompt; no usable per-frame annotations were found under %s",
                self.video_dir,
            )
            prompt_frame_idx = self._first_frame_index()
            return prompt_frame_idx, [], [], [], [], [], [], [], []

        return None, [], [], [], [], [], [], [], []

    def _apply_seed_prompts(
        self,
        *,
        frame_idx: int,
        session_id: Optional[str],
        boxes: List[List[float]],
        labels: List[int],
        mask_inputs: List[object],
        mask_labels: List[int],
        points: List[List[float]],
        point_labels: List[int],
        point_obj_ids: List[int],
        label_hints: List[str],
    ) -> int:
        """
        Apply initial prompts with SAM3.1-compatible sequencing:
        - optional text+box prompt first
        - point prompts grouped by stable obj_id
        """
        has_prior_record = False
        total_seed_masks = 0
        if self.text_prompt or boxes or mask_inputs:
            semantic_result = self.add_prompt(
                frame_idx=frame_idx,
                session_id=session_id,
                text=self.text_prompt,
                boxes=boxes or None,
                box_labels=labels or None,
                mask_inputs=mask_inputs or None,
                mask_labels=mask_labels or None,
                record_outputs=True,
                merge_existing_on_record=False,
                label_hints=label_hints,
            )
            total_seed_masks += self._prompt_result_mask_count(semantic_result)
            has_prior_record = True

        if points:
            grouped: Dict[int, Dict[str, List[float]]] = {}
            for idx, point in enumerate(points):
                obj_id = 1
                if idx < len(point_obj_ids):
                    try:
                        obj_id = max(1, int(point_obj_ids[idx]))
                    except Exception:
                        obj_id = 1
                plabel = point_labels[idx] if idx < len(point_labels) else 1
                bucket = grouped.setdefault(obj_id, {"points": [], "point_labels": []})
                bucket["points"].append(point)
                bucket["point_labels"].append(plabel)

            for local_idx, (obj_id, payload) in enumerate(sorted(grouped.items())):
                point_hint = self.id_to_labels.get(int(obj_id), str(obj_id))
                point_result = self.add_prompt(
                    frame_idx=frame_idx,
                    session_id=session_id,
                    text=None,
                    points=payload["points"],
                    point_labels=payload["point_labels"],
                    obj_id=int(obj_id),
                    record_outputs=True,
                    merge_existing_on_record=has_prior_record or local_idx > 0,
                    label_hints=[point_hint],
                )
                total_seed_masks += self._prompt_result_mask_count(point_result)
        return int(total_seed_masks)

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
            self._stop_requested = False
            total_frames = self.total_frames_estimate()
            if total_frames:
                removed = self.prune_out_of_range_prediction_frames(total_frames)
                if removed > 0:
                    logger.info(
                        "SAM3: pruned %d out-of-range prediction frame record(s) "
                        "above frame %d before run.",
                        int(removed),
                        int(total_frames) - 1,
                    )
            (
                prompt_frame_idx,
                boxes,
                labels,
                mask_inputs,
                mask_labels,
                points,
                point_labels,
                obj_ids,
                point_obj_ids,
            ) = self._prepare_prompts(
                annotations, self.text_prompt
            )
            if prompt_frame_idx is None:
                raise FileNotFoundError(
                    f"No usable prompts found under {self.video_dir}"
                )
            # Build label hints for semantic seeds (boxes + masks).
            label_hints = self._label_hints_from_ids(list(obj_ids), self.id_to_labels)

            # Reset tracking sets for this run.
            self._frames_processed.clear()
            self._frames_with_masks.clear()
            self._frame_masks.clear()
            self._frame_track_ids.clear()
            self._track_last_seen_frame.clear()

            resolved_device = self._resolve_runtime_device(target_device)
            window_size, _ = self._resolve_window_schedule(
                resolved_device=resolved_device,
                total_frames=total_frames,
            )
            use_windowed_annotations = bool(annotations) and bool(
                total_frames and total_frames > int(window_size)
            )
            if use_windowed_annotations:
                try:
                    return self._propagate_annotations_windowed(
                        annotations,
                        target_device=resolved_device,
                        propagation_direction=propagation_direction or self.propagation_direction,
                        max_frame_num_to_track=max_frame_num_to_track or self.max_frame_num_to_track,
                    )
                except RuntimeError as exc:
                    if _is_mps_oom(exc):
                        logger.warning(
                            "SAM3.1 annotated windowed mode hit MPS OOM; retrying on CPU: %s",
                            exc,
                        )
                        _clear_mps_cache()
                        return self._propagate_annotations_windowed(
                            annotations,
                            target_device="cpu",
                            propagation_direction=propagation_direction or self.propagation_direction,
                            max_frame_num_to_track=max_frame_num_to_track or self.max_frame_num_to_track,
                        )
                    raise

            if (
                self.use_sliding_window_for_text_prompt
                and self.text_prompt
                and not boxes
                and not mask_inputs
                and not points
            ):
                logger.info(
                    "SAM3.1: running windowed text propagation (window_size=%s, stride=%s).",
                    self.sliding_window_size,
                    self.sliding_window_stride or self.sliding_window_size,
                )
                try:
                    return self._propagate_text_prompt_windowed(
                        text_prompt=self.text_prompt,
                        target_device=target_device,
                    )
                except RuntimeError as exc:
                    if _is_mps_oom(exc):
                        logger.warning(
                            "SAM3.1 windowed mode hit MPS OOM; retrying on CPU: %s",
                            exc,
                        )
                        _clear_mps_cache()
                        return self._propagate_text_prompt_windowed(
                            text_prompt=self.text_prompt,
                            target_device="cpu",
                        )
                    raise

            with self._session_scope(target_device) as _:
                # Clear SAM3 action history so the first propagation performs a full
                # video pass instead of a partial/fetch-only update.
                self._reset_action_history_if_supported()

                self._apply_seed_prompts(
                    frame_idx=prompt_frame_idx,
                    session_id=self._session_id,
                    boxes=boxes,
                    labels=labels,
                    mask_inputs=mask_inputs,
                    mask_labels=mask_labels,
                    points=points,
                    point_labels=point_labels,
                    point_obj_ids=point_obj_ids,
                    label_hints=label_hints,
                )
                self._check_stop_requested()
                frames, masks = self.propagate(
                    start_frame_idx=prompt_frame_idx,
                    propagation_direction=propagation_direction or self.propagation_direction,
                    max_frame_num_to_track=max_frame_num_to_track or self.max_frame_num_to_track,
                )
                # Use text-guided recovery for frames that lost all or some recent
                # instances, and merge the recovered outputs into existing tracks.
                recovery_frames = sorted(
                    {
                        int(frame_idx)
                        for frame_idx in self._frames_processed
                        if (int(frame_idx) not in self._frames_with_masks)
                        or self._missing_track_ids_for_frame(int(frame_idx))
                    }
                )
                if recovery_frames and self.text_prompt:
                    logger.info(
                        "SAM3: %d frame(s) need text-guided recovery for missing/lost instances.",
                        len(recovery_frames),
                    )
                    self._reacquire_frames_with_visual_and_text(
                        recovery_frames, target_device
                    )
                expected_frames: Iterable[int]
                total_frames = self.total_frames_estimate()
                effective_limit = max_frame_num_to_track or self.max_frame_num_to_track
                if (
                    total_frames
                    and total_frames > 0
                    and (effective_limit is None or int(effective_limit) >= int(total_frames))
                ):
                    expected_frames = range(int(total_frames))
                else:
                    expected_frames = self._frames_processed
                repaired, invalid = self._ensure_prediction_json_coverage(
                    expected_frames=expected_frames
                )
                if repaired > 0:
                    logger.info(
                        "SAM3: repaired %d frame JSON file(s)%s during coverage finalization.",
                        int(repaired),
                        f" (invalid={int(invalid)})" if invalid > 0 else "",
                    )
            return frames, masks

        return _run()

    def _reacquire_frame_with_visual_and_text(
        self,
        frame_idx: int,
        target_device: Optional[torch.device | str] = None,
    ) -> None:
        """
        Run a lightweight SAM3 pass on a single frame using only the text
        prompt to recover tracking when the main tracker path produced none.
        """
        with torch.inference_mode():
            if not self.text_prompt:
                return

            missing_track_ids = self._missing_track_ids_for_frame(int(frame_idx))
            target_track_ids = missing_track_ids

            with self._session_scope(target_device, auto_close=True):
                self._reset_session_state()
                self._reset_action_history_if_supported()
                result = self.add_prompt(
                    frame_idx=frame_idx,
                    text=self.text_prompt,
                    record_outputs=False,
                    label_hints=self._label_hints_from_ids(target_track_ids, self.id_to_labels)
                    if target_track_ids
                    else None,
                )
                outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
                outputs = self._map_outputs_to_global_ids_at_frame(
                    outputs or {},
                    frame_idx=int(frame_idx),
                )
                self._handle_frame_outputs(
                    frame_idx=int(frame_idx),
                    outputs=outputs,
                    total_frames=self.total_frames_estimate(),
                    yielded_frames=len(self._frames_processed) + 1,
                    apply_score_threshold=False,
                    merge_existing=True,
                )

    def _reacquire_frames_with_visual_and_text(
        self,
        frame_indices: List[int],
        target_device: Optional[torch.device | str] = None,
    ) -> None:
        """
        Re-acquire multiple frames using a single temporary session to avoid
        reloading video data repeatedly, improving speed on CPU fallback.
        The recovery prompt is text-only; object identity comes from the
        cross-window matcher.
        """
        if not frame_indices or not self.text_prompt or not self._frame_masks:
            return

        with torch.inference_mode():
            with self._session_scope(target_device, auto_close=True):
                for frame_idx in frame_indices:
                    missing_track_ids = self._missing_track_ids_for_frame(int(frame_idx))
                    target_track_ids = missing_track_ids
                    self._reset_session_state()
                    self._reset_action_history_if_supported()
                    try:
                        result = self.add_prompt(
                            frame_idx=frame_idx,
                            text=self.text_prompt,
                            record_outputs=False,
                            label_hints=self._label_hints_from_ids(
                                target_track_ids, self.id_to_labels
                            )
                            if target_track_ids
                            else None,
                        )
                        outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
                        outputs = self._map_outputs_to_global_ids_at_frame(
                            outputs or {},
                            frame_idx=int(frame_idx),
                        )
                        self._handle_frame_outputs(
                            frame_idx=int(frame_idx),
                            outputs=outputs,
                            total_frames=self.total_frames_estimate(),
                            yielded_frames=len(self._frames_processed) + 1,
                            apply_score_threshold=False,
                            merge_existing=True,
                        )
                    except Exception as exc:
                        logger.warning(
                            "SAM3 per-frame reacquisition failed for frame %s: %s",
                            frame_idx,
                            exc,
                        )
