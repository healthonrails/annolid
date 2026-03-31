from __future__ import annotations

import importlib
import json
import tempfile
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch

from annolid.segmentation.SAM.sam_v2 import BaseSAMVideoProcessor
from annolid.utils.logger import logger
from .sam3.utils import set_default_device

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
        points: Optional[List[List[float]]],
        point_labels: Optional[List[int]],
        bounding_boxes: Optional[List[List[float]]],
        bounding_box_labels: Optional[List[int]],
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

    def _reset_action_history_if_supported(self):
        """
        Best-effort clearing of cached action history on the underlying predictor
        so the next propagation performs a full pass.
        """
        try:
            raw_predictor = self._predictor.raw if self._predictor is not None else None
            states = getattr(raw_predictor, "_ALL_INFERENCE_STATES", None)
            if states is None:
                states = getattr(raw_predictor, "_all_inference_states", None)
            state_entry = states.get(self._session_id) if isinstance(states, dict) else None
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
        normalized_boxes = boxes if boxes else None
        normalized_points = points if points else None
        # SAM3 geometric prompt embeddings are binary (foreground/background).
        # Normalize all external label ids (class ids, None, etc.) to {0,1}.
        safe_box_labels = self._sanitize_prompt_labels(
            box_labels,
            expected_len=len(normalized_boxes) if normalized_boxes else 0,
            default_value=1,
        )
        safe_point_labels = self._sanitize_prompt_labels(
            point_labels,
            expected_len=len(normalized_points) if normalized_points else 0,
            default_value=1,
        )
        result = self._predictor.add_prompt(
            session_id=self._session_id,
            frame_idx=frame_idx,
            text=text,
            points=normalized_points,
            point_labels=safe_point_labels,
            bounding_boxes=normalized_boxes,
            bounding_box_labels=safe_box_labels,
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
                apply_score_threshold=False,
            )
        return result

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

    def _reset_global_tracks(self) -> None:
        self._global_track_next_id = 1
        self._global_track_last_box.clear()

    def _assign_global_track_id(
        self,
        box_xywh: np.ndarray,
        used_ids: Optional[set[int]] = None,
        iou_threshold: float = 0.3,
    ) -> int:
        if used_ids is None:
            used_ids = set()

        best_gid: Optional[int] = None
        best_iou = 0.0
        for gid, prev_box in self._global_track_last_box.items():
            if gid in used_ids:
                continue
            iou = self._box_iou_xywh(prev_box, box_xywh)
            if iou > best_iou:
                best_iou = iou
                best_gid = gid

        if best_gid is not None and best_iou >= iou_threshold:
            self._global_track_last_box[best_gid] = box_xywh
            return best_gid

        gid = self._global_track_next_id
        self._global_track_next_id += 1
        self._global_track_last_box[gid] = box_xywh
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
    ) -> int:
        """
        Materialize a window using stable local filenames.

        Rewriting only the active positions and trimming stale tail files avoids
        full directory churn across heavily overlapping windows.
        """
        for local_idx, frame in enumerate(frames):
            out_path = window_dir / f"{local_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
        for stale_idx in range(len(frames), max(len(frames), int(previous_count))):
            stale_path = window_dir / f"{stale_idx:06d}.jpg"
            stale_path.unlink(missing_ok=True)
        return len(frames)

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
        mapped: List[int] = []
        for idx, _ in enumerate(obj_ids):
            gid = self._assign_global_track_id(
                box_xywh=np.asarray(boxes_arr[idx], dtype=float),
                used_ids=used,
            )
            used.add(gid)
            mapped.append(int(gid))
        outputs = dict(outputs)
        outputs["out_obj_ids"] = np.asarray(mapped, dtype=np.int64)
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

        if self._predictor is None or self._predictor_device != resolved_device:
            self._predictor = self._initialize_predictor(resolved_device)
            self._predictor_device = resolved_device

        with tempfile.TemporaryDirectory(prefix="annolid_sam3p1_windows_") as tmp_root:
            window_dir = Path(tmp_root) / "frames"
            window_dir.mkdir(parents=True, exist_ok=True)
            previous_window_frame_count = 0

            for start_idx, _, frames in self._iter_video_windows(
                window_size=window_size,
                stride=stride,
            ):
                self._check_stop_requested()
                previous_window_frame_count = self._write_window_frames(
                    window_dir,
                    frames,
                    previous_count=previous_window_frame_count,
                )

                session_resp = self._predictor.start_session(
                    resource_path=str(window_dir),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                )
                session_id = str(session_resp["session_id"])
                try:
                    # Use previous masks as a visual cue when available.
                    carry_boxes_abs = self._derive_boxes_from_neighbor_masks(
                        start_idx, max_boxes=min(4, self.max_num_objects)
                    )
                    if carry_boxes_abs:
                        h, w = frames[0].shape[:2]
                        carry_boxes = self._normalize_boxes(carry_boxes_abs, w, h)
                    else:
                        carry_boxes = None

                    prompt_result = self._predictor.add_prompt(
                        session_id=session_id,
                        frame_idx=0,
                        text=text_prompt,
                        points=None,
                        point_labels=None,
                        bounding_boxes=carry_boxes,
                        bounding_box_labels=[1] * len(carry_boxes)
                        if carry_boxes
                        else None,
                        obj_id=None,
                    )
                    prompt_outputs = (
                        prompt_result.get("outputs", {})
                        if isinstance(prompt_result, dict)
                        else {}
                    ) or {}
                    prompt_outputs = self._map_outputs_to_global_ids(prompt_outputs)
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
                    for result in self._predictor.propagate_in_video(
                        session_id=session_id,
                        propagation_direction="forward",
                        start_frame_idx=0,
                        max_frame_num_to_track=len(frames),
                    ):
                        self._check_stop_requested()
                        local_frame = int(result.get("frame_index", 0))
                        global_frame = start_idx + local_frame
                        outputs = result.get("outputs", {}) or {}
                        # Keep previously materialized non-empty frames stable:
                        # SAM3 can emit duplicate boundary frames with empty outputs.
                        if (
                            int(frame_to_masks.get(global_frame, 0)) > 0
                            and self._output_candidate_mask_count(outputs) == 0
                        ):
                            continue
                        outputs = self._map_outputs_to_global_ids(outputs)
                        masks_in_frame, _ = self._handle_frame_outputs(
                            frame_idx=global_frame,
                            outputs=outputs,
                            total_frames=total_frames,
                            yielded_frames=len(frame_to_masks) + 1,
                            apply_score_threshold=False,
                        )
                        frame_to_masks[global_frame] = max(
                            int(frame_to_masks.get(global_frame, 0)),
                            int(masks_in_frame),
                        )
                finally:
                    try:
                        self._predictor.close_session(session_id)
                    except Exception:
                        pass

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
                self._reacquire_frames_with_visual_and_text(
                    missing_frames, target_device
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

    def _prepare_prompts(
        self, annotations: Iterable[dict], text_prompt: Optional[str]
    ) -> Tuple[
        Optional[int],
        List[List[float]],
        List[int],
        List[List[float]],
        List[int],
        List[int],
    ]:
        """
        Build SAM3 prompts from cached annotations.

        Returns:
            (frame_idx, boxes, box_labels, points, point_labels, obj_ids)
        """
        if self.frame_shape is None:
            self.frame_shape = self.get_frame_shape()

        if not annotations and text_prompt:
            logger.info(
                "SAM3 using text-only prompt; no per-frame annotations found under %s",
                self.video_dir,
            )
            prompt_frame_idx = self._first_frame_index()
            return prompt_frame_idx, [], [], [], [], []

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
            box_labels: List[int] = []
            points: List[List[float]] = []
            point_labels: List[int] = []
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
                elif ann["type"] in {"polygon", "polyline"}:
                    poly_pts = ann.get("polygon") or ann.get("polyline") or []
                    if not poly_pts:
                        continue
                    try:
                        arr = np.asarray(poly_pts, dtype=float)
                        if arr.ndim != 2 or arr.shape[1] != 2:
                            continue
                        x1 = float(arr[:, 0].min())
                        x2 = float(arr[:, 0].max())
                        y1 = float(arr[:, 1].min())
                        y2 = float(arr[:, 1].max())
                    except Exception:
                        continue
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
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
                        added_any = True
                    if added_any:
                        obj_ids.append(int(ann.get("obj_id", label_val)))
                    continue
                else:
                    continue

                if w <= 0 or h <= 0:
                    continue

                boxes.append([x1 / width, y1 / height, w / width, h / height])
                box_labels.append(label_val)
                obj_ids.append(int(ann.get("obj_id", label_val)))

            if boxes or points:
                return frame_idx, boxes, box_labels, points, point_labels, obj_ids

        return None, [], [], [], [], []

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
            prompt_frame_idx, boxes, labels, points, point_labels, _ = self._prepare_prompts(
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
            self._frame_masks.clear()
            self._frame_track_ids.clear()
            self._track_last_seen_frame.clear()

            if (
                self.use_sliding_window_for_text_prompt
                and self.text_prompt
                and not boxes
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

                self.add_prompt(
                    frame_idx=prompt_frame_idx,
                    text=self.text_prompt,
                    boxes=boxes or None,
                    box_labels=labels or None,
                    points=points or None,
                    point_labels=point_labels or None,
                    record_outputs=True,
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
        Run a lightweight SAM3 pass on a single frame using the text prompt
        and the last available masks as visual prompts to recover tracking
        when the main tracker path produced none.
        """
        with torch.inference_mode():
            if not self.text_prompt:
                return

            missing_track_ids = self._missing_track_ids_for_frame(int(frame_idx))
            if missing_track_ids:
                boxes_abs, target_track_ids = self._derive_boxes_for_track_ids(
                    frame_idx,
                    missing_track_ids,
                    max_boxes=min(4, self.max_num_objects),
                )
            else:
                boxes_abs = self._derive_boxes_from_neighbor_masks(
                    frame_idx, max_boxes=min(4, self.max_num_objects)
                )
                target_track_ids = []
            if not boxes_abs:
                return

            with self._session_scope(target_device, auto_close=True):
                if self.frame_shape is None:
                    self.frame_shape = self.get_frame_shape()
                h, w = self.frame_shape[:2]
                boxes = self._normalize_boxes(boxes_abs, w, h)
                self._reset_session_state()
                self._reset_action_history_if_supported()
                result = self.add_prompt(
                    frame_idx=frame_idx,
                    text=self.text_prompt,
                    boxes=boxes,
                    box_labels=[1] * len(boxes),
                    record_outputs=False,
                    label_hints=self._label_hints_from_ids(target_track_ids, self.id_to_labels)
                    if target_track_ids
                    else None,
                )
                outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
                outputs = self._map_outputs_to_global_ids(outputs or {})
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
        """
        if not frame_indices or not self.text_prompt or not self._frame_masks:
            return

        with torch.inference_mode():
            with self._session_scope(target_device, auto_close=True):
                if self.frame_shape is None:
                    self.frame_shape = self.get_frame_shape()
                h, w = self.frame_shape[:2]

                for frame_idx in frame_indices:
                    missing_track_ids = self._missing_track_ids_for_frame(int(frame_idx))
                    if missing_track_ids:
                        boxes_abs, target_track_ids = self._derive_boxes_for_track_ids(
                            frame_idx,
                            missing_track_ids,
                            max_boxes=min(4, self.max_num_objects),
                        )
                    else:
                        boxes_abs = self._derive_boxes_from_neighbor_masks(
                            frame_idx, max_boxes=min(4, self.max_num_objects)
                        )
                        target_track_ids = []
                    if not boxes_abs:
                        continue
                    boxes = self._normalize_boxes(boxes_abs, w, h)
                    self._reset_session_state()
                    self._reset_action_history_if_supported()
                    try:
                        result = self.add_prompt(
                            frame_idx=frame_idx,
                            text=self.text_prompt,
                            boxes=boxes,
                            box_labels=[1] * len(boxes),
                            record_outputs=False,
                            label_hints=self._label_hints_from_ids(
                                target_track_ids, self.id_to_labels
                            )
                            if target_track_ids
                            else None,
                        )
                        outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
                        outputs = self._map_outputs_to_global_ids(outputs or {})
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
