from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from qtpy import QtWidgets

import yaml
from annolid.utils.logger import logger


class Sam3Manager:
    """Encapsulates SAM3 runtime state, prompts, and video processor setup."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        self.window = window
        self.sam3_session = None
        self._last_prompt_frame: Optional[int] = None
        self._initial_prompts: Optional[dict] = None

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

    def close_session(self) -> None:
        """Best-effort close of any active SAM3 session."""
        try:
            if self.sam3_session:
                self.sam3_session.close_session()
        except Exception as exc:  # pragma: no cover - shutdown best effort
            logger.warning("Error closing SAM3 session on exit: %s", exc)
        self.sam3_session = None

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
        self.max_frame_num_to_track = sam3_runtime.get(
            "max_frame_num_to_track")
        self.device_override = sam3_runtime.get("device")
        self.sliding_window_size = sam3_runtime.get("sliding_window_size")
        self.sliding_window_stride = sam3_runtime.get("sliding_window_stride")
        self.compile_model = sam3_runtime.get("compile_model")
        self.offload_video_to_cpu = sam3_runtime.get("offload_video_to_cpu")
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
        Extract SAM3-friendly prompts (boxes and points) from current canvas shapes.

        Returns a dict with:
          - frame_idx: current frame
          - boxes_abs: list of [x, y, w, h] in pixel space
          - box_labels: list of int labels (defaults to 1)
          - points_abs: list of [x, y]
          - point_labels: list of int labels (defaults to 1 for positive clicks)
        """
        frame_idx = max(getattr(self.window, "frame_number", 0), 0)
        boxes_abs = []
        box_labels = []
        points_abs = []
        point_labels = []
        canvas = getattr(self.window, "canvas", None)
        if canvas is None or not getattr(canvas, "shapes", None):
            return {
                "frame_idx": frame_idx,
                "boxes_abs": boxes_abs,
                "box_labels": box_labels,
                "points_abs": points_abs,
                "point_labels": point_labels,
            }

        for shape in canvas.shapes:
            if shape.shape_type == "rectangle" and len(shape.points) == 2:
                p1, p2 = shape.points
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()
                w = x2 - x1
                h = y2 - y1
                boxes_abs.append([x1, y1, w, h])
                box_labels.append(1)
            elif shape.shape_type == "polygon" and shape.points:
                xs = [pt.x() for pt in shape.points]
                ys = [pt.y() for pt in shape.points]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                w = x2 - x1
                h = y2 - y1
                boxes_abs.append([x1, y1, w, h])
                box_labels.append(1)
            elif shape.shape_type in ["points", "point"]:
                labels = shape.point_labels if shape.point_labels else [1] * len(
                    shape.points
                )
                for pt, lbl in zip(shape.points, labels):
                    points_abs.append([pt.x(), pt.y()])
                    point_labels.append(lbl if lbl is not None else 1)
        return {
            "frame_idx": frame_idx,
            "boxes_abs": boxes_abs,
            "box_labels": box_labels,
            "points_abs": points_abs,
            "point_labels": point_labels,
        }

    def build_video_processor(
        self, model_name: str, model_weight: str, text_prompt: Optional[str]
    ):
        """
        Construct the SAM3 video processor or agent runner callable.
        Returns a callable or None if setup fails (and shows a user warning).
        """
        try:
            from annolid.segmentation.SAM.sam3 import adapter as sam3_adapter
            from annolid.segmentation.SAM.sam_v2 import (
                load_annotations_from_video,
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
        text_prompt = text_prompt or self.window._current_text_prompt()

        logger.info(
            "Using SAM3 with checkpoint '%s'",
            sam3_checkpoint if sam3_checkpoint else "auto-download",
        )
        try:
            annotations, id_to_labels = load_annotations_from_video(
                self.window.video_file
            )
        except FileNotFoundError:
            annotations, id_to_labels = [], {}
            logger.info(
                "SAM3: no per-frame JSON annotations found under %s; "
                "continuing with text-only prompt if provided.",
                self.window.video_file,
            )

        if not annotations and not text_prompt:
            QtWidgets.QMessageBox.warning(
                self.window,
                "SAM3 prompts missing",
                "No per-frame annotations were found and no text prompt was "
                "provided. Please add a prompt and try again.",
            )
            return None

        sam3_cfg = dict((self.window._config or {}).get("sam3", {}) or {})
        propagation_direction = sam3_cfg.get("propagation_direction", "both")
        max_frame_num_to_track = sam3_cfg.get("max_frame_num_to_track")
        device_override = sam3_cfg.get("device")
        score_threshold_detection = sam3_cfg.get("score_threshold_detection")
        new_det_thresh = sam3_cfg.get("new_det_thresh")
        sliding_window_size = sam3_cfg.get("sliding_window_size", 5)
        sliding_window_stride = sam3_cfg.get("sliding_window_stride")
        compile_model_cfg = sam3_cfg.get("compile_model", False)
        offload_video_to_cpu_cfg = sam3_cfg.get("offload_video_to_cpu", True)
        agent_cfg = sam3_cfg.get("agent", {}) or {}
        agent_det_thresh_cfg = agent_cfg.get("det_thresh")
        agent_window_size_cfg = agent_cfg.get("window_size")
        agent_stride_cfg = agent_cfg.get("stride")
        agent_output_dir_cfg = agent_cfg.get("output_dir")
        try:
            if isinstance(max_frame_num_to_track, str):
                max_frame_num_to_track = int(max_frame_num_to_track)
        except Exception:
            max_frame_num_to_track = None
        for name, val in (
            ("score_threshold_detection", score_threshold_detection),
            ("new_det_thresh", new_det_thresh),
            ("sliding_window_size", sliding_window_size),
            ("agent_det_thresh_cfg", agent_det_thresh_cfg),
        ):
            try:
                if name == "sliding_window_size":
                    parsed = int(val) if isinstance(val, str) else int(val)
                elif name == "agent_det_thresh_cfg":
                    parsed = float(val) if val is not None else None
                else:
                    if isinstance(val, str):
                        parsed = float(val)
                    elif val is None:
                        parsed = None
                    else:
                        parsed = float(val)
                if name == "score_threshold_detection":
                    score_threshold_detection = parsed
                elif name == "new_det_thresh":
                    new_det_thresh = parsed
                elif name == "sliding_window_size":
                    sliding_window_size = parsed
                else:
                    agent_det_thresh_cfg = parsed
            except Exception:
                if name == "score_threshold_detection":
                    score_threshold_detection = None
                elif name == "new_det_thresh":
                    new_det_thresh = None
                elif name == "sliding_window_size":
                    sliding_window_size = 5
                else:
                    agent_det_thresh_cfg = None

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

        compile_model = _parse_bool(
            self.compile_model if self.compile_model is not None else compile_model_cfg,
            False,
        )
        offload_video_to_cpu = _parse_bool(
            self.offload_video_to_cpu if self.offload_video_to_cpu is not None else offload_video_to_cpu_cfg,
            True,
        )
        if self.score_threshold_detection is not None:
            score_threshold_detection = self.score_threshold_detection
        if self.new_det_thresh is not None:
            new_det_thresh = self.new_det_thresh
        if score_threshold_detection is None:
            score_threshold_detection = self.score_threshold_detection
        if new_det_thresh is None:
            new_det_thresh = self.new_det_thresh
        try:
            if isinstance(sliding_window_stride, str):
                sliding_window_stride = int(sliding_window_stride)
        except Exception:
            sliding_window_stride = None
        try:
            if isinstance(agent_window_size_cfg, str):
                agent_window_size_cfg = int(agent_window_size_cfg)
        except Exception:
            agent_window_size_cfg = None
        try:
            if isinstance(agent_stride_cfg, str):
                agent_stride_cfg = int(agent_stride_cfg)
        except Exception:
            agent_stride_cfg = None
        if self.propagation_direction:
            propagation_direction = self.propagation_direction
        if self.max_frame_num_to_track is not None:
            max_frame_num_to_track = self.max_frame_num_to_track
        if self.device_override:
            device_override = self.device_override
        if self.sliding_window_size is not None:
            sliding_window_size = self.sliding_window_size
        if self.sliding_window_stride is not None:
            sliding_window_stride = self.sliding_window_stride
        agent_det_thresh = (
            self.agent_det_thresh
            if self.agent_det_thresh is not None
            else agent_det_thresh_cfg
            if agent_det_thresh_cfg is not None
            else score_threshold_detection
        )
        agent_window_size = (
            self.agent_window_size
            if self.agent_window_size is not None
            else agent_window_size_cfg
            if agent_window_size_cfg is not None
            else sliding_window_size
        )
        agent_stride = (
            self.agent_stride
            if self.agent_stride is not None
            else agent_stride_cfg
            if agent_stride_cfg is not None
            else sliding_window_stride
        )
        agent_output_dir = (
            self.agent_output_dir or agent_output_dir_cfg or "sam3_agent_out"
        )

        if not annotations:
            prompts = self.extract_prompts_from_canvas()
            if prompts["boxes_abs"] or prompts["points_abs"]:
                annotations = []
                self._initial_prompts = prompts
            else:
                self._initial_prompts = None
        else:
            self._initial_prompts = None

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
                    compile_model=compile_model,
                    offload_video_to_cpu=offload_video_to_cpu,
                    sliding_window_size=sliding_window_size,
                    sliding_window_stride=sliding_window_stride,
                )
                self._last_prompt_frame = None
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    "SAM3 init error",
                    f"Failed to initialise SAM3 session.\n{exc}",
                )
                return None

            def _run_sam3_with_canvas_prompts():
                if self._initial_prompts:
                    prompts = self._initial_prompts
                    try:
                        if prompts["boxes_abs"]:
                            self.sam3_session.add_prompt_boxes_abs(
                                prompts["frame_idx"],
                                prompts["boxes_abs"],
                                prompts["box_labels"] or [1]
                                * len(prompts["boxes_abs"]),
                                text=text_prompt,
                            )
                            self._last_prompt_frame = prompts["frame_idx"]
                        if prompts["points_abs"]:
                            self.sam3_session.add_prompt_points_abs(
                                prompts["frame_idx"],
                                prompts["points_abs"],
                                prompts["point_labels"] or [1]
                                * len(prompts["points_abs"]),
                                text=text_prompt,
                            )
                            self._last_prompt_frame = prompts["frame_idx"]
                    except Exception as exc:
                        logger.warning(
                            "Failed to add SAM3 canvas prompts: %s", exc
                        )
                return self.sam3_session.run()

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
                        compile_model=compile_model,
                        offload_video_to_cpu=offload_video_to_cpu,
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
            if standard_runner is None:
                return "SAM3 init failed"
            return standard_runner()

        return _run_sam3_agent_first
