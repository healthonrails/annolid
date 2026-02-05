from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from annolid.data.videos import CV2Video
from annolid.segmentation.cutie_vos.predict import CutieCoreVideoProcessor
from annolid.utils.logger import logger


class CutieVideoProcessor:
    """CUTIE-first processor compatible with legacy VideoProcessor call sites."""

    _DEFAULT_MEM_EVERY = 5
    _DEFAULT_VISUALIZE_EVERY = 20
    _RUNTIME_KWARG_EXCLUDE = {
        "results_folder",
        "t_max_value",
        "use_cpu_only",
        "epsilon_for_polygon",
        "save_video_with_color_mask",
        "mem_every",
        "video_path",
        "model_name",
        "device",
        "save_image_to_disk",
        "point_tracking",
        "is_cutie",
        "is_new_segment",
        "step",
        "start_frame",
        "end_frame",
    }

    def __init__(self, video_path: str, *args: Any, **kwargs: Any) -> None:
        _ = args
        self.video_path = str(video_path)
        results_folder = kwargs.get("results_folder")
        self.video_folder = (
            Path(results_folder) if results_folder else Path(video_path).with_suffix("")
        )
        self.results_folder = self.video_folder

        self.t_max_value = kwargs.get("t_max_value", 5)
        self.use_cpu_only = kwargs.get("use_cpu_only", False)
        self.epsilon_for_polygon = kwargs.get("epsilon_for_polygon", 2.0)
        self.save_video_with_color_mask = kwargs.get(
            "save_video_with_color_mask", False
        )
        self._runtime_kwargs = {
            k: v for k, v in kwargs.items() if k not in self._RUNTIME_KWARG_EXCLUDE
        }

        self.video_loader = CV2Video(self.video_path)
        self.num_frames = int(self.video_loader.total_frames())

        self.pred_worker: Optional[Any] = None
        self.cutie_processor: Optional[CutieCoreVideoProcessor] = None
        self._cutie_mem_every: Optional[int] = None

    def set_pred_worker(self, pred_worker: Any) -> None:
        self.pred_worker = pred_worker

    def get_total_frames(self) -> int:
        return int(self.num_frames)

    def cleanup(self) -> None:
        self.cutie_processor = None
        if getattr(self, "video_loader", None) is not None:
            try:
                self.video_loader.release()
            except Exception:
                pass

    def __del__(self) -> None:
        self.cleanup()

    def _is_pred_worker_stopped(self) -> bool:
        if self.pred_worker is None:
            return False
        try:
                return bool(self.pred_worker.is_stopped())
        except Exception:
            return False

    @staticmethod
    def _resolve_mem_every(value: Any) -> int:
        try:
            mem_every = int(value)
            if mem_every <= 0:
                return CutieVideoProcessor._DEFAULT_MEM_EVERY
            return mem_every
        except (TypeError, ValueError):
            return CutieVideoProcessor._DEFAULT_MEM_EVERY

    def _resolve_target_end_frame(self, end_frame: Any) -> int:
        target_end = self.num_frames - 1
        if end_frame is None:
            return target_end
        try:
            return min(int(end_frame), target_end)
        except (TypeError, ValueError):
            return target_end

    def _ensure_cutie_processor(self, mem_every: int) -> None:
        if self.cutie_processor is None or self._cutie_mem_every != mem_every:
            self.reset_cutie_processor(mem_every=mem_every)

    def reset_cutie_processor(self, mem_every: int = 5) -> None:
        mem_every = self._resolve_mem_every(mem_every)
        logger.debug(
            "Resetting CutieVideoProcessor for video '%s' (mem_every=%s)",
            self.video_path,
            mem_every,
        )
        self.cutie_processor = CutieCoreVideoProcessor(
            self.video_path,
            mem_every=mem_every,
            debug=False,
            epsilon_for_polygon=self.epsilon_for_polygon,
            t_max_value=self.t_max_value,
            use_cpu_only=self.use_cpu_only,
            results_folder=self.results_folder,
            **self._runtime_kwargs,
        )
        self._cutie_mem_every = mem_every

    def process_video_frames(self, *args: Any, **kwargs: Any) -> str:
        _ = args
        if self._is_pred_worker_stopped():
            return "Tracking stopped by user."

        mem_every = self._resolve_mem_every(kwargs.get("mem_every", 5))
        start_frame = int(kwargs.get("start_frame", 0))
        end_frame = kwargs.get("end_frame", None)
        has_occlusion = bool(kwargs.get("has_occlusion", False))
        recording = bool(
            kwargs.get("save_video_with_color_mask", self.save_video_with_color_mask)
        )

        self._ensure_cutie_processor(mem_every=mem_every)
        target_end = self._resolve_target_end_frame(end_frame)

        return self.cutie_processor.process_video_from_seeds(
            end_frame=target_end,
            start_frame=max(0, start_frame),
            pred_worker=self.pred_worker,
            recording=recording,
            output_video_path=None,
            has_occlusion=has_occlusion,
            visualize_every=self._DEFAULT_VISUALIZE_EVERY,
        )
