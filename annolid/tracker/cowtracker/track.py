"""CoWTracker processor for Annolid.

CoWTracker dependencies are only loaded when load_model() is called,
ensuring no side effects on other trackers until the user explicitly uses this.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import torch

from annolid.tracker.point_tracking_processor import BasePointTrackingProcessor
from annolid.utils.logger import logger

if TYPE_CHECKING:
    from cowtracker import CoWTrackerWindowed

# Constants
_PATCH_LCM = 112  # LCM of 14 (ViT) and 16 (ResNet) for dimension alignment
_MODEL_CACHE: dict[tuple[str, int, int], torch.nn.Module] = {}


class CoWTrackerProcessor(BasePointTrackingProcessor):
    """CoWTracker processor with lazy dependency loading.

    Uses CoWTracker's dense tracking model in offline windowed mode.
    Dependencies are loaded only when load_model() is called.
    """

    def __init__(
        self,
        video_path: str,
        json_path: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        model_name: Optional[str] = None,
        window_len: int = 8,
        stride: int = 8,
        **kwargs: Any,
    ):
        """Initialize CoWTrackerProcessor.

        Args:
            video_path: Path to the video file.
            json_path: Optional path to JSON annotations.
            should_stop: Optional callback to check for stop requests.
            model_name: Optional model name (unused, for API compatibility).
            window_len: Number of frames per inference window (default: 8).
            stride: Step size between windows (default: 8).
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            video_path=video_path,
            json_path=json_path,
            is_online=False,
            should_stop=should_stop,
            model_name=model_name,
            **kwargs,
        )
        self.window_len = window_len
        self.stride = stride

    def load_model(self) -> "CoWTrackerWindowed":
        """Load CoWTracker model with lazy dependency import.

        Returns:
            Loaded CoWTrackerWindowed model.

        Raises:
            RuntimeError: If CoWTracker dependencies are not installed.
        """
        # Add cowtracker package to path only when needed
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        try:
            from cowtracker.dependencies import validate_cowtracker_runtime
            from cowtracker import CoWTrackerWindowed
        except ImportError as exc:
            raise RuntimeError(
                "CoWTracker dependencies not found. "
                "Please ensure the cowtracker package is properly installed."
            ) from exc
        validate_cowtracker_runtime()

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        cache_key = (str(self.device), self.window_len, self.stride)
        if cache_key in _MODEL_CACHE:
            logger.debug(
                "Reusing cached CoWTracker model on %s (window=%d, stride=%d)",
                self.device,
                self.window_len,
                self.stride,
            )
            return _MODEL_CACHE[cache_key]

        logger.info(
            "Loading CoWTracker model on %s (window=%d, stride=%d)",
            self.device,
            self.window_len,
            self.stride,
        )
        model = CoWTrackerWindowed.from_checkpoint(
            device=str(self.device),
            dtype=dtype,
            window_len=self.window_len,
            stride=self.stride,
        )
        _MODEL_CACHE[cache_key] = model
        return model

    def _process_video_bidirectional(
        self,
        start_frame: int = 0,
        end_frame: int = -1,
        grid_size: int = 10,
        grid_query_frame: int = 0,
    ):
        """Process video using CoWTracker with incremental saving.

        Overrides parent to use CoWTracker's dense tracking and save
        results after each window is processed.
        """
        if self._should_stop():
            self._stop_triggered = True
            return None, None, None

        del grid_size, grid_query_frame
        logger.info(
            "Running CoWTracker inference: frames %d to %d", start_frame, end_frame
        )

        # Load video frames
        video = self.video_loader.get_frames_between(start_frame, end_frame)
        video_tensor = (
            torch.from_numpy(video).permute(0, 3, 1, 2).float().to(self.device)
        )

        # Pad to multiple of 112 (LCM of ViT patch size 14 and ResNet stride 16)
        video_tensor = self._pad_to_multiple(video_tensor, _PATCH_LCM)

        model = self._ensure_model()
        return self._run_windowed_inference(model, video_tensor, video, start_frame)

    def _pad_to_multiple(self, tensor: torch.Tensor, multiple: int) -> torch.Tensor:
        """Pad tensor dimensions to be multiples of the given value."""
        h, w = tensor.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
            logger.info(
                "Padded video to (%d, %d) for model constraints", h + pad_h, w + pad_w
            )

        return tensor

    def _run_windowed_inference(
        self,
        model,
        video_tensor: torch.Tensor,
        original_video: np.ndarray,
        start_frame: int,
    ):
        """Run windowed inference with incremental saving."""
        # Normalize to [0, 1] as expected by CoWTracker
        images = (
            video_tensor.unsqueeze(0) / 255.0
            if video_tensor.ndim == 4
            else video_tensor / 255.0
        )
        # Keep input dtype aligned with model weights (e.g., fp16 on CUDA)
        # to avoid conv dtype mismatch errors on platforms like Windows.
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = images.dtype
        if images.dtype != model_dtype:
            images = images.to(dtype=model_dtype)

        B, T_total, C, H, W = images.shape
        device, dtype = images.device, images.dtype

        # Initialize output accumulators
        accumulated = {
            "track": torch.zeros((B, T_total, H, W, 2), device=device, dtype=dtype),
            "vis": torch.zeros((B, T_total, H, W), device=device, dtype=dtype),
            "conf": torch.zeros((B, T_total, H, W), device=device, dtype=dtype),
        }

        # Get windowing components
        win_inf = model.windowed
        model_inner = model.model
        windows = win_inf.compute_windows(T_total)
        first_frame = images[:, 0:1]

        # Prepare queries for sampling
        queries = self.queries.cpu().numpy()
        N = queries.shape[0]

        # Process each window
        for window_idx, (start, end) in enumerate(windows):
            if self._should_stop():
                self._stop_triggered = True
                break

            logger.info(
                "Processing window %d/%d: frames [%d, %d)",
                window_idx + 1,
                len(windows),
                start,
                end,
            )

            self._process_window(
                model_inner,
                win_inf,
                images,
                first_frame,
                window_idx,
                start,
                end,
                H,
                W,
                accumulated,
            )

            # Incrementally save results for this window
            self._save_window_results(
                accumulated, queries, N, window_idx, start, end, start_frame, win_inf
            )

        # Sample final outputs for compatibility with parent class
        return self._sample_final_outputs(
            accumulated, queries, N, T_total, original_video
        )

    def _process_window(
        self,
        model_inner,
        win_inf,
        images,
        first_frame,
        window_idx: int,
        start: int,
        end: int,
        H: int,
        W: int,
        accumulated: dict,
    ):
        """Process a single window and merge into accumulated results."""
        with torch.no_grad():
            # Gather frames: first + memory + current window
            memory_indices = win_inf.select_memory_frames(window_idx, start)
            parts = [first_frame]
            if memory_indices:
                parts.append(images[:, memory_indices])
            parts.append(images[:, start:end])
            frames = torch.cat(parts, dim=1)

            # Run model inference
            tokens, patch_idx = model_inner.aggregator(frames)
            features = model_inner.feature_extractor(tokens, frames, patch_idx)

            pred = model_inner.tracking_head(
                features[:, 1:],  # Exclude first frame from input
                image_size=(H, W),
                first_frame_features=features[:, 0:1],
            )

            # Extract window predictions (remove memory frames)
            num_memory = len(memory_indices)
            window_pred = {
                "track": pred["track"][:, num_memory:],
                "vis": pred["vis"][:, num_memory:],
                "conf": pred["conf"][:, num_memory:],
            }

            win_inf.merge_predictions(window_idx, start, end, window_pred, accumulated)

            # Cleanup for memory efficiency
            del features, tokens, pred

    def _save_window_results(
        self,
        accumulated: dict,
        queries: np.ndarray,
        N: int,
        window_idx: int,
        start: int,
        end: int,
        global_start_frame: int,
        win_inf,
    ):
        """Sample and save tracking results for a window.

        Handles both overlapping (stride < window_len) and non-overlapping
        (stride == window_len) window configurations correctly.
        """
        # First window always saves from the start
        if window_idx == 0:
            save_start = start
        else:
            # For subsequent windows, skip already-saved overlap region
            if win_inf.stride < win_inf.window_len:
                overlap_len = win_inf.window_len - win_inf.stride
                save_start = start + overlap_len
            else:
                # Non-overlapping: save entire window
                save_start = start

        if save_start >= end:
            return

        num_save = end - save_start
        incremental_tracks = torch.zeros((1, num_save, N, 2), device=self.device)
        incremental_vis = torch.zeros((1, num_save, N), device=self.device)

        # Sample dense tracks at query locations
        seg_tracks = accumulated["track"][0, save_start:end]
        seg_vis = accumulated["vis"][0, save_start:end]

        for i in range(N):
            q_x = int(np.clip(queries[i, 1], 0, self.video_width - 1))
            q_y = int(np.clip(queries[i, 2], 0, self.video_height - 1))
            incremental_tracks[0, :, i] = seg_tracks[:, q_y, q_x]
            incremental_vis[0, :, i] = seg_vis[:, q_y, q_x]

        self.extract_frame_points(
            incremental_tracks,
            incremental_vis,
            chunk_start_frame=global_start_frame + save_start,
        )

    def _sample_final_outputs(
        self,
        accumulated: dict,
        queries: np.ndarray,
        N: int,
        T_total: int,
        original_video: np.ndarray,
    ):
        """Sample final outputs from accumulated dense tracks."""
        output_tracks = torch.zeros((1, T_total, N, 2), device=self.device)
        output_vis = torch.zeros((1, T_total, N), device=self.device)

        for i in range(N):
            q_x = int(np.clip(queries[i, 1], 0, self.video_width - 1))
            q_y = int(np.clip(queries[i, 2], 0, self.video_height - 1))
            output_tracks[0, :, i] = accumulated["track"][0, :, q_y, q_x]
            output_vis[0, :, i] = accumulated["vis"][0, :, q_y, q_x]

        return output_tracks, output_vis, original_video
