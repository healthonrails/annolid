# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Windowed inference for long video processing."""

from typing import Dict, List, Tuple


class WindowedInference:
    """
    Manages windowed inference for long videos.

    Handles window computation, memory frame selection, and prediction merging.
    """

    def __init__(
        self,
        window_len: int = 100,
        stride: int = 100,
        num_memory_frames: int = 10,
    ):
        """
        Args:
            window_len: Number of frames per window.
            stride: Step size between windows.
            num_memory_frames: Maximum number of memory frames to use.
        """
        self.window_len = window_len
        self.stride = stride
        self.num_memory_frames = num_memory_frames

    def compute_windows(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Compute all window (start, end) indices.

        Args:
            total_frames: Total number of frames in the video.

        Returns:
            List of (start, end) tuples for each window.
        """
        S = self.window_len
        step = self.stride

        if total_frames <= S:
            return [(0, total_frames)]

        windows = []
        start = 0
        while start < total_frames:
            end = min(start + S, total_frames)
            windows.append((start, end))
            if end == total_frames:
                break
            start += step

        return windows

    def select_memory_frames(
        self,
        window_idx: int,
        window_start: int,
    ) -> List[int]:
        """
        Select memory frame indices using hybrid strategy.

        Strategy combines:
        - First frame (always included for global reference)
        - Recent frames (temporal continuity)
        - Uniformly sampled middle frames (long-range context)

        Args:
            window_idx: Current window index.
            window_start: Start frame index of current window.

        Returns:
            Sorted list of memory frame indices.
        """
        if window_idx == 0:
            return []

        memory_indices = [0]  # Always include first frame

        # Recent frames for temporal continuity
        for offset in [2, 1]:
            idx = window_start - offset
            if idx > 0 and idx not in memory_indices:
                memory_indices.append(idx)

        # Uniform sampling from middle history for long-range context
        if window_start > 10:
            mid_start, mid_end = 5, window_start - 3
            step = (mid_end - mid_start) / 6
            for i in range(5):
                idx = int(mid_start + (i + 1) * step)
                if idx not in memory_indices:
                    memory_indices.append(idx)

        # Limit to maximum number of memory frames
        if len(memory_indices) > self.num_memory_frames:
            memory_indices = sorted(memory_indices)[-self.num_memory_frames :]

        return sorted(memory_indices)

    def merge_predictions(
        self,
        window_idx: int,
        window_start: int,
        window_end: int,
        window_pred: Dict,
        accumulated: Dict,
    ) -> None:
        """
        Merge window predictions into accumulated results.

        Handles overlapping regions by using only non-overlapping parts
        from subsequent windows.

        Args:
            window_idx: Current window index.
            window_start: Start frame index.
            window_end: End frame index.
            window_pred: Predictions for current window (track, vis, conf).
            accumulated: Accumulated predictions to update in-place.
        """
        S_actual = window_end - window_start

        if window_idx > 0 and self.stride < self.window_len:
            # Has overlap with previous window - only take non-overlapping part
            overlap_len = min(self.window_len - self.stride, S_actual)
            if overlap_len < S_actual:
                start_offset = overlap_len
                for key in ["track", "vis", "conf"]:
                    accumulated[key][:, window_start + start_offset : window_end] = (
                        window_pred[key][:, start_offset:S_actual]
                    )
        else:
            # No overlap or first window - take everything
            for key in ["track", "vis", "conf"]:
                accumulated[key][:, window_start:window_end] = window_pred[key][
                    :, :S_actual
                ]
