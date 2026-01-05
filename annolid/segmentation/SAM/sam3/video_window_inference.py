import tempfile
from pathlib import Path
from typing import Generator, Iterable, List, Dict, Any, Optional, Tuple, Literal

import cv2
import numpy as np
from PIL import Image
import torch

from .sam3.model_builder import build_sam3_video_model
from .sam3.agent.agent_core import agent_inference


def _iter_video_windows(
    video_path: str,
    window_size: int,
    stride: Optional[int] = None,
) -> Generator[Tuple[int, int, List[np.ndarray]], None, None]:
    """
    Iterate over a video and yield small windows of frames as numpy arrays.

    Args:
        video_path: Path to the input video file.
        window_size: Number of frames in each window (e.g. 5).
        stride: Step between two consecutive windows. Defaults to window_size
            (non-overlapping windows).

    Yields:
        (start_frame_idx, end_frame_idx_exclusive, frames)
    """
    if stride is None:
        stride = window_size

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    start = 0
    while True:
        frames: List[np.ndarray] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(window_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if not frames:
            break

        end = start + len(frames)
        yield start, end, frames

        if end >= (total_frames or end):
            break

        start += stride

    cap.release()


def _frames_to_pil(frames: List[np.ndarray]) -> List["Image.Image"]:
    """Convert OpenCV BGR frames to RGB PIL images."""
    pil_frames: List[Image.Image] = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))
    return pil_frames


def run_sam3_video_sliding_window(
    video_path: str,
    text_prompt: str,
    window_size: int = 5,
    stride: Optional[int] = None,
    device: Optional[str] = None,
    model=None,
) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    """
    Run SAM3 dense tracking on a video using a sliding-window strategy that
    only keeps a small number of frames in memory at once.

    Args:
        video_path: Path to the input video file.
        text_prompt: Text prompt describing the object to track.
        window_size: Number of frames to hold in memory per window.
        stride: Step between two consecutive windows. Defaults to window_size.
        device: Optional device string for SAM3 (e.g. "cuda" or "cpu").
        model: Optional pre-built SAM3 video model to reuse; if None, a model
            is built once and reused.

    Yields:
        (global_frame_index, sam3_frame_output_dict)
    """
    sam3_model = model or build_sam3_video_model(device=device)

    stride = stride or window_size
    for start_idx, _, frames in _iter_video_windows(
        video_path=video_path,
        window_size=window_size,
        stride=stride,
    ):
        pil_frames: List[Image.Image] = _frames_to_pil(frames)
        inference_state = None
        try:
            inference_state = sam3_model.init_state(
                resource_path=pil_frames,
                offload_video_to_cpu=True,
                async_loading_frames=False,
                video_loader_type="cv2",
            )

            sam3_model.add_prompt(
                inference_state,
                frame_idx=0,
                text_str=text_prompt,
            )

            for local_frame_idx, out in sam3_model.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                reverse=False,
            ):
                global_idx = start_idx + local_frame_idx
                yield global_idx, out
        finally:
            # Best-effort cleanup between windows to avoid unbounded MPS memory growth.
            if inference_state is not None:
                try:
                    sam3_model.reset_state(inference_state)
                except Exception:
                    pass
            try:
                import gc

                gc.collect()
            except Exception:
                pass
            try:
                empty_cache = getattr(
                    getattr(torch, "mps", None), "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
            except Exception:
                pass


def run_sam3_agent_sliding_window(
    video_path: str,
    agent_prompt: str,
    window_size: int = 5,
    stride: Optional[int] = None,
    output_dir: str = "../../sam3_agent_out",
) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    """
    Run the SAM3 agent (LLM-driven) on a video using a sliding window.

    Each frame is processed independently by the agent to minimize RAM usage.
    Temporary frame files are written to disk so the agent can consume them.

    Args:
        video_path: Path to the input video file.
        agent_prompt: User prompt passed to the SAM3 agent.
        window_size: Number of frames to hold in memory per window.
        stride: Step between windows. Defaults to window_size (no overlap).
        output_dir: Directory where agent outputs are written.

    Yields:
        (global_frame_index, agent_result_dict)
        where agent_result_dict contains:
            - messages: conversation history
            - outputs: model outputs dict
            - rendered: rendered output image (PIL)
    """
    stride = stride or window_size

    with tempfile.TemporaryDirectory(prefix="sam3_agent_frames_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for start_idx, _, frames in _iter_video_windows(
            video_path=video_path,
            window_size=window_size,
            stride=stride,
        ):
            pil_frames = _frames_to_pil(frames)
            for local_idx, pil_img in enumerate(pil_frames):
                global_idx = start_idx + local_idx
                frame_path = tmpdir_path / f"frame_{global_idx:08d}.png"
                pil_img.save(frame_path)

                messages, outputs, rendered = agent_inference(
                    img_path=str(frame_path),
                    initial_text_prompt=agent_prompt,
                    output_dir=output_dir,
                )
                yield global_idx, {
                    "messages": messages,
                    "outputs": outputs,
                    "rendered": rendered,
                }


def run_video_sliding_window(
    *,
    video_path: str,
    mode: Literal["sam3", "agent"] = "sam3",
    text_prompt: Optional[str] = None,
    agent_prompt: Optional[str] = None,
    window_size: int = 5,
    stride: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: str = "../../sam3_agent_out",
    model=None,
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """
    Unified entrypoint to run sliding-window inference in two modes:
      - mode="sam3"  : dense tracking with the SAM3 video model (default)
      - mode="agent" : LLM-driven SAM3 agent per frame

    Returns a generator yielding (global_frame_idx, result_dict).
    """
    if mode == "sam3":
        if not text_prompt:
            raise ValueError("text_prompt is required for mode='sam3'")
        return run_sam3_video_sliding_window(
            video_path=video_path,
            text_prompt=text_prompt,
            window_size=window_size,
            stride=stride,
            device=device,
            model=model,
        )
    if mode == "agent":
        if not agent_prompt:
            raise ValueError("agent_prompt is required for mode='agent'")
        return run_sam3_agent_sliding_window(
            video_path=video_path,
            agent_prompt=agent_prompt,
            window_size=window_size,
            stride=stride,
            output_dir=output_dir,
        )
    raise ValueError(f"Unknown mode '{mode}'. Expected 'sam3' or 'agent'.")
