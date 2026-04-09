"""
Utilities to drive SAM3 video tracking with SAM3 Agent-corrected seeds on
window key frames. The agent refines the first frame of each window, then
`Sam3SessionManager` propagates within that window using the configured
tracking thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import tempfile

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from annolid.utils.llm_settings import resolve_llm_config
from annolid.core.agent.providers.openai_compat import resolve_openai_compat

from annolid.segmentation.SAM.sam3.session import Sam3SessionConfig, Sam3SessionManager
from .sam3.agent.client_llm import send_generate_request as _send_generate_request
from .video_window_inference import _iter_video_windows
from .window_refresh import run_mid_window_refresh
from .windowed_runner import compute_window_reuse_shift


@dataclass
class AgentConfig:
    """Configuration for running the SAM3 Agent on key frames."""

    prompt: str
    det_thresh: float = 0.3
    window_size: int = 5
    stride: Optional[int] = None
    output_dir: Optional[str] = None
    debug: bool = False
    max_generations: int = 100
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_profile: Optional[str] = None


@dataclass
class TrackingConfig:
    """Configuration for SAM3 video propagation."""

    checkpoint_path: Optional[str] = None
    propagation_direction: str = "forward"
    device: Optional[str] = None
    score_threshold_detection: Optional[float] = None
    new_det_thresh: Optional[float] = None
    max_num_objects: int = 16
    multiplex_count: int = 16
    compile_model: bool = False
    offload_video_to_cpu: bool = True
    use_explicit_window_reseed: bool = True
    boundary_mask_match_iou_threshold: float = 0.2
    use_vlm_boundary_id_correction: bool = True
    vlm_boundary_match_iou_threshold: float = 0.2
    allow_private_state_mutation: bool = False
    epsilon_for_polygon: float = 2.0
    max_frame_num_to_track: Optional[int] = None


def _save_frame_to_tmp(frame, dst_path: Path) -> None:
    """Persist a single BGR frame to disk for agent consumption."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(dst_path)


def _boxes_from_agent_output(outputs: Dict[str, object], det_thresh: float) -> Tuple[List[List[float]], List[float]]:
    """
    Convert agent JSON outputs (normalized xywh) to absolute xyxy boxes and
    filter by detection score.
    """
    boxes_norm = outputs.get("pred_boxes", []) or []
    scores = outputs.get("pred_scores", []) or []
    width = outputs.get("orig_img_w")
    height = outputs.get("orig_img_h")
    if not width or not height:
        return [], []

    boxes_abs: List[List[float]] = []
    kept_scores: List[float] = []
    for box, score in zip(boxes_norm, scores):
        try:
            if score < det_thresh:
                continue
            x, y, w, h = box
            x1 = float(x * width)
            y1 = float(y * height)
            x2 = float((x + w) * width)
            y2 = float((y + h) * height)
            boxes_abs.append([x1, y1, x2, y2])
            kept_scores.append(float(score))
        except Exception:
            continue
    return boxes_abs, kept_scores


def _box_iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _filter_duplicate_boxes(
    candidate_boxes: List[List[float]],
    existing_boxes: List[List[float]],
    *,
    iou_threshold: float = 0.55,
) -> List[List[float]]:
    filtered: List[List[float]] = []
    for candidate in candidate_boxes:
        if any(_box_iou_xyxy(candidate, prev) >= iou_threshold for prev in existing_boxes):
            continue
        filtered.append(candidate)
    return filtered


def _xyxy_to_xywh(box_xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]


def _boundary_box_to_xyxy(
    box_xywh_norm: Sequence[float],
    *,
    frame_width: float,
    frame_height: float,
) -> Optional[List[float]]:
    if len(box_xywh_norm) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in box_xywh_norm]
    except Exception:
        return None
    if frame_width <= 0.0 or frame_height <= 0.0:
        return None
    x1 = x * float(frame_width)
    y1 = y * float(frame_height)
    x2 = (x + w) * float(frame_width)
    y2 = (y + h) * float(frame_height)
    return [x1, y1, x2, y2]


def _resolve_vlm_boundary_box_labels(
    *,
    candidate_boxes_xyxy: List[List[float]],
    boundary_boxes_norm_xywh: Sequence[Sequence[float]],
    boundary_track_ids: Sequence[int],
    frame_width: float,
    frame_height: float,
    iou_threshold: float,
    prompt_prefix: str,
    id_to_labels: Dict[int, str],
) -> Tuple[List[int], List[str]]:
    """
    Match VLM detections on a new window's first frame to boundary carry IDs.

    Returns:
        (box_labels, label_hints), both aligned to candidate_boxes_xyxy order.
    """
    if not candidate_boxes_xyxy:
        return [], []
    labels: List[int] = []
    hints: List[str] = []
    if not boundary_boxes_norm_xywh or not boundary_track_ids:
        for idx in range(len(candidate_boxes_xyxy)):
            lid = int(idx + 1)
            labels.append(lid)
            hints.append(f"{prompt_prefix}_{lid}")
        return labels, hints

    reference_items: List[Tuple[int, List[float]]] = []
    for gid, norm_box in zip(boundary_track_ids, boundary_boxes_norm_xywh):
        xyxy = _boundary_box_to_xyxy(
            norm_box,
            frame_width=float(frame_width),
            frame_height=float(frame_height),
        )
        if xyxy is None:
            continue
        reference_items.append((int(gid), xyxy))
    if not reference_items:
        for idx in range(len(candidate_boxes_xyxy)):
            lid = int(idx + 1)
            labels.append(lid)
            hints.append(f"{prompt_prefix}_{lid}")
        return labels, hints

    iou_cost = np.ones((len(candidate_boxes_xyxy), len(reference_items)), dtype=float)
    valid = np.zeros_like(iou_cost, dtype=bool)
    for det_idx, det_box in enumerate(candidate_boxes_xyxy):
        for ref_idx, (_, ref_box) in enumerate(reference_items):
            iou = _box_iou_xyxy(det_box, ref_box)
            if iou < float(iou_threshold):
                continue
            valid[det_idx, ref_idx] = True
            iou_cost[det_idx, ref_idx] = 1.0 - float(iou)

    assigned: Dict[int, int] = {}
    if iou_cost.size > 0:
        row_ind, col_ind = linear_sum_assignment(iou_cost)
        for row, col in zip(row_ind, col_ind):
            if not valid[row, col]:
                continue
            assigned[int(row)] = int(reference_items[int(col)][0])

    used_ids = {int(v) for v in assigned.values()}
    next_local_id = max(1, len(reference_items) + 1)
    for det_idx in range(len(candidate_boxes_xyxy)):
        gid = assigned.get(int(det_idx))
        if gid is not None:
            labels.append(int(gid))
            hints.append(str(id_to_labels.get(int(gid)) or f"{prompt_prefix}_{int(gid)}"))
            continue
        while next_local_id in used_ids:
            next_local_id += 1
        labels.append(int(next_local_id))
        hints.append(f"{prompt_prefix}_{int(next_local_id)}")
        used_ids.add(int(next_local_id))
        next_local_id += 1
    return labels, hints


def _run_agent_on_frame(
    frame_path: Path,
    agent_cfg: AgentConfig,
    send_request=None,
) -> Tuple[List[List[float]], List[float]]:
    """
    Run the SAM3 Agent on a single frame image and return filtered boxes/scores.
    """
    from .sam3.agent.agent_core import agent_inference

    if send_request is None:
        send_request = _build_send_generate_request(agent_cfg)[0]

    try:
        _, outputs, _ = agent_inference(
            img_path=str(frame_path),
            initial_text_prompt=agent_cfg.prompt,
            debug=agent_cfg.debug,
            max_generations=agent_cfg.max_generations,
            send_generate_request=send_request,
            output_dir=agent_cfg.output_dir,
        )
    except Exception as exc:
        # Normalize assertion/missing tool responses into a clean RuntimeError for caller fallback.
        raise RuntimeError(
            f"Agent inference failed on {frame_path}: {exc} "
            "Ensure the configured VLM supports tool calls (e.g., qwen3-vl/vision-capable models)."
        ) from exc
    return _boxes_from_agent_output(outputs or {}, agent_cfg.det_thresh)


def _call_run_agent_on_frame(
    frame_path: Path,
    agent_cfg: AgentConfig,
    send_request,
) -> Tuple[List[List[float]], List[float]]:
    try:
        return _run_agent_on_frame(frame_path, agent_cfg, send_request)
    except TypeError as exc:
        message = str(exc)
        if "positional arguments" not in message and "positional argument" not in message:
            raise
        return _run_agent_on_frame(frame_path, agent_cfg)


def _build_send_generate_request(agent_cfg: AgentConfig):
    """
    Build a chat completion callable from the currently selected Annolid bot
    provider/model.

    The SAM3 agent expects an OpenAI-compatible text+image chat backend with
    tool-call support. We reuse the active bot provider/model selection rather
    than falling back to an unrelated default.
    """
    resolved_cfg = resolve_llm_config(
        profile=agent_cfg.llm_profile,
        provider=agent_cfg.llm_provider,
        model=agent_cfg.llm_model,
        persist=False,
    )
    resolved = resolve_openai_compat(resolved_cfg)

    def _send(messages):
        return _send_generate_request(
            messages,
            server_url=resolved.base_url,
            model=resolved.model,
            api_key=resolved.api_key,
        )

    return _send, resolved


def run_agent_seeded_sam3_video(
    video_path: str,
    agent_cfg: AgentConfig,
    tracking_cfg: TrackingConfig,
    *,
    id_to_labels: Optional[Dict[int, str]] = None,
) -> Tuple[int, int]:
    """
    Iterate over a video in windows. For each window, run the SAM3 Agent on the
    first frame to obtain refined boxes, then propagate within that window using
    SAM3 video tracking.

    Returns:
        (total_frames_processed, total_masks_written)
    """
    id_to_labels = id_to_labels or {}
    session_cfg = Sam3SessionConfig(
        checkpoint_path=tracking_cfg.checkpoint_path,
        text_prompt=agent_cfg.prompt,
        epsilon_for_polygon=tracking_cfg.epsilon_for_polygon,
        ndjson_filename=None,
        max_frame_num_to_track=tracking_cfg.max_frame_num_to_track,
        propagation_direction=tracking_cfg.propagation_direction,
        device=tracking_cfg.device,
        score_threshold_detection=tracking_cfg.score_threshold_detection,
        new_det_thresh=tracking_cfg.new_det_thresh,
        max_num_objects=tracking_cfg.max_num_objects,
        multiplex_count=tracking_cfg.multiplex_count,
        compile_model=tracking_cfg.compile_model,
        offload_video_to_cpu=tracking_cfg.offload_video_to_cpu,
        use_explicit_window_reseed=tracking_cfg.use_explicit_window_reseed,
        boundary_mask_match_iou_threshold=tracking_cfg.boundary_mask_match_iou_threshold,
        allow_private_state_mutation=tracking_cfg.allow_private_state_mutation,
        async_loading_frames=False,
        sliding_window_size=agent_cfg.window_size,
        sliding_window_stride=agent_cfg.stride,
        use_sliding_window_for_text_prompt=False,
    )
    session = Sam3SessionManager(
        video_dir=video_path,
        id_to_labels=id_to_labels,
        config=session_cfg,
    )
    send_request, _resolved_llm = _build_send_generate_request(agent_cfg)

    total_frames = 0
    total_masks = 0
    stride = agent_cfg.stride or agent_cfg.window_size

    with tempfile.TemporaryDirectory(prefix="sam3_agent_windows_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        previous_window_end_idx: Optional[int] = None
        previous_window_frame_count = 0
        previous_window_state: Optional[dict] = None
        for start_idx, end_idx, frames in _iter_video_windows(
            video_path=video_path,
            window_size=agent_cfg.window_size,
            stride=stride,
        ):
            if not frames:
                continue
            window_end_idx = int(end_idx)
            first_frame = frames[0]
            frame_path = tmpdir_path / f"frame_{start_idx:09}.png"
            _save_frame_to_tmp(first_frame, frame_path)
            boxes_abs, _scores = _call_run_agent_on_frame(
                frame_path,
                agent_cfg,
                send_request,
            )
            if not boxes_abs:
                continue

            # Keep the session-level tracking history alive across windows so
            # overlap can stabilize IDs and box carry-over.
            with session._session_scope(
                target_device=tracking_cfg.device, auto_close=True
            ):
                boundary_bundle = None
                boundary_allowed_gids = None
                if (
                    bool(tracking_cfg.use_explicit_window_reseed)
                    and int(start_idx) > 0
                    and previous_window_end_idx is not None
                    and hasattr(session, "_build_boundary_reseed_prompt_bundle")
                ):
                    frame_h, frame_w = first_frame.shape[:2]
                    try:
                        boundary_bundle = session._build_boundary_reseed_prompt_bundle(
                            frame_idx=int(start_idx),
                            frame_width=float(frame_w),
                            frame_height=float(frame_h),
                            source_frame_idx=int(previous_window_end_idx) - 1,
                        )
                    except Exception:
                        boundary_bundle = None
                    if boundary_bundle is not None:
                        boundary_allowed_gids = {
                            int(v)
                            for v in (getattr(boundary_bundle, "track_ids", []) or [])
                        }
                shift = compute_window_reuse_shift(
                    previous_window_end_idx=previous_window_end_idx,
                    window_start_idx=int(start_idx),
                    frame_count=len(frames),
                    previous_window_frame_count=previous_window_frame_count,
                )
                session._reset_action_history_if_supported()
                if previous_window_state is not None and shift > 0:
                    session._carry_forward_window_state(previous_window_state, shift=shift)
                propagation_direction = (tracking_cfg.propagation_direction or "forward").lower()

                def seed_first_frame() -> None:
                    seed_boxes = [_xyxy_to_xywh(v) for v in boxes_abs]
                    seed_labels = list(range(1, len(seed_boxes) + 1))
                    seed_hints = [f"{agent_cfg.prompt}_{i + 1}" for i in range(len(seed_boxes))]
                    if (
                        tracking_cfg.use_vlm_boundary_id_correction
                        and boundary_bundle is not None
                        and getattr(boundary_bundle, "boxes", None)
                        and getattr(boundary_bundle, "track_ids", None)
                    ):
                        frame_h, frame_w = first_frame.shape[:2]
                        seed_labels, seed_hints = _resolve_vlm_boundary_box_labels(
                            candidate_boxes_xyxy=boxes_abs,
                            boundary_boxes_norm_xywh=getattr(boundary_bundle, "boxes", []),
                            boundary_track_ids=getattr(boundary_bundle, "track_ids", []),
                            frame_width=float(frame_w),
                            frame_height=float(frame_h),
                            iou_threshold=float(
                                tracking_cfg.vlm_boundary_match_iou_threshold
                            ),
                            prompt_prefix=agent_cfg.prompt,
                            id_to_labels=getattr(session, "id_to_labels", {}),
                        )
                    session.add_prompt_boxes_abs(
                        frame_idx=start_idx,
                        boxes_abs=seed_boxes,
                        box_labels=seed_labels,
                        record_outputs=True,
                        label_hints=seed_hints,
                        boundary_bundle=boundary_bundle,
                        boundary_allowed_gids=boundary_allowed_gids,
                        boundary_max_new_ids=None,
                    )

                def propagate_segment(
                    segment_start_local_idx: int,
                    segment_len: int,
                ) -> Tuple[int, int]:
                    return session.propagate(
                        start_frame_idx=start_idx + int(segment_start_local_idx),
                        propagation_direction=propagation_direction,
                        max_frame_num_to_track=int(segment_len),
                    )

                def refresh_mid_frame(refresh_local_idx: int) -> Tuple[int, int]:
                    refresh_frame = frames[int(refresh_local_idx)]
                    refresh_path = tmpdir_path / f"frame_{start_idx + refresh_local_idx:09}.png"
                    _save_frame_to_tmp(refresh_frame, refresh_path)
                    refresh_boxes_abs, _ = _call_run_agent_on_frame(
                        refresh_path,
                        agent_cfg,
                        send_request,
                    )
                    refresh_boxes_abs = _filter_duplicate_boxes(
                        refresh_boxes_abs,
                        boxes_abs,
                    )
                    if refresh_boxes_abs:
                        refresh_boxes_xywh = [_xyxy_to_xywh(v) for v in refresh_boxes_abs]
                        refresh_label_offset = len(boxes_abs) + 1
                        refresh_labels = list(range(refresh_label_offset, refresh_label_offset + len(refresh_boxes_abs)))
                        refresh_hints = [f"{agent_cfg.prompt}_refresh_{i + 1}" for i in range(len(refresh_boxes_abs))]
                        for lid, hint in zip(refresh_labels, refresh_hints):
                            session.id_to_labels.setdefault(lid, hint)
                        refresh_result = session.add_prompt_boxes_abs(
                            frame_idx=start_idx + int(refresh_local_idx),
                            boxes_abs=refresh_boxes_xywh,
                            box_labels=refresh_labels,
                            record_outputs=True,
                            label_hints=refresh_hints,
                        )
                        refresh_outputs = (
                            refresh_result.get("outputs", {})
                            if isinstance(refresh_result, dict)
                            else {}
                        ) or {}
                        refresh_outputs = session._map_outputs_to_global_ids_at_frame(
                            refresh_outputs,
                            frame_idx=start_idx + int(refresh_local_idx),
                        )
                        refresh_obj_ids = refresh_outputs.get("out_obj_ids", []) or []
                        refresh_masks = len(refresh_obj_ids)
                    else:
                        refresh_masks = 0
                    return 1, int(refresh_masks)

                frames_processed, masks_written, _ = run_mid_window_refresh(
                    len(frames),
                    propagation_direction,
                    seed_first_frame=seed_first_frame,
                    propagate_segment=propagate_segment,
                    refresh_mid_frame=refresh_mid_frame,
                )
                total_frames += frames_processed
                total_masks += masks_written
                get_state = getattr(session, "_get_active_session_state", None)
                previous_window_state = get_state() if callable(get_state) else None
            previous_window_frame_count = len(frames)
            previous_window_end_idx = int(end_idx)
    if total_masks == 0:
        raise RuntimeError(
            "SAM3 agent-seeded run produced no masks; consider providing an API key "
            "for the LLM or falling back to standard tracking."
        )
    return total_frames, total_masks
