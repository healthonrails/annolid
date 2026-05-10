"""Reusable helpers for behavior segment labeling workflows."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from annolid.behavior.event_utils import parse_aggression_sub_event_counts
from annolid.behavior import prompting as behavior_prompting
from annolid.behavior.timeline_sampling import format_hhmmss
from annolid.core.media.video import build_segment_frame_grid, save_rgb_image

logger = logging.getLogger(__name__)


class BehaviorLabelRateLimitError(RuntimeError):
    """Raised when behavior segment labeling should pause after provider 429s."""


def infer_behavior_subject_term(
    video_path: str,
    subject: Optional[str] = None,
    *,
    explicit_subject_term: str = "",
    context_text: str = "",
) -> str:
    generic_subjects = {"agent", "subject", "subject 1"}
    explicit = str(explicit_subject_term or "").strip()
    if explicit and explicit.lower() not in generic_subjects:
        return explicit
    raw_subject = str(subject or "").strip()
    if raw_subject and raw_subject.lower() not in generic_subjects:
        return raw_subject
    text = f"{video_path or ''} {context_text or ''}".lower()
    hints = (
        ("fly", "fly"),
        ("drosophila", "fly"),
        ("mouse", "mouse"),
        ("mice", "mouse"),
        ("rat", "rat"),
        ("zebrafish", "zebrafish"),
        ("fish", "fish"),
        ("worm", "worm"),
        ("c elegans", "worm"),
        ("celegans", "worm"),
    )
    for token, term in hints:
        if token in text:
            return term
    return "animal"


def is_likely_non_vision_model(*, provider: str, model: str) -> bool:
    provider_text = str(provider or "").strip().lower()
    model_text = str(model or "").strip().lower()
    if not model_text:
        return False
    known_text_only_tokens = (
        "kimi-k2.5",
        "kimi-k2",
    )
    if any(token in model_text for token in known_text_only_tokens):
        return True
    if provider_text == "nvidia" and "moonshotai/kimi-k2.5" in model_text:
        return True
    return False


def behavior_label_provider_request_interval(provider: str, model: str) -> float:
    provider_text = str(provider or "").strip().lower()
    model_text = str(model or "").strip().lower()
    if not provider_text or provider_text in {"ollama", "local"}:
        return 0.0
    if provider_text == "nvidia" or "kimi" in model_text:
        return 1.0
    return 0.5


def behavior_label_rate_limit_backoff_seconds(exc: Exception) -> Optional[float]:
    text = str(exc or "").lower()
    if (
        "429" not in text
        and "rate limit" not in text
        and "too many request" not in text
    ):
        return None
    match = re.search(r"retry[-\s]?after[=: ]+(\d+(?:\.\d+)?)", text)
    if match:
        try:
            return max(1.0, min(60.0, float(match.group(1))))
        except Exception:
            pass
    return 8.0


def sleep_with_stop(stop_event: object, seconds: float) -> bool:
    wait_seconds = max(0.0, float(seconds or 0.0))
    if wait_seconds <= 0.0:
        return False
    if stop_event is not None:
        waiter = getattr(stop_event, "wait", None)
        if callable(waiter):
            return bool(waiter(wait_seconds))
    time.sleep(wait_seconds)
    return False


def behavior_grid_segment_label(
    *,
    start_frame: int,
    end_frame: int,
    frame_indices: List[int],
    fps: float,
) -> str:
    tiles = ", ".join(f"f{int(frame)}" for frame in frame_indices)
    fps_value = float(fps or 0.0)
    if fps_value > 0.0:
        start_time = format_hhmmss(float(start_frame) / fps_value)
        end_time = format_hhmmss(float(end_frame) / fps_value)
        time_text = f" (time {start_time}-{end_time})"
    else:
        time_text = ""
    return (
        f"segment frames {int(start_frame)}-{int(end_frame)}{time_text}. "
        "The provided image is one chronological frame grid for this segment; "
        f"read tiles left-to-right, top-to-bottom ({tiles})."
    )


def behavior_grid_output_path(
    *,
    video_path: str,
    segment_index: int,
    start_frame: int,
    end_frame: int,
) -> Path:
    source_path = Path(video_path)
    output_dir = source_path.parent / f"{source_path.stem}_behavior_segment_grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (
        f"{source_path.stem}_segment_{int(segment_index):06d}"
        f"_frames_{int(start_frame)}_{int(end_frame)}.png"
    )


def behavior_label_retry_prompt(prompt: str, labels: List[str]) -> str:
    labels_text = ", ".join(str(label) for label in labels)
    return "\n".join(
        [
            str(prompt or "").strip(),
            "",
            "Your previous response was empty or did not include a valid label.",
            f"Choose exactly one label from this list: {labels_text}.",
            'Return JSON only: {"label":"<one label exactly as listed>","classification":"<same label>","confidence":0.0,"description":"observable evidence from the frame grid"}.',
            "Do not return an empty message. Do not use a label outside the list.",
        ]
    )


def behavior_grid_system_prompt(labels: List[str]) -> str:
    labels_text = ", ".join(str(label) for label in labels if str(label).strip())
    return "\n".join(
        [
            "You are Annolid Bot.",
            "Analyze chronological frame-grid images for behavior labeling.",
            "You must ground output in visible evidence from the grid tiles.",
            f"Allowed labels: {labels_text}",
            "Classification must be exactly one allowed label.",
            "Always include a short behavior description.",
        ]
    )


def summarize_frame_grid_motion(
    grid_image: Any,
    *,
    rows: int,
    columns: int,
    tile_width: int,
    tile_height: int,
    frame_count: int,
) -> Dict[str, Any]:
    if frame_count < 2:
        return {"motion_score": 0.0, "mean_delta": 0.0, "pair_count": 0}
    try:
        import cv2  # type: ignore
    except Exception:
        return {"motion_score": 0.0, "mean_delta": 0.0, "pair_count": 0}

    tiles = []
    limit = min(int(frame_count), int(rows) * int(columns))
    for idx in range(limit):
        row = idx // int(columns)
        col = idx % int(columns)
        y0 = row * int(tile_height)
        x0 = col * int(tile_width)
        tile = grid_image[y0 : y0 + int(tile_height), x0 : x0 + int(tile_width)]
        if tile is None or getattr(tile, "size", 0) <= 0:
            continue
        crop = tile[min(24, max(0, int(tile_height) - 1)) :, :]
        if getattr(crop, "size", 0) <= 0:
            crop = tile
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        tiles.append(gray)
    if len(tiles) < 2:
        return {"motion_score": 0.0, "mean_delta": 0.0, "pair_count": 0}
    diffs = [
        float(cv2.absdiff(tiles[idx - 1], tiles[idx]).mean())
        for idx in range(1, len(tiles))
    ]
    motion_score = max(diffs) if diffs else 0.0
    mean_delta = (sum(diffs) / len(diffs)) if diffs else 0.0
    return {
        "motion_score": motion_score,
        "mean_delta": mean_delta,
        "pair_count": len(diffs),
    }


def run_behavior_segment_vlm_worker(
    *,
    video_path: str,
    intervals: List[Dict[str, Any]],
    labels: List[str],
    sample_frames_per_segment: int,
    llm_profile: str,
    llm_provider: str,
    llm_model: str,
    fps: float,
    prediction_parser: Callable[[str, List[str]], Dict[str, Any]],
    sleep_func: Callable[[object, float], bool] = sleep_with_stop,
    video_description: str = "",
    instance_count: Optional[int] = None,
    experiment_context: str = "",
    behavior_definitions: str = "",
    focus_points: str = "",
    subject_term: str = "",
    stop_event=None,
    pred_worker=None,
) -> Dict[str, Any]:
    from annolid.core.models.adapters.llm_chat import LLMChatAdapter
    from annolid.core.models.base import ModelRequest

    predictions: List[Dict[str, Any]] = []
    skipped_segments = 0
    rate_limited = False
    rate_limit_error = ""

    requested_profile = str(llm_profile or "").strip()
    requested_provider = str(llm_provider or "").strip()
    requested_model = str(llm_model or "").strip()
    route_to_caption_profile = is_likely_non_vision_model(
        provider=requested_provider,
        model=requested_model,
    )
    request_interval_seconds = behavior_label_provider_request_interval(
        requested_provider,
        requested_model,
    )
    last_request_at = 0.0
    rate_limit_backoffs = 0
    if route_to_caption_profile:
        logger.info(
            "Behavior segment routing to caption profile for image labeling "
            "because selected model appears non-vision provider=%s model=%s",
            requested_provider,
            requested_model,
        )

    with contextlib.ExitStack() as stack:
        if route_to_caption_profile:
            adapter = stack.enter_context(
                LLMChatAdapter(
                    profile="caption",
                    provider=requested_provider or None,
                    model=requested_model or None,
                    persist=False,
                )
            )
        else:
            adapter = stack.enter_context(
                LLMChatAdapter(
                    profile=requested_profile or None,
                    provider=requested_provider or None,
                    model=requested_model or None,
                    persist=False,
                )
            )
        caption_rescue_adapter: Optional[LLMChatAdapter] = None
        total = max(1, len(intervals))
        for idx, item in enumerate(intervals, start=1):
            if stop_event is not None and bool(stop_event.is_set()):
                break
            start_frame = int(item["start_frame"])
            end_frame = int(item["end_frame"])
            segment_subject_term = infer_behavior_subject_term(
                video_path,
                item.get("subject"),
                explicit_subject_term=subject_term,
                context_text=" ".join(
                    str(value or "")
                    for value in (
                        video_description,
                        experiment_context,
                        behavior_definitions,
                        focus_points,
                    )
                ),
            )
            progress_value = int((idx * 100) / total)
            try:
                grid = build_segment_frame_grid(
                    video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    sample_count=max(1, int(sample_frames_per_segment)),
                    tile_width=224,
                    annotate=True,
                    annotation_position="header",
                )
                image_path = save_rgb_image(
                    grid.image,
                    behavior_grid_output_path(
                        video_path=video_path,
                        segment_index=idx,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    ),
                )
                segment_text = behavior_grid_segment_label(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_indices=list(grid.frame_indices),
                    fps=float(fps),
                )
                prompt = behavior_prompting.build_behavior_classification_prompt(
                    behavior_labels=labels,
                    segment_label=segment_text,
                    subject_term=segment_subject_term,
                    video_description=video_description,
                    instance_count=instance_count,
                    experiment_context=experiment_context,
                    behavior_definitions=behavior_definitions,
                    focus_points=focus_points,
                )
                retry_prompt = behavior_label_retry_prompt(prompt, labels)
                system_prompt = behavior_grid_system_prompt(labels)
                if route_to_caption_profile:
                    attempts: List[Dict[str, Any]] = [
                        {
                            "name": "caption_profile_with_image",
                            "text": prompt,
                            "params": {
                                "temperature": 0.0,
                                "max_tokens": 180,
                                "system_prompt": system_prompt,
                            },
                        }
                    ]
                else:
                    attempts = [
                        {
                            "name": "json_with_image",
                            "text": prompt,
                            "params": {
                                "temperature": 0.0,
                                "max_tokens": 180,
                                "system_prompt": system_prompt,
                                "response_format": {"type": "json_object"},
                            },
                        },
                        {
                            "name": "plain_with_image",
                            "text": prompt,
                            "params": {
                                "temperature": 0.0,
                                "max_tokens": 180,
                                "system_prompt": system_prompt,
                            },
                        },
                        {
                            "name": "repair_with_image",
                            "text": retry_prompt,
                            "params": {
                                "temperature": 0.0,
                                "max_tokens": 180,
                                "use_annolid_bot_system": False,
                            },
                        },
                    ]

                def _predict_with_rate_limit(
                    local_adapter: LLMChatAdapter,
                    *,
                    attempt_name: str,
                    text: str,
                    params: Dict[str, Any],
                ):
                    nonlocal last_request_at, rate_limit_backoffs
                    if stop_event is not None and bool(stop_event.is_set()):
                        raise RuntimeError("Behavior labeling cancelled.")
                    elapsed = time.monotonic() - last_request_at
                    wait_seconds = max(0.0, float(request_interval_seconds) - elapsed)
                    if wait_seconds > 0 and sleep_func(stop_event, wait_seconds):
                        raise RuntimeError("Behavior labeling cancelled.")
                    try:
                        response = local_adapter.predict(
                            ModelRequest(
                                task="caption",
                                image_path=str(image_path),
                                text=str(text or "").strip(),
                                params=dict(params or {}),
                            )
                        )
                        last_request_at = time.monotonic()
                        return response
                    except BehaviorLabelRateLimitError:
                        raise
                    except Exception as exc:
                        last_request_at = time.monotonic()
                        backoff = behavior_label_rate_limit_backoff_seconds(exc)
                        if backoff is None:
                            raise
                        rate_limit_backoffs += 1
                        logger.warning(
                            "Behavior segment attempt '%s' hit rate limit for frames %s-%s; backing off %.1fs and pausing the run.",
                            attempt_name,
                            start_frame,
                            end_frame,
                            backoff,
                        )
                        if sleep_func(stop_event, backoff):
                            raise RuntimeError("Behavior labeling cancelled.")
                        raise BehaviorLabelRateLimitError(
                            "Provider rate limit reached while labeling "
                            f"frames {start_frame}-{end_frame}. "
                            "Behavior labeling was paused to avoid repeated "
                            "429 requests; retry later or use a provider/model "
                            "with higher capacity."
                        )

                raw = ""
                parsed: Dict[str, Any] = {}
                label = ""
                used_attempt_name = ""
                empty_attempts = 0
                model_description = ""
                for attempt in attempts:
                    attempt_name = str(attempt.get("name") or "").strip()
                    used_attempt_name = attempt_name or used_attempt_name
                    try:
                        resp = _predict_with_rate_limit(
                            adapter,
                            attempt_name=attempt_name,
                            text=str(attempt.get("text") or "").strip(),
                            params=dict(attempt.get("params") or {}),
                        )
                    except BehaviorLabelRateLimitError:
                        raise
                    except Exception as exc:
                        logger.info(
                            "Behavior segment attempt '%s' failed for frames %s-%s: %s",
                            attempt_name,
                            start_frame,
                            end_frame,
                            exc,
                        )
                        continue
                    raw = str(
                        resp.text or (resp.output or {}).get("text") or ""
                    ).strip()
                    if not raw:
                        empty_attempts += 1
                        logger.info(
                            "Behavior segment attempt '%s' returned empty text for frames %s-%s.",
                            attempt_name,
                            start_frame,
                            end_frame,
                        )
                        continue
                    parsed = prediction_parser(raw, labels)
                    label = str(parsed.get("label") or "").strip()
                    model_description = str(parsed.get("description") or "").strip()
                    if label:
                        if attempt_name == "repair_with_image":
                            parsed.setdefault("fallback_reason", "repair_prompt")
                        break
                if not label and not route_to_caption_profile:
                    if caption_rescue_adapter is None:
                        caption_rescue_adapter = stack.enter_context(
                            LLMChatAdapter(
                                profile="caption",
                                provider=requested_provider or None,
                                model=requested_model or None,
                                persist=False,
                            )
                        )
                    rescue_attempt_name = "rescue_caption_profile"
                    used_attempt_name = rescue_attempt_name
                    try:
                        rescue_resp = _predict_with_rate_limit(
                            caption_rescue_adapter,
                            attempt_name=rescue_attempt_name,
                            text=str(retry_prompt or "").strip(),
                            params={
                                "temperature": 0.0,
                                "max_tokens": 180,
                                "use_annolid_bot_system": False,
                            },
                        )
                        rescue_raw = str(
                            rescue_resp.text
                            or (rescue_resp.output or {}).get("text")
                            or ""
                        ).strip()
                        if not rescue_raw:
                            empty_attempts += 1
                            logger.info(
                                "Behavior segment attempt '%s' returned empty text for frames %s-%s.",
                                rescue_attempt_name,
                                start_frame,
                                end_frame,
                            )
                        else:
                            parsed = prediction_parser(rescue_raw, labels)
                            label = str(parsed.get("label") or "").strip()
                            model_description = str(
                                parsed.get("description") or ""
                            ).strip()
                            if label:
                                parsed.setdefault(
                                    "fallback_reason", "caption_profile_rescue"
                                )
                    except BehaviorLabelRateLimitError:
                        raise
                    except Exception as exc:
                        logger.info(
                            "Behavior segment attempt '%s' failed for frames %s-%s: %s",
                            rescue_attempt_name,
                            start_frame,
                            end_frame,
                            exc,
                        )
                motion_summary = summarize_frame_grid_motion(
                    grid.image,
                    rows=int(grid.rows),
                    columns=int(grid.columns),
                    tile_width=int(grid.tile_width),
                    tile_height=int(grid.tile_height),
                    frame_count=len(grid.frame_indices),
                )
                motion_score = float(motion_summary.get("motion_score") or 0.0)
                if not label:
                    raw_preview = raw[:240].replace("\n", " ")
                    raise RuntimeError(
                        "Model response did not contain a label from the defined "
                        f"list {labels!r}. Raw response: {raw_preview!r}"
                    )
                classification_raw = str(
                    parsed.get("classification") or parsed.get("label") or label
                ).strip()
                label_lookup = {
                    str(candidate).strip().casefold(): str(candidate).strip()
                    for candidate in labels
                    if str(candidate).strip()
                }
                parsed["classification"] = label_lookup.get(
                    classification_raw.casefold(), label
                )
                description = str(parsed.get("description") or "").strip()
                fallback_reason = str(parsed.get("fallback_reason") or "").strip()
                if not description:
                    raw_preview = raw[:240].replace("\n", " ")
                    raise RuntimeError(
                        "Model response did not include a non-empty description. "
                        f"Raw response: {raw_preview!r}"
                    )
                if not model_description:
                    model_description = description
                description_source = "model"
                parsed["description"] = description
                parsed["model_description"] = model_description
                parsed["description_source"] = description_source
                visual_evidence = {
                    "type": "frame_grid",
                    "grid_image_path": str(image_path),
                    "grid_frame_description": segment_text,
                    "subject_term": segment_subject_term,
                    "frame_indices": list(grid.frame_indices),
                    "rows": int(grid.rows),
                    "columns": int(grid.columns),
                    "tile_width": int(grid.tile_width),
                    "tile_height": int(grid.tile_height),
                    "model_attempt": used_attempt_name,
                    "empty_attempts": int(empty_attempts),
                    "description_source": description_source,
                    "request_interval_seconds": float(request_interval_seconds),
                    "rate_limit_backoffs": int(rate_limit_backoffs),
                }
                if route_to_caption_profile:
                    visual_evidence["model_routed_profile"] = "caption"
                    if requested_provider:
                        visual_evidence["requested_provider"] = requested_provider
                    if requested_model:
                        visual_evidence["requested_model"] = requested_model
                if fallback_reason:
                    visual_evidence["fallback_reason"] = fallback_reason
                if parsed.get("motion_score") is not None:
                    visual_evidence["motion_score"] = float(
                        parsed.get("motion_score") or 0.0
                    )
                elif motion_summary:
                    visual_evidence["motion_score"] = motion_score
                if motion_summary.get("mean_delta") is not None:
                    visual_evidence["mean_delta"] = float(
                        motion_summary.get("mean_delta") or 0.0
                    )
                predictions.append(
                    {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "subject": item.get("subject"),
                        "label": label,
                        "classification": str(
                            parsed.get("classification") or label
                        ).strip(),
                        "confidence": float(parsed.get("confidence") or 0.0),
                        "description": str(parsed.get("description") or "").strip(),
                        "model_description": str(
                            parsed.get("model_description") or ""
                        ).strip(),
                        "description_source": str(
                            parsed.get("description_source") or "model"
                        ).strip(),
                        "grid_image_path": str(image_path),
                        "grid_frame_description": segment_text,
                        "aggression_sub_events": parse_aggression_sub_event_counts(
                            parsed.get("aggression_sub_events")
                            or parsed.get("sub_events")
                            or parsed.get("subevents")
                        ),
                        "visual_evidence": visual_evidence,
                    }
                )
                if pred_worker is not None:
                    pred_worker.report_preview(
                        {
                            "index": int(idx),
                            "total": int(total),
                            "status": "labeled",
                            "progress": int(progress_value),
                            "prediction": dict(predictions[-1]),
                        }
                    )
                    pred_worker.report_progress(progress_value)
            except BehaviorLabelRateLimitError as exc:
                skipped_segments += 1
                rate_limited = True
                rate_limit_error = str(exc)
                logger.warning(
                    "Behavior segment grid labeling paused at frames %s-%s: %s",
                    start_frame,
                    end_frame,
                    exc,
                )
                if pred_worker is not None:
                    pred_worker.report_preview(
                        {
                            "index": int(idx),
                            "total": int(total),
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                            "status": "rate_limited",
                            "progress": int(progress_value),
                            "error": str(exc),
                        }
                    )
                    pred_worker.report_progress(progress_value)
                break
            except Exception as exc:
                skipped_segments += 1
                logger.warning(
                    "Behavior segment grid labeling skipped frames %s-%s: %s",
                    start_frame,
                    end_frame,
                    exc,
                )
                if pred_worker is not None:
                    pred_worker.report_preview(
                        {
                            "index": int(idx),
                            "total": int(total),
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                            "status": "skipped",
                            "progress": int(progress_value),
                            "error": str(exc),
                        }
                    )
                    pred_worker.report_progress(progress_value)

    return {
        "predictions": predictions,
        "skipped_segments": int(skipped_segments),
        "processed_segments": int(len(intervals)),
        "cancelled": bool(stop_event is not None and stop_event.is_set()),
        "rate_limited": bool(rate_limited),
        "error": str(rate_limit_error),
    }


def behavior_segment_labeling_log_path(video_path: str) -> Path:
    path = Path(str(video_path or ""))
    return path.with_name(f"{path.stem}_behavior_segment_labels.json")


def normalize_behavior_segment_prediction_for_log(
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    label = str(
        prediction.get("label") or prediction.get("classification") or ""
    ).strip()
    classification = str(
        prediction.get("classification") or prediction.get("label") or label
    ).strip()
    normalized: Dict[str, Any] = {
        "start_frame": int(prediction.get("start_frame") or 0),
        "end_frame": int(
            prediction.get("end_frame") or prediction.get("start_frame") or 0
        ),
        "subject": prediction.get("subject"),
        "label": label,
        "classification": classification,
        "confidence": float(prediction.get("confidence") or 0.0),
        "description": str(prediction.get("description") or "").strip(),
        "model_description": str(prediction.get("model_description") or "").strip(),
        "description_source": str(prediction.get("description_source") or "").strip(),
        "grid_image_path": str(prediction.get("grid_image_path") or "").strip(),
        "grid_frame_description": str(
            prediction.get("grid_frame_description") or ""
        ).strip(),
        "aggression_sub_events": parse_aggression_sub_event_counts(
            prediction.get("aggression_sub_events")
            or prediction.get("sub_events")
            or prediction.get("subevents")
        ),
    }
    visual_evidence = prediction.get("visual_evidence")
    if isinstance(visual_evidence, dict):
        normalized["visual_evidence"] = dict(visual_evidence)
        if not normalized["grid_image_path"]:
            normalized["grid_image_path"] = str(
                visual_evidence.get("grid_image_path") or ""
            ).strip()
        if not normalized["grid_frame_description"]:
            normalized["grid_frame_description"] = str(
                visual_evidence.get("grid_frame_description") or ""
            ).strip()
    return normalized


def load_resumable_behavior_segment_predictions(
    video_path: str,
    *,
    labels: List[str],
    segment_frames: int,
    segment_seconds: float,
    sample_frames_per_segment: int,
) -> Dict[str, Any]:
    output_path = behavior_segment_labeling_log_path(video_path)
    if not output_path.exists():
        return {"ok": True, "path": str(output_path), "predictions": []}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Could not load existing behavior segment log %s for resume: %s",
            output_path,
            exc,
        )
        return {"ok": False, "path": str(output_path), "predictions": []}
    if not isinstance(payload, dict):
        return {"ok": False, "path": str(output_path), "predictions": []}

    allowed = {str(label).strip().casefold() for label in labels if str(label).strip()}
    predictions: List[Dict[str, Any]] = []
    for raw in list(payload.get("predictions") or []):
        if not isinstance(raw, dict):
            continue
        try:
            normalized = normalize_behavior_segment_prediction_for_log(raw)
        except Exception:
            continue
        label = str(normalized.get("label") or "").strip()
        if allowed and label.casefold() not in allowed:
            continue
        try:
            start_frame = int(normalized.get("start_frame"))
            end_frame = int(normalized.get("end_frame"))
        except Exception:
            continue
        if start_frame < 0 or end_frame < start_frame:
            continue
        normalized["start_frame"] = start_frame
        normalized["end_frame"] = end_frame
        predictions.append(normalized)

    if predictions:
        expected_segment_frames = int(payload.get("segment_frames") or 0)
        expected_sample_frames = int(payload.get("sample_frames_per_segment") or 0)
        expected_segment_seconds = float(payload.get("segment_seconds") or 0.0)
        if expected_segment_frames and expected_segment_frames != int(segment_frames):
            logger.info(
                "Resuming behavior labels from %s with segment_frames mismatch old=%s new=%s; matching frame ranges will still be skipped.",
                output_path,
                expected_segment_frames,
                segment_frames,
            )
        if expected_sample_frames and expected_sample_frames != int(
            sample_frames_per_segment
        ):
            logger.info(
                "Resuming behavior labels from %s with frames-per-grid mismatch old=%s new=%s.",
                output_path,
                expected_sample_frames,
                sample_frames_per_segment,
            )
        if (
            expected_segment_seconds > 0
            and float(segment_seconds) > 0
            and abs(expected_segment_seconds - float(segment_seconds)) > 1e-6
        ):
            logger.info(
                "Resuming behavior labels from %s with segment_seconds mismatch old=%s new=%s.",
                output_path,
                expected_segment_seconds,
                segment_seconds,
            )

    return {
        "ok": True,
        "path": str(output_path),
        "predictions": predictions,
    }
