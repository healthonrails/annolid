"""Reusable helpers for behavior segment labeling workflows."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from annolid.behavior.event_utils import parse_aggression_sub_event_counts
from annolid.behavior import prompting as behavior_prompting
from annolid.behavior.timeline_sampling import format_hhmmss
from annolid.core.media.video import build_segment_frame_grid, save_rgb_image

logger = logging.getLogger(__name__)
_OLLAMA_MODEL_CAPABILITY_CACHE: Dict[tuple[str, str], Optional[bool]] = {}


class BehaviorLabelRateLimitError(RuntimeError):
    """Raised when behavior segment labeling should pause after provider 429s."""


class BehaviorLabelEmptyResponseError(RuntimeError):
    """Raised when a provider repeatedly returns empty behavior-label text."""


def behavior_label_model_controls(provider: str, model: str) -> Dict[str, Any]:
    provider_text = str(provider or "").strip().lower()
    controls: Dict[str, Any] = {"max_tokens": 512}
    if provider_text in {"ollama", "local"}:
        controls["extra_body"] = {"think": False}
    return controls


def behavior_label_prompt_text(prompt: str, *, provider: str, model: str) -> str:
    return str(prompt or "").strip()


def _ollama_host_from_settings() -> str:
    try:
        from annolid.utils.llm_settings import load_llm_settings

        settings = load_llm_settings()
        host = str((settings.get("ollama", {}) or {}).get("host") or "").strip()
    except Exception:
        host = ""
    return host or "http://localhost:11434"


def _extract_ollama_vision_capability(payload: Any) -> Optional[bool]:
    if not isinstance(payload, dict):
        return None
    caps = payload.get("capabilities")
    if isinstance(caps, list):
        normalized = {str(item or "").strip().lower() for item in caps}
        if "vision" in normalized:
            return True
        if "completion" in normalized or "tools" in normalized:
            return False

    details = payload.get("details")
    detail_text = ""
    if isinstance(details, dict):
        detail_text = json.dumps(details, ensure_ascii=False).lower()
    text = " ".join(
        str(payload.get(key) or "").lower()
        for key in ("modelfile", "template", "system")
    )
    combined = f"{detail_text} {text}"
    vision_markers = ("clip", "vision", "image", "mmproj", "projector", "multimodal")
    if any(marker in combined for marker in vision_markers):
        return True
    return None


def ollama_model_supports_vision(model: str, *, host: str = "") -> Optional[bool]:
    model_text = str(model or "").strip()
    if not model_text:
        return None
    base_host = str(host or "").strip() or _ollama_host_from_settings()
    base_host = base_host.rstrip("/")
    cache_key = (base_host, model_text)
    if cache_key in _OLLAMA_MODEL_CAPABILITY_CACHE:
        return _OLLAMA_MODEL_CAPABILITY_CACHE[cache_key]

    request = urllib.request.Request(
        f"{base_host}/api/show",
        data=json.dumps({"model": model_text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        logger.debug(
            "Could not inspect Ollama model capabilities for %s: %s",
            model_text,
            exc,
        )
        return None

    supports = _extract_ollama_vision_capability(payload)
    _OLLAMA_MODEL_CAPABILITY_CACHE[cache_key] = supports
    return supports


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
    model_name = str(model or "").strip()
    model_text = model_name.lower()
    if not model_name:
        return False
    if provider_text in {"ollama", "local"}:
        supports_vision = ollama_model_supports_vision(model_name)
        if supports_vision is True:
            return False
        if supports_vision is False:
            return True
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


def behavior_label_empty_response_segment_limit(
    *,
    provider: str,
    model: str,
    routed_to_caption_profile: bool,
) -> int:
    provider_text = str(provider or "").strip().lower()
    if routed_to_caption_profile or provider_text in {"ollama", "local"}:
        return 1
    return 3


def _is_local_llm_provider(provider: str) -> bool:
    return str(provider or "").strip().lower() in {"ollama", "local"}


def behavior_label_provider_request_interval(provider: str, model: str) -> float:
    provider_text = str(provider or "").strip().lower()
    model_text = str(model or "").strip().lower()
    if not provider_text or provider_text in {"ollama", "local"}:
        return 0.0
    if provider_text == "nvidia" or "kimi" in model_text:
        return 1.0
    return 0.5


def behavior_label_attempt_plan(
    *,
    prompt: str,
    retry_prompt: str,
    system_prompt: str,
    provider: str,
    model: str,
    routed_to_caption_profile: bool,
) -> List[Dict[str, Any]]:
    controls = behavior_label_model_controls(provider, model)
    if routed_to_caption_profile:
        return [
            {
                "name": "caption_profile_with_image",
                "text": behavior_label_prompt_text(
                    prompt,
                    provider=provider,
                    model=model,
                ),
                "params": {
                    "temperature": 0.0,
                    "system_prompt": system_prompt,
                    **controls,
                },
            }
        ]

    if _is_local_llm_provider(provider):
        return [
            {
                "name": "local_vlm_with_image",
                "text": behavior_label_prompt_text(
                    prompt,
                    provider=provider,
                    model=model,
                ),
                "params": {
                    "temperature": 0.0,
                    "system_prompt": system_prompt,
                    **controls,
                },
            }
        ]

    return [
        {
            "name": "json_with_image",
            "text": behavior_label_prompt_text(
                prompt,
                provider=provider,
                model=model,
            ),
            "params": {
                "temperature": 0.0,
                "system_prompt": system_prompt,
                "response_format": {"type": "json_object"},
                **controls,
            },
        },
        {
            "name": "plain_with_image",
            "text": behavior_label_prompt_text(
                prompt,
                provider=provider,
                model=model,
            ),
            "params": {
                "temperature": 0.0,
                "system_prompt": system_prompt,
                **controls,
            },
        },
        {
            "name": "repair_with_image",
            "text": behavior_label_prompt_text(
                retry_prompt,
                provider=provider,
                model=model,
            ),
            "params": {
                "temperature": 0.0,
                "use_annolid_bot_system": False,
                **controls,
            },
        },
    ]


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
            "Do not spend tokens on hidden reasoning; answer with the JSON object only.",
        ]
    )


def behavior_grid_description_fallback(
    *,
    segment_text: str,
    label: str,
    motion_score: float,
    mean_delta: float,
) -> str:
    label_text = str(label or "selected behavior").strip()
    return (
        f"{segment_text} The selected label is {label_text!r}. "
        "The model did not provide a description, so Annolid recorded a "
        "minimal frame-grid evidence summary instead "
        f"(motion_score={motion_score:.2f}, mean_delta={mean_delta:.2f})."
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
    prior_predictions: Optional[List[Dict[str, Any]]] = None,
    stop_event=None,
    pred_worker=None,
) -> Dict[str, Any]:
    from annolid.core.models.adapters.llm_chat import LLMChatAdapter
    from annolid.core.models.base import ModelRequest

    predictions: List[Dict[str, Any]] = []
    prior_prediction_records: List[Dict[str, Any]] = []
    for pred in list(prior_predictions or []):
        if not isinstance(pred, dict):
            continue
        try:
            prior_prediction_records.append(
                normalize_behavior_segment_prediction_for_log(pred)
            )
        except Exception:
            continue
    skipped_segments = 0
    rate_limited = False
    rate_limit_error = ""
    empty_response_paused = False
    empty_response_error = ""
    empty_response_streak = 0

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
    max_consecutive_empty_segments = behavior_label_empty_response_segment_limit(
        provider=requested_provider,
        model=requested_model,
        routed_to_caption_profile=route_to_caption_profile,
    )
    primary_provider = "" if route_to_caption_profile else requested_provider
    primary_model = "" if route_to_caption_profile else requested_model
    last_request_at = 0.0
    rate_limit_backoffs = 0
    if route_to_caption_profile:
        logger.info(
            "Behavior segment routing to caption profile for image labeling "
            "because selected model appears non-vision provider=%s model=%s; "
            "provider/model overrides will not be applied to the caption profile.",
            requested_provider,
            requested_model,
        )

    with contextlib.ExitStack() as stack:
        if route_to_caption_profile:
            adapter = stack.enter_context(
                LLMChatAdapter(
                    profile="caption",
                    provider=None,
                    model=None,
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
            active_progress_value = int(((idx - 1) * 100) / total)
            try:
                if pred_worker is not None:
                    pred_worker.report_preview(
                        {
                            "index": int(idx),
                            "total": int(total),
                            "status": "building_grid",
                            "progress": int(active_progress_value),
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                        }
                    )
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
                attempts = behavior_label_attempt_plan(
                    prompt=prompt,
                    retry_prompt=retry_prompt,
                    system_prompt=system_prompt,
                    provider=primary_provider,
                    model=primary_model,
                    routed_to_caption_profile=route_to_caption_profile,
                )

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
                        if pred_worker is not None:
                            pred_worker.report_preview(
                                {
                                    "index": int(idx),
                                    "total": int(total),
                                    "status": "model_request",
                                    "attempt": attempt_name,
                                    "progress": int(active_progress_value),
                                    "start_frame": int(start_frame),
                                    "end_frame": int(end_frame),
                                }
                            )
                        logger.info(
                            "Behavior segment attempt '%s' started for frames %s-%s.",
                            attempt_name,
                            start_frame,
                            end_frame,
                        )
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
                allow_caption_rescue = (
                    not route_to_caption_profile
                    and not _is_local_llm_provider(primary_provider)
                )
                if not label and allow_caption_rescue:
                    if caption_rescue_adapter is None:
                        caption_rescue_adapter = stack.enter_context(
                            LLMChatAdapter(
                                profile="caption",
                                provider=None,
                                model=None,
                                persist=False,
                            )
                        )
                    rescue_attempt_name = "rescue_caption_profile"
                    used_attempt_name = rescue_attempt_name
                    try:
                        rescue_resp = _predict_with_rate_limit(
                            caption_rescue_adapter,
                            attempt_name=rescue_attempt_name,
                            text=behavior_label_prompt_text(
                                retry_prompt,
                                provider="",
                                model="",
                            ),
                            params={
                                "temperature": 0.0,
                                "use_annolid_bot_system": False,
                                "max_tokens": 512,
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
                    if not raw_preview and empty_attempts:
                        adjacent_predictions = [
                            pred
                            for pred in [*prior_prediction_records, *predictions]
                            if isinstance(pred, dict)
                        ]
                        prior_adjacent = None
                        for pred in reversed(adjacent_predictions):
                            try:
                                pred_end = int(pred.get("end_frame"))
                                pred_start = int(pred.get("start_frame"))
                            except Exception:
                                continue
                            if pred_end + 1 == start_frame:
                                prior_adjacent = pred
                                break
                            if end_frame + 1 == pred_start and prior_adjacent is None:
                                prior_adjacent = pred
                        segment_length = int(end_frame) - int(start_frame) + 1
                        adjacent_length = 0
                        if prior_adjacent is not None:
                            try:
                                adjacent_length = (
                                    int(prior_adjacent.get("end_frame"))
                                    - int(prior_adjacent.get("start_frame"))
                                    + 1
                                )
                            except Exception:
                                adjacent_length = 0
                        adjacent_label = str(
                            (prior_adjacent or {}).get("label")
                            or (prior_adjacent or {}).get("classification")
                            or ""
                        ).strip()
                        allowed_labels = {
                            str(candidate).strip().casefold()
                            for candidate in labels
                            if str(candidate).strip()
                        }
                        is_short_adjacent_segment = (
                            prior_adjacent is not None
                            and segment_length > 0
                            and adjacent_length > 0
                            and segment_length < max(2, int(adjacent_length * 0.5))
                            and adjacent_label.casefold() in allowed_labels
                        )
                        if not is_short_adjacent_segment:
                            raise BehaviorLabelEmptyResponseError(
                                "Provider returned empty text for all behavior-label "
                                f"attempts on frames {start_frame}-{end_frame}."
                            )
                        label = adjacent_label
                        parsed = {
                            "label": adjacent_label,
                            "classification": adjacent_label,
                            "confidence": min(
                                0.5,
                                max(
                                    0.0,
                                    float(
                                        (prior_adjacent or {}).get("confidence") or 0.0
                                    ),
                                ),
                            ),
                            "description": "",
                            "fallback_reason": "empty_response_adjacent_tail",
                        }
                        model_description = ""
                        used_attempt_name = (
                            used_attempt_name or "empty_response_adjacent_tail"
                        )
                        logger.warning(
                            "Behavior segment frames %s-%s received empty provider text; appending adjacent-tail fallback label %r from frames %s-%s.",
                            start_frame,
                            end_frame,
                            adjacent_label,
                            (prior_adjacent or {}).get("start_frame"),
                            (prior_adjacent or {}).get("end_frame"),
                        )
                    if not label:
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
                    description = behavior_grid_description_fallback(
                        segment_text=segment_text,
                        label=label,
                        motion_score=motion_score,
                        mean_delta=float(motion_summary.get("mean_delta") or 0.0),
                    )
                    fallback_reason = fallback_reason or "description_fallback"
                if not model_description:
                    model_description = str(parsed.get("description") or "").strip()
                description_source = "model" if model_description else "fallback"
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
                empty_response_streak = 0
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
            except BehaviorLabelEmptyResponseError as exc:
                skipped_segments += 1
                empty_response_streak += 1
                logger.warning(
                    "Behavior segment grid labeling received empty text for frames %s-%s (%s/%s): %s",
                    start_frame,
                    end_frame,
                    empty_response_streak,
                    max_consecutive_empty_segments,
                    exc,
                )
                if pred_worker is not None:
                    pred_worker.report_preview(
                        {
                            "index": int(idx),
                            "total": int(total),
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                            "status": "empty_response",
                            "progress": int(progress_value),
                            "error": str(exc),
                        }
                    )
                    pred_worker.report_progress(progress_value)
                if empty_response_streak >= max_consecutive_empty_segments:
                    empty_response_paused = True
                    empty_response_error = (
                        f"{exc} Behavior labeling was paused to avoid repeatedly "
                        "sending grids that receive blank HTTP-200 responses."
                    )
                    logger.warning(
                        "Behavior segment labeling paused after %s consecutive empty provider responses.",
                        empty_response_streak,
                    )
                    break
            except Exception as exc:
                skipped_segments += 1
                empty_response_streak = 0
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
        "empty_response_paused": bool(empty_response_paused),
        "empty_response_segment_limit": int(max_consecutive_empty_segments),
        "routed_to_caption_profile": bool(route_to_caption_profile),
        "error": str(rate_limit_error or empty_response_error),
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
