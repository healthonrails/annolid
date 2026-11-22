"""VLM behavior segment labeling and behavior catalog helpers for AIChatWidget."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets

from annolid.behavior.event_utils import (
    aggregate_aggression_bout_summary,
    aggression_sub_event_schema,
    infer_aggression_sub_event_counts_from_text,
    parse_aggression_sub_event_counts,
)
from annolid.behavior.labels import (
    NO_BEHAVIOR_LABEL,
    behavior_label_key,
    canonicalize_behavior_label,
    is_no_behavior_label,
    normalize_behavior_label_list,
    text_indicates_no_behavior,
)
from annolid.behavior.segment_labeling import (
    behavior_grid_output_path,
    behavior_grid_segment_label,
    behavior_grid_system_prompt,
    behavior_label_provider_request_interval,
    behavior_label_rate_limit_backoff_seconds,
    behavior_label_retry_prompt,
    behavior_segment_labeling_log_path,
    behavior_segment_resume_key,
    infer_behavior_subject_term,
    is_likely_non_vision_model,
    load_resumable_behavior_segment_predictions,
    normalize_behavior_segment_prediction_for_log,
    run_behavior_segment_vlm_worker,
    summarize_frame_grid_motion,
)
from annolid.gui.widgets.behavior_slash_dialog import BehaviorSlashDialog
from annolid.gui.workers import FlexibleWorker

logger = logging.getLogger(__name__)


def _widget_module_attr(name: str, default: Any) -> Any:
    mod = sys.modules.get("annolid.gui.widgets.ai_chat_widget")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default


class AIChatBehaviorMixin:
    """Mixin encapsulating VLM behavior labeling and catalog operations."""

    @staticmethod
    def _normalize_behavior_labels(labels: List[str]) -> List[str]:
        return normalize_behavior_label_list(labels)

    @staticmethod
    def _is_placeholder_behavior_label(label: str) -> bool:
        value = str(label or "").strip().lower()
        if not value:
            return True
        key = behavior_label_key(value)
        if re.fullmatch(r"behavior[_\-\s]?\d+", value):
            return True
        if re.fullmatch(r"behavior_\d+", key):
            return True
        video_ext_tokens = {"mp4", "avi", "mov", "mkv", "m4v", "wmv", "flv"}
        if (
            value.startswith(("behavior in ", "behaviour in "))
            or key.startswith(("behavior_in_", "behaviour_in_"))
        ) and (
            re.search(r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b", value) is not None
            or bool(video_ext_tokens.intersection(key.split("_")))
        ):
            return True
        return value in {"behavior", "behaviour", "label", "placeholder"}

    def _labels_from_schema_or_flags(self) -> List[str]:
        host = self.host_window_widget or self.window()
        labels: List[str] = []
        schema = getattr(host, "project_schema", None)
        behaviors = getattr(schema, "behaviors", None)
        if isinstance(behaviors, list):
            for behavior in behaviors:
                name = str(getattr(behavior, "name", "") or "").strip()
                if name:
                    labels.append(name.replace("_", " "))
                    continue
                code = str(getattr(behavior, "code", "") or "").strip()
                if code:
                    labels.append(code.replace("_", " "))
        if not labels:
            flags = getattr(host, "flags", None)
            if isinstance(flags, dict):
                labels.extend([str(k).strip() for k in flags.keys() if str(k).strip()])
        flag_widget = getattr(host, "flag_widget", None)
        if flag_widget is not None:
            try:
                existing_flag_names = getattr(
                    flag_widget, "_get_existing_flag_names", None
                )
                if callable(existing_flag_names):
                    labels.extend(
                        [
                            str(name).strip()
                            for name in dict(existing_flag_names() or {}).keys()
                            if str(name).strip()
                        ]
                    )
            except Exception:
                pass
        flags_controller = getattr(host, "flags_controller", None)
        if flags_controller is not None:
            try:
                pinned = getattr(flags_controller, "pinned_flags", None)
                if isinstance(pinned, dict):
                    labels.extend(
                        [
                            str(name).strip()
                            for name in pinned.keys()
                            if str(name).strip()
                        ]
                    )
            except Exception:
                pass
        behavior_controller = getattr(host, "behavior_controller", None)
        if behavior_controller is not None:
            try:
                timeline_behaviors = list(
                    getattr(behavior_controller, "behavior_names", lambda: set())()
                    if callable(getattr(behavior_controller, "behavior_names", None))
                    else getattr(behavior_controller, "behavior_names", set())
                )
            except Exception:
                timeline_behaviors = []
            labels.extend(
                [str(name).strip() for name in timeline_behaviors if str(name).strip()]
            )
        normalized = self._normalize_behavior_labels(labels)
        filtered = [
            name for name in normalized if not self._is_placeholder_behavior_label(name)
        ]
        return filtered

    def _resolve_segment_label_candidates(
        self,
        explicit_labels: List[str],
        *,
        use_defined_behavior_list: bool,
    ) -> List[str]:
        defined_labels = self._labels_from_schema_or_flags()
        normalized_explicit = self._normalize_behavior_labels(explicit_labels)
        if not use_defined_behavior_list:
            return normalized_explicit or defined_labels
        if not defined_labels:
            return normalized_explicit
        if not normalized_explicit:
            return defined_labels
        return normalized_explicit

    @staticmethod
    def _behavior_ranges_from_events(events: List[object]) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, List[object]] = {}
        for event in events:
            behavior = str(getattr(event, "behavior", "") or "").strip()
            if not behavior:
                continue
            subject = getattr(event, "subject", None)
            key = (behavior, subject)
            grouped.setdefault(key, []).append(event)
        ranges: List[Dict[str, Any]] = []
        for (behavior, subject), group in grouped.items():
            group_sorted = sorted(
                group,
                key=lambda evt: (
                    int(getattr(evt, "frame", 0)),
                    str(getattr(evt, "event", "")),
                ),
            )
            open_frame: Optional[int] = None
            for evt in group_sorted:
                label = str(getattr(evt, "event", "") or "").strip().lower()
                frame = int(getattr(evt, "frame", 0))
                if label == "start":
                    open_frame = frame
                elif label == "end":
                    start = frame if open_frame is None else open_frame
                    end = frame
                    if end < start:
                        start, end = end, start
                    ranges.append(
                        {
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "subject": subject,
                            "seed_behavior": behavior,
                        }
                    )
                    open_frame = None
            if open_frame is not None:
                ranges.append(
                    {
                        "start_frame": int(open_frame),
                        "end_frame": int(open_frame),
                        "subject": subject,
                        "seed_behavior": behavior,
                    }
                )
        return sorted(
            ranges, key=lambda item: (int(item["start_frame"]), int(item["end_frame"]))
        )

    @staticmethod
    def _extract_label_from_model_text(
        text: str, labels: List[str]
    ) -> tuple[str, float]:
        prediction = AIChatBehaviorMixin._extract_prediction_from_model_text(
            text, labels
        )
        return str(prediction.get("label") or ""), float(
            prediction.get("confidence") or 0.0
        )

    @staticmethod
    def _extract_prediction_from_model_text(
        text: str,
        labels: List[str],
    ) -> Dict[str, Any]:
        raw = str(text or "").strip()
        labels = normalize_behavior_label_list(labels)
        if not labels:
            return {"label": "", "confidence": 0.0}

        # Prefer structured JSON responses if present, including fenced blocks.
        json_candidate = ""
        if raw.startswith("{"):
            json_candidate = raw
        else:
            fence_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if fence_match:
                json_candidate = str(fence_match.group(1) or "").strip()
            else:
                brace_match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
                if brace_match:
                    json_candidate = str(brace_match.group(1) or "").strip()
        if json_candidate:
            try:
                payload = json.loads(json_candidate)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                label_val = str(
                    payload.get("label")
                    or payload.get("behavior")
                    or payload.get("classification")
                    or payload.get("prediction")
                    or ""
                ).strip()
                conf_val = payload.get("confidence", 0.0)
                try:
                    conf = float(conf_val)
                except Exception:
                    conf = 0.0
                description = str(payload.get("description") or "").strip()
                if label_val:
                    canonical_label = canonicalize_behavior_label(label_val, labels)
                    if canonical_label == NO_BEHAVIOR_LABEL:
                        parsed = {
                            "label": "",
                            "classification": NO_BEHAVIOR_LABEL,
                            "confidence": max(0.0, min(1.0, conf)),
                            "no_behavior": True,
                            "model_label": label_val,
                        }
                        if description:
                            parsed["description"] = description
                        return parsed
                    if canonical_label:
                        parsed_sub_events = parse_aggression_sub_event_counts(
                            payload.get("sub_events")
                            or payload.get("subevents")
                            or payload.get("aggression_sub_events")
                        )
                        if not parsed_sub_events:
                            parsed_sub_events = (
                                infer_aggression_sub_event_counts_from_text(
                                    str(payload.get("description") or "")
                                )
                            )
                        parsed: Dict[str, Any] = {
                            "label": canonical_label,
                            "classification": canonical_label,
                            "confidence": max(0.0, min(1.0, conf)),
                        }
                        if description:
                            parsed["description"] = description
                        if parsed_sub_events:
                            parsed["aggression_sub_events"] = parsed_sub_events
                        return parsed
                    parsed = {
                        "label": "",
                        "classification": "",
                        "confidence": max(0.0, min(1.0, conf)),
                        "unmatched_label": label_val,
                        "unmatched_text": raw,
                    }
                    if description:
                        parsed["description"] = description
                    return parsed
                no_behavior_text = " ".join(
                    str(payload.get(key) or "").strip()
                    for key in ("description", "reason", "evidence", "explanation")
                    if str(payload.get(key) or "").strip()
                )
                if text_indicates_no_behavior(no_behavior_text):
                    parsed = {
                        "label": "",
                        "classification": NO_BEHAVIOR_LABEL,
                        "confidence": max(0.0, min(1.0, conf)),
                        "no_behavior": True,
                        "model_label": "",
                    }
                    if description:
                        parsed["description"] = description
                    return parsed

        def _label_mentioned(candidate: str, source: str) -> bool:
            normalized_candidate = re.sub(
                r"[^a-z0-9]+", " ", str(candidate or "").lower()
            ).strip()
            normalized_source = re.sub(
                r"[^a-z0-9]+", " ", str(source or "").lower()
            ).strip()
            if not normalized_candidate or not normalized_source:
                return False
            return bool(
                re.search(
                    rf"\b{re.escape(normalized_candidate)}\b",
                    normalized_source,
                )
            )

        explicit_label = ""
        explicit_match = re.search(
            r"(?:^|[\n\r])\s*"
            r"(?:label|classification|behavior|behaviour|prediction)"
            r"\s*(?::|=|is)\s*[\"'`]*(?P<label>[^\n\r,;\"'`{}]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if explicit_match is not None:
            explicit_label = str(explicit_match.group("label") or "").strip()
        else:
            terminal_match = re.search(
                r"(?:^|[\n\r])\s*[\"'`]*(?P<label>[A-Za-z][A-Za-z0-9 _-]{1,80})"
                r"[\"'`]*\s*$",
                raw,
                flags=re.IGNORECASE,
            )
            if terminal_match is not None:
                explicit_label = str(terminal_match.group("label") or "").strip()

        if explicit_label:
            canonical_label = canonicalize_behavior_label(explicit_label, labels)
            if canonical_label == NO_BEHAVIOR_LABEL:
                return {
                    "label": "",
                    "classification": NO_BEHAVIOR_LABEL,
                    "confidence": 0.6,
                    "description": raw,
                    "no_behavior": True,
                    "model_label": explicit_label,
                }
            if canonical_label:
                return {
                    "label": canonical_label,
                    "classification": canonical_label,
                    "confidence": 0.6,
                    "description": raw,
                    "aggression_sub_events": infer_aggression_sub_event_counts_from_text(
                        raw
                    ),
                }

        mention_fallback_blockers = (
            "allowed label",
            "available label",
            "defined behavior list",
            "choose exactly",
            "thinking process",
            "previous response",
        )
        allow_short_mention_fallback = len(raw) <= 240 and not any(
            token in raw.lower() for token in mention_fallback_blockers
        )
        for candidate in labels if allow_short_mention_fallback else []:
            if _label_mentioned(candidate, raw):
                return {
                    "label": candidate,
                    "classification": candidate,
                    "confidence": 0.6,
                    "description": raw,
                    "aggression_sub_events": infer_aggression_sub_event_counts_from_text(
                        raw
                    ),
                }
        if is_no_behavior_label(raw):
            return {
                "label": "",
                "classification": NO_BEHAVIOR_LABEL,
                "confidence": 0.6,
                "description": raw,
                "no_behavior": True,
                "model_label": raw,
            }
        return {
            "label": "",
            "confidence": 0.0,
            "unmatched_text": raw,
            "aggression_sub_events": infer_aggression_sub_event_counts_from_text(raw),
        }

    @staticmethod
    def _uniform_segment_frame_indices(
        start_frame: int, end_frame: int, sample_count: int
    ) -> List[int]:
        start = int(start_frame)
        end = int(end_frame)
        if end < start:
            start, end = end, start
        if start == end:
            return [start]
        count = max(1, int(sample_count))
        span = end - start + 1
        count = min(count, span)
        if count == span:
            return list(range(start, end + 1))
        step = (end - start) / float(max(1, count - 1))
        frames = [int(round(start + (idx * step))) for idx in range(count)]
        return sorted(set(max(start, min(end, frame)) for frame in frames))

    _infer_behavior_subject_term = staticmethod(infer_behavior_subject_term)

    _behavior_grid_segment_label = staticmethod(behavior_grid_segment_label)

    _behavior_grid_output_path = staticmethod(behavior_grid_output_path)

    _behavior_label_retry_prompt = staticmethod(behavior_label_retry_prompt)

    _behavior_grid_system_prompt = staticmethod(behavior_grid_system_prompt)

    _summarize_frame_grid_motion = staticmethod(summarize_frame_grid_motion)

    @staticmethod
    def _select_label_by_motion_affinity(
        *,
        labels: List[str],
        motion_score: float,
    ) -> Optional[str]:
        if not labels:
            return None

        active_hints = (
            "walk",
            "run",
            "move",
            "rear",
            "jump",
            "climb",
            "chase",
            "aggress",
            "attack",
            "explor",
            "travel",
            "locomot",
        )
        passive_hints = (
            "rest",
            "sleep",
            "still",
            "idle",
            "freeze",
            "groom",
            "sit",
            "eat",
            "drink",
            "sniff",
        )

        def score(label: str) -> float:
            text = str(label).strip().lower()
            active = float(sum(1 for token in active_hints if token in text))
            passive = float(sum(1 for token in passive_hints if token in text))
            return active - passive

        ranked = sorted(
            (str(label), score(str(label))) for label in labels if str(label).strip()
        )
        if not ranked:
            return None
        motion_threshold = 2.0
        if motion_score >= motion_threshold:
            return max(ranked, key=lambda item: item[1])[0]
        return min(ranked, key=lambda item: item[1])[0]

    @staticmethod
    def _build_grid_description_fallback(
        *,
        segment_text: str,
        motion_score: float,
        mean_delta: float,
    ) -> str:
        return (
            f"{segment_text} Observed temporal change across tiles "
            f"(motion_score={motion_score:.2f}, mean_delta={mean_delta:.2f})."
        )

    _is_likely_non_vision_model = staticmethod(is_likely_non_vision_model)

    _behavior_label_provider_request_interval = staticmethod(
        behavior_label_provider_request_interval
    )

    _behavior_label_rate_limit_backoff_seconds = staticmethod(
        behavior_label_rate_limit_backoff_seconds
    )

    @staticmethod
    def _sleep_with_stop(stop_event: object, seconds: float) -> bool:
        wait_seconds = max(0.0, float(seconds or 0.0))
        if wait_seconds <= 0.0:
            return False
        if stop_event is not None:
            waiter = getattr(stop_event, "wait", None)
            if callable(waiter):
                return bool(waiter(wait_seconds))
        time.sleep(wait_seconds)
        return False

    def _save_behavior_timestamps_csv(self, host: object) -> Dict[str, Any]:
        behavior_controller = getattr(host, "behavior_controller", None)
        if behavior_controller is None:
            return {"ok": False, "error": "Behavior controller unavailable."}
        video_file = str(getattr(host, "video_file", "") or "").strip()
        if not video_file:
            return {"ok": False, "error": "No video file is loaded."}

        video_path = Path(video_file)
        output_path = video_path.with_name(f"{video_path.stem}_timestamps.csv")
        estimator = getattr(host, "_estimate_recording_time", None)
        if not callable(estimator):
            estimator = self._behavior_label_timestamp_provider(host)
        rows = behavior_controller.export_rows(
            timestamp_fallback=lambda evt: estimator(evt.frame)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["Trial time", "Recording time", "Subject", "Behavior", "Event"]
            )
            for row in rows:
                writer.writerow(row)
        return {"ok": True, "path": str(output_path), "rows": len(rows)}

    def _save_behavior_segment_labeling_log(
        self,
        host: object,
        *,
        mode: str,
        labels_used: List[str],
        segment_frames: int,
        segment_seconds: float,
        sample_frames_per_segment: int,
        evaluated_segments: int,
        skipped_segments: int,
        predictions: List[Dict[str, Any]],
        skipped_predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        video_file = str(getattr(host, "video_file", "") or "").strip()
        if not video_file:
            return {"ok": False, "error": "No video file is loaded."}
        try:
            video_path = Path(video_file)
            output_path = video_path.with_name(
                f"{video_path.stem}_behavior_segment_labels.json"
            )
            normalized_predictions = [
                normalize_behavior_segment_prediction_for_log(pred)
                for pred in predictions
            ]
            normalized_skipped_predictions = [
                normalize_behavior_segment_prediction_for_log(pred)
                for pred in list(skipped_predictions or [])
            ]
            aggression_summary = aggregate_aggression_bout_summary(
                normalized_predictions
            )
            payload: Dict[str, Any] = {
                "video_path": str(video_path),
                "mode": str(mode or "uniform"),
                "labels_used": list(labels_used),
                "segment_frames": int(segment_frames),
                "segment_seconds": float(segment_seconds),
                "sample_frames_per_segment": int(sample_frames_per_segment),
                "visual_input_mode": "frame_grid",
                "evaluated_segments": int(evaluated_segments),
                "labeled_segments": int(len(normalized_predictions)),
                "skipped_segments": int(skipped_segments),
                "predictions": list(normalized_predictions),
                "skipped_predictions": list(normalized_skipped_predictions),
                "event_schema": aggression_sub_event_schema(),
                "aggression_bout_summary": aggression_summary,
                "generated_at": datetime.now().isoformat(),
            }
            output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return {
                "ok": True,
                "path": str(output_path),
                "rows": len(normalized_predictions),
                "skipped_rows": len(normalized_skipped_predictions),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    _behavior_segment_labeling_log_path = staticmethod(
        behavior_segment_labeling_log_path
    )

    _load_resumable_behavior_segment_predictions = staticmethod(
        load_resumable_behavior_segment_predictions
    )

    _normalize_behavior_segment_prediction_for_log = staticmethod(
        normalize_behavior_segment_prediction_for_log
    )

    def _update_behavior_label_skipped_overlay_records(
        self, context: Dict[str, Any]
    ) -> None:
        """Expose skipped behavior-label records to the canvas overlay mixin."""
        host = context.get("host")
        if host is None:
            return
        records: List[Dict[str, Any]] = []
        for pred in list(context.get("skipped_predictions") or []):
            if not isinstance(pred, dict):
                continue
            try:
                records.append(normalize_behavior_segment_prediction_for_log(pred))
            except Exception:
                continue
        video_file = str(getattr(host, "video_file", "") or "").strip()
        mtime_ns = None
        if video_file:
            try:
                log_path = behavior_segment_labeling_log_path(video_file)
                mtime_ns = log_path.stat().st_mtime_ns if log_path.exists() else None
            except Exception:
                mtime_ns = None
        setattr(host, "_behavior_label_skipped_overlay_records", records)
        setattr(host, "_behavior_label_skipped_overlay_video", video_file)
        setattr(host, "_behavior_label_skipped_overlay_mtime_ns", mtime_ns)

    def _run_behavior_segment_vlm_worker(
        self,
        *,
        video_path: str,
        intervals: List[Dict[str, Any]],
        labels: List[str],
        sample_frames_per_segment: int,
        llm_profile: str,
        llm_provider: str,
        llm_model: str,
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
        return run_behavior_segment_vlm_worker(
            video_path=video_path,
            intervals=intervals,
            labels=labels,
            sample_frames_per_segment=sample_frames_per_segment,
            llm_profile=llm_profile,
            llm_provider=llm_provider,
            llm_model=llm_model,
            fps=self._infer_video_fps(video_path),
            prediction_parser=self._extract_prediction_from_model_text,
            sleep_func=self._sleep_with_stop,
            video_description=video_description,
            instance_count=instance_count,
            experiment_context=experiment_context,
            behavior_definitions=behavior_definitions,
            focus_points=focus_points,
            subject_term=subject_term,
            prior_predictions=prior_predictions,
            stop_event=stop_event,
            pred_worker=pred_worker,
        )

    def _infer_video_fps(self, video_path: str) -> float:
        """Resolve a stable FPS value for segment-label prompts and logs."""
        host_fps = getattr(getattr(self, "host_window_widget", None), "fps", None)
        try:
            fps = float(host_fps or 0.0)
        except Exception:
            fps = 0.0
        if fps > 0.0:
            return fps
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = max(0.0, float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
            cap.release()
        except Exception:
            fps = 0.0
        return fps

    def _clear_behavior_label_run_context(self) -> None:
        self._behavior_label_run_context = {}

    def _clear_behavior_label_thread_refs(self) -> None:
        self._behavior_label_worker = None
        self._behavior_label_thread = None

    def _stop_behavior_label_worker(self, *, wait_ms: int = 3000) -> None:
        worker = getattr(self, "_behavior_label_worker", None)
        thread = getattr(self, "_behavior_label_thread", None)
        if worker is not None:
            try:
                worker._stop()
            except RuntimeError:
                pass
        if thread is not None and thread.isRunning():
            if not thread.wait(max(0, int(wait_ms))):
                logger.warning(
                    "Behavior labeling thread did not stop within %sms.", wait_ms
                )
                raise TimeoutError("Behavior labeling thread is still running.")
        if thread is not None and not thread.isRunning():
            self._clear_behavior_label_thread_refs()

    def _behavior_label_timestamp_provider(self, host: object):
        def _timestamp_provider(frame: int) -> Optional[float]:
            local_fps = getattr(host, "fps", None)
            if local_fps is None or float(local_fps) <= 0:
                return None
            return float(frame) / float(local_fps)

        return _timestamp_provider

    def _refresh_behavior_panels(self, host: object) -> None:
        refresh_log = getattr(host, "_refresh_behavior_log", None)
        if callable(refresh_log):
            refresh_log()
        timeline_panel = getattr(host, "timeline_panel", None)
        refresh_timeline = getattr(
            timeline_panel, "refresh_from_behavior_controller", None
        )
        if callable(refresh_timeline):
            refresh_timeline()

    def _commit_behavior_label_prediction(
        self,
        context: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        host = context.get("host")
        behavior_controller = context.get("behavior_controller")
        if host is None or behavior_controller is None:
            return None

        label = str(prediction.get("label") or "").strip()
        if not label:
            return None

        start_frame = int(prediction.get("start_frame") or 0)
        end_frame = int(prediction.get("end_frame") or start_frame)
        subject_value = prediction.get("subject")
        default_subject = context.get("default_subject")
        resolved_subject = subject_value or default_subject
        timestamp_provider = context.get("timestamp_provider")
        behavior_controller.create_interval(
            behavior=label,
            start_frame=start_frame,
            end_frame=end_frame,
            subject=resolved_subject,
            timestamp_provider=timestamp_provider,
        )

        normalized_prediction = {
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "subject": resolved_subject,
            "label": label,
            "classification": label,
            "confidence": float(prediction.get("confidence") or 0.0),
            "description": str(prediction.get("description") or "").strip(),
            "model_description": str(prediction.get("model_description") or "").strip(),
            "description_source": str(
                prediction.get("description_source") or ""
            ).strip(),
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
            normalized_prediction["visual_evidence"] = dict(visual_evidence)
            if not normalized_prediction["grid_image_path"]:
                normalized_prediction["grid_image_path"] = str(
                    visual_evidence.get("grid_image_path") or ""
                ).strip()
            if not normalized_prediction["grid_frame_description"]:
                normalized_prediction["grid_frame_description"] = str(
                    visual_evidence.get("grid_frame_description") or ""
                ).strip()
        context_predictions = list(context.get("predictions") or [])
        context_predictions.append(normalized_prediction)
        context["predictions"] = context_predictions

        self._refresh_behavior_panels(host)
        return normalized_prediction

    def _commit_behavior_label_skipped_prediction(
        self,
        context: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            normalized_prediction = normalize_behavior_segment_prediction_for_log(
                prediction
            )
        except Exception:
            return None
        normalized_prediction["subject"] = normalized_prediction.get(
            "subject"
        ) or context.get("default_subject")
        context_predictions = list(context.get("skipped_predictions") or [])
        existing_ranges = {
            behavior_segment_resume_key(pred)
            for pred in context_predictions
            if isinstance(pred, dict)
        }
        pred_range = behavior_segment_resume_key(normalized_prediction)
        if pred_range not in existing_ranges:
            context_predictions.append(normalized_prediction)
            context["skipped_predictions"] = context_predictions
            behavior_controller = context.get("behavior_controller")
            add_mark = getattr(behavior_controller, "add_generic_mark", None)
            if callable(add_mark):
                label = str(normalized_prediction.get("label") or "").strip()
                mark_type = (
                    "behavior_no_behavior"
                    if label == "no_behavior"
                    else "behavior_unclassified"
                )
                color = "#808080" if label == "no_behavior" else "#ff9800"
                try:
                    add_mark(
                        int(normalized_prediction.get("start_frame") or 0),
                        mark_type=mark_type,
                        color=color,
                    )
                except Exception:
                    pass
        self._update_behavior_label_skipped_overlay_records(context)
        host = context.get("host")
        refresh_overlay = getattr(host, "_refresh_behavior_overlay", None)
        if callable(refresh_overlay):
            try:
                refresh_overlay(getattr(host, "frame_number", pred_range[0]))
            except Exception:
                pass
        return normalized_prediction

    def _save_behavior_segment_progress(
        self, *, force_timestamps: bool = False
    ) -> Dict[str, Any]:
        context = dict(self._behavior_label_run_context or {})
        if not context:
            return {"ok": False, "error": "Behavior labeling run context is missing."}

        host = context.get("host")
        if host is None:
            return {"ok": False, "error": "Behavior labeling host is unavailable."}

        predictions = list(context.get("predictions") or [])
        skipped_predictions = list(context.get("skipped_predictions") or [])
        timestamp_result: Dict[str, Any] = {}
        processed = int(context.get("processed_segments") or 0)
        if force_timestamps or (processed > 0 and processed % 10 == 0):
            timestamp_result = self._save_behavior_timestamps_csv(host)

        behavior_log_result = self._save_behavior_segment_labeling_log(
            host,
            mode=str(context.get("mode") or "uniform"),
            labels_used=list(context.get("labels") or []),
            segment_frames=int(context.get("segment_frames") or 1),
            segment_seconds=float(context.get("segment_seconds") or 0.0),
            sample_frames_per_segment=int(
                context.get("sample_frames_per_segment") or 4
            ),
            evaluated_segments=int(context.get("evaluated_segments") or 0),
            skipped_segments=int(context.get("skipped_segments") or 0),
            predictions=predictions,
            skipped_predictions=skipped_predictions,
        )
        return {
            "ok": bool(behavior_log_result.get("ok", False)),
            "timestamp_result": timestamp_result,
            "behavior_log_result": behavior_log_result,
        }

    def _behavior_catalog_host(self) -> object:
        return self.host_window_widget or self.window()

    def _behavior_catalog_action(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        host = self._behavior_catalog_host()
        action_name = str(action or "").strip().lower()
        if not action_name:
            return {"ok": False, "error": "Behavior catalog action is required."}
        save = bool(kwargs.pop("save", True))

        if action_name == "list":
            getter = getattr(host, "list_behavior_catalog", None)
            if callable(getter):
                return dict(getter())
            schema = getattr(host, "project_schema", None)
            if schema is None:
                return {"ok": False, "error": "No project schema is loaded."}
            return {
                "ok": True,
                "count": len(schema.behaviors),
                "behavior_catalog": [
                    {
                        "code": behavior.code,
                        "name": behavior.name,
                        "description": behavior.description or "",
                        "category_id": behavior.category_id or "",
                        "modifier_ids": list(behavior.modifier_ids or []),
                        "key_binding": behavior.key_binding or "",
                        "is_state": bool(behavior.is_state),
                        "exclusive_with": list(behavior.exclusive_with or []),
                    }
                    for behavior in schema.behaviors
                ],
            }

        if action_name == "save":
            saver = getattr(host, "save_behavior_catalog", None)
            if callable(saver):
                return dict(saver())
            return {"ok": False, "error": "Behavior catalog saver is unavailable."}

        if action_name == "create":
            creator = getattr(host, "create_behavior_catalog_item", None)
            if callable(creator):
                return dict(creator(save=save, **kwargs))
            return {"ok": False, "error": "Behavior catalog creator is unavailable."}

        if action_name == "update":
            updater = getattr(host, "update_behavior_catalog_item", None)
            if callable(updater):
                code = str(kwargs.pop("code", "") or "").strip()
                if not code:
                    return {"ok": False, "error": "Behavior code is required."}
                return dict(updater(code=code, updates=kwargs, save=save))
            return {"ok": False, "error": "Behavior catalog updater is unavailable."}

        if action_name == "delete":
            deleter = getattr(host, "delete_behavior_catalog_item", None)
            if callable(deleter):
                code = str(kwargs.get("code", "") or "").strip()
                if not code:
                    return {"ok": False, "error": "Behavior code is required."}
                return dict(deleter(code=code, save=save))
            return {"ok": False, "error": "Behavior catalog deleter is unavailable."}

        return {
            "ok": False,
            "error": f"Unsupported behavior catalog action: {action_name}",
        }

    @QtCore.Slot(int)
    def _on_behavior_label_progress(self, progress_value: int) -> None:
        context = dict(self._behavior_label_run_context or {})
        if not context:
            return
        pct = max(0, min(100, int(progress_value)))
        processed = int(context.get("processed_segments") or 0)
        total = int(context.get("evaluated_segments") or 0)
        self.status_label.setText(
            f"Behavior labeling running in background… {pct}% ({processed}/{max(1, total)} segments)."
        )

    @QtCore.Slot(object)
    def _on_behavior_label_preview(self, payload: object) -> None:
        context = self._behavior_label_run_context
        if not context:
            return
        if not isinstance(payload, dict):
            return

        host = context.get("host")
        behavior_controller = context.get("behavior_controller")
        if host is None or behavior_controller is None:
            return

        status = str(payload.get("status") or "").strip().lower()
        evaluated_for_progress = int(context.get("evaluated_segments") or 0)
        progress_value = max(0, min(100, int(payload.get("progress") or 0)))
        if status in {"building_grid", "model_request"}:
            active_index = int(payload.get("index") or 0)
            total = int(payload.get("total") or evaluated_for_progress or 0)
            start_frame = int(payload.get("start_frame") or 0)
            end_frame = int(payload.get("end_frame") or start_frame)
            if status == "building_grid":
                message = (
                    f"Preparing behavior grid {active_index}/{max(1, total)} "
                    f"(frames {start_frame}-{end_frame}); {progress_value}% complete."
                )
            else:
                attempt = str(payload.get("attempt") or "model").strip()
                message = (
                    f"Labeling behavior grid {active_index}/{max(1, total)} "
                    f"with {attempt}; {progress_value}% complete."
                )
            self.status_label.setText(message)
            self._set_bot_action_result(
                "label_behavior_segments",
                {
                    "ok": True,
                    "queued": False,
                    "in_progress": True,
                    "mode": str(context.get("mode") or "uniform"),
                    "active_status": status,
                    "active_segment": active_index,
                    "active_attempt": str(payload.get("attempt") or ""),
                    "evaluated_segments": int(context.get("evaluated_segments") or 0),
                    "labeled_segments": len(context.get("predictions") or []),
                    "skipped_segments": int(context.get("skipped_segments") or 0),
                    "resumed_segments": int(context.get("resumed_segments") or 0),
                    "remaining_segments": max(
                        0,
                        int(context.get("evaluated_segments") or 0)
                        - int(context.get("resumed_segments") or 0),
                    ),
                    "labels_used": list(context.get("labels") or []),
                },
            )
            return

        context["processed_segments"] = int(context.get("processed_offset") or 0) + int(
            payload.get("index") or 0
        )
        context["skipped_segments"] = int(context.get("skipped_segments") or 0)
        if evaluated_for_progress > 0:
            progress_value = int(
                (int(context.get("processed_segments") or 0) * 100)
                / evaluated_for_progress
            )
            progress_value = max(0, min(100, progress_value))
        else:
            progress_value = max(0, min(100, int(payload.get("progress") or 0)))
        if status in {
            "skipped",
            "no_behavior",
            "unclassified",
            "rate_limited",
            "empty_response",
        }:
            context["skipped_segments"] = int(context["skipped_segments"]) + 1
            if status in {"no_behavior", "unclassified"}:
                skipped_prediction = payload.get("skipped_prediction")
                if isinstance(skipped_prediction, dict):
                    self._commit_behavior_label_skipped_prediction(
                        context, skipped_prediction
                    )
            if status == "rate_limited":
                context["rate_limited"] = True
                context["rate_limit_error"] = str(payload.get("error") or "")
                self.status_label.setText(
                    "Behavior labeling paused after provider rate limit; "
                    f"{progress_value}% complete."
                )
            elif status == "empty_response":
                context["empty_response_error"] = str(payload.get("error") or "")
                self.status_label.setText(
                    "Behavior labeling received empty model responses; "
                    f"{progress_value}% complete."
                )
            elif status == "no_behavior":
                self.status_label.setText(
                    f"Segment {context['processed_segments']}/{context.get('evaluated_segments')} has no listed behavior; {progress_value}% complete."
                )
            elif status == "unclassified":
                self.status_label.setText(
                    f"Segment {context['processed_segments']}/{context.get('evaluated_segments')} returned an unsupported label; saved model description; {progress_value}% complete."
                )
            else:
                self.status_label.setText(
                    f"Skipped segment {context['processed_segments']}/{context.get('evaluated_segments')}; {progress_value}% complete."
                )
            self._save_behavior_segment_progress(force_timestamps=False)
            return

        prediction = payload.get("prediction")
        if not isinstance(prediction, dict):
            return
        normalized_prediction = self._commit_behavior_label_prediction(
            context, prediction
        )
        if normalized_prediction is None:
            return

        save_result = self._save_behavior_segment_progress(force_timestamps=False)
        behavior_log_result = dict(save_result.get("behavior_log_result") or {})

        total = int(context.get("evaluated_segments") or 0)
        self.status_label.setText(
            f"Labeled segment {context['processed_segments']}/{max(1, total)} as '{normalized_prediction['label']}' ({progress_value}%)."
        )
        self._set_bot_action_result(
            "label_behavior_segments",
            {
                "ok": True,
                "queued": False,
                "in_progress": True,
                "mode": str(context.get("mode") or "uniform"),
                "labeled_segments": len(context.get("predictions") or []),
                "evaluated_segments": int(context.get("evaluated_segments") or 0),
                "skipped_segments": int(context.get("skipped_segments") or 0),
                "resumed_segments": int(context.get("resumed_segments") or 0),
                "remaining_segments": max(
                    0,
                    int(context.get("evaluated_segments") or 0)
                    - int(context.get("resumed_segments") or 0),
                ),
                "segment_frames": int(context.get("segment_frames") or 1),
                "segment_seconds": float(context.get("segment_seconds") or 0.0),
                "sample_frames_per_segment": int(
                    context.get("sample_frames_per_segment") or 4
                ),
                "visual_input_mode": str(
                    context.get("visual_input_mode") or "frame_grid"
                ),
                "use_defined_behavior_list": bool(
                    context.get("use_defined_behavior_list", True)
                ),
                "labels_used": list(context.get("labels") or []),
                "video_description": str(context.get("video_description") or ""),
                "instance_count": context.get("instance_count"),
                "experiment_context": str(context.get("experiment_context") or ""),
                "behavior_definitions": str(context.get("behavior_definitions") or ""),
                "focus_points": str(context.get("focus_points") or ""),
                "behavior_log_json": str(behavior_log_result.get("path") or ""),
                "behavior_log_rows": int(behavior_log_result.get("rows") or 0),
            },
        )

    @QtCore.Slot(object)
    def _on_behavior_label_finished(self, result: object) -> None:
        context = dict(self._behavior_label_run_context or {})
        self._clear_behavior_label_run_context()
        if not context:
            return

        host = context.get("host")
        behavior_controller = context.get("behavior_controller")
        if host is None or behavior_controller is None:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Behavior labeling context was lost."},
            )
            self.status_label.setText(
                "Bot action failed: behavior labeling context lost."
            )
            return

        if isinstance(result, Exception):
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": str(result)},
            )
            self.status_label.setText(f"Bot action failed: {result}")
            return

        payload = dict(result or {})
        if bool(payload.get("cancelled")):
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Behavior labeling was cancelled."},
            )
            self.status_label.setText("Behavior labeling cancelled.")
            return
        if bool(payload.get("rate_limited")):
            context["rate_limited"] = True
            context["rate_limit_error"] = str(
                payload.get("error")
                or context.get("rate_limit_error")
                or "Provider rate limit reached."
            )
        if bool(payload.get("empty_response_paused")):
            context["empty_response_paused"] = True
            context["empty_response_error"] = str(
                payload.get("error")
                or context.get("empty_response_error")
                or "Provider returned empty behavior-label responses."
            )

        predictions = list(context.get("predictions") or [])
        skipped_predictions = list(context.get("skipped_predictions") or [])
        fallback_predictions = list(payload.get("predictions") or [])
        if fallback_predictions:
            existing_ranges = {
                (
                    int(pred.get("start_frame") or -1),
                    int(pred.get("end_frame") or -1),
                )
                for pred in predictions
            }
            for pred in fallback_predictions:
                try:
                    pred_range = (
                        int(pred.get("start_frame") or -1),
                        int(pred.get("end_frame") or -1),
                    )
                except Exception:
                    pred_range = (-1, -1)
                if pred_range not in existing_ranges:
                    normalized_prediction = self._commit_behavior_label_prediction(
                        context, pred
                    )
                    if normalized_prediction is not None:
                        predictions.append(normalized_prediction)
                        existing_ranges.add(pred_range)
            context["predictions"] = predictions
        fallback_skipped_predictions = list(payload.get("skipped_predictions") or [])
        if fallback_skipped_predictions:
            existing_skipped_ranges = {
                (
                    int(pred.get("start_frame") or -1),
                    int(pred.get("end_frame") or -1),
                )
                for pred in skipped_predictions
                if isinstance(pred, dict)
            }
            for pred in fallback_skipped_predictions:
                try:
                    pred_range = (
                        int(pred.get("start_frame") or -1),
                        int(pred.get("end_frame") or -1),
                    )
                except Exception:
                    pred_range = (-1, -1)
                if pred_range not in existing_skipped_ranges:
                    normalized_prediction = (
                        self._commit_behavior_label_skipped_prediction(context, pred)
                    )
                    if normalized_prediction is not None:
                        skipped_predictions.append(normalized_prediction)
                        existing_skipped_ranges.add(pred_range)
            context["skipped_predictions"] = skipped_predictions

        if not predictions:
            if skipped_predictions:
                context["skipped_segments"] = max(
                    int(context.get("skipped_segments") or 0),
                    int(payload.get("skipped_segments") or 0),
                    len(skipped_predictions),
                )
                self._behavior_label_run_context = context
                save_result = self._save_behavior_segment_progress(
                    force_timestamps=True
                )
                self._behavior_label_run_context = {}
                behavior_log_result = dict(save_result.get("behavior_log_result") or {})
                self._set_bot_action_result(
                    "label_behavior_segments",
                    {
                        "ok": True,
                        "queued": False,
                        "in_progress": False,
                        "mode": str(context.get("mode") or "uniform"),
                        "labeled_segments": 0,
                        "evaluated_segments": int(
                            context.get("evaluated_segments") or 0
                        ),
                        "skipped_segments": int(context.get("skipped_segments") or 0),
                        "behavior_log_json": str(behavior_log_result.get("path") or ""),
                        "behavior_log_rows": int(behavior_log_result.get("rows") or 0),
                        "behavior_log_skipped_rows": int(
                            behavior_log_result.get("skipped_rows") or 0
                        ),
                    },
                )
                self.status_label.setText(
                    "Behavior labeling finished with no listed behaviors detected; saved segment log."
                )
                return
            if bool(context.get("rate_limited")):
                error_text = str(
                    context.get("rate_limit_error") or "Provider rate limit reached."
                )
                self._set_bot_action_result(
                    "label_behavior_segments",
                    {"ok": False, "rate_limited": True, "error": error_text},
                )
                self.status_label.setText(
                    "Behavior labeling paused after provider rate limit."
                )
                return
            if bool(context.get("empty_response_paused")):
                error_text = str(
                    context.get("empty_response_error")
                    or "Provider returned empty behavior-label responses."
                )
                self._set_bot_action_result(
                    "label_behavior_segments",
                    {"ok": False, "empty_response_paused": True, "error": error_text},
                )
                self.status_label.setText(
                    "Behavior labeling paused after empty model responses."
                )
                return
            self._set_bot_action_result(
                "label_behavior_segments",
                {
                    "ok": False,
                    "error": "Model did not return usable labels for any segment.",
                },
            )
            self.status_label.setText("Bot action failed: no usable segment labels.")
            return

        context["skipped_segments"] = max(
            int(context.get("skipped_segments") or 0),
            int(payload.get("skipped_segments") or 0),
        )
        self._behavior_label_run_context = context
        save_result = self._save_behavior_segment_progress(force_timestamps=True)
        self._behavior_label_run_context = {}
        timestamp_result = dict(save_result.get("timestamp_result") or {})
        if not bool(timestamp_result.get("ok", False)):
            self._set_bot_action_result(
                "label_behavior_segments",
                {
                    "ok": False,
                    "error": str(
                        timestamp_result.get("error")
                        or "Failed to save behavior timestamps."
                    ),
                },
            )
            self.status_label.setText("Bot action failed: timestamp export.")
            return

        behavior_log_result = dict(save_result.get("behavior_log_result") or {})
        action_payload = {
            "ok": True,
            "mode": str(context.get("mode") or "uniform"),
            "queued": False,
            "in_progress": False,
            "labeled_segments": len(predictions),
            "evaluated_segments": int(context.get("evaluated_segments") or 0),
            "skipped_segments": int(context.get("skipped_segments") or 0),
            "resumed_segments": int(context.get("resumed_segments") or 0),
            "remaining_segments": max(
                0,
                int(context.get("evaluated_segments") or 0)
                - int(context.get("resumed_segments") or 0),
            ),
            "segment_frames": int(context.get("segment_frames") or 1),
            "segment_seconds": float(context.get("segment_seconds") or 0.0),
            "sample_frames_per_segment": int(
                context.get("sample_frames_per_segment") or 4
            ),
            "visual_input_mode": str(context.get("visual_input_mode") or "frame_grid"),
            "use_defined_behavior_list": bool(
                context.get("use_defined_behavior_list", True)
            ),
            "labels_used": list(context.get("labels") or []),
            "video_description": str(context.get("video_description") or ""),
            "instance_count": context.get("instance_count"),
            "experiment_context": str(context.get("experiment_context") or ""),
            "behavior_definitions": str(context.get("behavior_definitions") or ""),
            "focus_points": str(context.get("focus_points") or ""),
            "timestamps_csv": str(timestamp_result.get("path") or ""),
            "timestamps_rows": int(timestamp_result.get("rows") or 0),
            "behavior_log_json": str(behavior_log_result.get("path") or ""),
            "behavior_log_rows": int(behavior_log_result.get("rows") or 0),
        }
        if bool(context.get("rate_limited")):
            action_payload["rate_limited"] = True
            action_payload["warning"] = str(
                context.get("rate_limit_error") or "Provider rate limit reached."
            )
        if bool(context.get("empty_response_paused")):
            action_payload["empty_response_paused"] = True
            action_payload["warning"] = str(
                context.get("empty_response_error")
                or "Provider returned empty behavior-label responses."
            )
        self._set_bot_action_result("label_behavior_segments", action_payload)
        timestamps_name = Path(str(timestamp_result.get("path") or "")).name
        behavior_log_name = Path(str(behavior_log_result.get("path") or "")).name
        if bool(context.get("rate_limited")):
            self.status_label.setText(
                f"Labeled {len(predictions)} segment(s) before provider rate limit; saved progress."
            )
        elif bool(context.get("empty_response_paused")):
            self.status_label.setText(
                f"Labeled {len(predictions)} segment(s) before empty model responses; saved progress."
            )
        elif behavior_log_name:
            self.status_label.setText(
                f"Labeled {len(predictions)} segment(s); saved {timestamps_name} and {behavior_log_name}."
            )
        else:
            self.status_label.setText(
                f"Labeled {len(predictions)} segment(s); saved timestamps to {timestamps_name}."
            )

    @QtCore.Slot(str)
    def bot_manage_behavior_catalog_json(self, payload_json: str = "") -> None:
        try:
            payload = json.loads(str(payload_json or "{}"))
        except Exception as exc:
            self._set_bot_action_result(
                "behavior_catalog",
                {"ok": False, "error": f"Invalid JSON payload: {exc}"},
            )
            self.status_label.setText(
                "Behavior catalog action failed: invalid payload."
            )
            return

        if not isinstance(payload, dict):
            self._set_bot_action_result(
                "behavior_catalog",
                {"ok": False, "error": "Invalid JSON payload type."},
            )
            self.status_label.setText(
                "Behavior catalog action failed: invalid payload."
            )
            return

        action = str(payload.get("action") or "list").strip().lower()
        kwargs = {
            "code": str(payload.get("code") or "").strip(),
            "name": str(payload.get("name") or "").strip() or None,
            "description": str(payload.get("description") or "").strip() or None,
            "category_id": str(payload.get("category_id") or "").strip() or None,
            "modifier_ids": [
                str(item).strip()
                for item in (payload.get("modifier_ids") or [])
                if str(item).strip()
            ],
            "key_binding": str(payload.get("key_binding") or "").strip() or None,
            "is_state": payload.get("is_state"),
            "exclusive_with": [
                str(item).strip()
                for item in (payload.get("exclusive_with") or [])
                if str(item).strip()
            ],
            "save": bool(payload.get("save", True)),
        }
        result = self._behavior_catalog_action(action, **kwargs)
        if action == "create" and result.get("ok", False):
            message = str(result.get("message") or "Behavior created.")
            self.status_label.setText(message)
        elif action == "update" and result.get("ok", False):
            message = str(result.get("message") or "Behavior updated.")
            self.status_label.setText(message)
        elif action == "delete" and result.get("ok", False):
            message = str(result.get("message") or "Behavior deleted.")
            self.status_label.setText(message)
        elif action == "save" and result.get("ok", False):
            path_text = str(result.get("path") or "").strip()
            if path_text:
                self.status_label.setText(
                    f"Behavior catalog saved to {Path(path_text).name}."
                )
            else:
                self.status_label.setText("Behavior catalog saved.")
        elif action == "list" and result.get("ok", False):
            self.status_label.setText(
                f"Behavior catalog has {int(result.get('count') or 0)} item(s)."
            )
        self._set_bot_action_result("behavior_catalog", result)

    def _behavior_slash_defaults(self) -> Dict[str, Any]:
        host = self.host_window_widget or self.window()
        active_video_path = str(getattr(host, "video_file", "") or "").strip()
        providers = getattr(self, "_providers", None)
        provider_names: List[str] = []
        if providers is not None and hasattr(providers, "available_providers"):
            try:
                provider_names = [
                    str(name or "").strip()
                    for name in providers.available_providers()
                    if str(name or "").strip()
                ]
            except Exception:
                provider_names = []
        if not provider_names:
            provider_names = list(getattr(self, "available_providers", []) or [])
        selected_provider = str(getattr(self, "selected_provider", "") or "").strip()
        if selected_provider and selected_provider not in provider_names:
            provider_names.append(selected_provider)
        provider_models: Dict[str, List[str]] = {}
        for provider in provider_names:
            try:
                provider_models[provider] = [
                    str(model or "").strip()
                    for model in providers.available_models(provider)
                    if str(model or "").strip()
                ]
            except Exception:
                provider_models[provider] = []
        selected_model = str(getattr(self, "selected_model", "") or "").strip()
        if selected_provider:
            models = provider_models.setdefault(selected_provider, [])
            if selected_model and selected_model not in models:
                models.append(selected_model)
        labels = self._labels_from_schema_or_flags()
        return {
            "video_path": active_video_path,
            "labels": labels,
            "providers": provider_names,
            "provider_models": provider_models,
            "selected_provider": selected_provider,
            "selected_model": selected_model,
            "segment_seconds": 1.0,
            "frames_per_grid": 9,
            "max_segments": 0,
        }

    @QtCore.Slot()
    def bot_open_behavior_slash_dialog(self) -> None:
        defaults = self._behavior_slash_defaults()
        dialog_cls = _widget_module_attr("BehaviorSlashDialog", BehaviorSlashDialog)
        dialog = dialog_cls(
            self,
            video_path=str(defaults.get("video_path") or ""),
            labels=list(defaults.get("labels") or []),
            providers=list(defaults.get("providers") or []),
            provider_models=dict(defaults.get("provider_models") or {}),
            selected_provider=str(defaults.get("selected_provider") or ""),
            selected_model=str(defaults.get("selected_model") or ""),
            segment_seconds=float(defaults.get("segment_seconds") or 1.0),
            frames_per_grid=int(defaults.get("frames_per_grid") or 9),
            max_segments=int(defaults.get("max_segments") or 0),
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self._set_bot_action_result(
                "open_behavior_dialog",
                {"ok": False, "cancelled": True},
            )
            self.status_label.setText("Behavior setup canceled.")
            return

        values = dialog.values()
        provider = str(values.get("llm_provider") or "").strip()
        model = str(values.get("llm_model") or "").strip()
        if provider and model:
            try:
                self.set_provider_and_model(provider, model)
            except Exception:
                logger.debug("Could not sync behavior model route.", exc_info=True)

        labels = list(values.get("behavior_labels") or [])
        self.bot_label_behavior_segments(
            video_path=str(values.get("video_path") or ""),
            behavior_labels_csv=", ".join(str(label) for label in labels),
            use_defined_behavior_list=True,
            segment_mode="uniform",
            segment_frames=60,
            segment_seconds=float(values.get("segment_seconds") or 1.0),
            sample_frames_per_segment=int(values.get("sample_frames_per_segment") or 9),
            max_segments=int(values.get("max_segments") or 0),
            subject="Agent",
            overwrite_existing=bool(values.get("overwrite_existing", False)),
            llm_profile="",
            llm_provider=provider,
            llm_model=model,
            video_description=str(values.get("video_description") or ""),
            instance_count=0,
            experiment_context="",
            behavior_definitions=str(values.get("behavior_definitions") or ""),
            focus_points=str(values.get("focus_points") or ""),
            subject_term=str(values.get("subject_term") or ""),
        )
        payload = {
            "ok": True,
            "opened": True,
            "queued": True,
            "video_path": str(values.get("video_path") or ""),
            "behavior_labels": labels,
            "llm_provider": provider,
            "llm_model": model,
            "segment_seconds": float(values.get("segment_seconds") or 1.0),
            "sample_frames_per_segment": int(
                values.get("sample_frames_per_segment") or 9
            ),
        }
        self._set_bot_action_result("open_behavior_dialog", payload)

    @QtCore.Slot(str, str, bool, str, int, float, int, int, str, bool, str, str, str)
    def bot_label_behavior_segments(
        self,
        video_path: str = "",
        behavior_labels_csv: str = "",
        use_defined_behavior_list: bool = True,
        segment_mode: str = "timeline",
        segment_frames: int = 60,
        segment_seconds: float = 0.0,
        sample_frames_per_segment: int = 4,
        max_segments: int = 0,
        subject: str = "Agent",
        overwrite_existing: bool = False,
        llm_profile: str = "",
        llm_provider: str = "",
        llm_model: str = "",
        video_description: str = "",
        instance_count: int = 0,
        experiment_context: str = "",
        behavior_definitions: str = "",
        focus_points: str = "",
        subject_term: str = "",
    ) -> None:
        host = self.host_window_widget or self.window()
        open_video = getattr(host, "openVideo", None)
        behavior_controller = getattr(host, "behavior_controller", None)
        if behavior_controller is None:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Behavior timeline APIs are unavailable."},
            )
            self.status_label.setText(
                "Bot action failed: behavior timeline unavailable."
            )
            return

        try:
            if (
                self._behavior_label_thread is not None
                and self._behavior_label_thread.isRunning()
            ):
                self._set_bot_action_result(
                    "label_behavior_segments",
                    {
                        "ok": False,
                        "error": "Behavior labeling is already running. Wait for completion or cancel the current run.",
                    },
                )
                self.status_label.setText(
                    "Behavior labeling already running. Please wait."
                )
                return

            video_text = str(video_path or "").strip()
            if video_text:
                if not callable(open_video):
                    raise RuntimeError("Video opening is unavailable.")
                open_video(
                    from_video_list=True,
                    video_path=video_text,
                    programmatic_call=True,
                )
            resolved_video_path = str(getattr(host, "video_file", "") or "").strip()
            if not resolved_video_path:
                resolved_video_path = video_text
            if not resolved_video_path:
                raise RuntimeError("No video is loaded.")

            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(resolved_video_path))
            try:
                total_frames = int(getattr(host, "num_frames", 0) or 0)
                fps = float(getattr(host, "fps", 0.0) or 0.0)
                if total_frames <= 0 and cap.isOpened():
                    total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
                if fps <= 0.0 and cap.isOpened():
                    fps = max(0.0, float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
            finally:
                cap.release()

            explicit_labels = self._normalize_behavior_labels(
                [
                    p.strip()
                    for p in str(behavior_labels_csv or "").split(",")
                    if p.strip()
                ]
            )
            labels = self._resolve_segment_label_candidates(
                explicit_labels,
                use_defined_behavior_list=bool(use_defined_behavior_list),
            )
            if not labels:
                raise RuntimeError(
                    "No behavior labels provided/found. Define behaviors in schema/flags/timeline or pass labels."
                )

            mode = str(segment_mode or "timeline").strip().lower()
            if total_frames <= 0:
                raise RuntimeError("No video is loaded.")
            max_segments = max(0, int(max_segments or 0))
            segment_frames = max(1, int(segment_frames))
            segment_seconds = max(0.0, float(segment_seconds or 0.0))
            try:
                instance_count_value = int(instance_count or 0)
            except Exception:
                instance_count_value = 0
            normalized_instance_count = (
                instance_count_value if instance_count_value > 0 else None
            )
            if mode == "uniform" and segment_seconds > 0.0 and fps > 0.0:
                segment_frames = max(1, int(round(segment_seconds * fps)))
            sample_frames_per_segment = max(1, int(sample_frames_per_segment))
            behavior_request_interval = self._behavior_label_provider_request_interval(
                str(llm_provider or ""),
                str(llm_model or ""),
            )

            intervals: List[Dict[str, Any]] = []
            if mode == "timeline":
                events = list(behavior_controller.iter_events())
                intervals = self._behavior_ranges_from_events(events)
            if not intervals:
                mode = "uniform"
                last = total_frames - 1
                start = 0
                while start <= last:
                    end = min(last, start + segment_frames - 1)
                    intervals.append(
                        {
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "subject": None,
                            "seed_behavior": None,
                        }
                    )
                    start = end + 1
            if max_segments > 0:
                intervals = intervals[:max_segments]
            if not intervals:
                raise RuntimeError("No segments available for labeling.")

            evaluated_segments = int(len(intervals))
            resumed_predictions: List[Dict[str, Any]] = []
            resumed_skipped_predictions: List[Dict[str, Any]] = []
            resume_log_path = ""
            if not bool(overwrite_existing):
                resume_payload = self._load_resumable_behavior_segment_predictions(
                    str(resolved_video_path),
                    labels=list(labels),
                    segment_frames=int(segment_frames),
                    segment_seconds=float(segment_seconds),
                    sample_frames_per_segment=int(sample_frames_per_segment),
                )
                if not resume_payload.get("ok"):
                    raise RuntimeError(
                        "Cannot resume from behavior progress log "
                        f"{resume_payload.get('path')}. Repair or back up the log "
                        "before retrying; existing progress has not been overwritten."
                    )
                resume_log_path = str(resume_payload.get("path") or "")
                resumed_predictions = list(resume_payload.get("predictions") or [])
                resumed_skipped_predictions = list(
                    resume_payload.get("skipped_predictions") or []
                )
                if resumed_predictions or resumed_skipped_predictions:
                    default_subject = str(subject or "").strip()

                    def resume_key(record):
                        return behavior_segment_resume_key(
                            record, default_subject=default_subject
                        )

                    current_ranges = {resume_key(item) for item in intervals}
                    resumed_predictions = [
                        pred
                        for pred in resumed_predictions
                        if resume_key(pred) in current_ranges
                    ]
                    resumed_skipped_predictions = [
                        pred
                        for pred in resumed_skipped_predictions
                        if resume_key(pred) in current_ranges
                    ]
                    existing_ranges = {
                        resume_key(pred)
                        for pred in [*resumed_predictions, *resumed_skipped_predictions]
                    }
                    intervals = [
                        item
                        for item in intervals
                        if resume_key(item) not in existing_ranges
                    ]

            if bool(overwrite_existing):
                behavior_controller.clear_behavior_data()
            clear_generic_marks = getattr(
                behavior_controller, "clear_generic_marks", None
            )
            if callable(clear_generic_marks):
                try:
                    clear_generic_marks(mark_type="behavior_no_behavior")
                    clear_generic_marks(mark_type="behavior_unclassified")
                except Exception:
                    pass

            if not intervals and (resumed_predictions or resumed_skipped_predictions):
                self._behavior_label_run_context = {
                    "host": host,
                    "behavior_controller": behavior_controller,
                    "mode": mode,
                    "labels": list(labels),
                    "segment_frames": int(segment_frames),
                    "segment_seconds": float(segment_seconds),
                    "sample_frames_per_segment": int(sample_frames_per_segment),
                    "visual_input_mode": "frame_grid",
                    "request_interval_seconds": float(behavior_request_interval),
                    "evaluated_segments": evaluated_segments,
                    "processed_segments": int(
                        len(resumed_predictions) + len(resumed_skipped_predictions)
                    ),
                    "skipped_segments": int(len(resumed_skipped_predictions)),
                    "predictions": list(resumed_predictions),
                    "skipped_predictions": list(resumed_skipped_predictions),
                    "use_defined_behavior_list": bool(use_defined_behavior_list),
                    "default_subject": str(subject or "").strip() or None,
                    "timestamp_provider": self._behavior_label_timestamp_provider(host),
                    "video_description": str(video_description or "").strip(),
                    "instance_count": normalized_instance_count,
                    "experiment_context": str(experiment_context or "").strip(),
                    "behavior_definitions": str(behavior_definitions or "").strip(),
                    "focus_points": str(focus_points or "").strip(),
                    "subject_term": str(subject_term or "").strip(),
                    "resumed_segments": int(len(resumed_predictions)),
                    "resume_log_json": resume_log_path,
                }
                self._update_behavior_label_skipped_overlay_records(
                    self._behavior_label_run_context
                )
                save_result = self._save_behavior_segment_progress(
                    force_timestamps=True
                )
                self._behavior_label_run_context = {}
                behavior_log_result = dict(save_result.get("behavior_log_result") or {})
                timestamp_result = dict(save_result.get("timestamp_result") or {})
                self._set_bot_action_result(
                    "label_behavior_segments",
                    {
                        "ok": True,
                        "queued": False,
                        "in_progress": False,
                        "mode": mode,
                        "labeled_segments": int(len(resumed_predictions)),
                        "evaluated_segments": evaluated_segments,
                        "skipped_segments": int(len(resumed_skipped_predictions)),
                        "resumed_segments": int(len(resumed_predictions)),
                        "remaining_segments": 0,
                        "segment_frames": int(segment_frames),
                        "segment_seconds": float(segment_seconds),
                        "sample_frames_per_segment": int(sample_frames_per_segment),
                        "visual_input_mode": "frame_grid",
                        "use_defined_behavior_list": bool(use_defined_behavior_list),
                        "labels_used": labels,
                        "timestamps_csv": str(timestamp_result.get("path") or ""),
                        "timestamps_rows": int(timestamp_result.get("rows") or 0),
                        "behavior_log_json": str(behavior_log_result.get("path") or ""),
                        "behavior_log_rows": int(behavior_log_result.get("rows") or 0),
                    },
                )
                self.status_label.setText(
                    f"Behavior labeling already complete for {len(resumed_predictions)} labeled and {len(resumed_skipped_predictions)} skipped segment(s); skipped model calls."
                )
                return

            self._set_bot_action_result(
                "label_behavior_segments",
                {
                    "ok": True,
                    "queued": True,
                    "in_progress": True,
                    "mode": mode,
                    "evaluated_segments": evaluated_segments,
                    "resumed_segments": int(len(resumed_predictions)),
                    "remaining_segments": int(len(intervals)),
                    "segment_frames": int(segment_frames),
                    "segment_seconds": float(segment_seconds),
                    "sample_frames_per_segment": int(sample_frames_per_segment),
                    "visual_input_mode": "frame_grid",
                    "request_interval_seconds": float(behavior_request_interval),
                    "use_defined_behavior_list": bool(use_defined_behavior_list),
                    "labels_used": labels,
                    "video_description": str(video_description or "").strip(),
                    "instance_count": normalized_instance_count,
                    "experiment_context": str(experiment_context or "").strip(),
                    "behavior_definitions": str(behavior_definitions or "").strip(),
                    "focus_points": str(focus_points or "").strip(),
                    "subject_term": str(subject_term or "").strip(),
                },
            )
            self.status_label.setText(
                f"Queued behavior labeling for {len(intervals)} remaining segment(s); skipped {len(resumed_predictions)} existing segment(s)."
            )

            self._behavior_label_run_context = {
                "host": host,
                "behavior_controller": behavior_controller,
                "mode": mode,
                "labels": list(labels),
                "segment_frames": int(segment_frames),
                "segment_seconds": float(segment_seconds),
                "sample_frames_per_segment": int(sample_frames_per_segment),
                "visual_input_mode": "frame_grid",
                "request_interval_seconds": float(behavior_request_interval),
                "evaluated_segments": evaluated_segments,
                "processed_segments": int(
                    len(resumed_predictions) + len(resumed_skipped_predictions)
                ),
                "processed_offset": int(
                    len(resumed_predictions) + len(resumed_skipped_predictions)
                ),
                "skipped_segments": int(len(resumed_skipped_predictions)),
                "predictions": list(resumed_predictions),
                "skipped_predictions": list(resumed_skipped_predictions),
                "use_defined_behavior_list": bool(use_defined_behavior_list),
                "default_subject": str(subject or "").strip() or None,
                "timestamp_provider": self._behavior_label_timestamp_provider(host),
                "video_description": str(video_description or "").strip(),
                "instance_count": normalized_instance_count,
                "experiment_context": str(experiment_context or "").strip(),
                "behavior_definitions": str(behavior_definitions or "").strip(),
                "focus_points": str(focus_points or "").strip(),
                "subject_term": str(subject_term or "").strip(),
                "resumed_segments": int(len(resumed_predictions)),
                "resume_log_json": resume_log_path,
            }
            self._update_behavior_label_skipped_overlay_records(
                self._behavior_label_run_context
            )

            thread = QtCore.QThread()
            worker = FlexibleWorker(
                self._run_behavior_segment_vlm_worker,
                video_path=str(resolved_video_path),
                intervals=list(intervals),
                labels=list(labels),
                sample_frames_per_segment=int(sample_frames_per_segment),
                llm_profile=str(llm_profile or ""),
                llm_provider=str(llm_provider or ""),
                llm_model=str(llm_model or ""),
                video_description=str(video_description or "").strip(),
                instance_count=normalized_instance_count,
                experiment_context=str(experiment_context or "").strip(),
                behavior_definitions=str(behavior_definitions or "").strip(),
                focus_points=str(focus_points or "").strip(),
                subject_term=str(subject_term or "").strip(),
                prior_predictions=list(resumed_predictions),
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
            worker.progress_signal.connect(
                self._on_behavior_label_progress, QtCore.Qt.QueuedConnection
            )
            worker.preview_signal.connect(
                self._on_behavior_label_preview, QtCore.Qt.QueuedConnection
            )
            worker.finished_signal.connect(
                self._on_behavior_label_finished, QtCore.Qt.QueuedConnection
            )
            worker.finished_signal.connect(thread.quit, QtCore.Qt.QueuedConnection)
            worker.finished_signal.connect(
                worker.deleteLater, QtCore.Qt.QueuedConnection
            )
            thread.finished.connect(self._clear_behavior_label_thread_refs)
            thread.finished.connect(thread.deleteLater)

            self._behavior_label_worker = worker
            self._behavior_label_thread = thread
            thread.start()
        except Exception as exc:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": str(exc)},
            )
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_label_behavior_segments_json(self, payload_json: str = "") -> None:
        try:
            payload = json.loads(str(payload_json or "{}"))
        except Exception as exc:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": f"Invalid JSON payload: {exc}"},
            )
            self.status_label.setText("Bot action failed: invalid labeling payload.")
            return

        if not isinstance(payload, dict):
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Invalid JSON payload type."},
            )
            self.status_label.setText("Bot action failed: invalid labeling payload.")
            return

        try:
            parsed_instance_count = int(payload.get("instance_count") or 0)
        except Exception:
            parsed_instance_count = 0

        self.bot_label_behavior_segments(
            video_path=str(payload.get("video_path") or ""),
            behavior_labels_csv=str(payload.get("behavior_labels_csv") or ""),
            use_defined_behavior_list=bool(
                payload.get("use_defined_behavior_list", True)
            ),
            segment_mode=str(payload.get("segment_mode") or "timeline"),
            segment_frames=int(payload.get("segment_frames") or 60),
            segment_seconds=float(payload.get("segment_seconds") or 0.0),
            sample_frames_per_segment=int(
                payload.get("sample_frames_per_segment")
                or payload.get("frames_per_grid")
                or 4
            ),
            max_segments=int(payload.get("max_segments") or 0),
            subject=str(payload.get("subject") or "Agent"),
            overwrite_existing=bool(payload.get("overwrite_existing", False)),
            llm_profile=str(payload.get("llm_profile") or ""),
            llm_provider=str(payload.get("llm_provider") or ""),
            llm_model=str(payload.get("llm_model") or ""),
            video_description=str(payload.get("video_description") or ""),
            instance_count=parsed_instance_count,
            experiment_context=str(payload.get("experiment_context") or ""),
            behavior_definitions=str(payload.get("behavior_definitions") or ""),
            focus_points=str(payload.get("focus_points") or ""),
            subject_term=str(payload.get("subject_term") or ""),
        )
