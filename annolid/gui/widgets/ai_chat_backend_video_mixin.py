"""Video tracking, behavior labeling, and analytics mixin for StreamingChatTask."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from qtpy import QtCore

from annolid.services.chat_agent_core import BusInboundMessage
from annolid.services.chat_tracking_correction import (
    correct_chat_tracking_ndjson_tool as gui_correct_tracking_ndjson_tool,
)
from annolid.services.chat_tracking_stats import (
    analyze_chat_tracking_stats_tool as gui_analyze_chat_tracking_stats_tool,
)
from annolid.services.chat_video import (
    behavior_catalog_tool as gui_behavior_catalog_tool,
    process_chat_video_behaviors_tool,
    sam3_agent_video_track_tool,
    score_chat_aggression_bouts_tool,
)


def _backend_attr(name: str, default: Any) -> Any:
    mod = sys.modules.get("annolid.gui.widgets.ai_chat_backend")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default


class StreamingChatVideoMixin:
    """Mixin encapsulating video tracking, behavior, and tracking-stats tools."""

    def _resolve_gui_track_fallback_model_name(self) -> str:
        # Prefer pre-snapshotted host model choice from inbound metadata to avoid
        # cross-thread Qt widget reads from the QThreadPool worker.
        if isinstance(self.inbound, BusInboundMessage):
            meta = dict(self.inbound.metadata or {})
            snap_model = str(meta.get("host_track_fallback_model") or "").strip()
            if snap_model and snap_model.lower() != "sam3":
                return snap_model
        widget = self.widget
        host = (
            getattr(widget, "host_window_widget", None) if widget is not None else None
        )
        if host is None and widget is not None and hasattr(widget, "window"):
            with contextlib.suppress(Exception):
                host = widget.window()
        combo = getattr(host, "_selectAiModelComboBox", None)
        candidates: list[str] = []
        current = ""
        if combo is not None:
            try:
                current = str(combo.currentText() or "").strip()
                for index in range(int(combo.count())):
                    item = str(combo.itemText(index) or "").strip()
                    if not item or item.lower() == "sam3":
                        continue
                    if item not in candidates:
                        candidates.append(item)
            except Exception:
                current = ""
                candidates = []
        if current and current.lower() != "sam3":
            return current
        if candidates:
            return candidates[0]
        return "Cutie"

    def _tool_sam3_agent_video_track(
        self,
        *,
        video_path: str,
        agent_prompt: str,
        window_size: int = 5,
        stride: int | None = None,
        output_dir: str | None = None,
        summary_filename: str | None = None,
        checkpoint_path: str | None = None,
        propagation_direction: str = "forward",
        device: str | None = None,
        agent_det_thresh: float = 0.3,
        score_threshold_detection: float | None = None,
        new_det_thresh: float | None = None,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
        compile_model: bool = False,
        offload_video_to_cpu: bool = True,
        use_explicit_window_reseed: bool = True,
        boundary_mask_match_iou_threshold: float = 0.2,
        allow_private_state_mutation: bool = False,
        reject_implausible_masks: bool = False,
        llm_provider: str = "",
        llm_model: str = "",
        llm_profile: str = "",
        max_generations: int = 100,
        debug: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        raw_result = _backend_attr(
            "sam3_agent_video_track_tool", sam3_agent_video_track_tool
        )(
            allowed_dir=Path(self.workspace),
            allowed_read_roots=self._vcs_read_roots(),
            video_path=video_path,
            agent_prompt=agent_prompt,
            window_size=window_size,
            stride=stride,
            output_dir=output_dir,
            summary_filename=summary_filename,
            checkpoint_path=checkpoint_path,
            propagation_direction=propagation_direction,
            device=device,
            agent_det_thresh=agent_det_thresh,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            compile_model=compile_model,
            offload_video_to_cpu=offload_video_to_cpu,
            use_explicit_window_reseed=use_explicit_window_reseed,
            boundary_mask_match_iou_threshold=boundary_mask_match_iou_threshold,
            allow_private_state_mutation=allow_private_state_mutation,
            reject_implausible_masks=reject_implausible_masks,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_profile=llm_profile,
            max_generations=max_generations,
            debug=debug,
            dry_run=dry_run,
        )
        if isinstance(raw_result, dict):
            return raw_result
        try:
            parsed = json.loads(str(raw_result or ""))
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Failed to parse SAM3 agent tracking response: {exc}",
                "video_path": str(video_path or "").strip(),
                "agent_prompt": str(agent_prompt or "").strip(),
            }
        if isinstance(parsed, dict):
            return parsed
        return {
            "ok": False,
            "error": "SAM3 agent tracking returned an unexpected response.",
            "video_path": str(video_path or "").strip(),
            "agent_prompt": str(agent_prompt or "").strip(),
        }

    def _tool_gui_sam3_agent_video_track(
        self,
        *,
        video_path: str,
        agent_prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        def invoke_segment_track(
            vpath: str,
            prompt: str,
            mode_norm: str,
            countgd: bool,
            model: str,
            frame: int,
        ) -> bool:
            return self._invoke_widget_slot(
                "bot_segment_track_video",
                QtCore.Q_ARG(str, vpath),
                QtCore.Q_ARG(str, prompt),
                QtCore.Q_ARG(str, mode_norm),
                QtCore.Q_ARG(bool, countgd),
                QtCore.Q_ARG(str, model),
                QtCore.Q_ARG(int, frame),
            )

        try:
            payload = self._tool_sam3_agent_video_track(
                video_path=video_path,
                agent_prompt=agent_prompt,
                **kwargs,
            )
        except Exception as exc:
            return self._fallback_from_sam3_agent_to_gui_tracking(
                path=video_path,
                text_prompt=agent_prompt,
                mode="track",
                use_countgd=False,
                to_frame=None,
                invoke_segment_track=invoke_segment_track,
                sam3_error=str(exc),
            )
        if isinstance(payload, dict) and bool(payload.get("ok", False)):
            return payload
        return self._fallback_from_sam3_agent_to_gui_tracking(
            path=video_path,
            text_prompt=agent_prompt,
            mode="track",
            use_countgd=False,
            to_frame=None,
            invoke_segment_track=invoke_segment_track,
            sam3_error=str(
                (payload or {}).get("error") or "SAM3 agent returned failure."
            ),
        )

    @staticmethod
    def _normalize_positive_instance_count(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    def _invoke_label_behavior_segments_slot(
        self,
        resolved_path: str,
        labels: str,
        use_defined: bool,
        mode_norm: str,
        frames: int,
        seconds: float,
        sample_frames: int,
        max_seg: int,
        subj: str,
        overwrite: bool,
        profile: str,
        provider: str,
        model: str,
        video_description_text: str,
        instance_count_value: Optional[int],
        experiment_context_text: str,
        behavior_definitions_text: str,
        focus_points_text: str,
        subject_term_text: str,
    ) -> bool:
        json_payload = json.dumps(
            {
                "video_path": str(resolved_path),
                "behavior_labels_csv": str(labels),
                "use_defined_behavior_list": bool(use_defined),
                "segment_mode": str(mode_norm),
                "segment_frames": int(frames),
                "segment_seconds": float(seconds),
                "sample_frames_per_segment": int(sample_frames),
                "max_segments": int(max_seg),
                "subject": str(subj),
                "overwrite_existing": bool(overwrite),
                "llm_profile": str(profile),
                "llm_provider": str(provider),
                "llm_model": str(model),
                "video_description": str(video_description_text or "").strip(),
                "instance_count": (
                    int(instance_count_value)
                    if instance_count_value is not None
                    and int(instance_count_value) > 0
                    else 0
                ),
                "experiment_context": str(experiment_context_text or "").strip(),
                "behavior_definitions": str(behavior_definitions_text or "").strip(),
                "focus_points": str(focus_points_text or "").strip(),
                "subject_term": str(subject_term_text or "").strip(),
            }
        )
        json_result = self._invoke_widget_json_slot(
            "bot_label_behavior_segments_json",
            QtCore.Q_ARG(str, json_payload),
        )
        if bool(json_result.get("ok", False)):
            return True
        if not bool(json_result.get("transport_error", False)):
            # The JSON slot executed and returned a semantic failure payload.
            # Do not fall back to positional signatures (which can emit noisy
            # invoke warnings). Let downstream action-result handling report
            # the real error to the user.
            return True
        # Prefer the current signature (includes use_defined_behavior_list).
        ok = self._invoke_widget_slot(
            "bot_label_behavior_segments",
            QtCore.Q_ARG(str, resolved_path),
            QtCore.Q_ARG(str, labels),
            QtCore.Q_ARG(bool, bool(use_defined)),
            QtCore.Q_ARG(str, mode_norm),
            QtCore.Q_ARG(int, frames),
            QtCore.Q_ARG(float, float(seconds)),
            QtCore.Q_ARG(int, int(sample_frames)),
            QtCore.Q_ARG(int, max_seg),
            QtCore.Q_ARG(str, subj),
            QtCore.Q_ARG(bool, overwrite),
            QtCore.Q_ARG(str, profile),
            QtCore.Q_ARG(str, provider),
            QtCore.Q_ARG(str, model),
        )
        if ok:
            return True
        # Backward-compatibility: older widgets may not yet include
        # the use_defined_behavior_list bool parameter.
        return self._invoke_widget_slot(
            "bot_label_behavior_segments",
            QtCore.Q_ARG(str, resolved_path),
            QtCore.Q_ARG(str, labels),
            QtCore.Q_ARG(str, mode_norm),
            QtCore.Q_ARG(int, frames),
            QtCore.Q_ARG(float, float(seconds)),
            QtCore.Q_ARG(int, int(sample_frames)),
            QtCore.Q_ARG(int, max_seg),
            QtCore.Q_ARG(str, subj),
            QtCore.Q_ARG(bool, overwrite),
            QtCore.Q_ARG(str, profile),
            QtCore.Q_ARG(str, provider),
            QtCore.Q_ARG(str, model),
        )

    def _tool_gui_process_video_behaviors(
        self,
        *,
        path: str,
        text_prompt: str = "animal",
        mode: str = "track",
        use_countgd: bool = False,
        model_name: str = "",
        to_frame: Optional[int] = None,
        behavior_labels: Any = None,
        use_defined_behavior_list: bool = True,
        segment_mode: str = "timeline",
        segment_frames: int = 60,
        segment_seconds: Optional[float] = None,
        sample_frames_per_segment: int = 4,
        frames_per_grid: Optional[int] = None,
        max_segments: int = 0,
        subject: str = "Agent",
        subject_term: str = "",
        overwrite_existing: bool = False,
        llm_profile: str = "",
        llm_provider: str = "",
        llm_model: str = "",
        video_description: str = "",
        instance_count: Optional[int] = None,
        experiment_context: str = "",
        behavior_definitions: str = "",
        focus_points: str = "",
        run_tracking: bool = True,
        run_behavior_labeling: bool = True,
    ) -> Dict[str, Any]:
        llm_provider, llm_model = self._resolve_behavior_labeling_llm_route(
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        _normalize_instance_count = self._normalize_positive_instance_count
        _invoke_label_behavior_slot = self._invoke_label_behavior_segments_slot

        if frames_per_grid is not None:
            sample_frames_per_segment = int(frames_per_grid)

        return _backend_attr(
            "process_chat_video_behaviors_tool", process_chat_video_behaviors_tool
        )(
            path=path,
            text_prompt=text_prompt,
            mode=mode,
            use_countgd=use_countgd,
            model_name=model_name,
            to_frame=to_frame,
            behavior_labels=behavior_labels,
            use_defined_behavior_list=use_defined_behavior_list,
            segment_mode=segment_mode,
            segment_frames=segment_frames,
            segment_seconds=segment_seconds,
            sample_frames_per_segment=sample_frames_per_segment,
            max_segments=max_segments,
            subject=subject,
            overwrite_existing=overwrite_existing,
            llm_profile=llm_profile,
            llm_provider=llm_provider,
            llm_model=llm_model,
            video_description=video_description,
            instance_count=instance_count,
            experiment_context=experiment_context,
            behavior_definitions=behavior_definitions,
            focus_points=focus_points,
            subject_term=subject_term,
            run_tracking=run_tracking,
            run_behavior_labeling=run_behavior_labeling,
            resolve_video_path=self._resolve_video_path_for_gui_tool,
            invoke_segment_track=lambda vpath,
            prompt,
            mode_norm,
            countgd,
            model,
            frame: self._invoke_widget_slot(
                "bot_segment_track_video",
                QtCore.Q_ARG(str, vpath),
                QtCore.Q_ARG(str, prompt),
                QtCore.Q_ARG(str, mode_norm),
                QtCore.Q_ARG(bool, countgd),
                QtCore.Q_ARG(str, model),
                QtCore.Q_ARG(int, frame),
            ),
            invoke_label_behavior=lambda resolved_path,
            labels,
            use_defined,
            mode_norm,
            frames,
            seconds,
            sample_frames,
            max_seg,
            subj,
            overwrite,
            profile,
            provider,
            model,
            video_description_text,
            instance_count_value,
            experiment_context_text,
            behavior_definitions_text,
            focus_points_text,
            subject_term_text: _invoke_label_behavior_slot(
                resolved_path=resolved_path,
                labels=labels,
                use_defined=bool(use_defined),
                mode_norm=mode_norm,
                frames=int(frames),
                seconds=float(seconds),
                sample_frames=int(sample_frames),
                max_seg=int(max_seg),
                subj=str(subj),
                overwrite=bool(overwrite),
                profile=str(profile),
                provider=str(provider),
                model=str(model),
                video_description_text=str(video_description_text or ""),
                instance_count_value=_normalize_instance_count(instance_count_value),
                experiment_context_text=str(experiment_context_text or ""),
                behavior_definitions_text=str(behavior_definitions_text or ""),
                focus_points_text=str(focus_points_text or ""),
                subject_term_text=str(subject_term_text or ""),
            ),
            get_action_result=self._get_widget_action_result,
        )

    def _tool_gui_score_aggression_bouts(
        self,
        *,
        path: str,
        artifacts_ndjson: str = "",
        run_id: str = "",
        episode_id: str = "",
        results_dir: str = "",
        context_prompt: str = "",
        assay: str = "",
        default_assay: str = "aggression",
        model_policy: str = "annolid_behavior_agent_v1",
        bout_frame_gap: int = 20,
        no_memory: bool = False,
        no_analysis: bool = False,
        fail_on_validation_error: bool = False,
    ) -> Dict[str, Any]:
        return _backend_attr(
            "score_chat_aggression_bouts_tool", score_chat_aggression_bouts_tool
        )(
            path=path,
            artifacts_ndjson=artifacts_ndjson,
            run_id=run_id,
            episode_id=episode_id,
            results_dir=results_dir,
            context_prompt=context_prompt,
            assay=assay,
            default_assay=default_assay,
            model_policy=model_policy,
            bout_frame_gap=bout_frame_gap,
            no_memory=no_memory,
            no_analysis=no_analysis,
            fail_on_validation_error=fail_on_validation_error,
            resolve_video_path=self._resolve_video_path_for_gui_tool,
        )

    def _tool_gui_behavior_catalog(
        self,
        *,
        action: str,
        code: str = "",
        name: str = "",
        description: str = "",
        category_id: str = "",
        modifier_ids: Any = None,
        key_binding: str = "",
        is_state: Optional[bool] = None,
        exclusive_with: Any = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        def _invoke_behavior_catalog_slot(payload_json: str) -> bool:
            json_result = self._invoke_widget_json_slot(
                "bot_manage_behavior_catalog_json",
                QtCore.Q_ARG(str, payload_json),
            )
            if isinstance(json_result, dict):
                if bool(json_result.get("transport_error", False)):
                    return False
                return True
            return bool(json_result)

        return _backend_attr("gui_behavior_catalog_tool", gui_behavior_catalog_tool)(
            action=action,
            code=code,
            name=name,
            description=description,
            category_id=category_id,
            modifier_ids=modifier_ids,
            key_binding=key_binding,
            is_state=is_state,
            exclusive_with=exclusive_with,
            save=save,
            invoke_behavior_catalog=_invoke_behavior_catalog_slot,
            get_action_result=self._get_widget_action_result,
        )

    def _tool_gui_analyze_tracking_stats(
        self,
        *,
        root_dir: str = "",
        output_dir: str = "",
        video_id: str = "",
        top_k: int = 10,
        include_plots: bool = True,
    ) -> Dict[str, Any]:
        initial_root = None
        host = getattr(self.widget, "host_window_widget", None) if self.widget else None
        if host is not None:
            for attr in ("video_results_folder", "annotation_dir", "lastOpenDir"):
                raw = getattr(host, attr, None)
                text = str(raw or "").strip()
                if text:
                    initial_root = Path(text).expanduser()
                    break
        if initial_root is None:
            initial_root = Path(self.workspace).expanduser()
        return _backend_attr(
            "gui_analyze_chat_tracking_stats_tool", gui_analyze_chat_tracking_stats_tool
        )(
            root_dir=root_dir,
            output_dir=output_dir,
            video_id=video_id,
            top_k=top_k,
            include_plots=include_plots,
            default_root_dir=initial_root,
        )

    def _tool_gui_correct_tracking_ndjson(
        self,
        *,
        ndjson_path: str,
        source_ndjson_path: str = "",
        output_ndjson_path: str = "",
        video_path: str = "",
        agent_prompt: str = "",
        run_sam3_agent: bool = False,
        window_size: int = 5,
        stride: Optional[int] = None,
        replace_only_empty_shapes: bool = True,
        allow_append_new_frames: bool = False,
        replace_all_shapes: bool = False,
        temporal_repair: bool = False,
        start_frame: int = 0,
        expected_instance_count: Optional[int] = None,
        max_gap_frames: int = 5,
        max_match_distance: float = 80.0,
    ) -> Dict[str, Any]:
        return _backend_attr(
            "gui_correct_tracking_ndjson_tool", gui_correct_tracking_ndjson_tool
        )(
            ndjson_path=ndjson_path,
            source_ndjson_path=source_ndjson_path,
            output_ndjson_path=output_ndjson_path,
            video_path=video_path,
            agent_prompt=agent_prompt,
            run_sam3_agent=run_sam3_agent,
            window_size=window_size,
            stride=stride,
            replace_only_empty_shapes=replace_only_empty_shapes,
            allow_append_new_frames=allow_append_new_frames,
            replace_all_shapes=replace_all_shapes,
            temporal_repair=temporal_repair,
            start_frame=start_frame,
            expected_instance_count=expected_instance_count,
            max_gap_frames=max_gap_frames,
            max_match_distance=max_match_distance,
            allowed_dir=Path(self.workspace),
            allowed_read_roots=self._vcs_read_roots(),
        )
