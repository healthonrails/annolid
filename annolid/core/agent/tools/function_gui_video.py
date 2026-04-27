from __future__ import annotations

from typing import Any, Optional

from .function_gui_base import ActionCallback, _run_callback
from .function_base import FunctionTool


class GuiOpenVideoTool(FunctionTool):
    def __init__(self, open_video_callback: Optional[ActionCallback] = None):
        self._open_video_callback = open_video_callback

    @property
    def name(self) -> str:
        return "gui_open_video"

    @property
    def description(self) -> str:
        return (
            "Open a video in Annolid GUI using an absolute or workspace-relative path."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_video_callback, **kwargs)


class GuiSetFrameTool(FunctionTool):
    def __init__(self, set_frame_callback: Optional[ActionCallback] = None):
        self._set_frame_callback = set_frame_callback

    @property
    def name(self) -> str:
        return "gui_set_frame"

    @property
    def description(self) -> str:
        return "Set the current frame index in the active Annolid video session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"frame_index": {"type": "integer", "minimum": 0}},
            "required": ["frame_index"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._set_frame_callback, **kwargs)


class GuiTrackNextFramesTool(FunctionTool):
    def __init__(self, track_callback: Optional[ActionCallback] = None):
        self._track_callback = track_callback

    @property
    def name(self) -> str:
        return "gui_track_next_frames"

    @property
    def description(self) -> str:
        return "Run Annolid tracking/prediction from current frame to target frame."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"to_frame": {"type": "integer", "minimum": 1}},
            "required": ["to_frame"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._track_callback, **kwargs)


class GuiSetAiTextPromptTool(FunctionTool):
    def __init__(self, set_ai_text_prompt_callback: Optional[ActionCallback] = None):
        self._set_ai_text_prompt_callback = set_ai_text_prompt_callback

    @property
    def name(self) -> str:
        return "gui_set_ai_text_prompt"

    @property
    def description(self) -> str:
        return "Set the GUI AI text prompt used by GroundingDINO + SAM segmentation."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "use_countgd": {"type": "boolean"},
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._set_ai_text_prompt_callback, **kwargs)


class GuiRunAiTextSegmentationTool(FunctionTool):
    def __init__(
        self, run_ai_text_segmentation_callback: Optional[ActionCallback] = None
    ):
        self._run_ai_text_segmentation_callback = run_ai_text_segmentation_callback

    @property
    def name(self) -> str:
        return "gui_run_ai_text_segmentation"

    @property
    def description(self) -> str:
        return (
            "Run GUI GroundingDINO + SAM segmentation using the current AI text prompt."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._run_ai_text_segmentation_callback)


class GuiSegmentTrackVideoTool(FunctionTool):
    def __init__(self, segment_track_video_callback: Optional[ActionCallback] = None):
        self._segment_track_video_callback = segment_track_video_callback

    @property
    def name(self) -> str:
        return "gui_segment_track_video"

    @property
    def description(self) -> str:
        return (
            "Open a video, run the legacy text-prompt GroundingDINO+SAM workflow, "
            "save, and optionally track to a target frame. Use sam3_agent_video_track "
            "when the selected tracking model is SAM3."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "text_prompt": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "enum": ["segment", "track"]},
                "use_countgd": {"type": "boolean"},
                "model_name": {"type": "string"},
                "to_frame": {"type": "integer", "minimum": 1},
            },
            "required": ["path", "text_prompt"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._segment_track_video_callback, **kwargs)


class GuiLabelBehaviorSegmentsTool(FunctionTool):
    def __init__(
        self, label_behavior_segments_callback: Optional[ActionCallback] = None
    ):
        self._label_behavior_segments_callback = label_behavior_segments_callback

    @property
    def name(self) -> str:
        return "gui_label_behavior_segments"

    @property
    def description(self) -> str:
        return (
            "Auto-label behavior intervals from video segments using an LLM model "
            "and write labels into the Annolid behavior timeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "behavior_labels": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "use_defined_behavior_list": {"type": "boolean"},
                "segment_mode": {"type": "string", "enum": ["timeline", "uniform"]},
                "segment_frames": {"type": "integer", "minimum": 1},
                "segment_seconds": {"type": "number", "minimum": 0.0},
                "sample_frames_per_segment": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 4,
                },
                "frames_per_grid": {"type": "integer", "minimum": 1, "default": 4},
                "max_segments": {"type": "integer", "minimum": 1},
                "subject": {"type": "string"},
                "overwrite_existing": {"type": "boolean"},
                "llm_profile": {"type": "string"},
                "llm_provider": {"type": "string"},
                "llm_model": {"type": "string"},
                "video_description": {"type": "string"},
                "instance_count": {"type": "integer", "minimum": 1},
                "experiment_context": {"type": "string"},
                "behavior_definitions": {"type": "string"},
                "focus_points": {"type": "string"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._label_behavior_segments_callback, **kwargs)


class GuiBehaviorCatalogTool(FunctionTool):
    def __init__(self, behavior_catalog_callback: Optional[ActionCallback] = None):
        self._behavior_catalog_callback = behavior_catalog_callback

    @property
    def name(self) -> str:
        return "gui_behavior_catalog"

    @property
    def description(self) -> str:
        return (
            "List, create, update, delete, and save the canonical Annolid behavior "
            "catalog shared by flags, timeline, and behavior labeling."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "save", "create", "update", "delete"],
                },
                "code": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "category_id": {"type": "string"},
                "modifier_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "key_binding": {"type": "string"},
                "is_state": {"type": "boolean"},
                "exclusive_with": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "save": {"type": "boolean"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._behavior_catalog_callback, **kwargs)


class GuiProcessVideoBehaviorsTool(FunctionTool):
    def __init__(
        self, process_video_behaviors_callback: Optional[ActionCallback] = None
    ):
        self._process_video_behaviors_callback = process_video_behaviors_callback

    @property
    def name(self) -> str:
        return "gui_process_video_behaviors"

    @property
    def description(self) -> str:
        return (
            "Run an end-to-end video behavior workflow in Annolid GUI: "
            "track/segment with a selected model and auto-label behavior segments."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "text_prompt": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "enum": ["segment", "track"]},
                "use_countgd": {"type": "boolean"},
                "model_name": {"type": "string"},
                "to_frame": {"type": "integer", "minimum": 1},
                "behavior_labels": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "use_defined_behavior_list": {"type": "boolean"},
                "segment_mode": {"type": "string", "enum": ["timeline", "uniform"]},
                "segment_frames": {"type": "integer", "minimum": 1},
                "segment_seconds": {"type": "number", "minimum": 0.0},
                "sample_frames_per_segment": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 4,
                },
                "frames_per_grid": {"type": "integer", "minimum": 1, "default": 4},
                "max_segments": {"type": "integer", "minimum": 1},
                "subject": {"type": "string"},
                "overwrite_existing": {"type": "boolean"},
                "llm_profile": {"type": "string"},
                "llm_provider": {"type": "string"},
                "llm_model": {"type": "string"},
                "video_description": {"type": "string"},
                "instance_count": {"type": "integer", "minimum": 1},
                "experiment_context": {"type": "string"},
                "behavior_definitions": {"type": "string"},
                "focus_points": {"type": "string"},
                "run_tracking": {"type": "boolean"},
                "run_behavior_labeling": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._process_video_behaviors_callback, **kwargs)


class GuiScoreAggressionBoutsTool(FunctionTool):
    def __init__(
        self, score_aggression_bouts_callback: Optional[ActionCallback] = None
    ):
        self._score_aggression_bouts_callback = score_aggression_bouts_callback

    @property
    def name(self) -> str:
        return "gui_score_aggression_bouts"

    @property
    def description(self) -> str:
        return (
            "Score aggression bouts from counted sub-events (slap in face, run away, "
            "fight initiation) and write a typed immutable analysis manifest."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "artifacts_ndjson": {"type": "string"},
                "run_id": {"type": "string"},
                "episode_id": {"type": "string"},
                "results_dir": {"type": "string"},
                "context_prompt": {"type": "string"},
                "assay": {"type": "string"},
                "default_assay": {"type": "string"},
                "model_policy": {"type": "string"},
                "bout_frame_gap": {"type": "integer", "minimum": 1},
                "no_memory": {"type": "boolean"},
                "no_analysis": {"type": "boolean"},
                "fail_on_validation_error": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._score_aggression_bouts_callback, **kwargs)


class GuiAnalyzeTrackingStatsTool(FunctionTool):
    def __init__(
        self, analyze_tracking_stats_callback: Optional[ActionCallback] = None
    ) -> None:
        self._analyze_tracking_stats_callback = analyze_tracking_stats_callback

    @property
    def name(self) -> str:
        return "gui_analyze_tracking_stats"

    @property
    def description(self) -> str:
        return (
            "Analyze Annolid *_tracking_stats.json files and return numeric summaries, "
            "per-video rankings, CSV artifact paths, and optional plot paths."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "root_dir": {"type": "string"},
                "output_dir": {"type": "string"},
                "video_id": {
                    "type": "string",
                    "description": "Optional case-insensitive substring filter.",
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                "include_plots": {"type": "boolean"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._analyze_tracking_stats_callback, **kwargs)


class GuiStartRealtimeStreamTool(FunctionTool):
    def __init__(self, start_realtime_stream_callback: Optional[ActionCallback] = None):
        self._start_realtime_stream_callback = start_realtime_stream_callback

    @property
    def name(self) -> str:
        return "gui_start_realtime_stream"

    @property
    def description(self) -> str:
        return (
            "Start realtime inference stream in Annolid and optionally enable "
            "MediaPipe face blink classification, bot reporting, and RTSP transport."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "camera_source": {"type": "string"},
                "model_name": {"type": "string"},
                "target_behaviors": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "viewer_type": {"type": "string", "enum": ["pyqt", "threejs"]},
                "rtsp_transport": {"type": "string", "enum": ["auto", "tcp", "udp"]},
                "classify_eye_blinks": {"type": "boolean"},
                "blink_ear_threshold": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.6,
                },
                "blink_min_consecutive_frames": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                },
                "bot_report_enabled": {"type": "boolean"},
                "bot_report_interval_sec": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 3600.0,
                },
                "bot_watch_labels": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "bot_email_report": {"type": "boolean"},
                "bot_email_to": {"type": "string"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._start_realtime_stream_callback, **kwargs)


class GuiStopRealtimeStreamTool(FunctionTool):
    def __init__(self, stop_realtime_stream_callback: Optional[ActionCallback] = None):
        self._stop_realtime_stream_callback = stop_realtime_stream_callback

    @property
    def name(self) -> str:
        return "gui_stop_realtime_stream"

    @property
    def description(self) -> str:
        return "Stop the current realtime inference stream in Annolid."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._stop_realtime_stream_callback)


class GuiGetRealtimeStatusTool(FunctionTool):
    def __init__(self, get_realtime_status_callback: Optional[ActionCallback] = None):
        self._get_realtime_status_callback = get_realtime_status_callback

    @property
    def name(self) -> str:
        return "gui_get_realtime_status"

    @property
    def description(self) -> str:
        return "Get realtime stream status, source, model, and active viewer details."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._get_realtime_status_callback)


class GuiListRealtimeModelsTool(FunctionTool):
    def __init__(self, list_realtime_models_callback: Optional[ActionCallback] = None):
        self._list_realtime_models_callback = list_realtime_models_callback

    @property
    def name(self) -> str:
        return "gui_list_realtime_models"

    @property
    def description(self) -> str:
        return "List available realtime model presets and their weight identifiers."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._list_realtime_models_callback)


class GuiListRealtimeLogsTool(FunctionTool):
    def __init__(self, list_realtime_logs_callback: Optional[ActionCallback] = None):
        self._list_realtime_logs_callback = list_realtime_logs_callback

    @property
    def name(self) -> str:
        return "gui_list_realtime_logs"

    @property
    def description(self) -> str:
        return "List current realtime detection and bot-event log file paths."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._list_realtime_logs_callback)


class GuiListLogsTool(FunctionTool):
    def __init__(self, list_logs_callback: Optional[ActionCallback] = None):
        self._list_logs_callback = list_logs_callback

    @property
    def name(self) -> str:
        return "gui_list_logs"

    @property
    def description(self) -> str:
        return (
            "List Annolid log folders (logs/app/realtime/runs/label_index) and paths."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._list_logs_callback)


class GuiOpenLogFolderTool(FunctionTool):
    def __init__(self, open_log_folder_callback: Optional[ActionCallback] = None):
        self._open_log_folder_callback = open_log_folder_callback

    @property
    def name(self) -> str:
        return "gui_open_log_folder"

    @property
    def description(self) -> str:
        return "Open a log folder in the system file browser."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["logs", "realtime", "runs", "label_index", "app"],
                }
            },
            "required": ["target"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_log_folder_callback, **kwargs)


class GuiRemoveLogFolderTool(FunctionTool):
    def __init__(self, remove_log_folder_callback: Optional[ActionCallback] = None):
        self._remove_log_folder_callback = remove_log_folder_callback

    @property
    def name(self) -> str:
        return "gui_remove_log_folder"

    @property
    def description(self) -> str:
        return "Delete a log folder recursively."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["logs", "realtime", "runs", "label_index", "app"],
                }
            },
            "required": ["target"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._remove_log_folder_callback, **kwargs)


class GuiListLogFilesTool(FunctionTool):
    def __init__(self, list_log_files_callback: Optional[ActionCallback] = None):
        self._list_log_files_callback = list_log_files_callback

    @property
    def name(self) -> str:
        return "gui_list_log_files"

    @property
    def description(self) -> str:
        return "List files within an Annolid log target with sorting controls."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["logs", "realtime", "runs", "label_index", "app"],
                },
                "pattern": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                "recursive": {"type": "boolean"},
                "sort_by": {"type": "string", "enum": ["name", "mtime", "size"]},
                "descending": {"type": "boolean"},
            },
            "required": ["target"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._list_log_files_callback, **kwargs)


class GuiReadLogFileTool(FunctionTool):
    def __init__(self, read_log_file_callback: Optional[ActionCallback] = None):
        self._read_log_file_callback = read_log_file_callback

    @property
    def name(self) -> str:
        return "gui_read_log_file"

    @property
    def description(self) -> str:
        return "Read log file content (tail-friendly) from allowed Annolid log roots."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "max_chars": {"type": "integer", "minimum": 200, "maximum": 200000},
                "tail_lines": {"type": "integer", "minimum": 1, "maximum": 100000},
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._read_log_file_callback, **kwargs)


class GuiSearchLogsTool(FunctionTool):
    def __init__(self, search_logs_callback: Optional[ActionCallback] = None):
        self._search_logs_callback = search_logs_callback

    @property
    def name(self) -> str:
        return "gui_search_logs"

    @property
    def description(self) -> str:
        return (
            "Search text across Annolid log files and return matched lines with "
            "optional regex and case-sensitivity."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "target": {
                    "type": "string",
                    "enum": ["logs", "realtime", "runs", "label_index", "app"],
                },
                "pattern": {"type": "string"},
                "case_sensitive": {"type": "boolean"},
                "use_regex": {"type": "boolean"},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 2000},
                "max_files": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._search_logs_callback, **kwargs)


class GuiCheckStreamSourceTool(FunctionTool):
    def __init__(self, check_stream_source_callback: Optional[ActionCallback] = None):
        self._check_stream_source_callback = check_stream_source_callback

    @property
    def name(self) -> str:
        return "gui_check_stream_source"

    @property
    def description(self) -> str:
        return (
            "Probe a camera/video stream source for connectivity and frame availability "
            "before starting realtime inference."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "camera_source": {"type": "string"},
                "rtsp_transport": {"type": "string", "enum": ["auto", "tcp", "udp"]},
                "timeout_sec": {"type": "number", "minimum": 0.5, "maximum": 30.0},
                "probe_frames": {"type": "integer", "minimum": 1, "maximum": 60},
                "save_snapshot": {"type": "boolean"},
                "email_to": {"type": "string"},
                "email_subject": {"type": "string"},
                "email_content": {"type": "string"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._check_stream_source_callback, **kwargs)
