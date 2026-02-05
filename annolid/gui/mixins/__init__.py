"""Mixin helpers for Annolid GUI windows."""

from .settings_timeline_mixin import SettingsTimelineMixin
from .video_workflow_mixin import VideoWorkflowMixin
from .canvas_workflow_mixin import CanvasWorkflowMixin
from .viewer_tools_mixin import ViewerToolsMixin
from .navigation_workflow_mixin import NavigationWorkflowMixin
from .lifecycle_mixin import LifecycleMixin
from .annotation_loading_mixin import AnnotationLoadingMixin
from .tracking_segment_mixin import TrackingSegmentMixin
from .workflow_actions_mixin import WorkflowActionsMixin
from .shape_editing_mixin import ShapeEditingMixin
from .depth_sam_proxy_mixin import DepthSamProxyMixin
from .model_identity_mixin import ModelIdentityMixin
from .prediction_progress_mixin import PredictionProgressMixin
from .csv_conversion_mixin import CsvConversionMixin
from .project_workflow_mixin import ProjectWorkflowMixin
from .agent_analysis_mixin import AgentAnalysisMixin
from .behavior_log_mixin import BehaviorLogMixin
from .frame_playback_mixin import FramePlaybackMixin
from .schema_behavior_loader_mixin import SchemaBehaviorLoaderMixin
from .behavior_interaction_mixin import BehaviorInteractionMixin
from .behavior_time_budget_mixin import BehaviorTimeBudgetMixin
from .training_workflow_mixin import TrainingWorkflowMixin
from .persistence_lifecycle_mixin import PersistenceLifecycleMixin
from .prediction_execution_mixin import PredictionExecutionMixin
from .ai_mask_prompt_mixin import AiMaskPromptMixin
from .flags_overlay_mixin import FlagsOverlayMixin
from .label_panel_mixin import LabelPanelMixin
from .media_workflow_mixin import MediaWorkflowMixin
from .tooling_dialogs_mixin import ToolingDialogsMixin
from .file_browser_mixin import FileBrowserMixin
from .color_timeline_mixin import ColorTimelineMixin
from .playback_draw_mixin import PlaybackDrawMixin
from .window_lifecycle_mixin import WindowLifecycleMixin
from .core_interaction_mixin import CoreInteractionMixin
from .window_mixin_bundle import AnnolidWindowMixinBundle

__all__ = [
    "SettingsTimelineMixin",
    "VideoWorkflowMixin",
    "CanvasWorkflowMixin",
    "ViewerToolsMixin",
    "NavigationWorkflowMixin",
    "LifecycleMixin",
    "AnnotationLoadingMixin",
    "TrackingSegmentMixin",
    "WorkflowActionsMixin",
    "ShapeEditingMixin",
    "DepthSamProxyMixin",
    "ModelIdentityMixin",
    "PredictionProgressMixin",
    "CsvConversionMixin",
    "ProjectWorkflowMixin",
    "AgentAnalysisMixin",
    "BehaviorLogMixin",
    "FramePlaybackMixin",
    "SchemaBehaviorLoaderMixin",
    "BehaviorInteractionMixin",
    "BehaviorTimeBudgetMixin",
    "TrainingWorkflowMixin",
    "PersistenceLifecycleMixin",
    "PredictionExecutionMixin",
    "AiMaskPromptMixin",
    "FlagsOverlayMixin",
    "LabelPanelMixin",
    "MediaWorkflowMixin",
    "ToolingDialogsMixin",
    "FileBrowserMixin",
    "ColorTimelineMixin",
    "PlaybackDrawMixin",
    "WindowLifecycleMixin",
    "CoreInteractionMixin",
    "AnnolidWindowMixinBundle",
]
