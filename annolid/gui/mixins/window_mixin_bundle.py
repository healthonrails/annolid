from __future__ import annotations

from .ai_mask_prompt_mixin import AiMaskPromptMixin
from .agent_analysis_mixin import AgentAnalysisMixin
from .annotation_loading_mixin import AnnotationLoadingMixin
from .behavior_interaction_mixin import BehaviorInteractionMixin
from .behavior_log_mixin import BehaviorLogMixin
from .behavior_time_budget_mixin import BehaviorTimeBudgetMixin
from .canvas_workflow_mixin import CanvasWorkflowMixin
from .color_timeline_mixin import ColorTimelineMixin
from .core_interaction_mixin import CoreInteractionMixin
from .csv_conversion_mixin import CsvConversionMixin
from .depth_sam_proxy_mixin import DepthSamProxyMixin
from .file_browser_mixin import FileBrowserMixin
from .flags_overlay_mixin import FlagsOverlayMixin
from .frame_playback_mixin import FramePlaybackMixin
from .label_panel_mixin import LabelPanelMixin
from .lifecycle_mixin import LifecycleMixin
from .media_workflow_mixin import MediaWorkflowMixin
from .model_identity_mixin import ModelIdentityMixin
from .navigation_workflow_mixin import NavigationWorkflowMixin
from .persistence_lifecycle_mixin import PersistenceLifecycleMixin
from .playback_draw_mixin import PlaybackDrawMixin
from .prediction_execution_mixin import PredictionExecutionMixin
from .prediction_progress_mixin import PredictionProgressMixin
from .project_workflow_mixin import ProjectWorkflowMixin
from .schema_behavior_loader_mixin import SchemaBehaviorLoaderMixin
from .settings_timeline_mixin import SettingsTimelineMixin
from .shape_editing_mixin import ShapeEditingMixin
from .tooling_dialogs_mixin import ToolingDialogsMixin
from .tracking_segment_mixin import TrackingSegmentMixin
from .training_workflow_mixin import TrainingWorkflowMixin
from .viewer_tools_mixin import ViewerToolsMixin
from .video_workflow_mixin import VideoWorkflowMixin
from .window_lifecycle_mixin import WindowLifecycleMixin
from .workflow_actions_mixin import WorkflowActionsMixin


class AnnolidWindowMixinBundle(
    CoreInteractionMixin,
    LabelPanelMixin,
    MediaWorkflowMixin,
    ToolingDialogsMixin,
    FileBrowserMixin,
    ColorTimelineMixin,
    PlaybackDrawMixin,
    WindowLifecycleMixin,
    FlagsOverlayMixin,
    AiMaskPromptMixin,
    PredictionExecutionMixin,
    PersistenceLifecycleMixin,
    TrainingWorkflowMixin,
    BehaviorTimeBudgetMixin,
    BehaviorInteractionMixin,
    SchemaBehaviorLoaderMixin,
    FramePlaybackMixin,
    BehaviorLogMixin,
    AgentAnalysisMixin,
    ProjectWorkflowMixin,
    CsvConversionMixin,
    PredictionProgressMixin,
    ModelIdentityMixin,
    DepthSamProxyMixin,
    ShapeEditingMixin,
    WorkflowActionsMixin,
    TrackingSegmentMixin,
    AnnotationLoadingMixin,
    LifecycleMixin,
    NavigationWorkflowMixin,
    ViewerToolsMixin,
    CanvasWorkflowMixin,
    VideoWorkflowMixin,
    SettingsTimelineMixin,
):
    """Ordered mixin composition for `AnnolidWindow`."""

    pass
