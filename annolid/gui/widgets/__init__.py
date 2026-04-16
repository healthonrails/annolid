"""Lazy exports for GUI widgets.

This package is imported on startup (including when importing submodules), so
we keep imports lazy to avoid pulling heavy optional dependencies too early.
"""

from __future__ import annotations

import importlib


_LAZY_EXPORTS = {
    "ExtractFrameDialog": (
        "annolid.gui.widgets.extract_frame_dialog",
        "ExtractFrameDialog",
    ),
    "ConvertCOODialog": ("annolid.gui.widgets.convert_coco_dialog", "ConvertCOODialog"),
    "ConvertCOCO2LabelMeDialog": (
        "annolid.gui.widgets.convert_coco2labelme_dialog",
        "ConvertCOCO2LabelMeDialog",
    ),
    "TrainModelDialog": ("annolid.gui.widgets.train_model_dialog", "TrainModelDialog"),
    "TrackDialog": ("annolid.gui.widgets.track_dialog", "TrackDialog"),
    "Glitter2Dialog": ("annolid.gui.widgets.glitter2_dialog", "Glitter2Dialog"),
    "ProgressingWindow": (
        "annolid.gui.widgets.progressing_dialog",
        "ProgressingWindow",
    ),
    "QualityControlDialog": (
        "annolid.gui.widgets.quality_control_dialog",
        "QualityControlDialog",
    ),
    "SystemInfoDialog": ("annolid.gui.widgets.about_dialog", "SystemInfoDialog"),
    "FlagTableWidget": ("annolid.gui.widgets.flags", "FlagTableWidget"),
    "RecordingWidget": ("annolid.gui.widgets.video_recording", "RecordingWidget"),
    "CanvasScreenshotWidget": (
        "annolid.gui.widgets.screen_shot",
        "CanvasScreenshotWidget",
    ),
    "ConvertDLCDialog": (
        "annolid.gui.widgets.convert_deeplabcut_dialog",
        "ConvertDLCDialog",
    ),
    "DepthSettingsDialog": (
        "annolid.gui.widgets.depth_settings_dialog",
        "DepthSettingsDialog",
    ),
    "AnnolidLabelDialog": ("annolid.gui.widgets.label_dialog", "AnnolidLabelDialog"),
    "LabelCollectionDialog": (
        "annolid.gui.widgets.label_collection_dialog",
        "LabelCollectionDialog",
    ),
    "LogManagerDialog": ("annolid.gui.widgets.log_manager_dialog", "LogManagerDialog"),
    "TrainingDashboardDialog": (
        "annolid.gui.widgets.training_dashboard",
        "TrainingDashboardDialog",
    ),
    "TrainingDashboardWidget": (
        "annolid.gui.widgets.training_dashboard",
        "TrainingDashboardWidget",
    ),
    "LabelingProgressDashboardDialog": (
        "annolid.gui.widgets.labeling_progress_dashboard",
        "LabelingProgressDashboardDialog",
    ),
    "LabelingProgressDashboardWidget": (
        "annolid.gui.widgets.labeling_progress_dashboard",
        "LabelingProgressDashboardWidget",
    ),
    "TrackingStatsDashboardDialog": (
        "annolid.gui.widgets.tracking_stats_dashboard_dialog",
        "TrackingStatsDashboardDialog",
    ),
    "TrackingStatsDashboardWidget": (
        "annolid.gui.widgets.tracking_stats_dashboard_dialog",
        "TrackingStatsDashboardWidget",
    ),
    "AgentRunDialog": ("annolid.gui.widgets.agent_run_dialog", "AgentRunDialog"),
    "EmbeddingSearchWidget": (
        "annolid.gui.widgets.embedding_search_widget",
        "EmbeddingSearchWidget",
    ),
    "BatchRelabelDialog": (
        "annolid.gui.widgets.batch_relabel_dialog",
        "BatchRelabelDialog",
    ),
    "IdentityGovernorDialog": (
        "annolid.gui.widgets.identity_governor_dialog",
        "IdentityGovernorDialog",
    ),
    "KeypointSequencerWidget": (
        "annolid.gui.widgets.keypoint_sequencer",
        "KeypointSequencerWidget",
    ),
    "ZoneDockWidget": ("annolid.gui.widgets.zone_dock", "ZoneDockWidget"),
    "ProjectWizard": ("annolid.gui.widgets.project_wizard", "ProjectWizard"),
    "DashboardWidget": ("annolid.gui.widgets.dashboard", "DashboardWidget"),
    "DatasetExportWizard": (
        "annolid.gui.widgets.dataset_wizard",
        "DatasetExportWizard",
    ),
    "TrainingWizard": ("annolid.gui.widgets.training_wizard", "TrainingWizard"),
    "InferenceWizard": ("annolid.gui.widgets.inference_wizard", "InferenceWizard"),
}


def __getattr__(name: str):
    if name == "RealtimeControlWidget":
        try:
            module = importlib.import_module(
                "annolid.gui.widgets.realtime_control_widget"
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RealtimeControlWidget requires optional dependency 'pyzmq'."
            ) from exc
        value = getattr(module, "RealtimeControlWidget")
        globals()[name] = value
        return value

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted([*_LAZY_EXPORTS.keys(), "RealtimeControlWidget"])
