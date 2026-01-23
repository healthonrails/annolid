from annolid.gui.widgets.extract_frame_dialog import ExtractFrameDialog
from annolid.gui.widgets.convert_coco_dialog import ConvertCOODialog
from annolid.gui.widgets.train_model_dialog import TrainModelDialog
from annolid.gui.widgets.track_dialog import TrackDialog
from annolid.gui.widgets.glitter2_dialog import Glitter2Dialog
from annolid.gui.widgets.progressing_dialog import ProgressingWindow
from annolid.gui.widgets.quality_control_dialog import QualityControlDialog
from annolid.gui.widgets.about_dialog import SystemInfoDialog
from annolid.gui.widgets.flags import FlagTableWidget
from annolid.gui.widgets.video_recording import RecordingWidget
from annolid.gui.widgets.screen_shot import CanvasScreenshotWidget
from annolid.gui.widgets.convert_deeplabcut_dialog import ConvertDLCDialog
from annolid.gui.widgets.realtime_control_widget import RealtimeControlWidget
from annolid.gui.widgets.depth_settings_dialog import DepthSettingsDialog
from annolid.gui.widgets.label_collection_dialog import LabelCollectionDialog
from annolid.gui.widgets.training_dashboard import (
    TrainingDashboardDialog,
    TrainingDashboardWidget,
)
from annolid.gui.widgets.agent_run_dialog import AgentRunDialog

# New streamlined wizards
from annolid.gui.widgets.project_wizard import ProjectWizard
from annolid.gui.widgets.dashboard import DashboardWidget
from annolid.gui.widgets.dataset_wizard import DatasetExportWizard
from annolid.gui.widgets.training_wizard import TrainingWizard
from annolid.gui.widgets.inference_wizard import InferenceWizard

__all__ = [
    "ExtractFrameDialog",
    "ConvertCOODialog",
    "TrainModelDialog",
    "TrackDialog",
    "Glitter2Dialog",
    "ProgressingWindow",
    "QualityControlDialog",
    "SystemInfoDialog",
    "FlagTableWidget",
    "RecordingWidget",
    "CanvasScreenshotWidget",
    "ConvertDLCDialog",
    "RealtimeControlWidget",
    "DepthSettingsDialog",
    "LabelCollectionDialog",
    "TrainingDashboardDialog",
    "TrainingDashboardWidget",
    "AgentRunDialog",
    "ProjectWizard",
    "DashboardWidget",
    "DatasetExportWizard",
    "TrainingWizard",
    "InferenceWizard",
]
