"""Feature setup entrypoints for Annolid GUI."""

from .container import GuiFeatureDeps
from .annotation import setup_annotation_feature
from .zones import setup_zone_feature
from .search import setup_search_feature
from .timeline import setup_timeline_feature
from .video import setup_video_feature
from .viewers import (
    ensure_ai_chat_manager,
    ensure_depth_manager,
    ensure_pdf_manager,
    ensure_realtime_manager,
    ensure_sam3d_manager,
    ensure_threejs_manager,
    ensure_web_manager,
    setup_viewers_feature,
)

__all__ = [
    "GuiFeatureDeps",
    "setup_annotation_feature",
    "setup_zone_feature",
    "setup_search_feature",
    "setup_timeline_feature",
    "setup_video_feature",
    "setup_viewers_feature",
    "ensure_pdf_manager",
    "ensure_web_manager",
    "ensure_threejs_manager",
    "ensure_depth_manager",
    "ensure_sam3d_manager",
    "ensure_realtime_manager",
    "ensure_ai_chat_manager",
]
