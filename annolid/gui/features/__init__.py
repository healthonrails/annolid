"""Feature setup entrypoints for Annolid GUI."""

from .container import GuiFeatureDeps
from .annotation import setup_annotation_feature
from .search import setup_search_feature
from .timeline import setup_timeline_feature
from .video import setup_video_feature
from .viewers import setup_viewers_feature

__all__ = [
    "GuiFeatureDeps",
    "setup_annotation_feature",
    "setup_search_feature",
    "setup_timeline_feature",
    "setup_video_feature",
    "setup_viewers_feature",
]
