from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from annolid.io.large_image.base import LargeImageBackendCapabilities
from annolid.gui.viewer_layers import ViewerLayerModel


@dataclass
class LargeImageViewport:
    zoom_percent: int = 100
    center_x: float = 0.0
    center_y: float = 0.0
    fit_mode: str = "fit_window"


@dataclass
class LargeImageSelectionState:
    selected_shape_count: int = 0
    selected_shape_labels: list[str] = field(default_factory=list)
    selected_overlay_id: str | None = None
    selected_landmark_pair_id: str | None = None
    selected_label_value: int | None = None


@dataclass
class LargeImageDocument:
    image_path: str = ""
    backend: Any = None
    backend_name: str = ""
    backend_capabilities: LargeImageBackendCapabilities = field(
        default_factory=LargeImageBackendCapabilities
    )
    current_page: int = 0
    page_count: int = 1
    surface: str = "canvas"
    draw_mode: str = "polygon"
    editing: bool = True
    viewport: LargeImageViewport = field(default_factory=LargeImageViewport)
    active_layers: list[ViewerLayerModel] = field(default_factory=list)
    active_label_layer_id: str | None = None
    label_overlay_state: dict[str, Any] = field(default_factory=dict)
    cache_metadata: dict[str, Any] = field(default_factory=dict)
    selection: LargeImageSelectionState = field(
        default_factory=LargeImageSelectionState
    )
