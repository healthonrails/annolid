from __future__ import annotations

from typing import Any, Optional

from .function_gui_base import ActionCallback, _run_callback
from .function_base import FunctionTool


class GuiOpenThreeJsTool(FunctionTool):
    def __init__(self, open_threejs_callback: Optional[ActionCallback] = None):
        self._open_threejs_callback = open_threejs_callback

    @property
    def name(self) -> str:
        return "gui_open_threejs"

    @property
    def description(self) -> str:
        return (
            "Open local/remote Three.js content in Annolid 3D view. "
            "Accepts model paths or HTML/URL targets."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path_or_url": {"type": "string", "minLength": 1}},
            "required": ["path_or_url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_threejs_callback, **kwargs)


class GuiOpenThreeJsExampleTool(FunctionTool):
    def __init__(self, open_threejs_example_callback: Optional[ActionCallback] = None):
        self._open_threejs_example_callback = open_threejs_example_callback

    @property
    def name(self) -> str:
        return "gui_open_threejs_example"

    @property
    def description(self) -> str:
        return (
            "Open a built-in Three.js example in Annolid 3D view. "
            "Known examples: two_mice_html, brain_viewer_html, helix_points_csv, "
            "wave_surface_obj, sphere_points_ply."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"example_id": {"type": "string"}},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_threejs_example_callback, **kwargs)
