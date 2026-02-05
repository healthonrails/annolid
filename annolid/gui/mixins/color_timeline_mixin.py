from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import imgviz
from qtpy import QtGui

try:
    LABEL_COLORMAP = imgviz.label_colormap(value=200)
except TypeError:
    LABEL_COLORMAP = imgviz.label_colormap()


def _hex_to_rgb(color: str) -> Optional[Tuple[int, int, int]]:
    color = color.strip()
    if not color.startswith("#"):
        return None
    hex_value = color[1:]
    if len(hex_value) == 6:
        try:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            return (r, g, b)
        except ValueError:
            return None
    return None


class ColorTimelineMixin:
    """Label color and timeline behavior catalog helpers."""

    def _get_rgb_by_label(self, label):
        schema = getattr(self, "project_schema", None)
        if schema is not None:
            behavior = schema.behavior_map().get(label)
            if behavior is None:
                behavior = next(
                    (
                        beh
                        for beh in schema.behaviors
                        if beh.name.lower() == label.lower()
                    ),
                    None,
                )
            if behavior is not None and behavior.category_id:
                category = schema.category_map().get(behavior.category_id)
                if category and category.color:
                    rgb = _hex_to_rgb(category.color)
                    if rgb is not None:
                        return rgb

        config = self._config
        if config.get("shape_color") == "auto":
            normalized_label = label.strip().lower()
            hash_digest = hashlib.md5(normalized_label.encode("utf-8")).hexdigest()
            hash_int = int(hash_digest, 16)
            shift_offset = config.get("shift_auto_shape_color", 0)
            index = (hash_int + shift_offset) % len(LABEL_COLORMAP)
            return (
                int(LABEL_COLORMAP[index][0]),
                int(LABEL_COLORMAP[index][1]),
                int(LABEL_COLORMAP[index][2]),
            )
        elif (
            config.get("shape_color") == "manual"
            and config.get("label_colors")
            and label in config["label_colors"]
        ):
            return config["label_colors"][label]
        elif config.get("default_shape_color"):
            return config["default_shape_color"]

    def _timeline_behavior_catalog(self) -> List[str]:
        behaviors: set[str] = set()
        schema = getattr(self, "project_schema", None)
        if schema is not None:
            try:
                behaviors.update(schema.behavior_map().keys())
            except Exception:
                pass
        try:
            behaviors.update(getattr(self, "pinned_flags", {}).keys())
        except Exception:
            pass
        try:
            if getattr(self, "flag_widget", None) is not None:
                behaviors.update(self.flag_widget._get_existing_flag_names().keys())
        except Exception:
            pass
        try:
            behaviors.update(
                getattr(self, "behavior_controller", None).behavior_names or set()
            )
        except Exception:
            pass
        return sorted({b for b in behaviors if b})

    def _timeline_add_behavior(self, name: str) -> None:
        name = str(name).strip()
        if not name:
            return
        try:
            if getattr(self, "flag_widget", None) is not None:
                self.flag_widget.add_row(name, False)
        except Exception:
            pass

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        if not shape.visible:
            shape.vertex_fill_color = QtGui.QColor(r, g, b, 0)
        else:
            shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)
        return r, g, b
