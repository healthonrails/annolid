from __future__ import annotations

import hashlib
import json
from typing import List, Optional, Tuple

import imgviz
from qtpy import QtGui

from annolid.core.behavior.catalog import behavior_catalog_entries

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


def _normalize_rgb(value) -> Optional[Tuple[int, int, int]]:
    if isinstance(value, QtGui.QColor):
        if value.isValid():
            return (int(value.red()), int(value.green()), int(value.blue()))
        return None
    if isinstance(value, str):
        return _hex_to_rgb(value)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            rgb = tuple(max(0, min(255, int(v))) for v in value[:3])
        except (TypeError, ValueError):
            return None
        return rgb
    return None


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = _normalize_rgb(rgb) or (0, 255, 0)
    return f"#{r:02x}{g:02x}{b:02x}"


class ColorTimelineMixin:
    """Label color and timeline behavior catalog helpers."""

    LABEL_COLOR_OVERRIDES_KEY = "labels/color_overrides_json"

    def _load_label_color_overrides_from_settings(self) -> None:
        settings = getattr(self, "settings", None)
        if settings is None:
            return
        raw = ""
        try:
            raw = settings.value(self.LABEL_COLOR_OVERRIDES_KEY, "", type=str) or ""
        except Exception:
            raw = ""
        overrides = {}
        if raw:
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                payload = {}
            if isinstance(payload, dict):
                for label, color in payload.items():
                    rgb = _normalize_rgb(color)
                    if str(label or "").strip() and rgb is not None:
                        overrides[str(label)] = rgb
        self._config.setdefault("label_color_overrides", {})
        self._config["label_color_overrides"] = overrides

    def _persist_label_color_overrides(self) -> None:
        settings = getattr(self, "settings", None)
        if settings is None:
            return
        overrides = dict(self._config.get("label_color_overrides") or {})
        payload = {
            str(label): _rgb_to_hex(rgb)
            for label, rgb in sorted(
                overrides.items(), key=lambda item: item[0].lower()
            )
            if str(label or "").strip() and _normalize_rgb(rgb) is not None
        }
        try:
            settings.setValue(self.LABEL_COLOR_OVERRIDES_KEY, json.dumps(payload))
            settings.sync()
        except Exception:
            pass

    def _set_label_color_override(self, label: str, color) -> bool:
        label = str(label or "").strip()
        rgb = _normalize_rgb(color)
        if not label or rgb is None:
            return False
        overrides = dict(self._config.get("label_color_overrides") or {})
        overrides[label] = rgb
        self._config["label_color_overrides"] = overrides
        self._persist_label_color_overrides()
        return True

    def _reset_label_color_override(self, label: str) -> bool:
        label = str(label or "").strip()
        if not label:
            return False
        overrides = dict(self._config.get("label_color_overrides") or {})
        if label not in overrides:
            return False
        overrides.pop(label, None)
        self._config["label_color_overrides"] = overrides
        self._persist_label_color_overrides()
        return True

    def _get_rgb_by_label(self, label):
        label = str(label or "").strip()
        overrides = self._config.get("label_color_overrides") or {}
        rgb = _normalize_rgb(overrides.get(label))
        if rgb is not None:
            return rgb

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
            normalized_label = label.lower()
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
            rgb = _normalize_rgb(config["label_colors"][label])
            if rgb is not None:
                return rgb
        elif config.get("default_shape_color"):
            rgb = _normalize_rgb(config["default_shape_color"])
            if rgb is not None:
                return rgb
        return (0, 255, 0)

    def _timeline_behavior_catalog(self) -> List[str]:
        behaviors: set[str] = set()
        schema = getattr(self, "project_schema", None)
        if schema is not None:
            try:
                entries = behavior_catalog_entries(schema)
                behaviors.update(
                    item["code"]
                    for item in entries
                    if str(item.get("code") or "").strip()
                )
                behaviors.update(
                    item["name"]
                    for item in entries
                    if str(item.get("name") or "").strip()
                )
            except Exception:
                pass
        try:
            behaviors.update(getattr(self, "pinned_flags", {}).keys())
        except Exception:
            pass
        try:
            if getattr(self, "flag_widget", None) is not None:
                if hasattr(self.flag_widget, "behavior_names"):
                    behaviors.update(self.flag_widget.behavior_names())
                else:
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
        creator = getattr(self, "create_behavior_catalog_item", None)
        if callable(creator):
            try:
                creator(code=name, name=name, save=True)
            except Exception:
                pass
        current_flags = dict(getattr(self, "pinned_flags", {}) or {})
        if name not in current_flags:
            current_flags[name] = False
        try:
            self.loadFlags(current_flags)
        except Exception:
            try:
                self.pinned_flags = current_flags
            except Exception:
                pass
        timeline_panel = getattr(self, "timeline_panel", None)
        if timeline_panel is not None:
            try:
                timeline_panel.refresh_behavior_catalog()
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
