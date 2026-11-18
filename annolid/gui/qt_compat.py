from __future__ import annotations

from typing import Any, Optional

from qtpy import QtCore, QtGui, QtWidgets


def _qt_enum(owner: Any, enum_name: str, member_name: str) -> Any:
    """Resolve Qt enums across Qt5-style and Qt6-style attribute layouts."""
    enum_type = getattr(owner, enum_name, None)
    if enum_type is not None and hasattr(enum_type, member_name):
        return getattr(enum_type, member_name)
    return getattr(owner, member_name)


def palette_color_role(name: str) -> Any:
    return _qt_enum(QtGui.QPalette, "ColorRole", name)


def palette_color_group(name: str) -> Any:
    return _qt_enum(QtGui.QPalette, "ColorGroup", name)


def painter_render_hint(name: str) -> Any:
    return _qt_enum(QtGui.QPainter, "RenderHint", name)


def normalize_orientation(value: Any) -> Any:
    enum_value = getattr(value, "value", value)
    enum_name = getattr(value, "name", None)
    horizontal = _qt_enum(QtCore.Qt, "Orientation", "Horizontal")
    vertical = _qt_enum(QtCore.Qt, "Orientation", "Vertical")
    if value == horizontal or enum_name == "Horizontal":
        return horizontal
    if value == vertical or enum_name == "Vertical":
        return vertical
    resolved_ints: dict[Any, int] = {}
    for normalized, enum_member in ((horizontal, horizontal), (vertical, vertical)):
        with_int = getattr(enum_member, "value", None)
        if enum_value == with_int:
            return normalized
        try:
            enum_int = int(enum_member)
            resolved_ints[normalized] = enum_int
            if enum_value == enum_int:
                return normalized
        except Exception:
            pass
    if enum_value == 0:
        return horizontal
    if not resolved_ints and enum_value == 1:
        return vertical
    return value


def configure_compact_combo(
    combo: QtWidgets.QComboBox,
    *,
    min_chars: int = 12,
    popup_min_width: int = 280,
    max_width: Optional[int] = None,
) -> None:
    """Keep a closed QComboBox narrow inside docks/toolbars while widening its popup."""
    adjust_policy = _qt_enum(
        QtWidgets.QComboBox,
        "SizeAdjustPolicy",
        "AdjustToMinimumContentsLengthWithIcon",
    )
    combo.setSizeAdjustPolicy(adjust_policy)
    combo.setMinimumContentsLength(max(4, int(min_chars)))
    h_policy = (
        _qt_enum(QtWidgets.QSizePolicy, "Policy", "Preferred")
        if max_width is not None
        else _qt_enum(QtWidgets.QSizePolicy, "Policy", "Expanding")
    )
    v_policy = _qt_enum(QtWidgets.QSizePolicy, "Policy", "Fixed")
    combo.setSizePolicy(h_policy, v_policy)
    if max_width is not None:
        combo.setMaximumWidth(int(max_width))
    view = combo.view()
    if view is not None:
        view.setMinimumWidth(max(120, int(popup_min_width)))
        elide_none = _qt_enum(QtCore.Qt, "TextElideMode", "ElideNone")
        try:
            view.setTextElideMode(elide_none)
        except Exception:
            pass
