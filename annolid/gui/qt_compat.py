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


def event_pos_f(event: Any) -> QtCore.QPointF:
    """Return event position as ``QPointF`` across Qt5 (``localPos``/``posF``) and Qt6 (``position``)."""
    position_fn = getattr(event, "position", None)
    if callable(position_fn):
        return QtCore.QPointF(position_fn())
    local_pos_fn = getattr(event, "localPos", None)
    if callable(local_pos_fn):
        return QtCore.QPointF(local_pos_fn())
    pos_f_fn = getattr(event, "posF", None)
    if callable(pos_f_fn):
        return QtCore.QPointF(pos_f_fn())
    pos_fn = getattr(event, "pos", None)
    if callable(pos_fn):
        return QtCore.QPointF(pos_fn())
    raise AttributeError(f"Unsupported Qt event type for position: {type(event)!r}")


def event_pos_i(event: Any) -> QtCore.QPoint:
    """Return integer widget position as ``QPoint`` across Qt5 and Qt6."""
    return event_pos_f(event).toPoint()


def event_global_pos_i(event: Any) -> QtCore.QPoint:
    """Return integer global position as ``QPoint`` across Qt5 and Qt6."""
    global_position_fn = getattr(event, "globalPosition", None)
    if callable(global_position_fn):
        return global_position_fn().toPoint()
    global_pos_fn = getattr(event, "globalPos", None)
    if callable(global_pos_fn):
        return QtCore.QPoint(global_pos_fn())
    raise AttributeError(
        f"Unsupported Qt event type for global position: {type(event)!r}"
    )


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
