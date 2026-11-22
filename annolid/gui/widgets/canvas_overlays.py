"""Overlay rendering mixin for Canvas (pose skeletons, RGBA maps, landmarks, behavior text)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.annotation.pose_schema import PoseSchema

_CURSOR_POINT = QtCore.Qt.PointingHandCursor

_VALID_BEHAVIOR_OVERLAY_POSITIONS = frozenset(
    {"top_left", "top_right", "bottom_left", "bottom_right"}
)

_FALLBACK_BEHAVIOR_PALETTE: tuple[tuple[int, int, int], ...] = (
    (56, 189, 248),  # sky
    (52, 211, 153),  # emerald
    (251, 191, 36),  # amber
    (248, 113, 113),  # coral
    (167, 139, 250),  # violet
    (244, 114, 182),  # rose
    (45, 212, 191),  # teal
    (251, 146, 60),  # orange
    (96, 165, 250),  # blue
    (163, 230, 53),  # lime
    (232, 121, 249),  # fuchsia
    (129, 140, 248),  # indigo
)


@dataclass(frozen=True)
class BehaviorOverlayItem:
    """Structured behavior badge rendered by CanvasOverlayMixin."""

    label: str
    kind: str = "state"  # "state", "point", or "skipped"
    subject: str | None = None
    color: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class _PositionedBehaviorPill:
    """Layout result for a single rendered behavior pill."""

    rect: QtCore.QRectF
    text: str
    item: BehaviorOverlayItem | None
    is_overflow: bool
    dot_diameter: float
    dot_gap: float
    pad_x: float


def _normalize_rgb_tuple(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, QtGui.QColor):
        if value.isValid():
            return (value.red(), value.green(), value.blue())
        return None
    if isinstance(value, str):
        color = QtGui.QColor(value.strip())
        if color.isValid():
            return (color.red(), color.green(), color.blue())
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        try:
            r = max(0, min(255, int(value[0])))
            g = max(0, min(255, int(value[1])))
            b = max(0, min(255, int(value[2])))
            return (r, g, b)
        except Exception:
            return None
    return None


class BehaviorOverlayMixin:
    """Shared viewport-pinned, color-coded behavior pill overlay for Canvas and TiledImageView."""

    def _init_behavior_overlay_state(
        self,
        *,
        show_behavior_overlay: bool = True,
        behavior_overlay_position: str = "top_left",
    ) -> None:
        raw_pos = str(behavior_overlay_position or "top_left").strip().lower()
        self.behavior_text_color = QtGui.QColor(248, 250, 252)
        self.behavior_text_background = QtGui.QColor(18, 22, 28, 205)
        self.current_behavior_text = None
        self._show_behavior_overlay = bool(show_behavior_overlay)
        self._behavior_overlay_position = (
            raw_pos if raw_pos in _VALID_BEHAVIOR_OVERLAY_POSITIONS else "top_left"
        )
        self._behavior_overlay_items = None
        self._behavior_color_provider = None
        self._last_behavior_overlay_rects: list[QtCore.QRectF] = []

    def _request_behavior_overlay_repaint(self) -> None:
        self.update()

    def _notify_behavior_overlay_changed(self) -> None:
        signal = getattr(self, "behaviorOverlayChanged", None)
        if signal is not None and hasattr(signal, "emit"):
            try:
                signal.emit()
            except Exception:
                pass

    def setBehaviorText(self, text):
        normalized = str(text).strip() if text is not None else None
        if normalized == "":
            normalized = None
        if normalized == self.current_behavior_text:
            return
        self.current_behavior_text = normalized
        # Clear any stale structured items when callers update via plain text.
        self._behavior_overlay_items = None
        self._request_behavior_overlay_repaint()
        self._notify_behavior_overlay_changed()

    def setBehaviorOverlayItems(
        self, items: Iterable[BehaviorOverlayItem] | None
    ) -> None:
        """Attach rich behavior badge metadata (kind, subject, color) for canvas rendering."""
        if items is None:
            normalized_items = None
        else:
            cleaned: list[BehaviorOverlayItem] = []
            for entry in items:
                if not isinstance(entry, BehaviorOverlayItem):
                    continue
                label = str(entry.label or "").strip()
                if not label:
                    continue
                kind = str(entry.kind or "state").strip().lower()
                if kind not in {"state", "point", "skipped"}:
                    kind = "state"
                subject = (
                    (str(entry.subject).strip() or None) if entry.subject else None
                )
                color = _normalize_rgb_tuple(entry.color)
                cleaned.append(
                    BehaviorOverlayItem(
                        label=label,
                        kind=kind,
                        subject=subject,
                        color=color,
                    )
                )
            normalized_items = tuple(cleaned) if cleaned else None
        if normalized_items == getattr(self, "_behavior_overlay_items", None):
            return
        self._behavior_overlay_items = normalized_items
        self._request_behavior_overlay_repaint()
        self._notify_behavior_overlay_changed()

    def setBehaviorColorProvider(self, provider: Callable[[str], Any] | None) -> None:
        """Register an optional label -> RGB/QColor resolver for behavior pills."""
        self._behavior_color_provider = provider
        if self.hasBehaviorOverlay():
            self._request_behavior_overlay_repaint()
        self._notify_behavior_overlay_changed()

    def showBehaviorOverlay(self) -> bool:
        return bool(getattr(self, "_show_behavior_overlay", True))

    def setShowBehaviorOverlay(self, enabled: bool) -> None:
        flag = bool(enabled)
        if flag == bool(getattr(self, "_show_behavior_overlay", True)):
            return
        self._show_behavior_overlay = flag
        self._request_behavior_overlay_repaint()
        self._notify_behavior_overlay_changed()

    def behaviorOverlayPosition(self) -> str:
        pos = str(getattr(self, "_behavior_overlay_position", "top_left") or "top_left")
        return pos if pos in _VALID_BEHAVIOR_OVERLAY_POSITIONS else "top_left"

    def setBehaviorOverlayPosition(self, position: str) -> None:
        normalized = str(position or "top_left").strip().lower()
        if normalized not in _VALID_BEHAVIOR_OVERLAY_POSITIONS:
            raise ValueError(
                f"Unsupported behavior overlay position: {position!r}. "
                f"Expected one of {sorted(_VALID_BEHAVIOR_OVERLAY_POSITIONS)}."
            )
        if normalized == self.behaviorOverlayPosition():
            return
        self._behavior_overlay_position = normalized
        if self.hasBehaviorOverlay():
            self._request_behavior_overlay_repaint()
        self._notify_behavior_overlay_changed()

    def _resolved_behavior_overlay_items(self) -> tuple[BehaviorOverlayItem, ...]:
        structured = getattr(self, "_behavior_overlay_items", None)
        if structured:
            return tuple(structured)
        raw_text = getattr(self, "current_behavior_text", None)
        if not raw_text:
            return ()
        items: list[BehaviorOverlayItem] = []
        seen: set[str] = set()
        for part in str(raw_text).split(","):
            label = part.strip()
            if not label or label in seen:
                continue
            seen.add(label)
            items.append(BehaviorOverlayItem(label=label, kind="state"))
        return tuple(items)

    def hasBehaviorOverlay(self) -> bool:
        """Return True when the viewer currently has visible behavior badges to paint."""
        if not self.showBehaviorOverlay():
            return False
        return bool(self._resolved_behavior_overlay_items())

    def _resolve_behavior_accent_color(self, item: BehaviorOverlayItem) -> QtGui.QColor:
        rgb = item.color
        if rgb is None:
            provider = getattr(self, "_behavior_color_provider", None)
            if callable(provider):
                try:
                    rgb = _normalize_rgb_tuple(provider(item.label))
                except Exception:
                    rgb = None
        if rgb is None:
            digest = hashlib.md5(item.label.lower().encode("utf-8")).hexdigest()
            rgb = _FALLBACK_BEHAVIOR_PALETTE[
                int(digest, 16) % len(_FALLBACK_BEHAVIOR_PALETTE)
            ]

        r, g, b = rgb
        if item.kind == "skipped" and item.label.lower() in {
            "no_behavior",
            "unclassified",
            "background",
            "none",
        }:
            return QtGui.QColor(148, 163, 184)

        # Lift overly dark label colors so indicator dot & border stay legible on dark pills.
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance < 92.0:
            blend = 0.48
            r = int(round(r + (255 - r) * blend))
            g = int(round(g + (255 - g) * blend))
            b = int(round(b + (255 - b) * blend))
        return QtGui.QColor(r, g, b)

    def _enclosing_scroll_viewport(self) -> QtWidgets.QWidget | None:
        """Return the enclosing QScrollArea viewport if Canvas is scrolled inside one."""
        try:
            parent = self.parentWidget()
            for _ in range(4):
                if parent is None:
                    break
                grandparent = parent.parentWidget()
                if isinstance(grandparent, QtWidgets.QScrollArea):
                    viewport = grandparent.viewport()
                    if viewport is parent:
                        return viewport
                parent = grandparent
        except Exception:
            return None
        return None

    def _behavior_overlay_anchor_rect(self, painter: QtGui.QPainter) -> QtCore.QRectF:
        """Return the visible anchor rectangle in widget-logical coordinates."""
        device = painter.device()
        vp_rect = painter.viewport()
        if isinstance(device, (QtGui.QImage, QtGui.QPixmap)) and hasattr(
            device, "devicePixelRatioF"
        ):
            try:
                dpr = max(1.0, float(device.devicePixelRatioF()))
            except Exception:
                dpr = 1.0
            vp_w = float(vp_rect.width()) / dpr
            vp_h = float(vp_rect.height()) / dpr
        else:
            vp_w = float(vp_rect.width())
            vp_h = float(vp_rect.height())

        if device is self and self.width() > 0 and self.height() > 0:
            widget_rect = QtCore.QRectF(
                0.0, 0.0, float(self.width()), float(self.height())
            )
        elif vp_w > 0.0 and vp_h > 0.0:
            widget_rect = QtCore.QRectF(0.0, 0.0, vp_w, vp_h)
        else:
            widget_rect = QtCore.QRectF(
                0.0,
                0.0,
                float(max(self.width(), 1)),
                float(max(self.height(), 1)),
            )

        visible_rect = QtCore.QRectF(widget_rect)
        if device is self:
            viewport = self._enclosing_scroll_viewport()
            if viewport is not None and viewport.width() > 0 and viewport.height() > 0:
                try:
                    origin = self.mapFrom(viewport, QtCore.QPoint(0, 0))
                    vp_in_canvas = QtCore.QRectF(
                        float(origin.x()),
                        float(origin.y()),
                        float(viewport.width()),
                        float(viewport.height()),
                    )
                    clipped = widget_rect.intersected(vp_in_canvas)
                    if clipped.width() >= 40.0 and clipped.height() >= 24.0:
                        visible_rect = clipped
                except Exception:
                    pass

        pixmap = getattr(self, "pixmap", None)
        if pixmap is not None and not pixmap.isNull():
            try:
                scale = max(float(getattr(self, "scale", 1.0) or 1.0), 1e-6)
                offset = self.offsetToCenter()
                img_rect = QtCore.QRectF(
                    float(offset.x()) * scale,
                    float(offset.y()) * scale,
                    float(pixmap.width()) * scale,
                    float(pixmap.height()) * scale,
                )
                frame_visible = visible_rect.intersected(img_rect)
                if frame_visible.width() >= 180.0 and frame_visible.height() >= 64.0:
                    return frame_visible
            except Exception:
                pass
        return visible_rect

    @staticmethod
    def _behavior_overlay_font(anchor_rect: QtCore.QRectF) -> QtGui.QFont:
        ref_dim = min(max(anchor_rect.width(), 1.0), max(anchor_rect.height(), 1.0))
        font_px = int(round(max(11.0, min(16.0, ref_dim * 0.025))))
        font = QtGui.QFont()
        font.setPixelSize(font_px)
        font.setBold(True)
        return font

    def _layout_behavior_pills(
        self,
        items: tuple[BehaviorOverlayItem, ...],
        anchor_rect: QtCore.QRectF,
        font: QtGui.QFont,
    ) -> list[_PositionedBehaviorPill]:
        if not items or anchor_rect.width() <= 4.0 or anchor_rect.height() <= 4.0:
            return []

        ref_dim = min(max(anchor_rect.width(), 1.0), max(anchor_rect.height(), 1.0))
        margin = max(8.0, min(14.0, round(ref_dim * 0.02)))
        avail_w = max(48.0, anchor_rect.width() - 2.0 * margin)
        avail_h = max(24.0, anchor_rect.height() - 2.0 * margin)

        metrics = QtGui.QFontMetricsF(font)
        font_px = max(
            10.0,
            float(font.pixelSize() if font.pixelSize() > 0 else metrics.height()),
        )
        text_h = max(float(metrics.height()), font_px + 2.0)
        pad_x = max(8.0, round(font_px * 0.65))
        pad_y = max(4.0, round(font_px * 0.35))
        pill_h = text_h + 2.0 * pad_y
        dot_d = max(6.0, round(font_px * 0.46))
        dot_gap = max(5.0, round(font_px * 0.42))
        gap_x = 6.0
        gap_y = 6.0

        max_rows = max(
            1,
            min(3, int((avail_h * 0.38 + gap_y) // (pill_h + gap_y))),
        )
        max_text_w = max(24.0, avail_w - 2.0 * pad_x - dot_d - dot_gap)

        def _measure_text(text: str) -> float:
            return max(
                float(metrics.horizontalAdvance(text)),
                float(metrics.boundingRect(text).width()),
            )

        measured: list[tuple[BehaviorOverlayItem, str, float]] = []
        for item in items:
            raw_display = (
                f"{item.label} \u00b7 {item.subject}" if item.subject else item.label
            )
            text_w = _measure_text(raw_display)
            if text_w > max_text_w:
                display = metrics.elidedText(
                    raw_display, QtCore.Qt.ElideRight, max_text_w
                )
                text_w = min(max_text_w, _measure_text(display))
            else:
                display = raw_display
            pill_w = min(avail_w, pad_x + dot_d + dot_gap + text_w + pad_x)
            measured.append((item, display, pill_w))

        # Flow pills into rows, keeping within max_rows and avail_w.
        rows: list[list[tuple[BehaviorOverlayItem | None, str, float, bool]]] = [[]]
        row_widths: list[float] = [0.0]
        overflow_start_idx: int | None = None

        for idx, (item, display, pill_w) in enumerate(measured):
            cur_row = rows[-1]
            needed = pill_w if not cur_row else (gap_x + pill_w)
            if not cur_row or row_widths[-1] + needed <= avail_w + 0.5:
                cur_row.append((item, display, pill_w, False))
                row_widths[-1] += needed
            elif len(rows) < max_rows:
                rows.append([(item, display, pill_w, False)])
                row_widths.append(pill_w)
            else:
                overflow_start_idx = idx
                break

        if overflow_start_idx is not None:
            hidden_count = len(measured) - overflow_start_idx
            last_row = rows[-1]

            def _overflow_entry(count: int) -> tuple[None, str, float, bool]:
                label = f"+{count} more"
                ow = min(avail_w, 2.0 * pad_x + _measure_text(label))
                return (None, label, ow, True)

            overflow_pill = _overflow_entry(hidden_count)
            while last_row:
                current_w = sum(entry[2] for entry in last_row) + gap_x * max(
                    0, len(last_row) - 1
                )
                if current_w + gap_x + overflow_pill[2] <= avail_w + 0.5:
                    break
                last_row.pop()
                hidden_count += 1
                overflow_pill = _overflow_entry(hidden_count)
            last_row.append(overflow_pill)

        rows = [row for row in rows if row]
        if not rows:
            return []

        position = self.behaviorOverlayPosition()
        align_right = position.endswith("right")
        align_bottom = position.startswith("bottom")

        total_h = len(rows) * pill_h + max(0, len(rows) - 1) * gap_y
        if align_bottom:
            start_y = max(
                anchor_rect.top() + margin,
                anchor_rect.bottom() - margin - total_h,
            )
        else:
            start_y = anchor_rect.top() + margin

        positioned: list[_PositionedBehaviorPill] = []
        y = start_y
        for row in rows:
            row_w = sum(entry[2] for entry in row) + gap_x * max(0, len(row) - 1)
            if align_right:
                x = max(
                    anchor_rect.left() + margin,
                    anchor_rect.right() - margin - row_w,
                )
            else:
                x = anchor_rect.left() + margin
            for item, display, pill_w, is_overflow in row:
                rect = QtCore.QRectF(x, y, pill_w, pill_h)
                positioned.append(
                    _PositionedBehaviorPill(
                        rect=rect,
                        text=display,
                        item=item,
                        is_overflow=is_overflow,
                        dot_diameter=dot_d,
                        dot_gap=dot_gap,
                        pad_x=pad_x,
                    )
                )
                x += pill_w + gap_x
            y += pill_h + gap_y
        return positioned

    def _draw_behavior_text_banner(self, painter: QtGui.QPainter) -> None:
        if not self.showBehaviorOverlay():
            self._last_behavior_overlay_rects = []
            return
        items = self._resolved_behavior_overlay_items()
        if not items:
            self._last_behavior_overlay_rects = []
            return

        painter.save()
        try:
            painter.resetTransform()
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            try:
                painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
            except AttributeError:
                pass

            anchor_rect = self._behavior_overlay_anchor_rect(painter)
            font = self._behavior_overlay_font(anchor_rect)
            painter.setFont(font)

            pills = self._layout_behavior_pills(items, anchor_rect, font)
            self._last_behavior_overlay_rects = [pill.rect for pill in pills]
            if not pills:
                return

            base_bg = self.behavior_text_background
            if base_bg is None:
                base_bg = QtGui.QColor(18, 22, 28, 205)
            else:
                base_bg = QtGui.QColor(base_bg)

            base_fg = (
                QtGui.QColor(self.behavior_text_color)
                if self.behavior_text_color is not None
                else QtGui.QColor(248, 250, 252)
            )

            for pill in pills:
                rect = pill.rect
                radius = min(rect.height() / 2.0, 7.0)
                kind = pill.item.kind if pill.item is not None else "overflow"
                accent = (
                    self._resolve_behavior_accent_color(pill.item)
                    if pill.item is not None
                    else QtGui.QColor(148, 163, 184)
                )

                if base_bg.alpha() > 0:
                    shadow_color = QtGui.QColor(0, 0, 0, min(90, base_bg.alpha() // 2))
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.setBrush(QtGui.QBrush(shadow_color))
                    painter.drawRoundedRect(rect.translated(0.0, 1.5), radius, radius)

                    pill_bg = QtGui.QColor(base_bg)
                    if pill.is_overflow:
                        pill_bg = QtGui.QColor(34, 40, 50, min(220, base_bg.alpha()))
                        border_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 75), 1.0)
                    elif kind == "skipped":
                        pill_bg.setAlpha(max(60, int(round(base_bg.alpha() * 0.84))))
                        border_color = QtGui.QColor(accent)
                        border_color.setAlpha(185)
                        border_pen = QtGui.QPen(border_color, 1.2, QtCore.Qt.DashLine)
                    else:
                        border_color = QtGui.QColor(accent)
                        border_color.setAlpha(205)
                        border_pen = QtGui.QPen(border_color, 1.25)

                    painter.setBrush(QtGui.QBrush(pill_bg))
                    painter.setPen(border_pen)
                    painter.drawRoundedRect(rect, radius, radius)

                if not pill.is_overflow:
                    dot_cx = rect.left() + pill.pad_x + pill.dot_diameter / 2.0
                    dot_cy = rect.center().y()
                    dot_r = pill.dot_diameter / 2.0
                    if kind == "point":
                        diamond_r = dot_r * 1.15
                        diamond = QtGui.QPolygonF(
                            [
                                QtCore.QPointF(dot_cx, dot_cy - diamond_r),
                                QtCore.QPointF(dot_cx + diamond_r, dot_cy),
                                QtCore.QPointF(dot_cx, dot_cy + diamond_r),
                                QtCore.QPointF(dot_cx - diamond_r, dot_cy),
                            ]
                        )
                        painter.setPen(QtCore.Qt.NoPen)
                        painter.setBrush(QtGui.QBrush(accent))
                        painter.drawPolygon(diamond)
                    elif kind == "skipped":
                        ring_pen = QtGui.QPen(accent, 1.5)
                        painter.setPen(ring_pen)
                        painter.setBrush(QtCore.Qt.NoBrush)
                        painter.drawEllipse(
                            QtCore.QPointF(dot_cx, dot_cy),
                            dot_r * 0.9,
                            dot_r * 0.9,
                        )
                    else:
                        painter.setPen(QtCore.Qt.NoPen)
                        painter.setBrush(QtGui.QBrush(accent))
                        painter.drawEllipse(
                            QtCore.QPointF(dot_cx, dot_cy), dot_r, dot_r
                        )

                text_left = (
                    rect.left()
                    + pill.pad_x
                    + (0.0 if pill.is_overflow else (pill.dot_diameter + pill.dot_gap))
                )
                text_rect = QtCore.QRectF(
                    text_left,
                    rect.top(),
                    max(0.0, rect.right() - pill.pad_x - text_left),
                    rect.height(),
                )
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 175)))
                painter.drawText(
                    text_rect.translated(0.0, 1.0),
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                    pill.text,
                )
                fg = QtGui.QColor(base_fg)
                if pill.is_overflow:
                    fg = QtGui.QColor(215, 224, 236, 240)
                elif kind == "skipped":
                    fg.setAlpha(min(fg.alpha(), 225))
                painter.setPen(QtGui.QPen(fg))
                painter.drawText(
                    text_rect,
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                    pill.text,
                )
        finally:
            painter.restore()


class CanvasOverlayMixin(BehaviorOverlayMixin):
    """Pose skeleton, RGBA preview maps, explicit landmark pairs, and behavior banner."""

    # ------------------------------------------------------------------
    # Pose skeleton overlay
    # ------------------------------------------------------------------
    def setPoseSchema(self, schema: PoseSchema | None) -> None:
        self._pose_schema = schema
        self.update()

    def setShowPoseEdges(self, enabled: bool) -> None:
        self._show_pose_edges = bool(enabled)
        self.update()

    def showPoseEdges(self) -> bool:
        return bool(self._show_pose_edges)

    def setShowPoseBBoxes(self, enabled: bool) -> None:
        self._show_pose_bboxes = bool(enabled)
        self.update()

    def showPoseBBoxes(self) -> bool:
        return bool(self._show_pose_bboxes)

    @staticmethod
    def _is_pose_bbox_shape(shape) -> bool:
        if str(getattr(shape, "shape_type", "")).lower() != "rectangle":
            return False
        other = getattr(shape, "other_data", None)
        if not isinstance(other, dict):
            return False
        return other.get("instance_id") is not None

    def _infer_pose_instance_label(
        self, shape, candidate_labels, default_label: str = "object"
    ) -> str:
        flags = getattr(shape, "flags", None)
        if isinstance(flags, dict):
            instance_label = flags.get("instance_label")
            if isinstance(instance_label, str) and instance_label.strip():
                return instance_label.strip()
        other = getattr(shape, "other_data", None)
        if isinstance(other, dict):
            instance_label = other.get("instance_label")
            if isinstance(instance_label, str) and instance_label.strip():
                return instance_label.strip()

        group_id = getattr(shape, "group_id", None)
        if group_id not in (None, ""):
            return str(group_id)

        label = str(getattr(shape, "label", "") or "").strip()
        lower_label = label.lower()
        for candidate in sorted(candidate_labels, key=len, reverse=True):
            if not candidate:
                continue
            if lower_label.startswith(candidate.lower()):
                return candidate
        for delimiter in ("_", "-", ":", "|", " "):
            if delimiter in label:
                return label.split(delimiter, 1)[0].strip() or default_label
        return default_label

    def _infer_pose_keypoint_label(self, shape, instance_label: str) -> str:
        flags = getattr(shape, "flags", None)
        if isinstance(flags, dict):
            display_label = flags.get("display_label")
            if isinstance(display_label, str) and display_label.strip():
                return display_label.strip()
        other = getattr(shape, "other_data", None)
        if isinstance(other, dict):
            display_label = other.get("display_label")
            if isinstance(display_label, str) and display_label.strip():
                return display_label.strip()

        label = str(getattr(shape, "label", "") or "").strip()
        if instance_label:
            inst_len = len(instance_label)
            if label.lower().startswith(instance_label.lower()) and inst_len < len(
                label
            ):
                suffix = label[inst_len:].lstrip("_-:| ")
                if suffix:
                    return suffix
        for delimiter in ("_", "-", ":", "|"):
            if delimiter in label:
                suffix = label.split(delimiter, 1)[1].strip()
                if suffix:
                    return suffix
        return label

    def _draw_pose_edges(self, painter: QtGui.QPainter) -> None:
        if (
            not self._show_pose_edges
            or not self._pose_schema
            or not self._pose_schema.edges
        ):
            return

        candidate_labels = {
            str(getattr(s, "label", "") or "").strip()
            for s in self.shapes
            if str(getattr(s, "shape_type", "") or "").lower() != "point"
        }
        candidate_labels.discard("")

        instances = {}
        for shape in self.shapes:
            if str(getattr(shape, "shape_type", "") or "").lower() != "point":
                continue
            if not self.isVisible(shape):
                continue
            points = getattr(shape, "points", None) or []
            if not points:
                continue
            point = points[0]
            instance_label = None
            kp_label = None

            try:
                inst_from_schema, base_from_schema = (
                    self._pose_schema.strip_instance_prefix(
                        str(getattr(shape, "label", "") or "")
                    )
                )
                if inst_from_schema and base_from_schema:
                    instance_label = inst_from_schema
                    kp_label = base_from_schema
            except Exception:
                instance_label = None
                kp_label = None

            if not instance_label:
                instance_label = self._infer_pose_instance_label(
                    shape, candidate_labels, default_label="object"
                )
            if not kp_label:
                kp_label = self._infer_pose_keypoint_label(shape, instance_label)
            if not kp_label:
                continue
            instances.setdefault(instance_label, {})[kp_label] = point

        if not instances:
            return

        edge_width = max(1, int(round(3.0 / max(self.scale, 1e-6))))
        shadow_width = edge_width + max(1, int(round(2.0 / max(self.scale, 1e-6))))

        shadow_pen = QtGui.QPen(self._pose_edge_shadow)
        shadow_pen.setWidth(shadow_width)
        shadow_pen.setCapStyle(QtCore.Qt.RoundCap)
        shadow_pen.setJoinStyle(QtCore.Qt.RoundJoin)

        edge_pen = QtGui.QPen(self._pose_edge_color)
        edge_pen.setWidth(edge_width)
        edge_pen.setCapStyle(QtCore.Qt.RoundCap)
        edge_pen.setJoinStyle(QtCore.Qt.RoundJoin)

        painter.save()
        try:
            schema_instances = [
                inst for inst in (self._pose_schema.instances or []) if inst
            ]
            active_instances = list(instances.keys())
            if schema_instances:
                instance_order = [
                    inst for inst in schema_instances if inst in instances
                ] + [inst for inst in active_instances if inst not in schema_instances]
            else:
                instance_order = active_instances

            for edge in self._pose_schema.edges:
                a, b = edge
                inst_a, kp_a = self._pose_schema.strip_instance_prefix(a)
                inst_b, kp_b = self._pose_schema.strip_instance_prefix(b)
                if not self._pose_schema.instances:
                    sep = getattr(self._pose_schema, "instance_separator", "_") or "_"
                    if inst_a is None and isinstance(a, str) and sep in a:
                        candidate_inst, candidate_kp = a.split(sep, 1)
                        if candidate_inst in instances:
                            inst_a = candidate_inst
                            kp_a = candidate_kp.strip()
                    if inst_b is None and isinstance(b, str) and sep in b:
                        candidate_inst, candidate_kp = b.split(sep, 1)
                        if candidate_inst in instances:
                            inst_b = candidate_inst
                            kp_b = candidate_kp.strip()

                if inst_a or inst_b:
                    inst = inst_a or inst_b
                    if not inst:
                        continue
                    if inst_a and inst_b and inst_a != inst_b:
                        continue
                    kp_map = instances.get(inst, {})
                    p1 = kp_map.get(kp_a)
                    p2 = kp_map.get(kp_b)
                    if p1 is None or p2 is None:
                        continue
                    painter.setPen(shadow_pen)
                    painter.drawLine(p1, p2)
                    painter.setPen(edge_pen)
                    painter.drawLine(p1, p2)
                    continue

                for inst in instance_order:
                    kp_map = instances.get(inst, {})
                    p1 = kp_map.get(kp_a)
                    p2 = kp_map.get(kp_b)
                    if p1 is None or p2 is None:
                        continue
                    painter.setPen(shadow_pen)
                    painter.drawLine(p1, p2)
                    painter.setPen(edge_pen)
                    painter.drawLine(p1, p2)
        finally:
            painter.restore()

    # ------------------------------------------------------------------
    # Raster RGBA overlays (DINO similarity, PCA, depth, optical flow)
    # ------------------------------------------------------------------
    def _set_rgba_overlay(self, attr_name: str, overlay_rgba) -> None:
        if overlay_rgba is None:
            setattr(self, attr_name, None)
        else:
            arr = np.ascontiguousarray(overlay_rgba, dtype=np.uint8)
            h, w, _ = arr.shape
            image = QtGui.QImage(
                arr.data,
                w,
                h,
                arr.strides[0],
                QtGui.QImage.Format_RGBA8888,
            )
            setattr(self, attr_name, QtGui.QPixmap.fromImage(image.copy()))
        self.update()

    def enablePatchSimilarityMode(self, callback):
        """Enable click-to-query mode for DINO patch similarity."""
        self._patch_similarity_active = True
        self._patch_similarity_callback = callback
        self.overrideCursor(_CURSOR_POINT)

    def disablePatchSimilarityMode(self):
        self._patch_similarity_active = False
        self._patch_similarity_callback = None
        self._patch_similarity_pixmap = None
        self.restoreCursor()
        self.update()

    def setPatchSimilarityOverlay(self, overlay_rgba):
        """Update or clear the rendered heatmap overlay."""
        self._set_rgba_overlay("_patch_similarity_pixmap", overlay_rgba)

    def setPCAMapOverlay(self, overlay_rgba):
        """Update or clear the PCA coloring overlay."""
        self._set_rgba_overlay("_pca_map_pixmap", overlay_rgba)

    def setDepthPreviewOverlay(self, overlay_rgba):
        """Update or clear the streaming depth preview overlay."""
        self._set_rgba_overlay("_depth_preview_pixmap", overlay_rgba)

    def setFlowPreviewOverlay(self, overlay_rgba):
        """Update or clear the optical-flow overlay."""
        self._set_rgba_overlay("_flow_preview_pixmap", overlay_rgba)

    def _draw_raster_overlays(self, painter: QtGui.QPainter) -> None:
        for attr in (
            "_patch_similarity_pixmap",
            "_pca_map_pixmap",
            "_depth_preview_pixmap",
            "_flow_preview_pixmap",
        ):
            pixmap = getattr(self, attr, None)
            if pixmap is not None:
                painter.drawPixmap(0, 0, pixmap)

    # ------------------------------------------------------------------
    # Explicit vector-overlay landmark pairs
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_explicit_landmark_pairs_from_shapes(shapes):
        overlay_points = {}
        image_points = {}
        pair_visible = {}
        for shape in list(shapes or []):
            if str(getattr(shape, "shape_type", "") or "").lower() != "point":
                continue
            points = getattr(shape, "points", None) or []
            if not points:
                continue
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if not pair_id:
                continue
            point = points[0]
            coords = (float(point.x()), float(point.y()))
            if "overlay_id" in other:
                overlay_points[pair_id] = coords
                pair_visible[pair_id] = bool(
                    other.get("overlay_landmarks_visible", True)
                )
            else:
                image_points[pair_id] = coords
        pairs = []
        for pair_id, src in overlay_points.items():
            dst = image_points.get(pair_id)
            if dst is None or not bool(pair_visible.get(pair_id, True)):
                continue
            pairs.append((pair_id, src, dst))
        return pairs

    @staticmethod
    def _explicit_landmark_pair_pen(scale: float, *, selected: bool = False):
        color = (
            QtGui.QColor(255, 140, 0, 240)
            if selected
            else QtGui.QColor(255, 215, 0, 220)
        )
        pen = QtGui.QPen(color)
        base_width = 3.0 if selected else 2.0
        pen.setWidthF(max(1.5, base_width / max(scale, 0.01)))
        pen.setStyle(QtCore.Qt.DashLine)
        return pen

    def setSelectedOverlayLandmarkPair(self, pair_id: str | None) -> None:
        normalized = str(pair_id or "") or None
        if normalized == self._selected_overlay_landmark_pair_id:
            return
        self._selected_overlay_landmark_pair_id = normalized
        self.update()

    def _nearest_explicit_landmark_pair(self, pos, tolerance: float | None = None):
        threshold = float(
            tolerance if tolerance is not None else self.epsilon / max(self.scale, 0.01)
        )
        best_pair_id = None
        best_distance = None
        query = QtCore.QPointF(pos)
        for pair_id, src, dst in self._collect_explicit_landmark_pairs_from_shapes(
            self.shapes
        ):
            line = QtCore.QLineF(
                QtCore.QPointF(src[0], src[1]), QtCore.QPointF(dst[0], dst[1])
            )
            length = line.length()
            if length <= 0:
                continue
            dx = line.dx()
            dy = line.dy()
            t = (((query.x() - line.x1()) * dx) + ((query.y() - line.y1()) * dy)) / (
                length * length
            )
            t = max(0.0, min(1.0, t))
            closest = QtCore.QPointF(line.x1() + (dx * t), line.y1() + (dy * t))
            distance = QtCore.QLineF(query, closest).length()
            if distance > threshold:
                continue
            if best_distance is None or distance < best_distance:
                best_pair_id = pair_id
                best_distance = distance
        return best_pair_id

    def _draw_explicit_landmark_pairs(self, painter):
        pairs = self._collect_explicit_landmark_pairs_from_shapes(self.shapes)
        if not pairs:
            return
        painter.save()
        for pair_id, src, dst in pairs:
            painter.setPen(
                self._explicit_landmark_pair_pen(
                    self.scale,
                    selected=pair_id == self._selected_overlay_landmark_pair_id,
                )
            )
            painter.drawLine(
                QtCore.QPointF(src[0], src[1]),
                QtCore.QPointF(dst[0], dst[1]),
            )
        painter.restore()

    def _selected_explicit_landmark_pair_points(self):
        selected_pair_id = str(self._selected_overlay_landmark_pair_id or "")
        if not selected_pair_id:
            return []
        for pair_id, src, dst in self._collect_explicit_landmark_pairs_from_shapes(
            self.shapes
        ):
            if pair_id == selected_pair_id:
                return [src, dst]
        return []

    def _draw_selected_explicit_landmark_pair_points(self, painter):
        points = self._selected_explicit_landmark_pair_points()
        if not points:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        outer_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220))
        outer_pen.setWidthF(max(1.5, 2.0 / max(self.scale, 0.01)))
        painter.setPen(outer_pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 140, 0, 180)))
        outer_radius = max(6.0, 8.0 / max(self.scale, 0.01))
        inner_radius = outer_radius * 0.45
        for x, y in points:
            center = QtCore.QPointF(x, y)
            painter.drawEllipse(center, outer_radius, outer_radius)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 240)))
            painter.drawEllipse(center, inner_radius, inner_radius)
            painter.setPen(outer_pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 140, 0, 180)))
        painter.restore()
