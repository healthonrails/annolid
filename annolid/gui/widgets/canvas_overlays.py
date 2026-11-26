"""Overlay rendering mixin for Canvas (pose skeletons, RGBA maps, landmarks, behavior text)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    keypoint_visibility_from_shape_object,
)
from annolid.annotation.pose_schema import PoseSchema

_POSE_INSTANCE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 229, 255),  # cyan
    (255, 179, 0),  # amber
    (255, 64, 129),  # magenta
    (105, 240, 174),  # emerald
    (179, 136, 255),  # violet
    (255, 110, 64),  # coral
    (64, 196, 255),  # sky
    (255, 238, 88),  # yellow
)

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
            digest = hashlib.md5(
                item.label.lower().encode("utf-8"), usedforsecurity=False
            ).hexdigest()
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

    def setPoseColorMode(self, mode: str) -> None:
        norm = str(mode or "instance").strip().lower()
        if norm not in {"instance", "bodypart", "confidence"}:
            norm = "instance"
        self._pose_color_mode = norm
        self.update()

    def poseColorMode(self) -> str:
        return str(getattr(self, "_pose_color_mode", "instance") or "instance")

    def setPoseConfidenceThreshold(self, threshold: float) -> None:
        self._pose_confidence_threshold = max(0.0, min(1.0, float(threshold)))
        self.update()

    def poseConfidenceThreshold(self) -> float:
        return float(getattr(self, "_pose_confidence_threshold", 0.0) or 0.0)

    def setShowPoseLabels(self, enabled: bool) -> None:
        self._show_pose_labels = bool(enabled)
        try:
            from annolid.gui.shape import Shape

            Shape.show_pose_labels = self._show_pose_labels
        except Exception:
            pass
        self.update()

    def showPoseLabels(self) -> bool:
        return bool(getattr(self, "_show_pose_labels", False))

    def poseInstanceShapesFor(self, seed_shape) -> list:
        """Return all visible point shapes belonging to the same pose instance as ``seed_shape``."""
        if str(getattr(seed_shape, "shape_type", "") or "").lower() != "point":
            return []
        candidate_labels = {
            str(getattr(s, "label", "") or "").strip()
            for s in self.shapes
            if str(getattr(s, "shape_type", "") or "").lower() != "point"
        }
        candidate_labels.discard("")
        known_kps_lower = self._known_pose_keypoints_lower()
        resolved_seed = self._resolve_pose_point_shape(
            seed_shape, candidate_labels, known_kps_lower
        )
        if resolved_seed is None:
            return [seed_shape]
        target_bucket = resolved_seed[0]
        matched = []
        for s in self.shapes:
            res = self._resolve_pose_point_shape(s, candidate_labels, known_kps_lower)
            if res is not None and res[0] == target_bucket:
                matched.append(s)
        return matched or [seed_shape]

    def selectPoseInstanceShapes(self, seed_shape=None) -> list:
        """Select all keypoints belonging to the pose instance of ``seed_shape``."""
        target = seed_shape
        if target is None:
            selected = list(getattr(self, "selectedShapes", []) or [])
            for s in selected:
                if str(getattr(s, "shape_type", "") or "").lower() == "point":
                    target = s
                    break
        if target is None:
            return []
        inst_shapes = self.poseInstanceShapesFor(target)
        if inst_shapes and hasattr(self, "selectShapes"):
            self.selectShapes(inst_shapes)
        return inst_shapes

    def transformSelectedPoseShapes(
        self,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        angle_deg: float = 0.0,
        scale: float = 1.0,
    ) -> int:
        """Apply rigid/similarity transform (translate, rotate, scale) around centroid of selected keypoints."""
        import math

        selected = [
            s
            for s in (getattr(self, "selectedShapes", []) or [])
            if str(getattr(s, "shape_type", "") or "").lower() == "point"
            and getattr(s, "points", None)
            and not (callable(getattr(s, "is_locked", None)) and s.is_locked())
        ]
        if not selected:
            return 0

        cx = sum(float(s.points[0].x()) for s in selected) / float(len(selected))
        cy = sum(float(s.points[0].y()) for s in selected) / float(len(selected))
        rad = math.radians(float(angle_deg))
        cos_a = math.cos(rad) * float(scale)
        sin_a = math.sin(rad) * float(scale)

        clamp_bounds = not bool(getattr(self, "_allow_points_outside_image", False))
        pixmap = getattr(self, "pixmap", None)
        max_w = (
            float(pixmap.width() - 1)
            if (pixmap is not None and not pixmap.isNull())
            else None
        )
        max_h = (
            float(pixmap.height() - 1)
            if (pixmap is not None and not pixmap.isNull())
            else None
        )

        if hasattr(self, "storeShapes"):
            try:
                self.storeShapes()
            except Exception:
                pass

        for s in selected:
            px = float(s.points[0].x()) - cx
            py = float(s.points[0].y()) - cy
            nx = cx + cos_a * px - sin_a * py + float(dx)
            ny = cy + sin_a * px + cos_a * py + float(dy)
            if clamp_bounds and max_w is not None and max_h is not None:
                nx = max(0.0, min(max_w, nx))
                ny = max(0.0, min(max_h, ny))
            s.points[0] = QtCore.QPointF(nx, ny)

        if hasattr(self, "shapeMoved"):
            try:
                self.shapeMoved.emit()
            except Exception:
                pass
        self.update()
        return len(selected)

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
            for key in ("instance_label", "parent_instance", "instance_name"):
                instance_label = flags.get(key)
                if isinstance(instance_label, str) and instance_label.strip():
                    return instance_label.strip()
        other = getattr(shape, "other_data", None)
        if isinstance(other, dict):
            for key in ("instance_label", "parent_instance", "instance_name"):
                instance_label = other.get(key)
                if isinstance(instance_label, str) and instance_label.strip():
                    return instance_label.strip()

        group_id = getattr(shape, "group_id", None)
        if group_id not in (None, ""):
            return str(group_id)

        label = str(getattr(shape, "label", "") or "").strip()
        if ":" in label:
            left, right = label.split(":", 1)
            if left.strip() and right.strip():
                return left.strip()

        known_kps_lower = self._known_pose_keypoints_lower()
        if label.lower() in known_kps_lower:
            return default_label

        lower_label = label.lower()
        for candidate in sorted(candidate_labels, key=len, reverse=True):
            if not candidate:
                continue
            cand_lower = candidate.lower()
            if lower_label.startswith(cand_lower) and len(label) > len(candidate):
                if label[len(candidate)] in ("_", "-", ":", "|", " "):
                    return candidate

        # Match trailing known keypoint suffix (e.g. "mouse_1_tail_base" -> "mouse_1").
        if known_kps_lower:
            sep = (
                getattr(self._pose_schema, "instance_separator", "_") or "_"
                if self._pose_schema is not None
                else "_"
            )
            for kp_lower in sorted(known_kps_lower.keys(), key=len, reverse=True):
                for delimiter in (sep, "_", "-", ":", "|", " "):
                    token = f"{delimiter}{kp_lower}"
                    if lower_label.endswith(token) and len(label) > len(token):
                        prefix = label[: -len(token)].strip()
                        if prefix:
                            return prefix

        return default_label

    def _infer_pose_keypoint_label(self, shape, instance_label: str) -> str:
        flags = getattr(shape, "flags", None)
        if isinstance(flags, dict):
            for key in ("display_label", "part_label", "body_part"):
                display_label = flags.get(key)
                if isinstance(display_label, str) and display_label.strip():
                    return display_label.strip()
        other = getattr(shape, "other_data", None)
        if isinstance(other, dict):
            for key in ("display_label", "part_label", "body_part"):
                display_label = other.get(key)
                if isinstance(display_label, str) and display_label.strip():
                    return display_label.strip()

        label = str(getattr(shape, "label", "") or "").strip()
        known_kps_lower = self._known_pose_keypoints_lower()
        if label.lower() in known_kps_lower:
            return known_kps_lower[label.lower()]

        if ":" in label:
            _left, right = label.split(":", 1)
            if right.strip():
                r_clean = right.strip()
                return known_kps_lower.get(r_clean.lower(), r_clean)

        if instance_label:
            inst_len = len(instance_label)
            if (
                label.lower().startswith(instance_label.lower())
                and inst_len < len(label)
                and label[inst_len] in ("_", "-", ":", "|", " ")
            ):
                suffix = label[inst_len:].lstrip("_-:| ").strip()
                if suffix:
                    return known_kps_lower.get(suffix.lower(), suffix)

        if known_kps_lower:
            sep = (
                getattr(self._pose_schema, "instance_separator", "_") or "_"
                if self._pose_schema is not None
                else "_"
            )
            lower_label = label.lower()
            for kp_lower in sorted(known_kps_lower.keys(), key=len, reverse=True):
                for delimiter in (sep, "_", "-", ":", "|", " "):
                    token = f"{delimiter}{kp_lower}"
                    if lower_label.endswith(token) and len(label) > len(token):
                        return known_kps_lower[kp_lower]

        return label

    def _known_pose_keypoints_lower(self) -> dict[str, str]:
        schema = self._pose_schema
        if schema is None:
            return {}
        known: dict[str, str] = {}
        for kp in list(getattr(schema, "keypoints", ()) or ()):
            text = str(kp or "").strip()
            if text:
                known.setdefault(text.lower(), text)
        for raw_a, raw_b in list(getattr(schema, "edges", ()) or ()):
            for raw in (raw_a, raw_b):
                _inst, base = schema.strip_instance_prefix(str(raw or ""))
                if base:
                    known.setdefault(base.lower(), base)
        return known

    def _resolve_pose_point_shape(
        self,
        shape,
        candidate_labels: set[str],
        known_kps_lower: dict[str, str],
    ) -> tuple[tuple[str, str | None], str, str, int, bool] | None:
        """Resolve a visible point shape to ``(bucket_key, inst_label, kp_label, vis, highlighted)``."""
        if str(getattr(shape, "shape_type", "") or "").lower() != "point":
            return None
        if not self.isVisible(shape):
            return None
        points = getattr(shape, "points", None) or []
        if not points:
            return None

        vis = keypoint_visibility_from_shape_object(shape)
        if vis <= int(KeypointVisibility.MISSING):
            return None

        raw_label = str(getattr(shape, "label", "") or "").strip()
        group_id = getattr(shape, "group_id", None)
        gid_key = str(group_id) if group_id not in (None, "") else None

        instance_label: str | None = None
        kp_label: str | None = None

        # 1. Explicit instance/keypoint metadata takes precedence when both are present.
        flags = getattr(shape, "flags", None)
        other = getattr(shape, "other_data", None)
        explicit_inst = ""
        explicit_kp = ""
        for container in (flags, other):
            if not isinstance(container, dict):
                continue
            if not explicit_inst:
                for key in ("instance_label", "parent_instance", "instance_name"):
                    val = container.get(key)
                    if isinstance(val, str) and val.strip():
                        explicit_inst = val.strip()
                        break
            if not explicit_kp:
                for key in ("display_label", "part_label", "body_part"):
                    val = container.get(key)
                    if isinstance(val, str) and val.strip():
                        explicit_kp = val.strip()
                        break

        if explicit_inst:
            instance_label = explicit_inst
        if explicit_kp:
            kp_label = known_kps_lower.get(explicit_kp.lower(), explicit_kp)

        # 2. Try configured PoseSchema instance prefixes (longest-match-first).
        if (not instance_label or not kp_label) and self._pose_schema is not None:
            try:
                inst_from_schema, base_from_schema = (
                    self._pose_schema.strip_instance_prefix(raw_label)
                )
                if inst_from_schema and base_from_schema:
                    instance_label = instance_label or inst_from_schema
                    kp_label = kp_label or known_kps_lower.get(
                        base_from_schema.lower(), base_from_schema
                    )
            except Exception:
                pass

        # 3. Fallback inference (respecting ":" hierarchy, known base keypoints, and candidates).
        if not instance_label:
            instance_label = self._infer_pose_instance_label(
                shape, candidate_labels, default_label="object"
            )
        if not kp_label:
            kp_label = self._infer_pose_keypoint_label(shape, instance_label)
        if not kp_label:
            return None

        kp_canonical = known_kps_lower.get(kp_label.lower(), kp_label)
        bucket_key = (instance_label, gid_key)
        highlighted = bool(
            getattr(shape, "selected", False) or shape is getattr(self, "hShape", None)
        )
        return bucket_key, instance_label, kp_canonical, int(vis), highlighted

    def _pose_instance_color(
        self,
        inst_index: int,
        total_instances: int,
        *,
        occluded: bool = False,
        highlighted: bool = False,
    ) -> QtGui.QColor:
        base_color = getattr(self, "_pose_edge_color", None)
        if total_instances <= 1 and isinstance(base_color, QtGui.QColor):
            color = QtGui.QColor(base_color)
        else:
            r, g, b = _POSE_INSTANCE_PALETTE[inst_index % len(_POSE_INSTANCE_PALETTE)]
            color = QtGui.QColor(r, g, b, 210)
        if highlighted:
            color = color.lighter(115)
            color.setAlpha(245)
        elif occluded:
            color.setAlpha(min(color.alpha(), 140))
        return color

    def _collect_pose_edge_segments(
        self,
    ) -> list[
        tuple[QtCore.QPointF, QtCore.QPointF, tuple[str, str | None], int, bool, bool]
    ]:
        """Collect drawable skeleton bone segments from current visible shapes and PoseSchema."""
        schema = self._pose_schema
        if not self._show_pose_edges or not schema or not schema.edges:
            return []

        candidate_labels = {
            str(getattr(s, "label", "") or "").strip()
            for s in self.shapes
            if str(getattr(s, "shape_type", "") or "").lower() != "point"
        }
        candidate_labels.discard("")
        known_kps_lower = self._known_pose_keypoints_lower()

        buckets: dict[
            tuple[str, str | None],
            dict[str, tuple[QtCore.QPointF, int, bool, float | None]],
        ] = {}
        buckets_by_inst_name: dict[str, list[tuple[str, str | None]]] = {}

        for shape in self.shapes:
            resolved = self._resolve_pose_point_shape(
                shape, candidate_labels, known_kps_lower
            )
            if resolved is None:
                continue
            bucket_key, inst_label, kp_label, vis, highlighted = resolved
            point = shape.points[0]
            pt_score: float | None = None
            other = getattr(shape, "other_data", None)
            if isinstance(other, dict) and other.get("score") is not None:
                try:
                    sf = float(other.get("score"))
                    if sf == sf:
                        pt_score = sf
                except (TypeError, ValueError):
                    pt_score = None
            if bucket_key not in buckets:
                buckets[bucket_key] = {}
                buckets_by_inst_name.setdefault(inst_label, []).append(bucket_key)
            kp_map = buckets[bucket_key]
            # Prefer visible over occluded if duplicate keypoint labels exist in same bucket.
            existing = kp_map.get(kp_label)
            if existing is None or (existing[1] < vis):
                kp_map[kp_label] = (point, vis, highlighted, pt_score)
            elif highlighted and not existing[2]:
                kp_map[kp_label] = (existing[0], existing[1], True, existing[3])

        if not buckets:
            self._last_pose_edge_meta = []
            return []

        schema_instances = [inst for inst in (schema.instances or []) if inst]
        ordered_buckets: list[tuple[str, str | None]] = []
        seen_buckets: set[tuple[str, str | None]] = set()
        for inst in schema_instances:
            for bkey in buckets_by_inst_name.get(inst, ()):
                if bkey not in seen_buckets:
                    seen_buckets.add(bkey)
                    ordered_buckets.append(bkey)
        for bkey in buckets.keys():
            if bkey not in seen_buckets:
                seen_buckets.add(bkey)
                ordered_buckets.append(bkey)

        bucket_to_index = {bkey: idx for idx, bkey in enumerate(ordered_buckets)}
        kp_order = list(getattr(schema, "keypoints", ()) or ())
        kp_to_idx = {name: i for i, name in enumerate(kp_order)}
        segments: list[
            tuple[
                QtCore.QPointF,
                QtCore.QPointF,
                tuple[str, str | None],
                int,
                bool,
                bool,
            ]
        ] = []
        edge_meta: list[tuple[int, float | None]] = []
        drawn_per_bucket: set[tuple[tuple[str, str | None], str, str]] = set()

        def _append_edge(bkey: tuple[str, str | None], kp_a: str, kp_b: str) -> None:
            ca = known_kps_lower.get(kp_a.lower(), kp_a)
            cb = known_kps_lower.get(kp_b.lower(), kp_b)
            if not ca or not cb or ca == cb:
                return
            edge_id = (bkey, ca, cb) if ca < cb else (bkey, cb, ca)
            if edge_id in drawn_per_bucket:
                return
            kp_map = buckets.get(bkey)
            if not kp_map:
                return
            rec_a = kp_map.get(ca)
            rec_b = kp_map.get(cb)
            if rec_a is None or rec_b is None:
                return
            drawn_per_bucket.add(edge_id)
            occluded = rec_a[1] == int(KeypointVisibility.OCCLUDED) or rec_b[1] == int(
                KeypointVisibility.OCCLUDED
            )
            highlighted = bool(rec_a[2] or rec_b[2])
            scores = [s for s in (rec_a[3], rec_b[3]) if s is not None]
            min_score = min(scores) if scores else None
            segments.append(
                (rec_a[0], rec_b[0], bkey, bucket_to_index[bkey], occluded, highlighted)
            )
            edge_meta.append((kp_to_idx.get(ca, len(edge_meta)), min_score))

        sep = getattr(schema, "instance_separator", "_") or "_"
        for a, b in schema.edges:
            inst_a, kp_a = schema.strip_instance_prefix(a)
            inst_b, kp_b = schema.strip_instance_prefix(b)
            if not schema.instances:
                if inst_a is None and isinstance(a, str) and sep in a:
                    candidate_inst, candidate_kp = a.split(sep, 1)
                    if candidate_inst in buckets_by_inst_name:
                        inst_a = candidate_inst
                        kp_a = candidate_kp.strip()
                if inst_b is None and isinstance(b, str) and sep in b:
                    candidate_inst, candidate_kp = b.split(sep, 1)
                    if candidate_inst in buckets_by_inst_name:
                        inst_b = candidate_inst
                        kp_b = candidate_kp.strip()

            if inst_a or inst_b:
                if inst_a and inst_b and inst_a != inst_b:
                    continue
                target_inst = inst_a or inst_b
                if not target_inst:
                    continue
                for bkey in buckets_by_inst_name.get(target_inst, ()):
                    _append_edge(bkey, kp_a, kp_b)
                continue

            for bkey in ordered_buckets:
                _append_edge(bkey, kp_a, kp_b)

        self._last_pose_edge_meta = edge_meta
        return segments

    def _draw_pose_edges(self, painter: QtGui.QPainter) -> None:
        segments = self._collect_pose_edge_segments()
        if not segments:
            return

        edge_meta = list(getattr(self, "_last_pose_edge_meta", ()) or ())
        color_mode = str(getattr(self, "_pose_color_mode", "instance") or "instance")
        total_instances = len({seg[2] for seg in segments})
        scale = max(float(getattr(self, "scale", 1.0) or 1.0), 1e-3)
        base_edge_width = max(0.75, 2.5 / scale)
        base_shadow_pad = max(0.75, 1.8 / scale)

        painter.save()
        try:
            for seg_i, (p1, p2, _bkey, inst_idx, occluded, highlighted) in enumerate(
                segments
            ):
                edge_width = base_edge_width + (0.9 / scale if highlighted else 0.0)
                shadow_width = edge_width + base_shadow_pad

                shadow_color = QtGui.QColor(self._pose_edge_shadow)
                if occluded:
                    shadow_color.setAlpha(min(shadow_color.alpha(), 145))
                shadow_pen = QtGui.QPen(shadow_color)
                shadow_pen.setWidthF(shadow_width)
                shadow_pen.setCapStyle(QtCore.Qt.RoundCap)
                shadow_pen.setJoinStyle(QtCore.Qt.RoundJoin)

                if color_mode == "bodypart" and seg_i < len(edge_meta):
                    kp_idx, _sc = edge_meta[seg_i]
                    edge_color = self._pose_instance_color(
                        kp_idx,
                        max(2, len(segments)),
                        occluded=occluded,
                        highlighted=highlighted,
                    )
                elif color_mode == "confidence" and seg_i < len(edge_meta):
                    _kp_idx, sc = edge_meta[seg_i]
                    if sc is None:
                        edge_color = QtGui.QColor(
                            56, 189, 248, 170 if occluded else 230
                        )
                    elif sc < 0.5:
                        t = max(0.0, min(1.0, sc / 0.5))
                        edge_color = QtGui.QColor(
                            int(round(239 + (245 - 239) * t)),
                            int(round(68 + (158 - 68) * t)),
                            int(round(68 + (11 - 68) * t)),
                            170 if occluded else 235,
                        )
                    else:
                        t = max(0.0, min(1.0, (sc - 0.5) / 0.5))
                        edge_color = QtGui.QColor(
                            int(round(245 + (34 - 245) * t)),
                            int(round(158 + (197 - 158) * t)),
                            int(round(11 + (94 - 11) * t)),
                            170 if occluded else 235,
                        )
                else:
                    edge_color = self._pose_instance_color(
                        inst_idx,
                        total_instances,
                        occluded=occluded,
                        highlighted=highlighted,
                    )
                edge_pen = QtGui.QPen(edge_color)
                edge_pen.setWidthF(edge_width)
                edge_pen.setCapStyle(QtCore.Qt.RoundCap)
                edge_pen.setJoinStyle(QtCore.Qt.RoundJoin)

                if occluded:
                    # Use matched image-coordinate dash/gap lengths so shadow and
                    # foreground dashes stay phase-locked despite different pen widths.
                    dash_img = max(3.5, 7.0 / scale)
                    gap_img = max(3.0, 5.5 / scale)
                    shadow_pen.setDashPattern(
                        [
                            max(0.5, dash_img / max(shadow_width, 1e-3)),
                            max(0.5, gap_img / max(shadow_width, 1e-3)),
                        ]
                    )
                    edge_pen.setDashPattern(
                        [
                            max(0.5, dash_img / max(edge_width, 1e-3)),
                            max(0.5, gap_img / max(edge_width, 1e-3)),
                        ]
                    )
                else:
                    shadow_pen.setStyle(QtCore.Qt.SolidLine)
                    edge_pen.setStyle(QtCore.Qt.SolidLine)

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
