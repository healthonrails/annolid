"""Overlay rendering mixin for Canvas (pose skeletons, RGBA maps, landmarks, behavior text)."""

from __future__ import annotations

import numpy as np
from qtpy import QtCore, QtGui

from annolid.annotation.pose_schema import PoseSchema

_CURSOR_POINT = QtCore.Qt.PointingHandCursor


class CanvasOverlayMixin:
    """Pose skeleton, RGBA preview maps, explicit landmark pairs, and behavior banner."""

    def setBehaviorText(self, text):
        normalized = str(text).strip() if text is not None else None
        if normalized == "":
            normalized = None
        if normalized == self.current_behavior_text:
            return
        self.current_behavior_text = normalized
        self.update()

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

    def _draw_behavior_text_banner(self, painter: QtGui.QPainter) -> None:
        if not self.current_behavior_text:
            return
        painter.save()
        try:
            painter.resetTransform()
            if self.pixmap and not self.pixmap.isNull():
                base_size = min(self.pixmap.width(), self.pixmap.height())
            else:
                base_size = min(max(self.width(), 1), max(self.height(), 1))
            font_size = max(20, base_size // 40)

            font = QtGui.QFont()
            font.setPointSize(font_size)
            font.setBold(True)
            painter.setFont(font)

            metrics = QtGui.QFontMetrics(font)
            text = self.current_behavior_text
            text_bounds = metrics.boundingRect(text)

            margin = 12
            padding_x = 8
            padding_y = 6
            background_rect = QtCore.QRect(
                margin,
                margin,
                text_bounds.width() + 2 * padding_x,
                text_bounds.height() + 2 * padding_y,
            )

            bg_color = self.behavior_text_background
            if bg_color is None:
                bg_color = QtGui.QColor(0, 0, 0, 180)
            else:
                bg_color = QtGui.QColor(bg_color)

            if bg_color.alpha() > 0:
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(bg_color))
                painter.drawRoundedRect(background_rect, 6, 6)

            text_rect = QtCore.QRect(
                background_rect.left() + padding_x,
                background_rect.top() + padding_y,
                text_bounds.width(),
                text_bounds.height(),
            )
            shadow_rect = text_rect.translated(1, 1)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 200)))
            painter.drawText(
                shadow_rect,
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                text,
            )
            painter.setPen(QtGui.QPen(QtGui.QColor(self.behavior_text_color)))
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                text,
            )
        finally:
            painter.restore()

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
