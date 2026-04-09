from __future__ import annotations

from qtpy import QtCore

from annolid.gui.shared_vertices import SharedTopologyRegistry
from annolid.gui.shared_vertices import insert_shared_vertex_on_edge
from annolid.gui.shared_vertices import rebuild_polygon_topology
from annolid.gui.shared_vertices import remove_shared_vertex_at
from annolid.gui.shared_vertices import synchronize_shared_vertex
from annolid.gui.polygon_tools import mark_polygon_shape_manual


class SharedPolygonEditMixin:
    """Shared topology helpers for polygon editing widgets."""

    def _shared_polygon_shapes(self):
        shapes = getattr(self, "shapes", None)
        if shapes is None:
            shapes = getattr(self, "_shapes", None)
        return list(shapes or [])

    def _refresh_shared_topology_registry(self):
        shapes = self._shared_polygon_shapes()
        self._shared_topology_registry = SharedTopologyRegistry.from_shapes(shapes)
        return self._shared_topology_registry

    def _shared_mark_polygon_shapes_manual(self, shapes) -> None:
        for shape in list(shapes or []):
            try:
                mark_polygon_shape_manual(shape)
            except Exception:
                continue

    def _shared_current_scale(self) -> float:
        scale_attr = getattr(self, "scale", None)
        if isinstance(scale_attr, (int, float)):
            return max(float(scale_attr), 1e-6)
        current_scale = getattr(self, "current_scale", None)
        if callable(current_scale):
            try:
                return max(float(current_scale()), 1e-6)
            except Exception:
                pass
        return 1.0

    def _shared_sync_vertex(self, shape, index, point=None):
        shapes = self._shared_polygon_shapes()
        epsilon = max(
            1.0,
            float(getattr(self, "epsilon", 10.0)) / self._shared_current_scale(),
        )
        try:
            registry = getattr(self, "_shared_topology_registry", None)
            if isinstance(registry, SharedTopologyRegistry):
                result = registry.move_vertex(
                    shape,
                    int(index),
                    shapes,
                    epsilon,
                    point=point,
                )
            else:
                result = synchronize_shared_vertex(
                    shape,
                    int(index),
                    shapes,
                    epsilon,
                    point=point,
                )
            self._refresh_shared_topology_registry()
            self._shared_mark_polygon_shapes_manual([shape])
            if isinstance(result, str) and result:
                registry = getattr(self, "_shared_topology_registry", None)
                if isinstance(registry, SharedTopologyRegistry):
                    self._shared_mark_polygon_shapes_manual(
                        [
                            candidate
                            for candidate, _ in registry.vertex_occurrences(result)
                        ]
                    )
            return result
        except Exception:
            return None

    def _shared_move_selected_shapes(self, shapes, delta) -> bool:
        if delta is None:
            return False
        delta_point = QtCore.QPointF(delta)
        if abs(delta_point.x()) < 1e-8 and abs(delta_point.y()) < 1e-8:
            return False
        shapes_list = [shape for shape in list(shapes or []) if shape is not None]
        if not shapes_list:
            return False
        registry = getattr(self, "_shared_topology_registry", None)
        if isinstance(registry, SharedTopologyRegistry):
            moved = registry.translate_shapes(shapes_list, delta_point)
        else:
            moved = False
            for shape in shapes_list:
                try:
                    shape.moveBy(delta_point)
                    moved = True
                except Exception:
                    continue
            if moved:
                rebuild_polygon_topology(self._shared_polygon_shapes())
                self._refresh_shared_topology_registry()
        if moved:
            self._refresh_shared_topology_registry()
            self._shared_mark_polygon_shapes_manual(shapes_list)
            registry = getattr(self, "_shared_topology_registry", None)
            if isinstance(registry, SharedTopologyRegistry):
                moved_ids = set()
                for shape in shapes_list:
                    for vertex_id in list(
                        getattr(shape, "shared_vertex_ids", []) or []
                    ):
                        vertex_key = str(vertex_id or "")
                        if vertex_key:
                            moved_ids.add(vertex_key)
                for vertex_id in moved_ids:
                    self._shared_mark_polygon_shapes_manual(
                        [
                            candidate
                            for candidate, _ in registry.vertex_occurrences(vertex_id)
                        ]
                    )
        return bool(moved)

    def _shared_insert_vertex_on_edge(self, shape, edge_index: int, point):
        shapes = self._shared_polygon_shapes()
        registry = getattr(self, "_shared_topology_registry", None)
        if isinstance(registry, SharedTopologyRegistry):
            result = registry.insert_vertex_on_edge(shape, edge_index, point)
        else:
            result = insert_shared_vertex_on_edge(shape, edge_index, point, shapes)
        if result is None:
            return False
        self._refresh_shared_topology_registry()
        inserted_shapes = [candidate for candidate, _ in result.get("inserted", [])]
        self._shared_mark_polygon_shapes_manual(inserted_shapes)
        return result

    def _shared_remove_vertex(self, shape, vertex_index: int) -> bool:
        shapes = self._shared_polygon_shapes()
        registry = getattr(self, "_shared_topology_registry", None)
        if isinstance(registry, SharedTopologyRegistry):
            result = registry.remove_vertex_at(shape, vertex_index)
        else:
            result = remove_shared_vertex_at(shape, vertex_index, shapes)
        if result is None:
            return False
        self._refresh_shared_topology_registry()
        removed_shapes = [candidate for candidate, _ in result.get("removed", [])]
        self._shared_mark_polygon_shapes_manual(removed_shapes)
        return True

    def _shared_reshape_boundary(self, shape, edge_index: int, delta) -> bool:
        registry = getattr(self, "_shared_topology_registry", None)
        if not isinstance(registry, SharedTopologyRegistry):
            self._refresh_shared_topology_registry()
            registry = getattr(self, "_shared_topology_registry", None)
        if not isinstance(registry, SharedTopologyRegistry):
            return False
        result = registry.reshape_edge(shape, int(edge_index), delta)
        if result is None:
            return False
        self._refresh_shared_topology_registry()
        affected = [candidate for candidate, _ in result.get("occurrences", [])]
        self._shared_mark_polygon_shapes_manual(affected)
        return True

    def _shared_finalize_topology_edit(self):
        """Rebuild and resync shared topology after a committed edit."""
        rebuild_polygon_topology(self._shared_polygon_shapes())
        return self._refresh_shared_topology_registry()

    def _shared_adjoining_seed_for_shape(self, shape, edge_index=None):
        if shape is None:
            return None
        seed_fn = getattr(shape, "adjoining_polygon_seed", None)
        if not callable(seed_fn):
            return None

        candidate_indices: list[int] = []
        if edge_index is not None:
            try:
                candidate_indices.append(int(edge_index))
            except Exception:
                pass

        for shape_attr, edge_attr in (
            ("hShape", "hEdge"),
            ("_active_shape", "_active_edge_index"),
        ):
            active_shape = getattr(self, shape_attr, None)
            if active_shape is shape:
                active_edge = getattr(self, edge_attr, None)
                if active_edge is not None:
                    try:
                        candidate_indices.append(int(active_edge))
                    except Exception:
                        pass

        registry = getattr(self, "_shared_topology_registry", None)
        if isinstance(registry, SharedTopologyRegistry):
            shared_edge_ids = list(getattr(shape, "shared_edge_ids", []) or [])
            for index, edge_id in enumerate(shared_edge_ids):
                if not edge_id:
                    continue
                try:
                    if len(registry.edge_occurrences(edge_id)) >= 2:
                        candidate_indices.append(int(index))
                except Exception:
                    continue

        points = list(getattr(shape, "points", []) or [])
        if not candidate_indices and len(points) >= 2:
            candidate_indices.append(1 if len(points) > 1 else 0)

        seen: set[int] = set()
        for index in candidate_indices:
            if index in seen:
                continue
            seen.add(index)
            try:
                seed = seed_fn(index)
            except Exception:
                seed = None
            if seed is not None:
                return seed
        return None

    def _shared_link_adjoining_point(
        self,
        current_shape,
        point_index: int,
        point,
        source_shape,
        feature,
    ) -> bool:
        if (
            current_shape is None
            or source_shape is None
            or not isinstance(feature, dict)
        ):
            return False
        try:
            normalized_index = int(point_index)
        except Exception:
            return False
        if normalized_index < 0 or normalized_index >= len(
            getattr(current_shape, "points", []) or []
        ):
            return False

        feature_kind = str(feature.get("kind") or "").lower()
        if feature_kind == "vertex":
            try:
                source_index = int(feature.get("index"))
            except Exception:
                return False
            try:
                return bool(
                    current_shape.share_vertex_with(
                        normalized_index,
                        source_shape,
                        source_index,
                    )
                )
            except Exception:
                return False

        if feature_kind != "edge":
            return False
        try:
            source_edge_index = int(feature.get("index"))
        except Exception:
            return False
        result = self._shared_insert_vertex_on_edge(
            source_shape,
            source_edge_index,
            QtCore.QPointF(point),
        )
        if not result:
            return False
        vertex_id = str(result.get("vertex_id") or "")
        if not vertex_id:
            try:
                vertex_id = str(source_shape.shared_vertex_id(source_edge_index) or "")
            except Exception:
                vertex_id = ""
        if not vertex_id:
            return False
        setter = getattr(current_shape, "set_shared_vertex_id", None)
        if callable(setter):
            try:
                setter(normalized_index, vertex_id)
            except Exception:
                return False
        try:
            current_shape.points[normalized_index] = QtCore.QPointF(point)
        except Exception:
            return False
        try:
            # Commit the shared topology immediately so tiled and canvas views
            # both reflect the inserted boundary point before the next drag.
            self._shared_finalize_topology_edit()
        except Exception:
            return False
        return True
