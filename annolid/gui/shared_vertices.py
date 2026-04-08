from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from qtpy import QtCore


def _shape_points(shape):
    return list(getattr(shape, "points", []) or [])


def _is_polygon_shape(shape) -> bool:
    return str(getattr(shape, "shape_type", "") or "").lower() == "polygon"


def _is_closed_polygon(shape) -> bool:
    if not _is_polygon_shape(shape):
        return False
    is_closed = getattr(shape, "isClosed", None)
    if callable(is_closed):
        try:
            return bool(is_closed())
        except Exception:
            return False
    return True


def _shape_shared_ids(shape):
    points = _shape_points(shape)
    shared_ids = list(getattr(shape, "shared_vertex_ids", []) or [])
    if len(shared_ids) < len(points):
        shared_ids.extend([""] * (len(points) - len(shared_ids)))
    if len(shared_ids) > len(points):
        shared_ids = shared_ids[: len(points)]
    return shared_ids


def _shape_shared_edge_ids(shape):
    points = _shape_points(shape)
    shared_ids = list(getattr(shape, "shared_edge_ids", []) or [])
    if len(shared_ids) < len(points):
        shared_ids.extend([""] * (len(points) - len(shared_ids)))
    if len(shared_ids) > len(points):
        shared_ids = shared_ids[: len(points)]
    return shared_ids


def _ensure_shared_ids(shape) -> list[str]:
    ensure = getattr(shape, "_ensure_shared_vertex_ids", None)
    if callable(ensure):
        try:
            ensure()
        except Exception:
            pass
    return _shape_shared_ids(shape)


def _ensure_shared_edge_ids(shape) -> list[str]:
    ensure = getattr(shape, "_ensure_shared_edge_ids", None)
    if callable(ensure):
        try:
            ensure()
        except Exception:
            pass
    return _shape_shared_edge_ids(shape)


def _set_shared_ids(shape, shared_ids: list[str]) -> None:
    try:
        shape.shared_vertex_ids = [str(value or "") for value in shared_ids]
    except Exception:
        pass


def _set_shared_edge_ids(shape, shared_ids: list[str]) -> None:
    try:
        shape.shared_edge_ids = [str(value or "") for value in shared_ids]
    except Exception:
        pass


def _set_point(shape, index: int, point: QtCore.QPointF) -> None:
    try:
        shape.points[index] = QtCore.QPointF(point)
    except Exception:
        pass


def _shared_vertex_id(shape, index: int) -> str:
    getter = getattr(shape, "shared_vertex_id", None)
    if callable(getter):
        try:
            return str(getter(index) or "")
        except Exception:
            return ""
    shared_ids = _shape_shared_ids(shape)
    if 0 <= index < len(shared_ids):
        return str(shared_ids[index] or "")
    return ""


def _set_shared_vertex_id(shape, index: int, vertex_id: str) -> None:
    setter = getattr(shape, "set_shared_vertex_id", None)
    if callable(setter):
        try:
            setter(index, vertex_id)
            return
        except Exception:
            pass
    shared_ids = _shape_shared_ids(shape)
    if 0 <= index < len(shared_ids):
        shared_ids[index] = str(vertex_id or "")
        _set_shared_ids(shape, shared_ids)


def _shared_edge_id(shape, index: int) -> str:
    getter = getattr(shape, "shared_edge_id", None)
    if callable(getter):
        try:
            return str(getter(index) or "")
        except Exception:
            return ""
    shared_ids = _shape_shared_edge_ids(shape)
    if 0 <= index < len(shared_ids):
        return str(shared_ids[index] or "")
    return ""


def _set_shared_edge_id(shape, index: int, edge_id: str) -> None:
    setter = getattr(shape, "set_shared_edge_id", None)
    if callable(setter):
        try:
            setter(index, edge_id)
            return
        except Exception:
            pass
    shared_ids = _shape_shared_edge_ids(shape)
    if 0 <= index < len(shared_ids):
        shared_ids[index] = str(edge_id or "")
        _set_shared_edge_ids(shape, shared_ids)


def _relabel_shared_vertex_group(shapes, old_id: str, new_id: str) -> None:
    old_id = str(old_id or "")
    new_id = str(new_id or "")
    if not old_id or not new_id or old_id == new_id:
        return
    for shape in shapes or []:
        ids = _ensure_shared_ids(shape)
        changed = False
        for index, vertex_id in enumerate(ids):
            if str(vertex_id or "") == old_id:
                ids[index] = new_id
                changed = True
        if changed:
            _set_shared_ids(shape, ids)


def _propagate_shared_vertex_position(shapes, shared_id: str, point) -> None:
    shared_id = str(shared_id or "")
    if not shared_id:
        return
    shared_point = QtCore.QPointF(point)
    for shape in shapes or []:
        ids = _ensure_shared_ids(shape)
        for index, vertex_id in enumerate(ids):
            if str(vertex_id or "") == shared_id and index < len(_shape_points(shape)):
                _set_point(shape, index, shared_point)


def _shared_vertex_occurrences(shapes, shared_id: str) -> list[tuple[object, int]]:
    shared_id = str(shared_id or "")
    if not shared_id:
        return []
    occurrences: list[tuple[object, int]] = []
    for shape in shapes or []:
        ids = _ensure_shared_ids(shape)
        for index, vertex_id in enumerate(ids):
            if str(vertex_id or "") == shared_id:
                occurrences.append((shape, index))
    return occurrences


def _edge_vertex_ids(shape, edge_index: int):
    points = _shape_points(shape)
    if len(points) < 2:
        return None
    try:
        index = int(edge_index)
    except Exception:
        return None
    if index < 0 or index >= len(points):
        return None
    start_index = index - 1
    if start_index < 0:
        start_index = len(points) - 1
    start_id = _shared_vertex_id(shape, start_index)
    end_id = _shared_vertex_id(shape, index)
    if not start_id or not end_id:
        return None
    if start_id == end_id:
        return None
    return start_id, end_id


def _canonical_edge_key(start_id: str, end_id: str):
    start_id = str(start_id or "")
    end_id = str(end_id or "")
    if not start_id or not end_id or start_id == end_id:
        return None
    return tuple(sorted((start_id, end_id)))


def _matching_edge_index(shape, shared_edge_id: str):
    shared_edge_id = str(shared_edge_id or "")
    if not shared_edge_id:
        return None
    edge_ids = _ensure_shared_edge_ids(shape)
    for index, edge_id in enumerate(edge_ids):
        if str(edge_id or "") == shared_edge_id:
            return int(index)
    return None


def _matching_vertex_indices(shape, shared_vertex_id: str):
    shared_vertex_id = str(shared_vertex_id or "")
    if not shared_vertex_id:
        return []
    ids = _ensure_shared_ids(shape)
    return [
        int(index)
        for index, vertex_id in enumerate(ids)
        if str(vertex_id or "") == shared_vertex_id
    ]


@dataclass
class SharedTopologyRegistry:
    """Explicit shared-vertex and shared-edge topology for polygon layers."""

    shapes: list = field(default_factory=list)
    vertex_members: dict[str, list[tuple[object, int]]] = field(default_factory=dict)
    edge_members: dict[str, list[tuple[object, int]]] = field(default_factory=dict)
    edge_vertex_pairs: dict[str, tuple[str, str]] = field(default_factory=dict)
    assigned_edge_ids: dict[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def from_shapes(cls, shapes):
        registry = cls(list(shapes or []))
        registry.rebuild()
        return registry

    def rebuild(self):
        self.vertex_members.clear()
        self.edge_members.clear()
        self.edge_vertex_pairs.clear()
        self.assigned_edge_ids.clear()

        edge_groups: dict[tuple[str, str], list[tuple[object, int]]] = {}
        ordered_refs: list[tuple[object, int, tuple[str, str] | None]] = []

        for shape in self.shapes or []:
            if not _is_closed_polygon(shape):
                continue
            points = _shape_points(shape)
            if len(points) < 3:
                continue
            vertex_ids = _ensure_shared_ids(shape)
            edge_ids = _ensure_shared_edge_ids(shape)
            if len(edge_ids) != len(points):
                edge_ids = edge_ids[: len(points)] + [""] * max(
                    0, len(points) - len(edge_ids)
                )
            _set_shared_edge_ids(shape, edge_ids)

            for vertex_index, vertex_id in enumerate(vertex_ids):
                vertex_key = str(vertex_id or "")
                if not vertex_key:
                    continue
                self.vertex_members.setdefault(vertex_key, []).append(
                    (shape, vertex_index)
                )

            for edge_index in range(len(points)):
                vertex_ids_pair = _edge_vertex_ids(shape, edge_index)
                key = _canonical_edge_key(*vertex_ids_pair) if vertex_ids_pair else None
                ordered_refs.append((shape, edge_index, key))
                if key is not None:
                    edge_groups.setdefault(key, []).append((shape, edge_index))

        for key, refs in edge_groups.items():
            existing = None
            for shape, edge_index in refs:
                current = _shared_edge_id(shape, edge_index)
                if current:
                    existing = current
                    break
            edge_id = existing or uuid.uuid4().hex
            self.assigned_edge_ids[key] = edge_id
            self.edge_vertex_pairs[edge_id] = key
            for shape, edge_index in refs:
                _set_shared_edge_id(shape, edge_index, edge_id)
                self.edge_members.setdefault(edge_id, []).append((shape, edge_index))

        for shape, edge_index, key in ordered_refs:
            if key is not None:
                continue
            current = _shared_edge_id(shape, edge_index)
            if not current:
                current = uuid.uuid4().hex
                _set_shared_edge_id(shape, edge_index, current)
            self.edge_members.setdefault(current, []).append((shape, edge_index))
        return self.assigned_edge_ids

    def vertex_occurrences(self, vertex_id: str):
        return list(self.vertex_members.get(str(vertex_id or ""), []) or [])

    def edge_occurrences(self, edge_id: str):
        return list(self.edge_members.get(str(edge_id or ""), []) or [])

    def move_vertex(
        self,
        shape,
        index: int,
        shapes,
        epsilon: float,
        *,
        point=None,
        exclude_shape=None,
    ):
        result = synchronize_shared_vertex(
            shape,
            index,
            shapes,
            epsilon,
            point=point,
            exclude_shape=exclude_shape,
        )
        if result is not None:
            self.shapes = list(shapes or [])
            self.rebuild()
        return result

    def insert_vertex_on_edge(self, shape, edge_index: int, point):
        result = insert_shared_vertex_on_edge(shape, edge_index, point, self.shapes)
        if result is not None:
            self.shapes = list(self.shapes or [])
            self.rebuild()
        return result

    def remove_vertex_at(self, shape, vertex_index: int):
        result = remove_shared_vertex_at(shape, vertex_index, self.shapes)
        if result is not None:
            self.shapes = list(self.shapes or [])
            self.rebuild()
        return result

    def translate_shapes(self, moving_shapes, delta):
        if delta is None:
            return False
        delta_point = QtCore.QPointF(delta)
        if abs(delta_point.x()) < 1e-8 and abs(delta_point.y()) < 1e-8:
            return False

        moving_shapes = [
            shape for shape in list(moving_shapes or []) if shape is not None
        ]
        if not moving_shapes:
            return False

        # Refresh topology membership before translating so shared-vertex groups
        # are accurate for the current shape set.
        self.rebuild()

        moved_shared_vertex_ids: set[str] = set()
        for shape in moving_shapes:
            ids = _ensure_shared_ids(shape)
            for vertex_id in ids:
                vertex_key = str(vertex_id or "")
                if vertex_key:
                    moved_shared_vertex_ids.add(vertex_key)

        moved = False
        for shared_vertex_id in moved_shared_vertex_ids:
            for candidate, candidate_index in self.vertex_occurrences(shared_vertex_id):
                points = _shape_points(candidate)
                if candidate_index < 0 or candidate_index >= len(points):
                    continue
                _set_point(
                    candidate,
                    candidate_index,
                    QtCore.QPointF(points[candidate_index]) + delta_point,
                )
                moved = True

        for shape in moving_shapes:
            ids = _ensure_shared_ids(shape)
            points = _shape_points(shape)
            for index, point in enumerate(points):
                if index < len(ids) and str(ids[index] or ""):
                    continue
                _set_point(shape, index, QtCore.QPointF(point) + delta_point)
                moved = True

        if moved:
            self.rebuild()
        return moved

    def reshape_edge(self, shape, edge_index: int, delta):
        if shape is None or not _is_polygon_shape(shape):
            return None
        try:
            normalized_index = int(edge_index)
        except Exception:
            return None
        edge_id = _shared_edge_id(shape, normalized_index)
        if not edge_id:
            return None
        occurrences = self.edge_occurrences(edge_id)
        if len(occurrences) < 2:
            return None
        vertex_ids = _edge_vertex_ids(shape, normalized_index)
        if vertex_ids is None:
            return None
        start_id, end_id = vertex_ids
        delta_point = QtCore.QPointF(delta)
        if abs(delta_point.x()) < 1e-8 and abs(delta_point.y()) < 1e-8:
            return None
        points = _shape_points(shape)
        if normalized_index < 0 or normalized_index >= len(points):
            return None
        start_point = QtCore.QPointF(points[normalized_index - 1]) + delta_point
        end_point = QtCore.QPointF(points[normalized_index]) + delta_point
        for candidate, candidate_edge_index in occurrences:
            ids = _ensure_shared_ids(candidate)
            for vertex_index, vertex_id in enumerate(ids):
                vertex_key = str(vertex_id or "")
                if vertex_key == start_id:
                    _set_point(candidate, vertex_index, start_point)
                elif vertex_key == end_id:
                    _set_point(candidate, vertex_index, end_point)
        self.rebuild()
        return {
            "edge_id": edge_id,
            "vertex_ids": (start_id, end_id),
            "delta": QtCore.QPointF(delta_point),
            "occurrences": occurrences,
        }


def nearest_vertex_match(
    shapes,
    point: QtCore.QPointF,
    epsilon: float,
    *,
    exclude_shape=None,
):
    """Return the nearest vertex in other shapes within epsilon.

    Returns a tuple of (shape, index, point, distance) or None.
    """
    best = None
    best_distance = None
    for shape in shapes or []:
        if shape is None or shape is exclude_shape or not _is_polygon_shape(shape):
            continue
        points = _shape_points(shape)
        if not points:
            continue
        nearest = getattr(shape, "nearestVertex", None)
        if not callable(nearest):
            continue
        try:
            vertex_index = nearest(point, epsilon)
        except Exception:
            vertex_index = None
        if vertex_index is None:
            continue
        try:
            candidate = QtCore.QPointF(points[int(vertex_index)])
        except Exception:
            continue
        distance = QtCore.QLineF(point, candidate).length()
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = (shape, int(vertex_index), candidate, distance)
    return best


def rebuild_polygon_topology(shapes):
    """Assign explicit shared edge IDs from shared vertex IDs.

    Every polygon edge gets a stable edge identifier. When two polygons share
    the same pair of vertex IDs, they also receive the same edge ID, turning the
    boundary into a shared topological edge object.
    """
    registry = SharedTopologyRegistry(list(shapes or []))
    return registry.rebuild()


def insert_shared_vertex_on_edge(shape, edge_index: int, point, shapes):
    """Insert a vertex on an edge and propagate it to matching shared boundaries.

    The clicked polygon edge is inserted in-place. If the edge belongs to a
    shared topological boundary, every polygon that carries that same edge ID
    receives the same inserted vertex and the shared topology is rebuilt after
    the edit.
    """
    if shape is None or not _is_polygon_shape(shape):
        return None
    points = _shape_points(shape)
    if len(points) < 2:
        return None
    try:
        insert_index = int(edge_index)
    except Exception:
        return None
    if insert_index < 0 or insert_index >= len(points):
        return None

    shared_edge_id = _shared_edge_id(shape, insert_index)
    edge_refs: list[tuple[object, int]] = []
    if shared_edge_id:
        for candidate in shapes or []:
            if candidate is None or not _is_closed_polygon(candidate):
                continue
            candidate_index = _matching_edge_index(candidate, shared_edge_id)
            if candidate_index is not None:
                edge_refs.append((candidate, candidate_index))
    if not edge_refs:
        edge_refs = [(shape, insert_index)]

    shared_vertex_id = uuid.uuid4().hex
    inserted: list[tuple[object, int]] = []
    inserted_point = QtCore.QPointF(point)
    for candidate, candidate_index in edge_refs:
        try:
            candidate.insertPoint(candidate_index, QtCore.QPointF(inserted_point))
            candidate.set_shared_vertex_id(candidate_index, shared_vertex_id)
            inserted.append((candidate, candidate_index))
        except Exception:
            continue

    rebuild_polygon_topology(shapes)
    return {
        "vertex_id": shared_vertex_id,
        "edge_id": shared_edge_id,
        "inserted": inserted,
    }


def remove_shared_vertex_at(shape, vertex_index: int, shapes):
    """Remove a vertex and propagate the deletion to matching shared vertices."""
    if shape is None or not _is_polygon_shape(shape):
        return None
    points = _shape_points(shape)
    if not points:
        return None
    try:
        normalized_index = int(vertex_index)
    except Exception:
        return None
    if normalized_index < 0 or normalized_index >= len(points):
        return None

    shared_vertex_id = _shared_vertex_id(shape, normalized_index)
    if shared_vertex_id:
        candidate_refs: list[tuple[object, list[int]]] = []
        for candidate in shapes or []:
            if candidate is None or not _is_polygon_shape(candidate):
                continue
            indices = _matching_vertex_indices(candidate, shared_vertex_id)
            if indices:
                candidate_refs.append((candidate, indices))
        if not candidate_refs:
            candidate_refs = [(shape, [normalized_index])]
    else:
        candidate_refs = [(shape, [normalized_index])]

    removed: list[tuple[object, int]] = []
    for candidate, indices in candidate_refs:
        for index in sorted(set(indices), reverse=True):
            candidate_points = _shape_points(candidate)
            if index < 0 or index >= len(candidate_points):
                continue
            try:
                candidate.removePoint(index)
                removed.append((candidate, index))
            except Exception:
                continue

    rebuild_polygon_topology(shapes)
    return {
        "vertex_id": shared_vertex_id,
        "removed": removed,
    }


def synchronize_shared_vertex(
    shape,
    index: int,
    shapes,
    epsilon: float,
    *,
    point=None,
    exclude_shape=None,
):
    """Merge or propagate a moved vertex into shared vertex groups.

    If the moved vertex already belongs to a shared group, all linked vertices
    are updated to the new coordinate. Otherwise, if the moved vertex lands
    within epsilon of another polygon vertex, the two vertices are merged and
    assigned the same shared vertex id.
    """
    if shape is None:
        return None
    if not _is_polygon_shape(shape):
        return None
    points = _shape_points(shape)
    if index < 0 or index >= len(points):
        return None

    _ensure_shared_ids(shape)
    moved_point = QtCore.QPointF(point if point is not None else points[index])
    shared_id = _shared_vertex_id(shape, index)
    if len(_shared_vertex_occurrences(shapes, shared_id)) > 1:
        _propagate_shared_vertex_position(shapes, shared_id, moved_point)
        return shared_id

    match = nearest_vertex_match(
        shapes,
        moved_point,
        epsilon,
        exclude_shape=exclude_shape if exclude_shape is not None else shape,
    )

    if match is None:
        return None

    other_shape, other_index, other_point, _ = match
    other_id = _shared_vertex_id(other_shape, other_index)
    merged_id = shared_id or other_id or uuid.uuid4().hex

    if shared_id and other_id and shared_id != other_id:
        _relabel_shared_vertex_group(shapes, other_id, shared_id)
        merged_id = shared_id
    if other_id and other_id != merged_id:
        _relabel_shared_vertex_group(shapes, other_id, merged_id)
    _set_shared_vertex_id(shape, index, merged_id)
    _set_shared_vertex_id(other_shape, other_index, merged_id)
    _propagate_shared_vertex_position(shapes, merged_id, other_point)
    return merged_id


def synchronize_shared_topology(
    shape,
    index: int,
    shapes,
    epsilon: float,
    *,
    point=None,
    exclude_shape=None,
):
    vertex_id = synchronize_shared_vertex(
        shape,
        index,
        shapes,
        epsilon,
        point=point,
        exclude_shape=exclude_shape,
    )
    rebuild_polygon_topology(shapes)
    return vertex_id
