from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET
from functools import lru_cache

from annolid.gui.flybody_support import resolve_local_flybody_repo
from annolid.utils.logger import logger

THREEJS_EXAMPLE_IDS = (
    "helix_points_csv",
    "wave_surface_obj",
    "sphere_points_ply",
    "flybody_simulation_json",
    "brain_viewer_html",
    "two_mice_html",
    "swarm_visualizer_html",
)

_FLYBODY_PART_STYLES = {
    "thorax": {
        "category": "thorax",
        "color": "#9c6b3f",
        "roughness": 0.58,
        "metalness": 0.04,
    },
    "head": {
        "category": "head",
        "color": "#b17b4a",
        "roughness": 0.56,
        "metalness": 0.04,
    },
    "antenna": {
        "category": "antenna",
        "color": "#d6a96f",
        "roughness": 0.52,
        "metalness": 0.03,
    },
    "wing": {
        "category": "wing",
        "color": "#d9c7a1",
        "roughness": 0.34,
        "metalness": 0.01,
    },
    "abdomen": {
        "category": "abdomen",
        "color": "#7a4b2a",
        "roughness": 0.62,
        "metalness": 0.03,
    },
    "haltere": {
        "category": "haltere",
        "color": "#c99b62",
        "roughness": 0.5,
        "metalness": 0.03,
    },
    "leg": {"category": "leg", "color": "#8b5a32", "roughness": 0.6, "metalness": 0.03},
    "mouth": {
        "category": "mouth",
        "color": "#a87043",
        "roughness": 0.56,
        "metalness": 0.03,
    },
    "default": {
        "category": "body",
        "color": "#c8ab72",
        "roughness": 0.55,
        "metalness": 0.04,
    },
}


def _flybody_part_style(body_name: str) -> dict[str, object]:
    label = str(body_name or "").strip().lower()
    if label == "thorax":
        return dict(_FLYBODY_PART_STYLES["thorax"])
    if "head" in label:
        return dict(_FLYBODY_PART_STYLES["head"])
    if "antenna" in label:
        return dict(_FLYBODY_PART_STYLES["antenna"])
    if "wing" in label:
        return dict(_FLYBODY_PART_STYLES["wing"])
    if "abdomen" in label:
        return dict(_FLYBODY_PART_STYLES["abdomen"])
    if "haltere" in label:
        return dict(_FLYBODY_PART_STYLES["haltere"])
    if any(term in label for term in ("coxa", "femur", "tibia", "tarsus", "claw")):
        return dict(_FLYBODY_PART_STYLES["leg"])
    if any(term in label for term in ("rostrum", "haustellum", "labrum")):
        return dict(_FLYBODY_PART_STYLES["mouth"])
    return dict(_FLYBODY_PART_STYLES["default"])


def attach_flybody_mesh(payload: dict, out_dir: str | Path) -> dict:
    mesh_filename = _maybe_build_flybody_mesh_example(Path(out_dir))
    if mesh_filename:
        payload = dict(payload)
        payload["mesh"] = {"type": "obj", "path": mesh_filename}
    return payload


def attach_flybody_mesh_parts(payload: dict, out_dir: str | Path) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    parts = _maybe_build_flybody_mesh_parts(out_path)
    if parts:
        payload = dict(payload)
        payload["mesh"] = {"type": "flybody_parts", "parts": parts}
    return payload


def attach_flybody_floor(payload: dict) -> dict:
    floor = _load_flybody_floor()
    if not floor:
        return payload
    payload = dict(payload)
    environment = dict(payload.get("environment") or {})
    environment["floor"] = floor
    payload["environment"] = environment
    return payload


def generate_threejs_example(example_id: str, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    builders: dict[str, Callable[[Path], Path]] = {
        "helix_points_csv": _build_helix_points_csv,
        "wave_surface_obj": _build_wave_surface_obj,
        "sphere_points_ply": _build_sphere_points_ply,
        "flybody_simulation_json": _build_flybody_simulation_json,
        "brain_viewer_html": _build_brain_viewer_html,
        "two_mice_html": _build_two_mice_html,
        "swarm_visualizer_html": _build_swarm_visualizer_html,
    }
    builder = builders.get(example_id)
    if builder is not None:
        return builder(out)
    raise ValueError(f"Unknown Three.js example: {example_id}")


def _build_brain_viewer_html(_out: Path) -> Path:
    return _resolve_or_generate_bundled_html(
        "points_3d.html",
        _out,
        title="Annolid Three.js Brain Viewer Example",
    )


def _build_two_mice_html(_out: Path) -> Path:
    return _resolve_or_generate_bundled_html(
        "two_mice.html",
        _out,
        title="Annolid Three.js Two Mice Example",
    )


def _build_swarm_visualizer_html(_out: Path) -> Path:
    return _resolve_or_generate_bundled_html(
        "swarm_visualizer.html",
        _out,
        title="Annolid Three.js Swarm Visualizer Example",
    )


def _resolve_or_generate_bundled_html(filename: str, out: Path, *, title: str) -> Path:
    # Prefer bundled viewer HTML when present.
    bundled = Path(__file__).resolve().parent / "assets" / "threejs" / filename
    if bundled.exists():
        return bundled

    # Fallback for slim package builds missing optional example assets.
    fallback = out / filename
    fallback.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang='en'>",
                "<head>",
                "  <meta charset='utf-8'/>",
                f"  <title>{title}</title>",
                "</head>",
                "<body>",
                f"  <h1>{title}</h1>",
                "  <p>Bundled Three.js example asset was not packaged in this install.</p>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return fallback


def _build_helix_points_csv(out: Path) -> Path:
    path = out / "helix_points.csv"
    _write_helix_points_csv(path)
    return path


def _build_wave_surface_obj(out: Path) -> Path:
    path = out / "wave_surface.obj"
    _write_wave_surface_obj(path)
    return path


def _build_sphere_points_ply(out: Path) -> Path:
    path = out / "sphere_points.ply"
    _write_sphere_points_ply(path)
    return path


def _build_flybody_simulation_json(out: Path) -> Path:
    path = out / "flybody_simulation.json"
    labels = [
        "head_site",
        "left_antenna_site",
        "right_antenna_site",
        "thorax_site",
        "abdomen_tip_site",
        "left_front_tarsus_site",
        "right_front_tarsus_site",
        "left_middle_tarsus_site",
        "right_middle_tarsus_site",
        "left_hind_tarsus_site",
        "right_hind_tarsus_site",
    ]
    edges = [
        ["head_site", "thorax_site"],
        ["thorax_site", "abdomen_tip_site"],
        ["head_site", "left_antenna_site"],
        ["head_site", "right_antenna_site"],
        ["thorax_site", "left_front_tarsus_site"],
        ["thorax_site", "right_front_tarsus_site"],
        ["thorax_site", "left_middle_tarsus_site"],
        ["thorax_site", "right_middle_tarsus_site"],
        ["thorax_site", "left_hind_tarsus_site"],
        ["thorax_site", "right_hind_tarsus_site"],
    ]
    frames = []
    for i in range(48):
        phase = i / 47.0
        sway = 0.08 * math.sin(phase * math.tau * 2.0)
        lift = 0.04 * math.sin(phase * math.tau * 4.0)
        points = [
            {"label": "head_site", "x": 0.0 + sway, "y": 0.22, "z": 0.06 + lift * 0.25},
            {"label": "left_antenna_site", "x": -0.06 + sway, "y": 0.29, "z": 0.08},
            {"label": "right_antenna_site", "x": 0.06 + sway, "y": 0.29, "z": 0.08},
            {"label": "thorax_site", "x": 0.0, "y": 0.0, "z": 0.0},
            {
                "label": "abdomen_tip_site",
                "x": -0.01 - sway * 0.5,
                "y": -0.26,
                "z": -0.02,
            },
            {
                "label": "left_front_tarsus_site",
                "x": -0.18,
                "y": 0.04,
                "z": -0.08 - lift,
            },
            {
                "label": "right_front_tarsus_site",
                "x": 0.18,
                "y": 0.04,
                "z": -0.08 + lift,
            },
            {
                "label": "left_middle_tarsus_site",
                "x": -0.21,
                "y": -0.05,
                "z": -0.1 + lift * 0.5,
            },
            {
                "label": "right_middle_tarsus_site",
                "x": 0.21,
                "y": -0.05,
                "z": -0.1 - lift * 0.5,
            },
            {
                "label": "left_hind_tarsus_site",
                "x": -0.2,
                "y": -0.18,
                "z": -0.12 - lift * 0.35,
            },
            {
                "label": "right_hind_tarsus_site",
                "x": 0.2,
                "y": -0.18,
                "z": -0.12 + lift * 0.35,
            },
        ]
        # Walk in a gentle circle on the ground plane.
        # model_pose.position uses MuJoCo world units (un-scaled); scale=7.5 is
        # applied by the JS. z=0.0 means thorax at origin; feet reach ~z=-0.12
        # which maps to ~-0.9 in scene space — right at the floor level.
        walk_x = 0.55 * math.cos(phase * math.tau)
        walk_y = 0.55 * math.sin(phase * math.tau)
        bounce = 0.008 * math.sin(phase * math.tau * 8.0)  # subtle walking bounce
        # Yaw so the fly always faces its direction of travel
        yaw = phase * math.tau + math.pi / 2
        frames.append(
            {
                "frame_index": i,
                "timestamp_sec": round(i / 30.0, 4),
                "points": points,
                "qpos": [round(phase, 4), round(sway, 4)],
                "diagnostics": {},
                "dry_run": True,
                "model_pose": {
                    "position": [walk_x, walk_y, bounce],
                    "rotation": [0.0, 0.0, yaw],
                    "scale": 7.5,
                },
            }
        )
    payload: dict = {
        "kind": "annolid-simulation-v1",
        "title": "FlyBody Simulation Example",
        "adapter": "flybody",
        "labels": labels,
        "edges": edges,
        "display": {
            "show_points": False,
            "show_labels": False,
            "show_edges": False,
            "show_trails": False,
        },
        "metadata": {
            "run_metadata": {"source": "example"},
            "mapping_metadata": {"metadata": {"template": "flybody-example"}},
            "coordinate_system": {"units": "meters"},
        },
        "frames": frames,
    }
    payload = attach_flybody_mesh(payload, out)
    payload = attach_flybody_floor(payload)

    # Convert MuJoCo Z-up floor position to Three.js Y-up and scale it.
    # MuJoCo: floor at Z=-0.132 → Three.js Y=-0.132 (Z-up becomes Y-up).
    # JS reads floor.position[1] as the Y height of the horizontal plane.
    model_scale = 7.5
    env = payload.get("environment") or {}
    floor_env = env.get("floor")
    if floor_env and isinstance(floor_env.get("position"), list):
        fp = floor_env["position"]
        # fp is [mujoco_x, mujoco_y, mujoco_z]; floor Y in Three.js = mujoco_z * scale
        threejs_floor_y = fp[2] * model_scale  # e.g. -0.132 * 7.5 = -0.99
        payload = dict(payload)
        payload["environment"] = dict(env)
        payload["environment"]["floor"] = dict(floor_env)
        payload["environment"]["floor"]["position"] = [0.0, threejs_floor_y, 0.0]

    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def _parse_vector(value: str | None, *, count: int) -> list[float]:
    parts = [float(part) for part in str(value or "").split()[:count]]
    while len(parts) < count:
        parts.append(0.0)
    return parts


def _load_flybody_floor() -> dict[str, object] | None:
    repo_root = resolve_local_flybody_repo()
    if repo_root is None:
        return None
    floor_xml = repo_root / "flybody" / "fruitfly" / "assets" / "floor.xml"
    if not floor_xml.exists():
        return None
    try:
        root = ET.parse(floor_xml).getroot()
    except ET.ParseError:
        return None
    floor_geom = None
    for geom in root.findall(".//geom"):
        if str(geom.attrib.get("type") or "").strip() == "plane":
            floor_geom = geom
            break
    if floor_geom is None:
        return None
    size = _parse_vector(floor_geom.attrib.get("size"), count=3)
    pos = _parse_vector(floor_geom.attrib.get("pos"), count=3)
    return {
        "type": "plane",
        "size": size[:2],
        "position": pos,
        "color": "#314759",
        "gridColor": "#3f5f73",
    }


def _quat_mul(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product of two quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _compose_transform(
    parent_pos: tuple[float, float, float],
    parent_quat: tuple[float, float, float, float],
    child_pos: tuple[float, float, float],
    child_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compose two rigid transforms: world_T_parent * parent_T_child."""
    # Rotate child_pos into parent frame then add parent_pos
    rotated = _transform_vertex(child_pos, (0.0, 0.0, 0.0), parent_quat)
    new_pos = (
        parent_pos[0] + rotated[0],
        parent_pos[1] + rotated[1],
        parent_pos[2] + rotated[2],
    )
    new_quat = _quat_mul(parent_quat, child_quat)
    return new_pos, new_quat


def _quat_to_matrix(
    quat: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    w, x, y, z = quat
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _transform_vertex(
    vertex: tuple[float, float, float],
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    matrix = _quat_to_matrix(quat)
    x, y, z = vertex
    rx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + pos[0]
    ry = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + pos[1]
    rz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + pos[2]
    return (rx, ry, rz)


def _parse_obj_vertices_and_faces(
    path: Path,
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    stat = path.stat()
    return _parse_obj_vertices_and_faces_cached(str(path), int(stat.st_mtime_ns))


@lru_cache(maxsize=512)
def _parse_obj_vertices_and_faces_cached(
    path_str: str, _mtime_ns: int
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, ...], ...]]:
    path = Path(path_str)
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            face: list[int] = []
            for token in line.split()[1:]:
                try:
                    face.append(int(token.split("/")[0]))
                except ValueError:
                    continue
            if len(face) >= 3:
                faces.append(face)
    return tuple(vertices), tuple(tuple(face) for face in faces)


@lru_cache(maxsize=8)
def _load_flybody_asset_index(
    xml_path_str: str, xml_mtime_ns: int
) -> tuple[
    dict[str, str],
    float,
    ET.Element | None,
    dict[str, tuple[float, float, float, float]],
]:
    """Return (mesh_name->filename, default_mesh_scale, worldbody, xml_materials)."""
    del xml_mtime_ns
    xml_path = Path(xml_path_str)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mesh_files: dict[str, str] = {}
    xml_materials: dict[str, tuple[float, float, float, float]] = {}
    asset_node = root.find("asset")
    if asset_node is not None:
        for mesh in asset_node.findall("mesh"):
            name = str(mesh.attrib.get("name") or "").strip()
            file_name = str(mesh.attrib.get("file") or "").strip()
            if name and file_name:
                mesh_files[name] = file_name
        for mat in asset_node.findall("material"):
            mat_name = str(mat.attrib.get("name") or "").strip()
            rgba_str = str(mat.attrib.get("rgba") or "1 1 1 1").split()
            if mat_name and len(rgba_str) >= 4:
                try:
                    xml_materials[mat_name] = (
                        float(rgba_str[0]),
                        float(rgba_str[1]),
                        float(rgba_str[2]),
                        float(rgba_str[3]),
                    )
                except ValueError:
                    pass
    # Read global default mesh scale (e.g. "0.1 0.1 0.1" → 0.1)
    default_mesh_scale = 1.0
    default_node = root.find("default")
    if default_node is not None:
        default_mesh = default_node.find("mesh")
        if default_mesh is not None:
            scale_str = str(default_mesh.attrib.get("scale") or "").split()
            if scale_str:
                try:
                    default_mesh_scale = float(scale_str[0])
                except ValueError:
                    pass
    worldbody = root.find("worldbody")
    return mesh_files, default_mesh_scale, worldbody, xml_materials


def _maybe_build_flybody_mesh_example(out: Path) -> str | None:
    repo_root = resolve_local_flybody_repo()
    if repo_root is None:
        return None
    xml_path = repo_root / "flybody" / "fruitfly" / "assets" / "fruitfly.xml"
    asset_dir = xml_path.parent
    mesh_files, default_mesh_scale, worldbody, xml_materials = (
        _load_flybody_asset_index(
            str(xml_path),
            int(xml_path.stat().st_mtime_ns),
        )
    )
    if worldbody is None:
        return None

    selected_terms = (
        "thorax",
        "head",
        "antenna",
        "wing_left",
        "wing_right",
        "abdomen",
        "coxa_",
        "femur_",
        "tibia_",
        "tarsus_",
        "tarsal_claw_",
    )

    all_vertices: list[tuple[float, float, float]] = []
    all_faces: list[list[int]] = []
    _identity_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _identity_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def _parse_pos(s: str | None) -> tuple[float, float, float]:
        parts = str(s or "0 0 0").split()[:3]
        while len(parts) < 3:
            parts.append("0")
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            return (0.0, 0.0, 0.0)

    def _parse_quat(s: str | None) -> tuple[float, float, float, float]:
        parts = str(s or "1 0 0 0").split()[:4]
        while len(parts) < 4:
            parts.append("0")
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            return (1.0, 0.0, 0.0, 0.0)

    # Fallback material if geom has no material attr (use the XML body color)
    _FALLBACK_MAT = "body"
    # Each entry: (faces_list, xml_material_name)
    face_groups: list[tuple[list[list[int]], str]] = []

    def _walk_bodies(
        node: ET.Element,
        world_pos: tuple[float, float, float],
        world_quat: tuple[float, float, float, float],
        body_name: str = "",
    ) -> None:
        """Recursively walk the MuJoCo body tree, accumulating world transforms."""
        for body in node.findall("body"):
            bname = str(body.attrib.get("name") or body_name or "").strip()
            body_pos = _parse_pos(body.attrib.get("pos"))
            body_quat = _parse_quat(body.attrib.get("quat"))
            cur_pos, cur_quat = _compose_transform(
                world_pos, world_quat, body_pos, body_quat
            )

            for geom in body.findall("geom"):
                mesh_name = str(geom.attrib.get("mesh") or "").strip()
                if not mesh_name:
                    continue
                if not any(term in mesh_name for term in selected_terms):
                    continue
                mesh_file = mesh_files.get(mesh_name)
                if not mesh_file:
                    continue
                obj_path = asset_dir / mesh_file
                if not obj_path.exists():
                    continue

                geom_pos = _parse_pos(geom.attrib.get("pos"))
                geom_quat = _parse_quat(geom.attrib.get("quat"))
                full_pos, full_quat = _compose_transform(
                    cur_pos, cur_quat, geom_pos, geom_quat
                )

                vertices, faces = _parse_obj_vertices_and_faces(obj_path)
                if not vertices or not faces:
                    continue
                offset = len(all_vertices)
                for vx, vy, vz in vertices:
                    sv = (
                        vx * default_mesh_scale,
                        vy * default_mesh_scale,
                        vz * default_mesh_scale,
                    )
                    all_vertices.append(_transform_vertex(sv, full_pos, full_quat))
                # Use geom's material attribute directly (e.g. "body", "membrane", "red")
                xml_mat = str(geom.attrib.get("material") or _FALLBACK_MAT).strip()
                if xml_mat not in xml_materials:
                    xml_mat = _FALLBACK_MAT
                shifted_faces = [[offset + idx for idx in face] for face in faces]
                face_groups.append((shifted_faces, xml_mat))

            _walk_bodies(body, cur_pos, cur_quat, bname)

    _walk_bodies(worldbody, _identity_pos, _identity_quat)

    # Flatten faces from groups
    for group_faces, _cat in face_groups:
        all_faces.extend(group_faces)

    if not all_vertices or not all_faces:
        return None

    mtl_name = "flybody_combined.mtl"
    out_path = out / "flybody_combined.obj"
    mtl_path = out / mtl_name
    if (
        out_path.exists()
        and mtl_path.exists()
        and out_path.stat().st_mtime_ns >= xml_path.stat().st_mtime_ns
    ):
        logger.info("Reusing cached FlyBody combined mesh: %s", out_path)
        return out_path.name
    started = time.perf_counter()

    # --- Write MTL file directly from XML material RGBA values ---
    # XML shininess 0–1 → MTL Ns 0–1000; mem=translucent uses alpha as d
    _FLY_MAT_SHININESS = {
        "body": 0.6,
        "red": 0.692,
        "ocelli": 1.0,
        "black": 0.727,
        "bristle-brown": 0.907,
        "lower": 0.6,
        "brown": 0.6,
        "membrane": 0.907,
    }
    mtl_lines = ["# Annolid FlyBody materials — sourced from fruitfly.xml"]
    # Collect only materials actually used in face_groups
    used_mats = {mat for _, mat in face_groups}
    for mat_name in used_mats:
        rgba = xml_materials.get(mat_name)
        if rgba is None:
            rgba = (0.674, 0.35, 0.143, 1.0)  # fallback to body color
        r, g, b, a = rgba
        ns = round(_FLY_MAT_SHININESS.get(mat_name, 0.6) * 1000)
        ka_scale = 0.12
        mtl_lines += [
            f"newmtl {mat_name}",
            f"Ka {r * ka_scale:.4f} {g * ka_scale:.4f} {b * ka_scale:.4f}",
            f"Kd {r:.4f} {g:.4f} {b:.4f}",
            "Ks 0.05 0.05 0.05",
            f"Ns {ns}",
            f"d {a:.4f}",
            "illum 2",
            "",
        ]
    mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")

    # --- Write OBJ file with mtllib + per-group usemtl directives ---
    lines = [
        "# Annolid FlyBody example mesh",
        f"mtllib {mtl_name}",
    ]
    for x, y, z in all_vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    current_mat: str | None = None
    for group_faces, category in face_groups:
        if category != current_mat:
            lines.append(f"usemtl {category}")
            current_mat = category
        for face in group_faces:
            lines.append("f " + " ".join(str(index) for index in face))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "Built FlyBody combined mesh with %d vertices, %d face groups in %.1fms",
        len(all_vertices),
        len(face_groups),
        (time.perf_counter() - started) * 1000.0,
    )
    return out_path.name


def _maybe_build_flybody_mesh_parts(out: Path) -> list[dict[str, str]] | None:
    repo_root = resolve_local_flybody_repo()
    if repo_root is None:
        return None
    started = time.perf_counter()
    xml_path = repo_root / "flybody" / "fruitfly" / "assets" / "fruitfly.xml"
    asset_dir = xml_path.parent
    mesh_files, _mesh_scale, worldbody, _xml_mats = _load_flybody_asset_index(
        str(xml_path),
        int(xml_path.stat().st_mtime_ns),
    )
    bodies = worldbody.findall(".//body") if worldbody is not None else []

    selected_terms = (
        "thorax",
        "head",
        "antenna",
        "wing_left",
        "wing_right",
        "abdomen",
        "coxa_",
        "femur_",
        "tibia_",
        "tarsus_",
        "tarsal_claw_",
    )
    parts: list[dict[str, str]] = []
    rebuilt_parts = 0
    for body in bodies:
        body_name = str(body.attrib.get("name") or "").strip()
        if not body_name:
            continue
        part_vertices: list[tuple[float, float, float]] = []
        part_faces: list[list[int]] = []
        for geom in body.findall("geom"):
            mesh_name = str(geom.attrib.get("mesh") or "").strip()
            if not mesh_name or not any(term in mesh_name for term in selected_terms):
                continue
            mesh_file = mesh_files.get(mesh_name)
            if not mesh_file:
                continue
            obj_path = asset_dir / mesh_file
            if not obj_path.exists():
                continue
            pos_values = tuple(
                float(part)
                for part in str(geom.attrib.get("pos") or "0 0 0").split()[:3]
            )
            quat_parts = [
                float(part)
                for part in str(geom.attrib.get("quat") or "1 0 0 0").split()[:4]
            ]
            while len(quat_parts) < 4:
                quat_parts.append(0.0)
            quat_values = (quat_parts[0], quat_parts[1], quat_parts[2], quat_parts[3])
            vertices, faces = _parse_obj_vertices_and_faces(obj_path)
            if not vertices or not faces:
                continue
            offset = len(part_vertices)
            part_vertices.extend(
                _transform_vertex(vertex, pos_values, quat_values)
                for vertex in vertices
            )
            for face in faces:
                part_faces.append([offset + index for index in face])
        if not part_vertices or not part_faces:
            continue
        safe_body = body_name.replace("/", "_")
        out_path = out / f"flybody_part_{safe_body}.obj"
        if (
            not out_path.exists()
            or out_path.stat().st_mtime_ns < xml_path.stat().st_mtime_ns
        ):
            lines = [f"# Annolid FlyBody body part: {body_name}"]
            for x, y, z in part_vertices:
                lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            for face in part_faces:
                lines.append("f " + " ".join(str(index) for index in face))
            out_path.write_text("\n".join(lines), encoding="utf-8")
            rebuilt_parts += 1
        part = {"body": body_name, "path": out_path.name}
        part.update(_flybody_part_style(body_name))
        parts.append(part)
    if parts:
        if rebuilt_parts:
            logger.info(
                "Prepared %d FlyBody mesh parts (%d rebuilt) in %.1fms",
                len(parts),
                rebuilt_parts,
                (time.perf_counter() - started) * 1000.0,
            )
        else:
            logger.info("Reusing cached FlyBody mesh parts (%d parts)", len(parts))
    return parts or None


def _write_helix_points_csv(path: Path) -> None:
    lines = ["x,y,z"]
    turns = 8
    samples = 2400
    for i in range(samples):
        t = (i / max(1, samples - 1)) * turns * math.tau
        r = 0.8 + 0.22 * math.cos(6.0 * t)
        x = r * math.cos(t)
        y = r * math.sin(t)
        z = -1.2 + (2.4 * i / max(1, samples - 1))
        lines.append(f"{x:.6f},{y:.6f},{z:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_wave_surface_obj(path: Path) -> None:
    n = 72
    xs = [(-1.8 + (3.6 * i / max(1, n - 1))) for i in range(n)]
    ys = [(-1.8 + (3.6 * j / max(1, n - 1))) for j in range(n)]
    lines: list[str] = ["# Annolid Three.js example: wave surface"]

    for y in ys:
        for x in xs:
            r = math.sqrt(x * x + y * y)
            z = 0.28 * math.sin(5.0 * r) * math.exp(-0.18 * r * r)
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    def vid(i: int, j: int) -> int:
        return j * n + i + 1

    for j in range(n - 1):
        for i in range(n - 1):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i + 1, j + 1)
            d = vid(i, j + 1)
            lines.append(f"f {a} {b} {c}")
            lines.append(f"f {a} {c} {d}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sphere_points_ply(path: Path) -> None:
    rings = 80
    segments = 140
    points: list[tuple[float, float, float]] = []
    for r in range(rings):
        v = r / max(1, rings - 1)
        phi = (v - 0.5) * math.pi
        radius = math.cos(phi)
        z = math.sin(phi)
        for s in range(segments):
            u = s / max(1, segments - 1)
            theta = u * math.tau
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            jitter = 0.015 * math.sin(10.0 * theta) * math.cos(8.0 * phi)
            points.append((x * (1.0 + jitter), y * (1.0 + jitter), z))

    header = [
        "ply",
        "format ascii 1.0",
        "comment Annolid Three.js example: sphere point cloud",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    body = [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]
    path.write_text("\n".join(header + body), encoding="utf-8")
