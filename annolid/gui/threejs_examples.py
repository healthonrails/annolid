from __future__ import annotations

import math
from pathlib import Path

THREEJS_EXAMPLE_IDS = (
    "helix_points_csv",
    "wave_surface_obj",
    "sphere_points_ply",
    "brain_viewer_html",
)


def generate_threejs_example(example_id: str, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if example_id == "brain_viewer_html":
        # This is a standalone HTML example, return the asset path
        asset_path = (
            Path(__file__).resolve().parent / "assets" / "threejs" / "points_3d.html"
        )
        return asset_path

    if example_id == "helix_points_csv":
        path = out / "helix_points.csv"
        _write_helix_points_csv(path)
        return path
    if example_id == "wave_surface_obj":
        path = out / "wave_surface.obj"
        _write_wave_surface_obj(path)
        return path
    if example_id == "sphere_points_ply":
        path = out / "sphere_points.ply"
        _write_sphere_points_ply(path)
        return path
    raise ValueError(f"Unknown Three.js example: {example_id}")


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
