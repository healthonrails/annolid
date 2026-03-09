from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from annolid.simulation.io import read_pose_frames, write_simulation_ndjson
from annolid.simulation.lifting import lift_pose_frames_with_depth, load_depth_records
from annolid.simulation.mapping import load_simulation_mapping
from annolid.simulation.smoothing import smooth_pose_frames


def build_simulation_adapter(name: str):
    label = str(name or "").strip().lower()
    if label == "identity":
        from annolid.simulation.adapters.identity import IdentitySimulationAdapter

        return IdentitySimulationAdapter()
    if label == "flybody":
        from annolid.simulation.adapters.flybody import FlyBodyAdapter

        return FlyBodyAdapter()
    raise ValueError(f"Unsupported simulation backend: {name}")


@dataclass(frozen=True)
class SimulationRunRequest:
    backend: str
    input_path: str
    mapping_path: str
    out_ndjson: str
    pose_schema: str | None = None
    depth_ndjson: str | None = None
    video_name: str | None = None
    default_z: float = 0.0
    dry_run: bool = False
    env_factory: str | None = None
    ik_function: str | None = None
    ik_max_steps: int = 2000
    smooth_mode: str = "none"
    fps: float = 30.0
    max_gap_frames: int = 0
    min_score: float = 0.0
    ema_alpha: float = 0.7


def build_default_output_path(
    input_path: str | Path, *, backend: str, out_dir: str | Path | None = None
) -> Path:
    source = Path(input_path).expanduser()
    target_dir = (
        Path(out_dir).expanduser()
        if out_dir is not None
        else Path(tempfile.gettempdir()) / "annolid_simulation_runs"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{source.stem}.{backend}.ndjson"


def run_simulation_workflow(request: SimulationRunRequest) -> Path:
    mapping = load_simulation_mapping(request.mapping_path)
    pose_frames = read_pose_frames(
        request.input_path,
        pose_schema=request.pose_schema,
        video_name=request.video_name,
    )
    pose_frames = smooth_pose_frames(
        pose_frames,
        mode=request.smooth_mode,
        fps=float(request.fps),
        max_gap_frames=int(request.max_gap_frames),
        min_score=float(request.min_score),
        ema_alpha=float(request.ema_alpha),
    )
    adapter = build_simulation_adapter(request.backend)
    adapter_config = {
        "mapping_path": str(Path(request.mapping_path).expanduser()),
        "keypoint_to_site": dict(mapping.keypoint_to_site),
        "site_to_joint": dict(mapping.site_to_joint),
        "coordinate_system": dict(mapping.coordinate_system),
        "dry_run": bool(request.dry_run),
        "default_z": float(request.default_z),
        "ik_kwargs": {"max_steps": int(request.ik_max_steps)},
    }
    if request.env_factory:
        adapter_config["environment_factory"] = str(request.env_factory)
    if request.ik_function:
        adapter_config["ik_function"] = str(request.ik_function)

    adapter.configure(adapter_config)
    adapter.initialize()
    if request.depth_ndjson:
        depth_records = load_depth_records(request.depth_ndjson)
        pose_frames_3d = lift_pose_frames_with_depth(
            pose_frames,
            depth_records=depth_records,
            coordinate_system=dict(mapping.coordinate_system),
        )
        result = adapter.fit_3d(pose_frames_3d)
    else:
        result = adapter.fit_2d(pose_frames)
    out_path = Path(request.out_ndjson).expanduser()
    write_simulation_ndjson(
        out_path,
        pose_frames=pose_frames,
        result=result,
        adapter_name=adapter.name,
        extra_metadata={
            "backend": request.backend,
            "mapping_path": str(Path(request.mapping_path).expanduser()),
            "depth_ndjson": str(Path(request.depth_ndjson).expanduser())
            if request.depth_ndjson
            else None,
            "coordinate_system": dict(mapping.coordinate_system),
            "backend_metadata": dict(mapping.metadata),
            "smoothing": {
                "mode": str(request.smooth_mode),
                "fps": float(request.fps),
                "max_gap_frames": int(request.max_gap_frames),
                "min_score": float(request.min_score),
                "ema_alpha": float(request.ema_alpha),
            },
        },
    )
    return out_path
