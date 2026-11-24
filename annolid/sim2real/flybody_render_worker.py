"""Standalone FlyBody MuJoCo RGB + segmentation + keypoint dataset render worker.

Can be imported directly when `flybody`, `dm_control`, and `mujoco` are installed in
the active interpreter, or invoked as a module in a dedicated FlyBody Python
environment (for example `.venv311/bin/python -m annolid.sim2real.flybody_render_worker`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

FLY_PART_ID_TO_NAME: dict[int, str] = {
    1: "head",
    2: "thorax",
    3: "abdomen",
    4: "wings",
    5: "legs",
}

FLY_KEYPOINT_BODY_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("head", ("walker/head", "walker/rostrum")),
    ("thorax", ("walker/thorax",)),
    ("abdomen", ("walker/abdomen_4", "walker/abdomen_3", "walker/abdomen")),
    ("wing_left", ("walker/wing_left",)),
    ("wing_right", ("walker/wing_right",)),
    (
        "leg_front_left",
        ("walker/tibia_T1_left", "walker/tarsus_T1_left", "walker/femur_T1_left"),
    ),
    (
        "leg_front_right",
        ("walker/tibia_T1_right", "walker/tarsus_T1_right", "walker/femur_T1_right"),
    ),
    (
        "leg_mid_left",
        ("walker/tibia_T2_left", "walker/tarsus_T2_left", "walker/femur_T2_left"),
    ),
    (
        "leg_mid_right",
        ("walker/tibia_T2_right", "walker/tarsus_T2_right", "walker/femur_T2_right"),
    ),
    (
        "leg_hind_left",
        ("walker/tibia_T3_left", "walker/tarsus_T3_left", "walker/femur_T3_left"),
    ),
    (
        "leg_hind_right",
        ("walker/tibia_T3_right", "walker/tarsus_T3_right", "walker/femur_T3_right"),
    ),
)


def _classify_fly_body_part(body_name: str) -> int:
    """Map a MuJoCo walker/* body name to a coarse FlyBody part ID (1..5)."""
    raw = str(body_name or "").lower()
    if not raw.startswith("walker/"):
        return 0
    part = raw.split("/", 1)[-1]
    if any(
        k in part
        for k in (
            "head",
            "rostrum",
            "haustellum",
            "labrum",
            "antenna",
            "eye",
            "proboscis",
        )
    ):
        return 1
    if "wing" in part:
        return 4
    if "abdomen" in part:
        return 3
    if any(
        k in part for k in ("coxa", "femur", "tibia", "tarsus", "claw", "trochanter")
    ):
        return 5
    return 2  # thorax / haltere / default core body


def _resolve_camera_id(physics: Any, preferred_camera: str) -> tuple[int, str]:
    ncam = int(physics.model.ncam)
    cam_names = [
        str(physics.model.id2name(i, "camera") or f"cam_{i}") for i in range(ncam)
    ]
    if preferred_camera in cam_names:
        idx = cam_names.index(preferred_camera)
        if cam_names[idx] == "top_camera":
            # FlyBody's default top_camera spans the full arena so the 3mm fly is <10px;
            # tighten its FOV to frame the fly cleanly for segmentation & keypoint training.
            try:
                physics.model.cam_fovy[idx] = min(
                    float(physics.model.cam_fovy[idx]), 4.5
                )
            except Exception:
                pass
        return idx, cam_names[idx]
    for fallback in ("walker/track1", "walker/track2", "walker/hero", "walker/side"):
        if fallback in cam_names:
            idx = cam_names.index(fallback)
            return idx, cam_names[idx]
    return 0, cam_names[0] if cam_names else "camera_0"


def _project_world_point_to_camera(
    point_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot_3x3: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
) -> tuple[float, float, int]:
    """Project a 3D world point into 2D MuJoCo OpenGL camera pixel coordinates.

    In MuJoCo, camera +X is right, +Y is up, and -Z is the viewing direction.
    """
    rel = np.asarray(point_world, dtype=np.float64) - np.asarray(
        cam_pos, dtype=np.float64
    )
    cam_local = cam_rot_3x3.T @ rel
    depth = -float(cam_local[2])
    if depth <= 1e-6:
        return 0.0, 0.0, 0
    f = 0.5 * float(height) / math.tan(math.radians(float(fovy_deg)) * 0.5)
    u = float(width) * 0.5 + f * (float(cam_local[0]) / depth)
    v = float(height) * 0.5 - f * (float(cam_local[1]) / depth)
    in_bounds = 0.0 <= u < float(width) and 0.0 <= v < float(height)
    return float(u), float(v), (2 if in_bounds else 0)


def _extract_body_world_pos(
    physics: Any, candidates: Sequence[str]
) -> np.ndarray | None:
    for name in candidates:
        try:
            body_id = physics.model.name2id(name, "body")
            if body_id >= 0:
                return np.asarray(physics.data.xpos[body_id], dtype=np.float64).copy()
        except Exception:
            continue
    return None


def render_flybody_scenarios(
    output_dir: Path,
    *,
    behaviors: Sequence[str],
    frames_per_behavior: int = 20,
    width: int = 640,
    height: int = 480,
    camera_view: str = "top_camera",
    seed: int = 42,
    brain_dynamics: str = "spiking",
) -> dict[str, Any]:
    """Run real FlyBody MuJoCo rollouts and save RGB, part-ID masks, and 2D keypoints."""
    from annolid.simulation.fly_brain import FlyBrainController
    from annolid.simulation.flybody_live import (
        _build_behavior_env,
        _extend_inference_trajectory_if_needed,
    )

    out_root = Path(output_dir).expanduser().resolve()
    raw_img_dir = out_root / "raw_images"
    raw_mask_dir = out_root / "raw_masks"
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    raw_mask_dir.mkdir(parents=True, exist_ok=True)

    frames_meta: list[dict[str, Any]] = []
    used_cameras: list[str] = []
    rng = np.random.default_rng(int(seed))

    for b_idx, behavior in enumerate(behaviors):
        env = _build_behavior_env(behavior)
        try:
            steps_per_frame = 4
            min_steps = int(frames_per_behavior) * steps_per_frame + 20
            _extend_inference_trajectory_if_needed(
                env, behavior=behavior, min_control_steps=min_steps
            )
            ts = env.reset()
            physics = env.physics
            cam_id, cam_name = _resolve_camera_id(physics, camera_view)
            if cam_name not in used_cameras:
                used_cameras.append(cam_name)

            # Build geom_id -> coarse_part_id lookup table once per env.
            ngeom = int(physics.model.ngeom)
            geom_to_part = np.zeros(ngeom + 1, dtype=np.uint8)
            for gid in range(ngeom):
                bid = int(physics.model.geom_bodyid[gid])
                bname = str(physics.model.id2name(bid, "body") or "")
                geom_to_part[gid] = _classify_fly_body_part(bname)

            action_spec = env.action_spec()
            dt_fn = getattr(env, "control_timestep", None)
            dt = float(dt_fn()) if callable(dt_fn) else 0.002
            dyn = str(brain_dynamics or "spiking").strip().lower()
            if dyn not in ("spiking", "rate", "canonical"):
                dyn = "spiking"
            controller = FlyBrainController(
                action_spec,
                behavior=behavior,
                dynamics="rate" if dyn == "canonical" else dyn,
                seed=int(seed) + b_idx * 101,
                control_timestep=dt,
            )

            # Warmup steps so the fly enters natural posture.
            for _ in range(6):
                act = controller.step(getattr(ts, "observation", None), dt=dt)
                ts = env.step(act)

            for f_idx in range(int(frames_per_behavior)):
                for _ in range(steps_per_frame):
                    act = controller.step(getattr(ts, "observation", None), dt=dt)
                    # Add slight seeded exploratory jitter for posture diversity.
                    if rng.random() < 0.35:
                        jitter = rng.normal(0.0, 0.04, size=act.shape).astype(act.dtype)
                        act = np.clip(
                            act + jitter,
                            action_spec.minimum,
                            action_spec.maximum,
                        ).astype(action_spec.dtype)
                    ts = env.step(act)

                rgb = physics.render(
                    height=int(height), width=int(width), camera_id=cam_id
                )
                seg = physics.render(
                    height=int(height),
                    width=int(width),
                    camera_id=cam_id,
                    segmentation=True,
                )
                geom_map = seg[:, :, 0]
                valid_geom = (geom_map >= 0) & (geom_map < ngeom)
                coarse_map = np.zeros((int(height), int(width)), dtype=np.uint8)
                coarse_map[valid_geom] = geom_to_part[geom_map[valid_geom]]
                fg_mask = (coarse_map > 0).astype(np.uint8)

                cam_pos = np.asarray(physics.data.cam_xpos[cam_id], dtype=np.float64)
                cam_rot = np.asarray(
                    physics.data.cam_xmat[cam_id], dtype=np.float64
                ).reshape(3, 3)
                fovy = float(physics.model.cam_fovy[cam_id])

                keypoints: list[dict[str, Any]] = []
                for kp_name, candidates in FLY_KEYPOINT_BODY_CANDIDATES:
                    pos3d = _extract_body_world_pos(physics, candidates)
                    if pos3d is None:
                        keypoints.append({"name": kp_name, "x": 0.0, "y": 0.0, "v": 0})
                        continue
                    u, v, vis = _project_world_point_to_camera(
                        pos3d, cam_pos, cam_rot, fovy, int(width), int(height)
                    )
                    keypoints.append(
                        {
                            "name": kp_name,
                            "x": round(u, 2),
                            "y": round(v, 2),
                            "v": int(vis),
                        }
                    )

                stem = f"{behavior}_f{f_idx:04d}"
                img_path = raw_img_dir / f"{stem}.png"
                mask_path = raw_mask_dir / f"{stem}_mask.png"
                Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
                    img_path
                )
                # Pack [coarse_part_id, fg_instance_id, 0] into an RGB PNG
                # (when cv2 reads it back as BGR, channel 2 is coarse_part_id,
                # so store [0, fg_mask, coarse_map] in RGB -> cv2 BGR channels [0]=coarse_map, [1]=fg_mask, [2]=0).
                packed_rgb = np.stack(
                    [np.zeros_like(coarse_map), fg_mask, coarse_map], axis=-1
                )
                Image.fromarray(packed_rgb, mode="RGB").save(mask_path)

                frames_meta.append(
                    {
                        "stem": stem,
                        "behavior": behavior,
                        "frame_index": f_idx,
                        "camera": cam_name,
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "foreground_pixels": int(np.count_nonzero(fg_mask)),
                        "keypoints": keypoints,
                    }
                )
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    manifest = {
        "renderer": "flybody_mujoco",
        "width": int(width),
        "height": int(height),
        "cameras": used_cameras,
        "part_id_to_name": {str(k): v for k, v in FLY_PART_ID_TO_NAME.items()},
        "frames": frames_meta,
    }
    manifest_path = out_root / "flybody_raw_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render FlyBody MuJoCo RGB, segmentation masks, and 2D keypoints."
    )
    parser.add_argument("--output", required=True, help="Output work directory")
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=["walk_imitation", "forage_walk"],
        help="FlyBody behaviors to roll out",
    )
    parser.add_argument("--frames-per-behavior", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", default="walker/track1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brain-dynamics", default="spiking")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = render_flybody_scenarios(
        Path(args.output),
        behaviors=list(args.behaviors),
        frames_per_behavior=int(args.frames_per_behavior),
        width=int(args.width),
        height=int(args.height),
        camera_view=str(args.camera),
        seed=int(args.seed),
        brain_dynamics=str(args.brain_dynamics),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "renderer": manifest["renderer"],
                "frames": len(manifest["frames"]),
                "manifest": str(
                    Path(args.output).resolve() / "flybody_raw_manifest.json"
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
