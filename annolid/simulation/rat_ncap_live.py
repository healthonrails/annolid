from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from annolid.simulation.rat_ncap import (
    RAT_NCAP_METADATA,
    RAT_V2_ACTUATOR_NAMES,
    RAT_V2_COAT_PATTERNS,
    RAT_V2_KEYPOINT_NAMES,
    RAT_V2_LOCOMOTION_ACTUATOR_NAMES,
    RAT_V2_SKELETON_EDGES,
    SUPPORTED_RAT_ABLATIONS,
    SUPPORTED_RAT_BEHAVIORS,
    SUPPORTED_RAT_INTERFACES,
    run_rat_ncap_rollout,
)

LIVE_PAYLOAD_VERSION = 1


def probe_rat_ncap_runtime() -> dict[str, Any]:
    """Probe built-in Rat NCAP controller plus optional MuJoCo & external rat-ncap repo."""
    has_mujoco = importlib.util.find_spec("mujoco") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    has_jax = importlib.util.find_spec("jax") is not None
    rat_xml_path: str | None = None
    repo_root: str | None = None
    try:
        from annolid.gui.rat_ncap_support import (
            resolve_local_rat_ncap_repo,
            resolve_rat_xml_path,
        )

        repo = resolve_local_rat_ncap_repo()
        repo_root = str(repo) if repo is not None else None
        xml = resolve_rat_xml_path()
        rat_xml_path = str(xml) if xml is not None else None
    except Exception:
        pass

    return {
        "python": sys.executable,
        "ready": True,
        "builtin_ready": True,
        "mujoco_available": bool(has_mujoco),
        "torch_available": bool(has_torch),
        "jax_available": bool(has_jax),
        "repo_root": repo_root,
        "rat_xml_path": rat_xml_path,
        "mujoco_rat_ready": bool(has_mujoco and rat_xml_path),
        "model_id": RAT_NCAP_METADATA["id"],
        "dof_total": len(RAT_V2_ACTUATOR_NAMES),
        "dof_locomotion": len(RAT_V2_LOCOMOTION_ACTUATOR_NAMES),
        "supported_behaviors": list(SUPPORTED_RAT_BEHAVIORS),
        "supported_interfaces": list(SUPPORTED_RAT_INTERFACES),
        "supported_coats": list(RAT_V2_COAT_PATTERNS),
        "supported_ablations": list(SUPPORTED_RAT_ABLATIONS),
    }


def build_live_rat_ncap_payload(
    *,
    steps: int = 180,
    behavior: str = "trot",
    target_speed: float | None = None,
    interface: str = "flexor_extensor",
    coat: str = "black_hooded",
    ablation: str = "none",
    ablation_scale: float = 0.0,
    rg_noise_std: float = 0.0,
    terrain_bump_cm: float = 0.0,
    dt: float = 0.02,
    seed: int = 7,
) -> dict[str, Any]:
    """Generate an `annolid-simulation-v1` payload for the 3D viewer."""
    rollout = run_rat_ncap_rollout(
        steps=steps,
        behavior=behavior,
        target_speed=target_speed,
        interface=interface,
        coat=coat,
        ablation=ablation,
        ablation_scale=ablation_scale,
        rg_noise_std=rg_noise_std,
        terrain_bump_cm=terrain_bump_cm,
        dt=dt,
        seed=seed,
    )
    meta = rollout["metadata"]
    viewer_frames: list[dict[str, Any]] = []
    # Convert MuJoCo Z-up meters -> Three.js Y-up scaled viewer coordinates (10x for visibility).
    scale = 10.0
    for idx, frame in enumerate(rollout["frames"]):
        pts = []
        for label in RAT_V2_KEYPOINT_NAMES:
            coords = frame["keypoints_3d"].get(label)
            if not coords or len(coords) < 3:
                continue
            pts.append(
                {
                    "label": label,
                    "x": round(float(coords[0]) * scale, 5),
                    "y": round(float(coords[2]) * scale, 5),
                    "z": round(-float(coords[1]) * scale, 5),
                }
            )
        viewer_frames.append(
            {
                "frame_index": idx,
                "timestamp_sec": frame["time_sec"],
                "points": pts,
                "qpos": frame["qpos"],
                "body_poses": frame["body_poses"],
                "diagnostics": {
                    "gait": frame["gait"],
                    "target_speed_mps": frame["target_speed_mps"],
                    "forward_speed_mps": frame["forward_speed_mps"],
                    "step_frequency_hz": frame["step_frequency_hz"],
                    "step_length_m": frame["step_length_m"],
                    "duty_factor": frame["duty_factor"],
                    "phase_offsets_rel_fr": frame["phase_offsets_rel_fr"],
                    "foot_contacts": frame["foot_contacts"],
                    "mechanical_power": frame["mechanical_power"],
                    "rg_flx": frame["rg"]["flx"],
                    "rg_ext": frame["rg"]["ext"],
                },
                "dry_run": False,
            }
        )

    return {
        "kind": "annolid-simulation-v1",
        "title": f"Rat NCAP · {meta['behavior']} ({meta['interface']})",
        "adapter": "rat_ncap",
        "labels": list(RAT_V2_KEYPOINT_NAMES),
        "edges": [list(edge) for edge in RAT_V2_SKELETON_EDGES],
        "metadata": {
            "payload_version": LIVE_PAYLOAD_VERSION,
            "rat_ncap": meta,
            "coordinate_system": {"units": "decimeters", "up_axis": "Y"},
            "live_controls": {
                "kind": "rat_ncap",
                "behavior": meta["behavior"],
                "interface": meta["interface"],
                "coat": meta["coat"],
                "ablation": meta["ablation"],
                "ablation_scale": meta["ablation_scale"],
                "rg_noise_std": meta["rg_noise_std"],
                "terrain_bump_cm": meta["terrain_bump_cm"],
                "target_speed": target_speed,
                "steps": int(steps),
                "seed": int(seed),
            },
        },
        "environment": {
            "floor": {
                "type": "plane",
                "size": [12.0, 12.0],
                "position": [0.0, 0.0, 0.0],
            }
        },
        "display": {
            "show_points": True,
            "show_labels": True,
            "show_edges": True,
            "show_trails": True,
        },
        "frames": viewer_frames,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe or run live Rat NCAP 3D simulation rollouts."
    )
    parser.add_argument(
        "--probe", action="store_true", help="Probe runtime readiness and exit."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit probe result as JSON."
    )
    parser.add_argument(
        "--out", default="", help="Output JSON path for live viewer payload."
    )
    parser.add_argument(
        "--steps", type=int, default=180, help="Number of simulation steps."
    )
    parser.add_argument("--behavior", default="trot", choices=SUPPORTED_RAT_BEHAVIORS)
    parser.add_argument(
        "--target-speed", type=float, default=None, help="Target speed in m/s [0, 1]."
    )
    parser.add_argument(
        "--interface", default="flexor_extensor", choices=SUPPORTED_RAT_INTERFACES
    )
    parser.add_argument("--coat", default="black_hooded", choices=RAT_V2_COAT_PATTERNS)
    parser.add_argument("--ablation", default="none", choices=SUPPORTED_RAT_ABLATIONS)
    parser.add_argument("--ablation-scale", type=float, default=0.0)
    parser.add_argument("--rg-noise-std", type=float, default=0.0)
    parser.add_argument("--terrain-bump-cm", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.probe:
        probe = probe_rat_ncap_runtime()
        if args.json:
            print(json.dumps(probe))
        else:
            print(
                f"Rat NCAP ready: {probe['ready']} (mujoco={probe['mujoco_available']})"
            )
        return 0

    payload = build_live_rat_ncap_payload(
        steps=int(args.steps),
        behavior=str(args.behavior),
        target_speed=args.target_speed,
        interface=str(args.interface),
        coat=str(args.coat),
        ablation=str(args.ablation),
        ablation_scale=float(args.ablation_scale),
        rg_noise_std=float(args.rg_noise_std),
        terrain_bump_cm=float(args.terrain_bump_cm),
        dt=float(args.dt),
        seed=int(args.seed),
    )
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
    else:
        print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
