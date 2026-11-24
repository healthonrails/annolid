from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from annolid.colab.auth import ColabAuthRequiredError

from .models import (
    FLY_BEHAVIORS,
    MOUSE_ARENAS,
    MOUSE_COAT_PAIRS,
    MOUSE_HEADSTAGES,
    MOUSE_LIGHTING_PRESETS,
    MOUSE_OCCLUDERS,
    MOUSE_SCENARIOS,
    SimDatasetConfig,
)
from .service import Sim2RealWorkflowService


def _print_json_or_text(payload: object, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                print(f"{k}: {json.dumps(v, ensure_ascii=False)}")
            else:
                print(f"{k}: {v}")
    else:
        print(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annolid-run sim2real",
        description=(
            "Match real example videos to Annolid's 3D Mouse or FlyBody simulation, "
            "generate synthetic training datasets, train/fine-tune on Google Colab (or locally), "
            "evaluate on example videos, iterate to improve, and generate reports."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_an = sub.add_parser(
        "analyze",
        help="Analyze example video(s) and infer matching 3D simulation environment parameters.",
    )
    p_an.add_argument(
        "--video",
        action="append",
        required=True,
        help="Example video path or image folder (repeat --video for multiple examples).",
    )
    p_an.add_argument(
        "--project",
        required=True,
        help="Project directory to store environment profile, datasets, iterations, and reports.",
    )
    p_an.add_argument(
        "--species",
        choices=("auto", "mouse", "fly"),
        default="auto",
        help="Target simulation species (auto infers from video cues).",
    )
    p_an.add_argument(
        "--task",
        choices=("yolo_seg", "yolo_pose", "yolo_detect"),
        default="yolo_seg",
    )
    p_an.add_argument(
        "--export-level",
        choices=("instances", "parts"),
        default="instances",
    )
    p_an.add_argument("--frames-per-scenario", type=int, default=24)

    # generate
    p_gen = sub.add_parser(
        "generate",
        help="Generate an environment-matched Mouse 3D or FlyBody synthetic dataset.",
    )
    p_gen.add_argument(
        "--project", required=True, help="Sim-to-Real project directory."
    )
    p_gen.add_argument(
        "--output", default=None, help="Optional custom dataset output dir."
    )
    p_gen.add_argument("--species", choices=("mouse", "fly"), default=None)
    p_gen.add_argument(
        "--task", choices=("yolo_seg", "yolo_pose", "yolo_detect"), default=None
    )
    p_gen.add_argument("--export-level", choices=("instances", "parts"), default=None)
    p_gen.add_argument("--arena", choices=MOUSE_ARENAS, default=None)
    p_gen.add_argument("--camera", default=None, help="Camera view override.")
    p_gen.add_argument("--lighting", choices=MOUSE_LIGHTING_PRESETS, default=None)
    p_gen.add_argument("--coat-pair", choices=MOUSE_COAT_PAIRS, default=None)
    p_gen.add_argument("--occluder", choices=MOUSE_OCCLUDERS, default=None)
    p_gen.add_argument("--headstage", choices=MOUSE_HEADSTAGES, default=None)
    p_gen.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=f"Mouse scenarios ({', '.join(MOUSE_SCENARIOS)}) or FlyBody behaviors ({', '.join(FLY_BEHAVIORS)}).",
    )
    p_gen.add_argument("--frames-per-scenario", type=int, default=None)
    p_gen.add_argument("--width", type=int, default=None)
    p_gen.add_argument("--height", type=int, default=None)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--photometry-strength", type=float, default=None)
    p_gen.add_argument("--domain-randomization", type=float, default=None)
    p_gen.add_argument("--composite-bg", action="store_true", default=None)
    p_gen.add_argument("--real-labeled-dir", default=None)
    p_gen.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow explicit procedural fallback if Playwright/FlyBody runtime is unavailable.",
    )

    # iterate
    p_it = sub.add_parser(
        "iterate",
        help="Run closed-loop Generate -> Train/Fine-Tune (Colab/local) -> Evaluate -> Refine -> Report.",
    )
    p_it.add_argument("--project", required=True, help="Sim-to-Real project directory.")
    p_it.add_argument(
        "--iterations", type=int, default=2, help="Number of iterations to run."
    )
    p_it.add_argument(
        "--train-backend",
        choices=("colab", "local"),
        default="colab",
        help="Where to run training & evaluation inference.",
    )
    p_it.add_argument(
        "--colab-backend",
        choices=("auto", "cli", "api", "notebook"),
        default="auto",
    )
    p_it.add_argument("--gpu", "--accelerator", dest="accelerator", default="T4")
    p_it.add_argument("--high-mem", action="store_true")
    p_it.add_argument("--session", default=None, help="Reuse or name a Colab session.")
    p_it.add_argument("--keep-runtime", action="store_true")
    p_it.add_argument("--annolid-spec", default="annolid")
    p_it.add_argument("--model", default="yolo")
    p_it.add_argument(
        "--weights", default=None, help="Initial checkpoint to fine-tune from."
    )
    p_it.add_argument("--epochs", type=int, default=20)
    p_it.add_argument("--imgsz", type=int, default=640)
    p_it.add_argument("--batch", type=int, default=None)
    p_it.add_argument("--conf", type=float, default=0.25)
    p_it.add_argument("--eval-max-frames", type=int, default=24)
    p_it.add_argument("--gt-labelme-dir", default=None)
    p_it.add_argument("--target-score", type=float, default=0.88)
    p_it.add_argument(
        "--no-auto-adjust",
        action="store_true",
        help="Do not automatically apply advisor adjustments between iterations.",
    )
    p_it.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow explicit procedural fallback renderer if Playwright/FlyBody is unavailable.",
    )

    # evaluate
    p_ev = sub.add_parser(
        "evaluate",
        help="Evaluate a trained checkpoint directly on the project's example videos.",
    )
    p_ev.add_argument("--project", required=True)
    p_ev.add_argument(
        "--weights", required=True, help="Path to trained .pt checkpoint."
    )
    p_ev.add_argument("--train-backend", choices=("colab", "local"), default="colab")
    p_ev.add_argument(
        "--colab-backend", choices=("auto", "cli", "api", "notebook"), default="auto"
    )
    p_ev.add_argument("--gpu", "--accelerator", dest="accelerator", default="T4")
    p_ev.add_argument("--session", default=None)
    p_ev.add_argument("--keep-runtime", action="store_true")
    p_ev.add_argument("--conf", type=float, default=0.25)
    p_ev.add_argument("--eval-max-frames", type=int, default=24)
    p_ev.add_argument("--gt-labelme-dir", default=None)

    # report
    p_rep = sub.add_parser(
        "report",
        help="Write Markdown, JSON, and HTML reports with visual contact sheets.",
    )
    p_rep.add_argument("--project", required=True)
    p_rep.add_argument("--output", default=None)

    # status
    p_st = sub.add_parser(
        "status",
        help="Inspect project state, environment profile, iterations, and best checkpoint.",
    )
    p_st.add_argument("--project", required=True)

    return parser


def _apply_generate_overrides(
    cfg: SimDatasetConfig, args: argparse.Namespace
) -> SimDatasetConfig:
    if getattr(args, "species", None):
        cfg.species = str(args.species)
    if getattr(args, "task", None):
        cfg.task = str(args.task)
    if getattr(args, "export_level", None):
        cfg.export_level = str(args.export_level)
    if getattr(args, "arena", None):
        cfg.arena = str(args.arena)
    if getattr(args, "camera", None):
        cfg.camera_view = str(args.camera)
    if getattr(args, "lighting", None):
        cfg.lighting = str(args.lighting)
    if getattr(args, "coat_pair", None):
        cfg.coat_pair = str(args.coat_pair)
    if getattr(args, "occluder", None):
        cfg.occluder = str(args.occluder)
    if getattr(args, "headstage", None):
        cfg.headstage = str(args.headstage)
    if getattr(args, "scenarios", None):
        cfg.scenarios = list(args.scenarios)
    if getattr(args, "frames_per_scenario", None) is not None:
        cfg.frames_per_scenario = int(args.frames_per_scenario)
    if getattr(args, "width", None) is not None:
        cfg.width = int(args.width)
    if getattr(args, "height", None) is not None:
        cfg.height = int(args.height)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "photometry_strength", None) is not None:
        cfg.photometry_strength = float(args.photometry_strength)
    if getattr(args, "domain_randomization", None) is not None:
        cfg.domain_randomization = float(args.domain_randomization)
    if getattr(args, "composite_bg", None):
        cfg.composite_on_real_background = True
    if getattr(args, "real_labeled_dir", None):
        cfg.real_labeled_dir = str(args.real_labeled_dir)
    if getattr(args, "allow_fallback", False):
        cfg.allow_fallback = True
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    as_json = bool(args.json)
    log_cb = (lambda m: print(m, file=sys.stderr)) if as_json else print

    try:
        svc = Sim2RealWorkflowService(Path(args.project))

        if args.command == "analyze":
            profile, sim_cfg = svc.analyze_videos(
                args.video,
                species_hint=args.species,
                task=args.task,
                export_level=args.export_level,
                frames_per_scenario=args.frames_per_scenario,
                log_callback=log_cb,
            )
            _print_json_or_text(
                {
                    "project_dir": str(svc.project_dir),
                    "profile": profile.to_dict(),
                    "sim_config": sim_cfg.to_dict(),
                },
                as_json=as_json,
            )
            return 0

        if args.command == "generate":
            state = svc.load_state()
            base_cfg = (
                SimDatasetConfig.from_dict(state.current_sim_config)
                if isinstance(state.current_sim_config, dict)
                else SimDatasetConfig()
            )
            cfg = _apply_generate_overrides(base_cfg, args)
            artifact = svc.generate_dataset(
                cfg,
                output_dir=args.output,
                log_callback=log_cb,
            )
            _print_json_or_text(artifact.to_dict(), as_json=as_json)
            return 0

        if args.command == "iterate":
            state = svc.load_state()
            cfg = (
                SimDatasetConfig.from_dict(state.current_sim_config)
                if isinstance(state.current_sim_config, dict)
                else SimDatasetConfig()
            )
            if args.allow_fallback:
                cfg.allow_fallback = True
            records = svc.run_loop(
                iterations=args.iterations,
                auto_apply_recommendations=not bool(args.no_auto_adjust),
                sim_config=cfg,
                train_backend=args.train_backend,
                colab_backend=args.colab_backend,
                accelerator=args.accelerator,
                high_mem=args.high_mem,
                session_name=args.session,
                keep_runtime=args.keep_runtime,
                annolid_install_spec=args.annolid_spec,
                model=args.model,
                weights=args.weights,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                conf=args.conf,
                eval_max_frames=args.eval_max_frames,
                ground_truth_labelme_dir=args.gt_labelme_dir,
                target_score=args.target_score,
                log_callback=log_cb,
            )
            final_state = svc.load_state()
            _print_json_or_text(
                {
                    "project_dir": str(svc.project_dir),
                    "completed_iterations": [r.to_dict() for r in records],
                    "best_iteration": final_state.best_iteration,
                    "best_score": final_state.best_score,
                    "best_weights_path": final_state.best_weights_path,
                    "report_paths": final_state.report_paths,
                },
                as_json=as_json,
            )
            return 0

        if args.command == "evaluate":
            ev = svc.evaluate_existing_weights(
                weights_path=args.weights,
                train_backend=args.train_backend,
                colab_backend=args.colab_backend,
                accelerator=args.accelerator,
                session_name=args.session,
                keep_runtime=args.keep_runtime,
                conf=args.conf,
                eval_max_frames=args.eval_max_frames,
                ground_truth_labelme_dir=args.gt_labelme_dir,
                log_callback=log_cb,
            )
            _print_json_or_text(ev.to_dict(), as_json=as_json)
            return 0

        if args.command == "report":
            paths = svc.write_report(output_dir=args.output)
            _print_json_or_text(paths, as_json=as_json)
            return 0

        if args.command == "status":
            state = svc.load_state()
            _print_json_or_text(state.to_dict(), as_json=as_json)
            return 0

        parser.error(f"Unknown sim2real command: {args.command}")
        return 2
    except ColabAuthRequiredError as exc:
        print(
            f"Google Colab authentication required: {exc}\n"
            "Sign in with your personal Google account via:\n"
            "  annolid-run colab auth login --method personal",
            file=sys.stderr,
        )
        return 3
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
