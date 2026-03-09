from __future__ import annotations

import argparse

from annolid.engine.registry import ModelPluginBase, register_model


@register_model
class SimulationRunnerPlugin(ModelPluginBase):
    name = "simulation_runner"
    description = "Run backend-neutral simulation fitting against Annolid LabelMe/NDJSON pose data."
    predict_examples = (
        "annolid-run predict simulation_runner --help-model",
        "annolid-run predict simulation_runner --backend identity --input pose.json --mapping sim.json --out-ndjson sim.ndjson",
        "annolid-run predict simulation_runner --backend flybody --input pose.ndjson --mapping flybody.yaml --out-ndjson flybody.ndjson --dry-run",
    )
    predict_help_sections = (
        (
            "Required inputs",
            (
                "--backend",
                "--input",
                "--mapping",
                "--out-ndjson",
                "--write-mapping-template",
            ),
        ),
        (
            "Schema and labeling",
            (
                "--pose-schema",
                "--depth-ndjson",
                "--video-name",
                "--template-keypoints",
            ),
        ),
        (
            "Backend controls",
            (
                "--default-z",
                "--dry-run",
                "--env-factory",
                "--ik-function",
                "--ik-max-steps",
                "--smooth-mode",
                "--fps",
                "--max-gap-frames",
            ),
        ),
    )

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--backend",
            choices=("identity", "flybody"),
            required=True,
            help="Simulation backend to run.",
        )
        parser.add_argument(
            "--input",
            default=None,
            help="Input LabelMe JSON or Annolid NDJSON containing point keypoints.",
        )
        parser.add_argument(
            "--mapping",
            default=None,
            help="JSON/YAML mapping from Annolid keypoints to simulator sites.",
        )
        parser.add_argument(
            "--out-ndjson",
            default=None,
            help="Output NDJSON path for simulation-enriched Annolid records.",
        )
        parser.add_argument(
            "--pose-schema",
            default=None,
            help="Optional pose schema file for instance-prefixed keypoints.",
        )
        parser.add_argument(
            "--depth-ndjson",
            default=None,
            help="Optional Annolid depth sidecar NDJSON used to lift 2D keypoints into 3D.",
        )
        parser.add_argument(
            "--video-name",
            default=None,
            help="Optional video name override when the input file is LabelMe JSON.",
        )
        parser.add_argument(
            "--write-mapping-template",
            default=None,
            help="Write a backend-specific mapping template to this JSON/YAML path and exit.",
        )
        parser.add_argument(
            "--template-keypoints",
            default="",
            help="Comma-separated keypoints used when generating a mapping template without reading pose input.",
        )
        parser.add_argument(
            "--default-z",
            type=float,
            default=0.0,
            help="Default Z value when lifting 2D keypoints into 3D backend targets.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate mapping and emit target sites without invoking the backend runtime.",
        )
        parser.add_argument(
            "--env-factory",
            default=None,
            help="Optional dotted callable override for FlyBody environment creation.",
        )
        parser.add_argument(
            "--ik-function",
            default=None,
            help="Optional dotted callable override for FlyBody IK solving.",
        )
        parser.add_argument(
            "--ik-max-steps",
            type=int,
            default=2000,
            help="Backend IK iteration cap when the adapter supports it.",
        )
        parser.add_argument(
            "--smooth-mode",
            choices=("none", "ema", "one_euro", "kalman"),
            default="none",
            help="Temporal smoothing mode for per-keypoint trajectories before lifting/fitting.",
        )
        parser.add_argument(
            "--fps",
            type=float,
            default=30.0,
            help="Frame rate used by temporal smoothing filters.",
        )
        parser.add_argument(
            "--max-gap-frames",
            type=int,
            default=0,
            help="Linearly fill missing keypoints across gaps up to this many frames.",
        )
        parser.add_argument(
            "--min-score",
            type=float,
            default=0.0,
            help="Minimum score used by temporal smoothing gates.",
        )
        parser.add_argument(
            "--ema-alpha",
            type=float,
            default=0.7,
            help="EMA alpha when --smooth-mode ema.",
        )

    def predict(self, args: argparse.Namespace) -> int:
        from annolid.simulation import (
            SimulationRunRequest,
            build_default_output_path,
            generate_flybody_mapping_template,
            read_pose_frames,
            run_simulation_workflow,
            save_simulation_mapping_template,
        )

        pose_frames = []
        if args.input:
            pose_frames = read_pose_frames(
                args.input,
                pose_schema=args.pose_schema,
                video_name=args.video_name,
            )
        if args.write_mapping_template:
            if args.backend != "flybody":
                raise RuntimeError(
                    "Mapping template generation is currently implemented for the FlyBody backend only."
                )
            keypoints = []
            if args.pose_schema:
                from annolid.annotation.pose_schema import PoseSchema

                keypoints = list(PoseSchema.load(args.pose_schema).keypoints)
            if not keypoints and args.template_keypoints:
                keypoints = [
                    str(part).strip()
                    for part in str(args.template_keypoints).split(",")
                    if str(part).strip()
                ]
            if not keypoints and pose_frames:
                seen = set()
                for frame in pose_frames:
                    for label in frame.points:
                        if label not in seen:
                            seen.add(label)
                            keypoints.append(label)
            if not keypoints:
                raise RuntimeError(
                    "Cannot generate a mapping template without keypoints. "
                    "Provide --pose-schema, --template-keypoints, or --input."
                )
            template = generate_flybody_mapping_template(keypoints=keypoints)
            save_simulation_mapping_template(template, args.write_mapping_template)
            return 0

        if not args.mapping or not args.out_ndjson or not args.input:
            raise RuntimeError(
                "--input, --mapping, and --out-ndjson are required unless generating a mapping template."
            )

        request = SimulationRunRequest(
            backend=str(args.backend),
            input_path=str(args.input),
            mapping_path=str(args.mapping),
            out_ndjson=str(
                args.out_ndjson
                or build_default_output_path(args.input, backend=str(args.backend))
            ),
            pose_schema=args.pose_schema,
            depth_ndjson=args.depth_ndjson,
            video_name=args.video_name,
            default_z=float(args.default_z),
            dry_run=bool(args.dry_run),
            env_factory=args.env_factory,
            ik_function=args.ik_function,
            ik_max_steps=int(args.ik_max_steps),
            smooth_mode=str(args.smooth_mode),
            fps=float(args.fps),
            max_gap_frames=int(args.max_gap_frames),
            min_score=float(args.min_score),
            ema_alpha=float(args.ema_alpha),
        )
        run_simulation_workflow(request)
        return 0
