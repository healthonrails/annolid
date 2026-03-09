from __future__ import annotations

import argparse

from annolid.engine.models.simulation_runner import SimulationRunnerPlugin
from annolid.engine.registry import ModelPluginBase, register_model


@register_model
class FlyBodyPlugin(ModelPluginBase):
    name = "flybody"
    description = "Fit Annolid keypoints into FlyBody-compatible site targets and optional IK outputs."
    predict_examples = (
        "annolid-run predict flybody --help-model",
        "annolid-run predict flybody --pose-schema pose_schema.json --write-mapping-template flybody.yaml",
        "annolid-run predict flybody --input pose.ndjson --mapping flybody.yaml --out-ndjson flybody.ndjson --dry-run",
        "annolid-run predict flybody --input pose.ndjson --depth-ndjson depth.ndjson --mapping flybody.yaml --out-ndjson flybody.ndjson --smooth-mode ema --max-gap-frames 2 --ik-max-steps 4000",
    )
    predict_help_sections = (
        (
            "Required inputs",
            (
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
            "FlyBody controls",
            (
                "--dry-run",
                "--default-z",
                "--env-factory",
                "--ik-function",
                "--ik-max-steps",
                "--smooth-mode",
                "--max-gap-frames",
                "--fps",
            ),
        ),
    )

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        SimulationRunnerPlugin().add_predict_args(parser)
        for action in getattr(parser, "_actions", []):
            if getattr(action, "dest", "") == "backend":
                action.default = "flybody"
                action.required = False
                action.help = "Fixed to 'flybody' for this plugin."

    def predict(self, args: argparse.Namespace) -> int:
        args.backend = "flybody"
        return int(SimulationRunnerPlugin().predict(args))
