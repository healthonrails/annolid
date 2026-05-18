from __future__ import annotations

import argparse

from annolid.engine.registry import ModelPluginBase, register_model


@register_model
class PolygonClassifierPlugin(ModelPluginBase):
    name = "polygon_classifier"
    description = (
        "Polygon behavior classifier from Annolid tracking annotations "
        "(TCN or ConvNet)."
    )
    train_help_sections = (
        (
            "Required inputs",
            (
                "--train-video",
                "--test-video",
                "--train-csv",
                "--test-csv",
                "--run-config",
            ),
        ),
        (
            "Outputs and run location",
            (
                "--csv-output-dir",
                "--output-dir",
                "--run-name",
            ),
        ),
        (
            "Model and runtime",
            (
                "--model-type",
                "--device",
                "--num-points",
            ),
        ),
        (
            "Training controls",
            (
                "--num-epochs",
                "--batch-size",
                "--learning-rate",
                "--window-size",
                "--hidden-dim",
                "--num-residual-blocks",
                "--dropout",
            ),
        ),
    )
    train_examples = (
        "annolid-run train polygon_classifier --help-model",
        "annolid-run train polygon_classifier --model-type tcn --train-csv train_polygon_points.csv --test-csv test_polygon_points.csv",
        "annolid-run train polygon_classifier --train-video /path/to/train_video /path/to/train_labels.csv --test-video /path/to/test_video /path/to/test_labels.csv",
    )

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        from annolid.behavior.polygon_classifier_cli import add_train_args

        parser.add_argument(
            "--run-config",
            default=None,
            help="Path to run config YAML. CLI flags override YAML values.",
        )
        add_train_args(parser)

    def train(self, args: argparse.Namespace) -> int:
        from annolid.behavior.polygon_classifier_cli import run_train_command

        return int(run_train_command(args))
